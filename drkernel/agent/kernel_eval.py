# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""KernelGym caller for the rollout-side multi-turn agent loop.

Why not reuse `recipe.drkernel.rewards.reward_client.KernelRewardClient`?
- That client lives in the `RewardLoopWorker` Ray actors and spawns its
  own `_HybridHttpWorker` actor pool to fan out batched evals across
  many trajectories. Calling it from the rollout-side agent loop would
  mean either re-spawning the same actor pool inside every
  `AgentLoopWorker` (wasteful, creates ownership puzzles) or piggy-
  backing on the reward-side pool (cross-actor ref-passing complexity
  we already paid for once).
- This caller does the per-turn job in httpx: POST /evaluate, poll
  /status, fetch /results. One trajectory at a time, in the agent
  loop's own event loop. No Ray actors involved.

The reward path remains the canonical training-signal computation; this
is just the "feedback message between turns" call. Both ultimately hit
the same KernelGym server with the same payload schema.

Faithful-to-original payload: matches the reward-path defaults
(`reward_client.py:621-636`) so the per-turn feedback the model sees
contains the same compile/correctness/speedup/profiling fields the
training reward is computed against. The original DR.Kernel pipeline
(`vllm_async_engine_multi_iter.py:1965-1989`) JSON-dumps the full
`env_state` (= reward manager `results` dict) into the user-feedback
template; `format_feedback(..., mode="full_json")` mirrors that.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import time
from typing import Any, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


_DEFAULT_TIMEOUT = httpx.Timeout(connect=60.0, read=600.0, write=60.0, pool=5.0)


def _submit_backoff(attempt: int, base: int = 2, cap: int = 30) -> float:
    """Exponential backoff with a cap. Matches `reward_client.py:51-52`."""
    return min(base**attempt, cap)


async def evaluate_kernel(
    *,
    server_url: str,
    reference_code: str,
    kernel_code: str,
    entry_point: str = "Model",
    task_timeout: int = 600,
    client_timeout: int = 1200,
    poll_interval: float = 1.0,
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    enable_profiling: bool = True,
    verbose_errors: bool = True,
    detect_decoy_kernel: bool = True,
    toolkit: str = "kernelbench",
    is_valid: bool = False,
    summarizer: Optional[Any] = None,
    extra_payload: Optional[dict[str, Any]] = None,
    submit_max_retries: int = 3,
) -> dict[str, Any]:
    """Evaluate a kernel candidate against the reference and return the
    KernelGym result dict.

    Defaults match `recipe.drkernel.rewards.reward_client.KernelRewardClient`
    so the rollout-side feedback uses the same payload as the training
    reward path. In particular `num_perf_trials >= 1` (server-side
    `EvaluationRequest` validator: `Field(default=100, ge=1, le=1000)`)
    — sending 0 used to trigger an HTTP 400 from the server's
    `validation_exception_handler`, masking every per-turn evaluation.

    Result schema (matches what `_HybridHttpWorker.submit_and_poll` returns):
        {"status": "completed"|"failed"|"timeout"|"cancelled", ...}

    Failure modes (non-`completed` status, network errors, JSON errors)
    return a dict with `status="failed"` and an `error_message` so the
    caller can format a feedback string instead of raising. On HTTP
    error responses, the server-provided error body is included so
    schema/validation problems surface in the feedback rather than as
    an opaque "HTTP NNN" string.
    """
    task_id = f"agent_loop_{uuid4().hex[:12]}"
    payload: dict[str, Any] = {
        "task_id": task_id,
        "reference_code": reference_code,
        "kernel_code": kernel_code,
        "backend": "triton",
        "entry_point": entry_point,
        "timeout": int(task_timeout),
        "priority": "normal",
        "verbose_errors": bool(verbose_errors),
        "enable_profiling": bool(enable_profiling),
        "detect_decoy_kernel": bool(detect_decoy_kernel),
        # KernelGym evaluation toolkit. Must be sent explicitly: the server
        # pre-fills toolkit="kernelbench" for /evaluate and never falls back
        # to its DEFAULT_TOOLKIT env. "sandbox_v3" adds AST decoy validation.
        "toolkit": str(toolkit or "kernelbench"),
        "num_correct_trials": int(num_correct_trials),
        "num_perf_trials": int(num_perf_trials),
        "is_valid": bool(is_valid),
    }
    if extra_payload:
        payload.update(extra_payload)

    start_ts = time.time()
    try:
        async with httpx.AsyncClient(
            timeout=_DEFAULT_TIMEOUT,
            limits=httpx.Limits(max_keepalive_connections=4, max_connections=8, keepalive_expiry=30.0),
            headers={"Content-Type": "application/json"},
        ) as client:
            # POST /evaluate with bounded retry on transient failures
            # (429/503 backpressure + httpx.TimeoutException/ConnectError).
            # Mirrors `reward_client.py::_HybridHttpWorker.submit_and_poll`
            # lines 73-100. Without this a transient KernelGym hiccup
            # zeros out the turn's reward.
            resp = None
            last_error = None
            for attempt in range(submit_max_retries):
                try:
                    resp = await client.post(f"{server_url}/evaluate", json=payload)
                    if resp.status_code == 200:
                        break
                    if resp.status_code in (429, 503):
                        # Server-side backpressure — back off and retry.
                        backoff = _submit_backoff(attempt, base=2 if resp.status_code == 429 else 5)
                        await asyncio.sleep(backoff)
                        last_error = f"HTTP {resp.status_code} (will retry)"
                        continue
                    # Other non-200: don't retry, return error.
                    body_excerpt = _extract_error_body(resp)
                    return _maybe_summarize(
                        {
                            "status": "failed",
                            "error_message": f"HTTP {resp.status_code} on /evaluate: {body_excerpt}",
                        },
                        summarizer,
                    )
                except (httpx.TimeoutException, httpx.ConnectError) as exc:
                    last_error = str(exc)
                    if attempt < submit_max_retries - 1:
                        await asyncio.sleep(_submit_backoff(attempt))
                        continue
                    return _maybe_summarize(
                        {
                            "status": "failed",
                            "error_message": f"submit_and_poll transient error: {exc}",
                        },
                        summarizer,
                    )
            if resp is None or resp.status_code != 200:
                return _maybe_summarize(
                    {
                        "status": "failed",
                        "error_message": f"HTTP {resp.status_code if resp else 'no_response'} "
                        f"on /evaluate after {submit_max_retries} attempts: {last_error}",
                    },
                    summarizer,
                )

            while time.time() - start_ts < client_timeout:
                try:
                    s = await client.get(f"{server_url}/status/{task_id}")
                    if s.status_code == 200:
                        data = s.json()
                        status = data.get("status", "unknown")
                        if status in ("completed", "failed", "timeout", "cancelled"):
                            if status == "completed":
                                r = await client.get(f"{server_url}/results/{task_id}")
                                if r.status_code == 200:
                                    result = r.json()
                                    result["status"] = status
                                    return _maybe_summarize(result, summarizer)
                                return _maybe_summarize(
                                    {
                                        "status": "failed",
                                        "error_message": f"HTTP {r.status_code} on /results: {_extract_error_body(r)}",
                                    },
                                    summarizer,
                                )
                            return _maybe_summarize(
                                {
                                    "status": status,
                                    "error_message": data.get("error_message", f"Task {status}"),
                                },
                                summarizer,
                            )
                except Exception as exc:
                    logger.debug("kernel_eval poll error: %s", exc)
                await asyncio.sleep(poll_interval)

            return _maybe_summarize(
                {"status": "timeout", "error_message": f"Client timeout after {client_timeout}s"},
                summarizer,
            )
    except Exception as exc:
        return _maybe_summarize(
            {"status": "failed", "error_message": f"kernel_eval exception: {exc}"},
            summarizer,
        )


def _maybe_summarize(result: dict[str, Any], summarizer: Optional[Any]) -> dict[str, Any]:
    """Apply the reward summarizer (if provided) so the dict that goes
    into `{feedback}` matches the original DR.Kernel `env_state`
    (raw KernelGym fields + `_merge_reward_result(reward_func(raw))`).

    Errors are also routed through the summarizer because the original
    reward funcs (`calculate_reward_speedup`, `calculate_reward_weighted`)
    handle non-`completed` status and return a dict with `reward`,
    `score`, etc. set to the configured penalty — preserving that
    means the model sees the same penalty-shaped feedback the original
    pipeline shows.
    """
    if summarizer is None:
        return result
    try:
        return summarizer.summarize(result)
    except Exception as exc:
        logger.warning("[kernel_eval] summarizer.summarize failed (%s); returning raw result", exc)
        return result


def _extract_error_body(resp: httpx.Response) -> str:
    try:
        body = resp.json()
        msg = body.get("message") or body.get("detail") or body
        return json.dumps(msg, ensure_ascii=False)[:512]
    except Exception:
        try:
            return resp.text[:512]
        except Exception:
            return ""


# Keys stripped from the model-facing feedback dict in `drop_keys` mode.
# The same dict is still used elsewhere (metrics, reward path) — only the
# string the model sees is trimmed.
_FEEDBACK_DROP_KEYS = frozenset(
    {
        "metadata",
        "num_custom_kernel",
        "num_total_kernels",
        "custom_kernel_cuda_time_in_profiling_us",
        "total_kernel_run_time_in_profiling_us",
        "task_id",
        "submitted_at",
        "completed_at",
        "processing_time",
        "profiling",
    }
)


def format_feedback(
    result: dict[str, Any],
    *,
    mode: str = "full_json",
    drop_keys: Optional[Any] = None,
) -> str:
    """Render a KernelGym result dict into the feedback string that
    fills `{feedback}` in the per-turn user prompt.

    Modes:
      - "full_json" (default, faithful to original): JSON-dump the full
        result dict the same way the original
        `vllm_async_engine_multi_iter.py:1968` does
        (`json.dumps(env_state, ensure_ascii=False, indent=2)`).
      - "drop_keys": same as `full_json`, but strip a configured key
        list (verbose IDs/timestamps/coverage internals that bloat the
        prompt without informing the next turn). The list is taken from
        the `drop_keys` argument if non-empty, else from
        `_FEEDBACK_DROP_KEYS`.
      - "summary": legacy compact key:value form (status, compiled,
        correctness, [speedup], [error]) — keeps the option around for
        ablations but is NOT what the original DR.Kernel pipeline does.
      - "coach": natural-language, model-facing rendering. Replaces the
        snake_case JSON with plain-English sections (verdict, performance,
        per-operation NPU device-time breakdown tagged Triton-vs-PyTorch,
        coverage, next step) so a small (8B) model can act on the multi-operational
        signal: which ops it Tritonized vs which expensive ops are still on
        PyTorch. See `_format_coach`. (The per-turn wrapper template in
        `config/prompt_config/multi_turn_kernel.yaml` is mode-agnostic and
        unchanged; coach output renders inside it as-is.)
    """
    if mode == "summary":
        return _format_summary(result)
    if mode == "coach":
        return _format_coach(result)
    if mode == "drop_keys" and isinstance(result, dict):
        keys_to_drop = frozenset(drop_keys) if drop_keys else _FEEDBACK_DROP_KEYS
        feedback_dict = {k: v for k, v in result.items() if k not in keys_to_drop}
    else:
        feedback_dict = result
    return _dump_json(feedback_dict, result)


def _dump_json(feedback_dict: Any, result: dict[str, Any]) -> str:
    """JSON-dump the feedback dict (the full_json / drop_keys paths, and the
    coach-mode fallback for not-yet-specced branches). Falls back to the
    compact summary if the dict is not JSON-serializable."""
    try:
        return json.dumps(feedback_dict, ensure_ascii=False, indent=2, default=str)
    except Exception as exc:
        logger.warning("[kernel_eval] json.dumps failed (%s); falling back to summary", exc)
        return _format_summary(result)


def _format_summary(result: dict[str, Any]) -> str:
    status = result.get("status", "unknown")
    if status == "completed":
        compiled = result.get("compiled", False)
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", None)
        err = result.get("error", "") or result.get("error_message", "") or ""
        parts = [
            "status: completed",
            f"compiled: {compiled}",
            f"correctness: {correctness}",
        ]
        if speedup is not None:
            parts.append(f"speedup: {speedup}")
        if err:
            parts.append(f"error: {err}")
        return "\n".join(parts)
    err = result.get("error_message", "") or result.get("error", "") or ""
    return f"status: {status}\nerror: {err}"


# ---------------------------------------------------------------------------
# "coach" feedback mode — natural-language, model-facing rendering.
#
# Goal: a smaller (8B) model struggles to act on the raw snake_case JSON
# (verbose IDs/timestamps, nested profiling, cryptic keys like
# `custom_kernel_cuda_time_in_profiling_us`). The coach rendering turns the
# result into plain-English sections with an explicit Triton-vs-PyTorch
# per-operation time breakdown, so the multi-operational signal ("you only
# Tritonized the cheap ops; the expensive ones are still PyTorch") is handed
# to the model directly rather than left for it to infer.
#
# Spec status: the CORRECT-kernel (`_coach_correct`), compile/correctness-fail
# (`_coach_error`), and decoy / reward-hacking (`_coach_decoy`) branches are
# fully rendered. The non-completed (failed/timeout/cancelled) branch still
# falls back to the full JSON dump (`_dump_json`) until its template is
# specced — see the TODO in `_format_coach`.
# ---------------------------------------------------------------------------


def _format_coach(result: dict[str, Any]) -> str:
    """Render a KernelGym result as natural-language coaching feedback.

    See the module-level "coach" comment and the `format_feedback` docstring.
    """
    if not isinstance(result, dict):
        return _dump_json(result, result if isinstance(result, dict) else {})

    status = str(result.get("status", "unknown"))
    if status != "completed":
        # TODO(coach): spec the non-completed (failed/timeout/cancelled) template.
        return _dump_json(result, result)
    if result.get("decoy_kernel", False):
        return _coach_decoy(result)
    if not bool(result.get("compiled", False)):
        return _coach_error(
            result,
            "RESULT: Your kernel FAILED TO COMPILE.",
        )
    if not bool(result.get("correctness", False)):
        return _coach_error(
            result,
            "RESULT: Your kernel COMPILED but FAILED the correctness check. Its output does not match the reference.",
        )
    return _coach_correct(result)


def _coach_correct(result: dict[str, Any]) -> str:
    """Render the correct-kernel coaching feedback (verdict, performance,
    per-op NPU device-time breakdown, coverage, next step)."""
    metadata = result.get("metadata") or {}

    speedup = _coach_to_float(result.get("speedup"))
    ref_rt = _coach_to_float(result.get("reference_runtime"))
    ker_rt = _coach_to_float(result.get("kernel_runtime"))

    # --- verdict ---
    trials_raw = metadata.get("correctness_trials")
    if trials_raw:
        trials = str(trials_raw).strip().lstrip("(").rstrip(")").replace(" ", "")
        trials_phrase = f"passed {trials} correctness trials"
    else:
        nc = result.get("num_correct_trials") or metadata.get("num_correct_trials")
        trials_phrase = f"passed all {nc} correctness trials" if nc else "passed the correctness check"

    if speedup is not None and speedup > 1.0:
        verdict = f"RESULT: CORRECT — {trials_phrase}. Your kernel is FASTER than the reference."
    elif speedup is not None and 0.0 < speedup < 1.0:
        verdict = f"RESULT: CORRECT — {trials_phrase}. But your kernel is SLOWER than the reference."
    else:
        verdict = f"RESULT: CORRECT — {trials_phrase}."

    lines = [
        verdict,
        "",
        "PERFORMANCE",
        f"  Reference runtime: {_coach_fmt_ms(ref_rt)}",
        f"  Your runtime:      {_coach_fmt_ms(ker_rt)}",
        f"  Speedup:           {_coach_speedup_phrase(speedup)}",
        f"  Reward:            {_coach_fmt_score(result.get('reward'))}",
        f"  Score:             {_coach_fmt_score(result.get('score'))}",
    ]

    # --- per-operation NPU device-time breakdown ---
    # and counted harness/self-test ops.
    rows = _coach_profile_rows(result, metadata)
    custom_names = _coach_custom_kernel_names(metadata)
    if rows:
        lines.append("")
        lines.append(
            "PER-OPERATION NPU TIME (device time per forward pass of YOUR kernel; the reference model is not listed)"
        )
        namew = max((len(name) for name, _ in rows), default=0)
        for name, dt in rows:
            tag = "Triton" if _coach_is_custom(name, custom_names) else "PyTorch"
            lines.append(f"  {name.ljust(namew)}  {dt:>8.2f} us   [{tag}]")

    # --- coverage (device time only, from the rows above) ---
    custom_us = sum(t for n, t in rows if _coach_is_custom(n, custom_names))
    total_us = sum(t for _, t in rows)
    num_custom = sum(1 for n, _ in rows if _coach_is_custom(n, custom_names))
    num_total = len(rows)

    pct_time = (100.0 * custom_us / total_us) if total_us > 0 else None
    if rows:
        lines.append("")
        lines.append("COVERAGE")
        pct_ops = 100.0 * num_custom / num_total
        lines.append(f"  Ops in Triton:           {num_custom} of {num_total}   ({pct_ops:.2f}%)")
        if pct_time is not None:
            lines.append(
                f"  Custom-kernel NPU time:  {pct_time:.2f}%   (your Triton kernels = "
                f"{custom_us:.2f} us of {total_us:.2f} us total NPU device time)"
            )

    # --- next step ---
    torch_ops = [(n, t) for n, t in rows if not _coach_is_custom(n, custom_names)]
    largest_torch = max(torch_ops, key=lambda t: t[1]) if torch_ops else None

    pct_time_str = f"{pct_time:.2f}%" if pct_time is not None else "a small fraction"
    lines.append("")
    lines.append("NEXT STEP")
    if speedup is not None and speedup > 1.0:
        nxt = "  Your kernel already beats the reference."
        if pct_time is not None:
            nxt += f" Your custom Triton kernels cover {pct_time_str} of its NPU device time."
        if largest_torch is not None:
            nxt += (
                f" To go faster still, the largest operation still on PyTorch is "
                f"{largest_torch[0]} ({largest_torch[1]:.2f} us) — implement it in Triton "
                f"and/or fuse the remaining expensive ops."
            )
        else:
            nxt += " To go faster still, optimize your kernel's tiling/memory access — no PyTorch ops remain."
    else:
        if pct_time is not None and pct_time >= 50.0:
            nxt = (
                f"  Your Triton kernels already account for {pct_time_str} of the NPU device time, "
                f"but the kernel is slower than the reference — the bottleneck is your kernel itself. "
                f"Optimize its tiling, block sizes and memory access pattern."
            )
        else:
            nxt = (
                f"  Only {pct_time_str} of NPU device time is in your custom Triton kernels — "
                f"the rest runs as PyTorch operators."
            )
            if largest_torch is not None:
                nxt += f" The largest operation still on PyTorch is {largest_torch[0]} ({largest_torch[1]:.2f} us)."
            nxt += (
                " Implement the most expensive PyTorch ops above in Triton (and/or fuse adjacent ops) "
                "to raise coverage and beat the reference runtime."
            )
    lines.append(nxt)

    return "\n".join(lines)


def _coach_profile_rows(result: dict[str, Any], metadata: dict[str, Any]):
    """Build the cleaned (name, device_time_us) rows for `_coach_correct`."""
    raw = []
    profiling = metadata.get("profiling") or result.get("profiling") or {}
    kernels = profiling.get("kernels") if isinstance(profiling, dict) else None
    if kernels:
        max_count = 0
        for k in kernels:
            try:
                max_count = max(max_count, int(k.get("count") or 0))
            except (TypeError, ValueError):
                pass
        for k in kernels:
            try:
                cnt = int(k.get("count") or 0)
            except (TypeError, ValueError):
                cnt = 0
            if max_count >= 5 and cnt == 1:
                continue  # harness one-shot (profiler self-test), not a model op
            raw.append((str(k.get("name", "")), _coach_to_float(k.get("device_time_us")) or 0.0))
    else:
        ops = metadata.get("kernels") or result.get("kernels")
        if isinstance(ops, dict):
            raw = [(str(n), _coach_to_float(v) or 0.0) for n, v in ops.items()]

    if any(n.lower().startswith("aclnn") for n, _ in raw):
        raw = [(n, t) for n, t in raw if not n.startswith("aten::")]
    raw.sort(key=lambda r: -r[1])
    return raw


def _coach_error(result: dict[str, Any], headline: str) -> str:
    """Render the compile-fail / correctness-fail coaching feedback.

    Design constraints (per spec):
      - No interpretation of what the error *means* — we only state the
        factual verdict (compiled / correctness, from the result flags).
      - No over-summarization — the traceback is kept in full.
      - Preserve where execution failed: absolute harness file paths are
        shortened to their basename, but line numbers and the offending
        code lines are kept verbatim.
      - NPU-native: no GPU/CUDA wording.
    """
    metadata = result.get("metadata") or {}
    error_code = result.get("error_code") or metadata.get("error_code")
    error_type = metadata.get("runtime_error_name") or result.get("runtime_error_name")

    # Prefer metadata.runtime_error (the clean traceback) over error_message
    # (which prefixes it with "Kernel execution failed: ").
    traceback_text = (
        metadata.get("runtime_error")
        or result.get("error_message")
        or result.get("error")
        or metadata.get("error_message")
        or ""
    )
    traceback_text = str(traceback_text).strip()
    traceback_text = _coach_strip_paths(traceback_text) if traceback_text else "(no error detail provided)"

    header_bits = []
    if error_code:
        header_bits.append(f"code: {error_code}")
    if error_type:
        header_bits.append(f"type: {error_type}")
    error_header = "ERROR" + (f" ({', '.join(header_bits)})" if header_bits else "")

    speedup = _coach_to_float(result.get("speedup"))
    speedup_str = f"{speedup:.2f}x" if speedup is not None else "n/a"
    metrics_line = (
        f"  Speedup: {speedup_str}   "
        f"Reward: {_coach_fmt_score(result.get('reward'))}   "
        f"Score: {_coach_fmt_score(result.get('score'))}"
    )

    return "\n".join(
        [
            headline,
            "",
            "METRICS",
            metrics_line,
            "",
            error_header,
            traceback_text,
            "",
            "NEXT STEP",
            "  Use the traceback above to locate and fix the failure, then return an improved Triton ModelNew.",
        ]
    )


# ---------------------------------------------------------------------------
# Decoy / reward-hacking rendering.

# ---------------------------------------------------------------------------

# regression_type -> plain-English description of how the kernel degenerated.
# Mirrors the Type 1/2/3 taxonomy in `validate_triton_impl.py`.
_DECOY_TYPE_DESC = {
    1: "There is NO real Triton kernel — the whole forward() ran as PyTorch "
    "operators (either no @triton.jit kernel at all, or one that never uses "
    "the tl.* API).",
    2: "A @triton.jit kernel is DEFINED but forward() never launches it, so the "
    "actual computation still ran as PyTorch.",
    3: "forward() launches a Triton kernel but STILL does real computation with "
    "PyTorch ops — only buffer allocation and shape/layout ops are allowed "
    "outside the kernel.",
}

# regression_type -> the concrete fix to put in NEXT STEP.
_DECOY_TYPE_NEXT = {
    1: "Implement the core computation inside a @triton.jit kernel using "
    "tl.load / tl.store, and launch it from forward() via kernel[grid](...).",
    2: "Launch your existing @triton.jit kernel from forward() via "
    "kernel[grid](...) so the computation actually runs on the NPU.",
    3: "Move every PyTorch operation flagged above into your Triton kernel. "
    "forward() may keep only the allocation/shape ops in ALLOWED above; all "
    "arithmetic must happen inside the kernel.",
}

# sandbox_v4 overrides. v4 relaxes ONLY the Type-3 policy: besides allocation and
# shape/layout ops, the vendor/extern ops (matmul/conv/attention/linalg/fft) may
# stay in PyTorch outside the kernel (the TorchInductor extern_kernels pattern).
# Type 1/2 describe the ABSENCE of a real Triton kernel, which v4 still requires,
# so they are intentionally NOT overridden and inherit the base text above. These
# maps are spliced over the base ones in `_coach_decoy`, but only when the result
# metadata carries the v4 extern allowlist (otherwise the v3 wording stands).
_DECOY_TYPE_DESC_V4 = {
    3: "forward() launches a Triton kernel but STILL does real computation in "
    "PyTorch with ops that are not allowed outside the kernel. Only the ops "
    "listed in ALLOWED below may stay in PyTorch — every other operation must "
    "run inside the kernel.",
}

_DECOY_TYPE_NEXT_V4 = {
    3: "Move every PyTorch operation flagged above into your Triton kernel. "
    "forward() may keep the ops "
    "listed in ALLOWED above. "
    "All OTHER arithmetic must happen "
    "inside the kernel.",
}

# Chinese call-tokens emitted by `check_forbidden_torch_ops` for control-flow /
# multi-launch violations, mapped to English. ASCII call tokens (e.g.
# "torch.matmul", "@", "F.relu", "self.fc(...)") pass through unchanged.
_DECOY_CALL_TRANSLATE = {
    "for 循环": "Python for-loop",
    "while 循环": "Python while-loop",
}

# The torch.* calls the validator permits inside forward() — buffer/tensor
# allocation only. Literal mirror of `ALLOWED_TORCH_FUNCS` in
# `validate_triton_impl.py`; kernel_eval runs on the rollout side and does NOT
# import the kernelGym server package, so keep this in sync by hand if the
# validator's allowlist changes.
_DECOY_ALLOWED_TORCH_FUNCS = (
    "empty",
    "empty_like",
    "empty_strided",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    "tensor",
    "arange",
    "linspace",
    "as_tensor",
)

# Representative shape/layout tensor methods the validator permits (no compute).
# A readable subset of `ALLOWED_TENSOR_METHODS` in `validate_triton_impl.py`.
_DECOY_ALLOWED_TENSOR_METHODS = (
    "view",
    "reshape",
    "permute",
    "transpose",
    "contiguous",
    "to",
    "expand",
    "squeeze",
    "unsqueeze",
    "flatten",
)

# Cap on violations listed in the feedback, so a pathological forward() can't
# blow up the prompt. Truncation is announced rather than silent.
_DECOY_MAX_VIOLATIONS = 12


def _coach_to_int(value: Any) -> Optional[int]:
    """Best-effort int conversion; returns None for None/non-numeric."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coach_decoy_call(call: Any) -> str:
    """Render a violation's `call` token in English. Translates the handful of
    Chinese control-flow tokens; passes ASCII op names through verbatim."""
    text = str(call).strip()
    if text in _DECOY_CALL_TRANSLATE:
        return _DECOY_CALL_TRANSLATE[text]
    if text.startswith("kernel 启动"):  # validator Rule B token: "kernel 启动 N 次"
        # The validator counts every `kernel[grid](...)` call in forward() —
        # the same kernel re-launched in a host-side loop AND sequential
        # launches of distinct kernels both trip this rule.
        m = re.search(r"(\d+)", text)
        times = f"{m.group(1)} Triton kernel launches" if m else "multiple Triton kernel launches"
        return (
            f"{times} in forward(), but only ONE launch is allowed — whether one "
            "kernel is re-launched in a host-side loop or several different kernels "
            "run in sequence, the orchestration stays on the host. Fuse the whole "
            "computation into a single @triton.jit kernel launched once."
        )
    return text


def _coach_decoy(result: dict[str, Any]) -> str:
    """Render the decoy / reward-hacking coaching feedback.

    See the module-level "Decoy / reward-hacking rendering" comment for the
    metadata contract and fallback behavior.
    """
    metadata = result.get("metadata") or {}

    headline = (
        "RESULT: REWARD HACKING DETECTED — your kernel was rejected as a DECOY. "
        "It falls back to PyTorch operators instead of computing in Triton, so it "
        "earns the maximum penalty regardless of correctness or speed."
    )
    metrics_line = (
        f"  Reward: {_coach_fmt_score(result.get('reward'))}   "
        f"Score: {_coach_fmt_score(result.get('score'))}   "
        "(decoy penalty — speed and correctness are not counted)"
    )

    rtype = _coach_to_int(result.get("decoy_regression_type", metadata.get("decoy_regression_type")))

    # sandbox_v4 detection. The server validator ships its extern allowlist in
    # metadata (`decoy_allowed_extern_funcs`); when present, the policy permits the
    # vendor/extern ops (matmul/conv/attention/linalg/fft) to stay in PyTorch
    # outside the kernel, so the WHY / NEXT-STEP wording must NOT assert the v3
    # "only allocation/shape ops are allowed" rule. Splice the v4 overrides over
    # the base maps (Type 1/2 inherit the base). Absent for sandbox_v3 -> base text.
    _extern_funcs = result.get("decoy_allowed_extern_funcs") or metadata.get("decoy_allowed_extern_funcs")
    is_v4 = bool(_extern_funcs)
    desc_map = {**_DECOY_TYPE_DESC, **_DECOY_TYPE_DESC_V4} if is_v4 else _DECOY_TYPE_DESC
    next_map = {**_DECOY_TYPE_NEXT, **_DECOY_TYPE_NEXT_V4} if is_v4 else _DECOY_TYPE_NEXT

    why_lines = ["WHY IT WAS FLAGGED"]
    if rtype in desc_map:
        why_lines.append(f" {desc_map[rtype]}")
    else:
        why_lines.append(
            "  No genuine Triton computation was detected — the core work runs as "
            "PyTorch operators rather than inside a @triton.jit kernel."
        )

    violations = result.get("decoy_violations", metadata.get("decoy_violations"))
    viol_lines = []
    if isinstance(violations, (list, tuple)) and violations:
        viol_lines.append("")
        viol_lines.append("PROHIBITED PYTORCH OPS IN forward() (these OPS should be implemented as Triton kernels)")
        for v in violations[:_DECOY_MAX_VIOLATIONS]:
            if not isinstance(v, dict):
                continue
            line = v.get("line")
            call = _coach_decoy_call(v.get("call", ""))
            loc = f"line {line}: " if line is not None else ""
            viol_lines.append(f"  {loc}{call}")
        extra = len(violations) - _DECOY_MAX_VIOLATIONS
        if extra > 0:
            viol_lines.append(f"  ... and {extra} more.")

    allowed_lines = [
        "",
        "LIST OF ALLOWED OPS IN forward() (everything else must be implemented as @triton.jit kernel)",
        "  buffer/tensor allocation: " + ", ".join(f"torch.{f}" for f in _DECOY_ALLOWED_TORCH_FUNCS),
        "  shape/layout only (no math): " + ", ".join(f".{m}" for m in _DECOY_ALLOWED_TENSOR_METHODS),
    ]

    # Vendor/extern ops (sandbox_v4): appended to ALLOWED only when the server's
    # validator shipped the extern allowlist in metadata (detected as `is_v4`
    # above, reusing the same `_extern_funcs`). These are the hard ops
    # torch.compile itself delegates to vendor libs (extern_kernels.* / fused
    # attention / cuSOLVER / cuFFT); the model MAY use them ALONGSIDE a genuine
    # Triton kernel, either called functionally (e.g. F.conv2d(x, self.conv.weight,
    # ...)) OR as a whitelisted layer submodule (self.conv(x) — the validator now
    # accepts self.<name>(x) when <name> is bound to a Conv/Linear/Attention layer
    # in __init__; see `decoy_allowed_module_types`). Absent for sandbox_v3 -> this
    # block is skipped.
    if is_v4:
        _functional_funcs = (
            result.get("decoy_allowed_functional_funcs") or metadata.get("decoy_allowed_functional_funcs") or []
        )
        _torch_namespaces = (
            result.get("decoy_allowed_torch_namespaces") or metadata.get("decoy_allowed_torch_namespaces") or []
        )
        _allow_matmul_op = result.get("decoy_allowed_matmul_operator")
        if _allow_matmul_op is None:
            _allow_matmul_op = metadata.get("decoy_allowed_matmul_operator")
        _module_types = result.get("decoy_allowed_module_types") or metadata.get("decoy_allowed_module_types") or []
        allowed_lines.append(
            "  extern ops — OK as the sub-op ALONGSIDE a real Triton kernel (these heavy ops may stay in PyTorch):"
        )
        allowed_lines.append("    torch.*: " + ", ".join(f"torch.{f}" for f in _extern_funcs))
        if _functional_funcs:
            allowed_lines.append("    F.*: " + ", ".join(f"F.{f}" for f in _functional_funcs))
        if _torch_namespaces:
            allowed_lines.append("    namespaces: " + ", ".join(f"{ns}.*" for ns in _torch_namespaces))
        if _allow_matmul_op:
            allowed_lines.append("    the @ (matmul) operator")
        if _module_types:
            # Hide the rarely-used Lazy* variants to keep the prompt readable.
            _disp = [t for t in _module_types if not t.startswith("Lazy")] or list(_module_types)
            allowed_lines.append(
                "    layer submodules — self.<layer>(x) is OK when <layer> is one of: "
                + ", ".join(f"nn.{t}" for t in _disp)
                + " (e.g. self.conv(x), self.conv_transpose(x); or call them "
                "functionally, F.conv2d(x, self.conv.weight, ...))"
            )

    next_lines = [
        "NEXT STEP",
        f"  {next_map[rtype]}"
        if rtype in next_map
        else "  "
        + (
            "Implement the core computations as @triton.jit kernels and launch from "
            "forward(). forward() may keep only the ops "
            "in ALLOWED above — all other arithmetic must be inside "
            "the kernel."
            if is_v4
            else "Implement the core computations as @triton.jit kernels "
            "and launch it from forward(). forward() may only "
            "allocate buffers and reshape tensors (see ALLOWED above) — no arithmetic in PyTorch."
        ),
    ]

    return "\n".join(
        [
            headline,
            "",
            "METRICS",
            metrics_line,
            "",
            *why_lines,
            *viol_lines,
            *allowed_lines,
            "",
            *next_lines,
        ]
    )


# Matches a Python-traceback ``File "<path>"`` reference. Used to shorten
# absolute harness paths to their basename while leaving the line number and
# the offending code line untouched.
_FILE_PATH_RE = re.compile(r'File "([^"]*)"')


def _coach_basename(path: str) -> str:
    """Last path component, handling both POSIX and Windows separators."""
    normalized = path.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1] if "/" in normalized else normalized


def _coach_strip_paths(text: str) -> str:
    """Shorten ``File "<abs path>"`` references in a traceback to their
    basename. Keeps line numbers, function names, and code lines verbatim."""
    return _FILE_PATH_RE.sub(lambda m: f'File "{_coach_basename(m.group(1))}"', text)


def _coach_to_float(value: Any) -> Optional[float]:
    """Best-effort float conversion; returns None for None/non-numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coach_fmt_ms(value: Optional[float]) -> str:
    """Format a runtime in milliseconds (KernelGym runtimes come from
    `event.elapsed_time()`, which returns ms). 4 significant digits."""
    if value is None:
        return "unknown"
    return f"{value:.4g} ms"


def _coach_fmt_score(value: Any) -> str:
    """Format a reward/score scalar for the feedback. 3 decimals; 'n/a' if
    missing/non-numeric."""
    f = _coach_to_float(value)
    return f"{f:.3f}" if f is not None else "n/a"


def _coach_speedup_phrase(speedup: Optional[float]) -> str:
    """Plain-English speedup line. speedup = reference_runtime / kernel_runtime
    (>1 faster, <1 slower)."""
    if speedup is None:
        return "unknown"
    if speedup <= 0.0:
        return f"{speedup:.2f}x  (no measurable speedup)"
    if speedup > 1.0:
        return f"{speedup:.2f}x  (above 1.0x = faster; about {speedup:.2f}x faster than the reference)"
    if speedup == 1.0:
        return "1.00x  (same speed as the reference)"
    slower = 1.0 / speedup
    return f"{speedup:.2f}x  (below 1.0x = slower; your kernel is about {slower:.1f}x slower than the reference)"


def _coach_custom_kernel_names(metadata: dict[str, Any]) -> set[str]:
    """Extract the set of custom (Triton) kernel names from
    `metadata.triton_profiler_matches`. That field is a list (or a
    string-repr of a list) of entries like
    ``"log_softmax_kernel grid=<...> module=temp_module file=/tmp/...py"``;
    the leading token is the kernel name. Entries whose leading token is the
    literal ``function`` (the wrapped grid lambda) are skipped. Returns an
    empty set if the field is missing/unparseable — callers then fall back to
    name-prefix heuristics in `_coach_is_custom`."""
    matches = metadata.get("triton_profiler_matches")
    names: set[str] = set()
    if not matches:
        return names
    items: Any = matches
    if isinstance(matches, str):
        try:
            items = ast.literal_eval(matches)
        except (ValueError, SyntaxError):
            items = [matches]
    if isinstance(items, (list, tuple)):
        for entry in items:
            text = str(entry).strip()
            if not text:
                continue
            token = text.split()[0]
            if token and token != "function":
                names.add(token)
    return names


def _coach_is_custom(name: str, custom_names: set[str]) -> bool:
    """Classify a profiled op as a custom Triton kernel vs a PyTorch operator.

    Primary signal: membership in `custom_names` (from triton_profiler_matches).
    Fallback when that set is empty: treat `aten::*` / `aclnn*` / namespaced
    (`::`) names as PyTorch, everything else as a custom Triton kernel."""
    if custom_names:
        return name in custom_names
    low = name.lower()
    if name.startswith("aten::") or low.startswith("aclnn") or "::" in name:
        return False
    return True
