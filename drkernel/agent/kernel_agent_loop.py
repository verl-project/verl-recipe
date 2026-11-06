# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""DR.Kernel-style multi-turn agent loop on top of verl's `AgentLoopBase`.

Behavior (per assistant turn):
  1. Generate the assistant response (verl's server_manager).
  2. Extract a code block (DR.Kernel's `KernelAgent` regexes).
  3. If an explicit ```answer``` block is present → terminal, stop.
  4. Otherwise, evaluate the extracted kernel against KernelGym.
  5. Render a "tool_response" template with the feedback and append it
     as a user message; tokenize with `response_mask=0` for those tokens.
  6. Loop back to (1) up to `max_assistant_turns` (or `max_user_turns`).

This produces the alternating-1s/0s `response_mask` pattern that TRLOO
needs to compute meaningful turn-level baselines.

Config wiring (under `actor_rollout_ref.rollout`):

    agent:
      default_agent_loop: kernel_agent
      agent_loop_config_path: pkg://recipe.drkernel.config.agent_loop_config

    multi_turn:
      enable: True
      max_assistant_turns: 3
      max_user_turns: 3
      prompt_config_path: pkg://recipe.drkernel.config.prompt_config.multi_turn_kernel
      kernel_eval:
        server_url: ${reward_model.server_url}
        task_timeout: 600
        client_timeout: 1200
        enabled: True
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Any
from uuid import uuid4

from recipe.drkernel.agent.kernel_eval import evaluate_kernel, format_feedback
from recipe.drkernel.agent.kernel_extract import extract_answer_block, extract_kernel_code
from recipe.drkernel.agent.per_turn_prompts import PerTurnPrompts
from recipe.drkernel.rewards.reward_summary import KernelRewardSummarizer

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class _State(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    EVALUATING = "evaluating"
    TERMINATED = "terminated"


class _Data:
    """Per-trajectory mutable state — mirrors `tool_agent_loop.AgentData`
    but specialized for the kernel-RL loop (no tool calls, just code
    extraction + KernelGym call)."""

    __slots__ = (
        "messages",
        "reference_code",
        "entry_point",
        "request_id",
        "metrics",
        "extra_fields",
        "prompt_ids",
        "response_ids",
        "response_mask",
        "response_logprobs",
        "assistant_turns",
        "user_turns",
        "last_kernel_code",
        "last_eval_status",
        "turn_results",
        "turn_rewards",
        "should_terminate_after_eval",
    )

    def __init__(
        self,
        messages: list,
        reference_code: str,
        entry_point: str,
        request_id: str,
    ):
        self.messages = messages
        self.reference_code = reference_code
        self.entry_point = entry_point
        self.request_id = request_id
        self.metrics: dict[str, Any] = {}
        self.extra_fields: dict[str, Any] = {}
        self.prompt_ids: list = []
        self.response_ids: list = []
        self.response_mask: list = []
        self.response_logprobs: list = []
        self.assistant_turns = 0
        self.user_turns = 0
        self.last_kernel_code = ""
        self.last_eval_status = ""
        # Per-turn evaluation results (one entry per assistant turn / per
        # response_mask=1 span). Each entry is either a KernelGym result
        # dict (post `_maybe_summarize`) or a synthetic penalty-shaped
        # dict for turns where no kernel was extractable. TRLOO consumes
        # these via the response_mask-span redistribution in
        # `DrKernelFullyAsyncTrainerImpl._fit_compute_reward`.
        self.turn_results: list = []
        self.turn_rewards: list = []
        # Set in `_handle_generating` when the assistant turn just produced
        # is the final one: `_handle_evaluating` still runs (so the last
        # turn gets a real reward) but skips appending feedback.
        self.should_terminate_after_eval: bool = False


class KernelAgentLoop(AgentLoopBase):
    """DR.Kernel multi-turn agent loop (registered as ``kernel_agent`` via
    `recipe/drkernel/config/agent_loop_config.yaml` `_target_:`)."""

    def __init__(self, *args, **kwargs):
        # Pop fields injected from `agent_loop_config.yaml` BEFORE calling
        # super().__init__ so they don't propagate as unknown kwargs.
        # `MultiTurnConfig` is a strict dataclass and refuses unknown
        # fields under `actor_rollout_ref.rollout.multi_turn`, which is
        # why these live in the agent_loop_config registry entry instead.
        prompt_config_path = kwargs.pop("prompt_config_path", None)
        kernel_eval_cfg = kwargs.pop("kernel_eval", None)

        super().__init__(*args, **kwargs)

        mt = self.rollout_config.multi_turn
        self.max_assistant_turns = mt.max_assistant_turns or 0
        self.max_user_turns = mt.max_user_turns or 0
        self.max_tool_response_length = mt.get("max_tool_response_length", 4096)
        self.tool_response_truncate_side = mt.get("tool_response_truncate_side", "middle")

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        self.prompts = PerTurnPrompts.load(prompt_config_path)

        ke = kernel_eval_cfg
        self.kernel_eval_enabled = bool(ke and ke.get("enabled", True)) if ke is not None else False

        # Resolve server_url with multiple fallbacks. The agent_loop_config
        # YAML can't `${reward_model.server_url}`-interpolate (loaded
        # out-of-tree), so we walk known post-migrate locations:
        #   1. registry entry's kernel_eval.server_url           (explicit override)
        #   2. KERNELGYM_SERVER_URL env var                       (launcher convention)
        #   3. config.reward.reward_kwargs.server_url             (native YAML, post-migrate)
        #   4. config.reward.custom_reward_function.reward_kwargs.reward_config.server_url
        #      (verl-blessed YAML, post `${reward_model}` resolve+migrate)
        server_url = ke.get("server_url", None) if ke else None
        if not server_url:
            server_url = os.environ.get("KERNELGYM_SERVER_URL", "") or None
        if not server_url:
            server_url = self._discover_server_url_from_config()
        self.server_url = server_url or None

        self.task_timeout = int(ke.get("task_timeout", 600)) if ke else 600
        self.client_timeout = int(ke.get("client_timeout", 1200)) if ke else 1200

        # Eval payload knobs — defaults match the reward path
        # (`recipe.drkernel.rewards.reward_client.KernelRewardClient`) so
        # the per-turn feedback hits the same KernelGym validators and
        # carries the same fields the model trains against. Overridable
        # via `agent_loop_config.yaml::kernel_eval`.
        self.num_correct_trials = int(ke.get("num_correct_trials", 5)) if ke else 5
        self.num_perf_trials = int(ke.get("num_perf_trials", 100)) if ke else 100
        self.enable_profiling = bool(ke.get("enable_profiling", True)) if ke else True
        self.verbose_errors = bool(ke.get("verbose_errors", True)) if ke else True
        self.detect_decoy_kernel = bool(ke.get("detect_decoy_kernel", True)) if ke else True
        # KernelGym evaluation toolkit ("kernelbench" | "sandbox_v3"). The
        # server ignores its own DEFAULT_TOOLKIT for /evaluate, so the choice
        # must be sent in the payload. `KERNELGYM_TOOLKIT` env var (launcher
        # convention) wins over yaml — same pattern as feedback_mode below.
        self.toolkit = os.environ.get("KERNELGYM_TOOLKIT", "").strip() or (
            str(ke.get("toolkit", "kernelbench")) if ke else "kernelbench"
        )
        # "full_json" mirrors `vllm_async_engine.py:1750`
        # (`json.dumps(env_state, ensure_ascii=False, indent=2)`) — the
        # original DR.Kernel `MultiTurnAsyncvLLMEngine` selected by the
        # 8b/14b runs. "drop_keys" mirrors full_json but strips a
        # configured key list. "summary" keeps the legacy compact form
        # for ablations.
        #
        # `KERNELGYM_FEEDBACK_MODE` env var (set by the launcher script)
        # wins over yaml — agent_loop_config.yaml is loaded via
        # `OmegaConf.load`, not Hydra-composed, so Hydra CLI overrides
        # can't reach these fields. Same pattern as `KERNELGYM_SERVER_URL`.
        self.feedback_mode = os.environ.get("KERNELGYM_FEEDBACK_MODE", "").strip() or (
            str(ke.get("feedback_mode", "full_json")) if ke else "full_json"
        )

        # Drop-keys list for `feedback_mode=drop_keys`:
        #   env `KERNELGYM_FEEDBACK_DROP_KEYS` (comma-separated) >
        #   yaml `kernel_eval.feedback_drop_keys` (list) >
        #   None → `format_feedback` uses its built-in default
        #   (`recipe/drkernel/agent/kernel_eval.py::_FEEDBACK_DROP_KEYS`).
        env_drop_keys = os.environ.get("KERNELGYM_FEEDBACK_DROP_KEYS", "").strip()
        if env_drop_keys:
            self.feedback_drop_keys = tuple(k.strip() for k in env_drop_keys.split(",") if k.strip())
        else:
            yaml_drop_keys = ke.get("feedback_drop_keys", None) if ke else None
            self.feedback_drop_keys = tuple(str(k) for k in yaml_drop_keys) if yaml_drop_keys else None

        # Reward-summary merge: when True, the rollout-side feedback
        # dict is run through `KernelRewardSummarizer.summarize` so the
        # `{feedback}` payload between turns is the same merged dict
        # the original engine puts into `env_state` (raw KernelGym
        # `/results` + `_merge_reward_result(reward_func(raw))`,
        # including `reward`/`score`/`num_custom_kernel`/`time_coverage`
        # / etc.). When False, only the raw `/results` JSON is used.
        #
        # When this is the per-turn training reward source (Option A
        # plan), `reward_summarizer` MUST be available — otherwise the
        # result dicts won't carry `reward`/`score` keys and TRLOO will
        # see zero per-turn rewards. We warn loudly if it's None.
        merge_reward_summary = bool(ke.get("merge_reward_summary", True)) if ke else True
        self.reward_summarizer = self._build_reward_summarizer() if merge_reward_summary else None

        # Anomaly-rerun threshold (mirrors `kernel_async.py:261-263`):
        # if a per-turn eval comes back with speedup > this bound, we
        # call evaluate_kernel a second time and keep the rerun result.
        # Catches transient measurement noise that would otherwise be
        # rewarded at the clamped upper bound. Defaults: pull from the
        # summarizer's config; fall back to a sentinel large value (no
        # rerun) if no summarizer.
        if self.reward_summarizer is not None:
            self.speedup_reward_upper_bound = float(self.reward_summarizer.speedup_reward_upper_bound)
            self.penalty_score = float(self.reward_summarizer.penalty_score)
        else:
            self.speedup_reward_upper_bound = float("inf")
            self.penalty_score = 0.0

        if self.kernel_eval_enabled and not self.server_url:
            logger.warning(
                "[KernelAgentLoop] kernel_eval.enabled=True but neither config "
                "server_url nor KERNELGYM_SERVER_URL env is set; multi-turn "
                "feedback will be skipped (single-turn behavior)."
            )
            self.kernel_eval_enabled = False

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # Reference code + entry_point come from the dataset's
        # `reward_model.ground_truth` and `extra_info.entry_point`.
        # `naive` reward manager pulls them via the same paths.
        rm = kwargs.get("reward_model") or {}
        reference_code = rm.get("ground_truth", "") if isinstance(rm, dict) else ""
        if not reference_code:
            reference_code = kwargs.get("ground_truth", "") or kwargs.get("reference_code", "")
        extra_info = kwargs.get("extra_info") or {}
        entry_point = (
            (extra_info.get("entry_point") if isinstance(extra_info, dict) else None)
            or kwargs.get("entry_point")
            or "Model"
        )

        data = _Data(
            messages=messages,
            reference_code=str(reference_code),
            entry_point=str(entry_point),
            request_id=uuid4().hex,
        )

        # State machine. PENDING -> GENERATING -> {EVALUATING, TERMINATED}
        # -> GENERATING -> ... mirrors the upstream tool_agent_loop layout.
        state = _State.PENDING
        while state != _State.TERMINATED:
            if state == _State.PENDING:
                state = await self._handle_pending(data)
            elif state == _State.GENERATING:
                state = await self._handle_generating(data, sampling_params)
            elif state == _State.EVALUATING:
                state = await self._handle_evaluating(data)
            else:
                logger.error("[KernelAgentLoop] invalid state: %s", state)
                state = _State.TERMINATED

        # Slice prompt/response from accumulated prompt_ids the same way
        # tool_agent_loop does (response is the last `len(response_mask)`
        # tokens of `prompt_ids`).
        response_ids = data.prompt_ids[-len(data.response_mask) :]
        prompt_ids = data.prompt_ids[: len(data.prompt_ids) - len(data.response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=data.response_mask[: self.response_length],
            multi_modal_data={},
            response_logprobs=(data.response_logprobs[: self.response_length] if data.response_logprobs else None),
            num_turns=data.user_turns + data.assistant_turns + 1,
            metrics=data.metrics,
            routed_experts=None,
            extra_fields=data.extra_fields,
        )
        # Surface per-turn results to the reward manager and the
        # trainer-side TRLOO redistribution. `agent_loop.py:981-995`
        # turns each key in `extra_fields` into a length-bs object array
        # in `non_tensor_batch`, so consumers can read these as
        # `data_item.non_tensor_batch["turn_results"]` etc.
        output.extra_fields.update(
            {
                "last_kernel_code": data.last_kernel_code,
                "last_eval_status": data.last_eval_status,
                "turn_results": list(data.turn_results),
                "turn_rewards": list(data.turn_rewards),
            }
        )
        return output

    async def _handle_pending(self, data: _Data) -> _State:
        prompt_ids = await self.apply_chat_template(data.messages)
        data.prompt_ids = prompt_ids
        return _State.GENERATING

    async def _handle_generating(self, data: _Data, sampling_params: dict[str, Any]) -> _State:
        with simple_timer("generate_sequences", data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=data.request_id,
                prompt_ids=data.prompt_ids,
                sampling_params=sampling_params,
            )

        if data.metrics.get("num_preempted") is None:
            data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        else:
            data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not data.extra_fields:
            data.extra_fields.update(output.extra_fields)

        data.assistant_turns += 1
        data.response_ids = output.token_ids
        data.prompt_ids += data.response_ids
        data.response_mask += [1] * len(data.response_ids)
        if output.log_probs:
            data.response_logprobs += output.log_probs

        # Decode this turn to inspect for code / answer block.
        response_text = self.tokenizer.decode(data.response_ids, skip_special_tokens=True)

        # Detect terminal signals (mirroring the reference DR.Kernel engine:
        # the per-turn reward call still runs, but no feedback is appended).
        is_answer_block = extract_answer_block(response_text) is not None
        hard_cap_hit = (
            len(data.response_mask) >= self.response_length
            or (self.max_assistant_turns and data.assistant_turns >= self.max_assistant_turns)
            or (self.max_user_turns and data.user_turns >= self.max_user_turns)
        )
        data.should_terminate_after_eval = bool(is_answer_block or hard_cap_hit)

        # If KernelGym eval is disabled (server_url missing), there's no
        # per-turn reward source. Leave `turn_rewards` empty so the
        # reward manager's legacy single-call path kicks in to score the
        # (single-turn) response — same as the old behavior. Multi-turn
        # rollouts require kernel_eval enabled.
        if not self.kernel_eval_enabled:
            return _State.TERMINATED

        # No extractable kernel → no KernelGym call. Synthesize a
        # penalty-shaped result so TRLOO still sees a per-turn entry,
        # and stop the loop (continuing would feed a malformed turn
        # back into the model with no information gain).
        kernel_code = extract_kernel_code(response_text)
        if not kernel_code:
            data.turn_results.append(
                self._make_synthetic_result(
                    status="no_kernel_extracted",
                    error="no kernel block found in response",
                )
            )
            data.turn_rewards.append(float(self.penalty_score))
            return _State.TERMINATED

        data.last_kernel_code = kernel_code
        return _State.EVALUATING

    async def _handle_evaluating(self, data: _Data) -> _State:
        """Run KernelGym for this assistant turn, persist the result as
        the per-turn training reward, and (unless this is the final turn)
        render a feedback message and append it as a user turn.

        This is the only place per-turn rewards are produced — the
        downstream reward manager just reads `data.turn_results` /
        `data.turn_rewards` from `extra_fields`. Faithful to the
        reference DR.Kernel engine, which also computes one reward per
        turn inline during rollout (see
        `drkernel/KernelGYM/.../vllm_async_engine_multi_iter.py:1942-1996`).
        """
        with simple_timer("kernel_eval", data.metrics):
            result = await evaluate_kernel(
                server_url=self.server_url,
                reference_code=data.reference_code,
                kernel_code=data.last_kernel_code,
                entry_point=data.entry_point,
                task_timeout=self.task_timeout,
                client_timeout=self.client_timeout,
                num_correct_trials=self.num_correct_trials,
                num_perf_trials=self.num_perf_trials,
                enable_profiling=self.enable_profiling,
                verbose_errors=self.verbose_errors,
                detect_decoy_kernel=self.detect_decoy_kernel,
                toolkit=self.toolkit,
                summarizer=self.reward_summarizer,
            )

            # Anomaly re-execution: mirrors `kernel_async.py:261-263`.
            # A speedup above the upper-bound clamp is almost always
            # measurement noise (compiler caching, contention, etc.) —
            # re-run once and keep the rerun result regardless. Without
            # this, the model gets the clamped max reward for a flake.
            raw_speedup = result.get("speedup", 0.0)
            try:
                raw_speedup_f = float(raw_speedup) if raw_speedup is not None else 0.0
            except (TypeError, ValueError):
                raw_speedup_f = 0.0
            if raw_speedup_f > self.speedup_reward_upper_bound:
                logger.warning(
                    "[KernelAgentLoop] anomalous speedup=%.3f > upper_bound=%.3f; re-executing",
                    raw_speedup_f,
                    self.speedup_reward_upper_bound,
                )
                result = await evaluate_kernel(
                    server_url=self.server_url,
                    reference_code=data.reference_code,
                    kernel_code=data.last_kernel_code,
                    entry_point=data.entry_point,
                    task_timeout=self.task_timeout,
                    client_timeout=self.client_timeout,
                    num_correct_trials=self.num_correct_trials,
                    num_perf_trials=self.num_perf_trials,
                    enable_profiling=self.enable_profiling,
                    verbose_errors=self.verbose_errors,
                    detect_decoy_kernel=self.detect_decoy_kernel,
                    toolkit=self.toolkit,
                    summarizer=self.reward_summarizer,
                )

        data.last_eval_status = str(result.get("status", "unknown"))
        data.turn_results.append(result)
        data.turn_rewards.append(self._extract_turn_reward(result))

        # Skip feedback rendering on the final turn — no next turn to
        # consume it. Matches the reference engine (no feedback render
        # after the last turn either).
        if data.should_terminate_after_eval:
            return _State.TERMINATED

        feedback = format_feedback(
            result,
            mode=self.feedback_mode,
            drop_keys=self.feedback_drop_keys,
        )
        feedback = self._truncate_feedback(feedback)

        feedback_msg = self.prompts.render_feedback_message(feedback=feedback, turn_idx=data.assistant_turns - 1)
        new_messages = [{"role": "user", "content": feedback_msg}]
        data.messages.extend(new_messages)

        # Tokenize the new user turn the same way tool_agent_loop does.
        # `remove_system_prompt=True` so the chat template doesn't re-emit the system block.
        feedback_ids = await self.apply_chat_template(new_messages, remove_system_prompt=True)

        if len(data.response_mask) + len(feedback_ids) >= self.response_length:
            # No room left in the response budget for another assistant turn.
            return _State.TERMINATED

        data.prompt_ids += feedback_ids
        data.response_mask += [0] * len(feedback_ids)
        if data.response_logprobs:
            data.response_logprobs += [0.0] * len(feedback_ids)
        data.user_turns += 1

        return _State.GENERATING

    def _extract_turn_reward(self, result: dict[str, Any]) -> float:
        """Pull the per-turn scalar reward out of an `evaluate_kernel`
        result. With `merge_reward_summary=True` (default), the result
        dict carries `reward`/`score` keys produced by `calculate_reward_*`
        in `KernelRewardSummarizer`. Falls back to `penalty_score` if
        neither key is present (e.g. summarizer disabled, or summarize()
        failed and returned the raw dict)."""
        for key in ("score", "reward"):
            value = result.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(self.penalty_score)

    def _make_synthetic_result(self, *, status: str, error: str) -> dict[str, Any]:
        """Build a penalty-shaped result dict for turns where no real
        KernelGym call was made (no extractable kernel, eval disabled).
        Shape matches what `KernelRewardSummarizer.calculate_reward_*`
        returns on the failure branch so downstream consumers don't
        need a special case."""
        return {
            "status": status,
            "reward": float(self.penalty_score),
            "score": float(self.penalty_score),
            "speedup": 0.0,
            "success": False,
            "correctness": False,
            "compiled": False,
            "error": error,
        }

    def _discover_server_url_from_config(self) -> str | None:
        """Walk known post-migrate config locations for the KernelGym URL.
        Returns None if not found at any known path."""
        for cfg_node in self._reward_config_candidates():
            try:
                url = getattr(cfg_node, "server_url", None)
                if url:
                    return str(url)
            except Exception:
                continue
        return None

    def _reward_config_candidates(self) -> list:
        """Return the candidate `reward_config` DictConfig nodes in the
        order the launcher might have populated them. Used by both
        `_discover_server_url_from_config` and `_build_reward_summarizer`
        so the agent loop reads from the same place the reward path does.

        Order:
          1. cfg.reward.reward_kwargs                                 (native YAML, post-migrate)
          2. cfg.reward.custom_reward_function.reward_kwargs.reward_config
                                                                     (verl-blessed YAML, post-migrate)
          3. cfg.reward_model                                         (pre-migrate fallback)
        """
        cfg = self.config
        candidates = []
        try:
            candidates.append(cfg.reward.reward_kwargs)
        except Exception:
            pass
        try:
            candidates.append(cfg.reward.custom_reward_function.reward_kwargs.reward_config)
        except Exception:
            pass
        try:
            candidates.append(cfg.reward_model)
        except Exception:
            pass
        return candidates

    def _build_reward_summarizer(self) -> KernelRewardSummarizer | None:
        """Build a `KernelRewardSummarizer` from the same `reward_config`
        node the reward path consumes. Returns None if no candidate
        exposes the required fields (e.g. `reward_func_name`,
        `init_correct_weight`, ...) — in that case the agent loop falls
        back to feeding the raw `/results` JSON between turns."""
        for cfg_node in self._reward_config_candidates():
            try:
                _ = cfg_node.reward_func_name
                _ = cfg_node.init_correct_weight
                _ = cfg_node.init_performance_weight
                _ = cfg_node.speedup_eps
                _ = cfg_node.reward_policy.penalties.penalty_score
                _ = cfg_node.speedup_reward_upper_bound
                _ = cfg_node.speedup_reward_lower_bound
            except Exception:
                continue
            try:
                return KernelRewardSummarizer(cfg_node)
            except Exception as exc:
                logger.warning(
                    "[KernelAgentLoop] reward_summarizer init failed on candidate (%s); trying next candidate",
                    exc,
                )
                continue
        logger.warning(
            "[KernelAgentLoop] no reward_config node exposes the fields needed for "
            "KernelRewardSummarizer; per-turn feedback will use raw /results JSON only."
        )
        return None

    def _truncate_feedback(self, text: str) -> str:
        n = self.max_tool_response_length
        if not text or len(text) <= n:
            return text
        if self.tool_response_truncate_side == "left":
            return text[:n] + "...(truncated)"
        if self.tool_response_truncate_side == "right":
            return "(truncated)..." + text[-n:]
        half = n // 2
        return text[:half] + "...(truncated)..." + text[-half:]
