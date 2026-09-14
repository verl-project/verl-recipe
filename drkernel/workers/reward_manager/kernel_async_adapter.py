# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
RewardManagerBase adapter on top of DR.Kernel's per-trajectory
AsyncKernelRewardManager.

Why this exists:
    The async fully_async_policy spawns `RewardLoopWorker` Ray actors that
    look up reward managers via the **experimental** registry at
    `verl.experimental.reward_loop.reward_manager.registry`. That registry
    expects subclasses of `RewardManagerBase` with an `async run_single(data)`
    contract — completely different from the legacy `AbstractRewardManager`.

    This adapter:
    - Subclasses `RewardManagerBase` and registers as ``"kernel_async"``
      via the experimental registry.
    - Implements `async run_single(data)` — pulls one trajectory's
      response_ids / ground_truth / entry_point / uuid out of `data`,
      decodes the response, calls the per-trajectory
      `AsyncKernelRewardManager.__call__`, returns the standard
      ``{"reward_score": float, "reward_extra_info": dict}`` dict the
      reward loop expects.

The per-trajectory logic in `AsyncKernelRewardManager` stays untouched so
upstream DR.Kernel updates can be re-pulled by overwriting `kernel_async.py`
without touching this adapter.
"""

from __future__ import annotations

from typing import Any

from recipe.drkernel.rewards.fast_at import compute_fast_at_indicators

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase

from .kernel_async import AsyncKernelRewardManager


@register("kernel_async")
class KernelAsyncRewardManager(RewardManagerBase):
    """Async reward manager (RewardLoopWorker-compatible) wrapping
    DR.Kernel's per-trajectory `AsyncKernelRewardManager`."""

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score)

        # Resolve `reward_config` from one of three places, in priority order:
        #
        #   1. `kwargs["reward_config"]` — set when the manager is built from
        #      a custom reward fn closure that pre-binds it (legacy DR.Kernel
        #      RayPPOTrainer wiring path).
        #   2. `config.reward.reward_kwargs` — present after
        #      `migrate_legacy_reward_impl` forwards `reward_model.reward_kwargs`
        #      into `config.reward.reward_kwargs`. This is the path our
        #      `drkernel_kernel_trainer_native.yaml` uses.
        #   3. The raw `config` we were handed — last-resort fallback for
        #      ad-hoc setups.
        reward_config = kwargs.get("reward_config")
        if reward_config is None:
            try:
                reward_config = config.reward.reward_kwargs
            except Exception:
                reward_config = None
        if reward_config is None:
            reward_config = config

        # `is_valid` follows the same precedence: explicit kwarg, then
        # the value embedded in the reward_kwargs blob, then default False.
        is_valid = kwargs.get("is_valid")
        if is_valid is None:
            is_valid = bool(getattr(reward_config, "is_valid", False) or False)

        self._inner = AsyncKernelRewardManager(
            tokenizer=tokenizer,
            num_examine=kwargs.get("num_examine", 0),
            compute_score=compute_score,
            reward_fn_key=kwargs.get("reward_fn_key", "data_source"),
            reward_config=reward_config,
            is_valid=is_valid,
        )

    async def run_single(self, data: DataProto) -> dict[str, Any]:
        assert len(data) == 1, "Only support single data item (RewardLoopWorker convention)"
        data_item = data[0]
        non_tensor = data_item.non_tensor_batch

        # Per-turn reward path (preferred): the agent loop (`kernel_agent_loop.py`)
        # already evaluated each assistant turn against KernelGym and surfaced
        # `turn_rewards` / `turn_results` via `extra_fields`. Use those directly
        # so the training signal matches the per-turn feedback the model saw,
        # and skip the duplicate KernelGym roundtrip.
        #
        # `turn_rewards`/`turn_results` may live at the top level of
        # `non_tensor_batch` (when this manager is invoked via the trainer's
        # batch-reward path), OR be wrapped inside `tool_extra_fields` — which
        # is how `AgentLoopWorker._compute_score` packs `output.extra_fields`
        # when it streams per-trajectory reward requests to the reward-loop
        # worker (`verl/experimental/agent_loop/agent_loop.py:877-881`). Check
        # both locations so the fast path triggers in either invocation mode.
        #
        # Legacy single-call path is kept as a fallback for callers that don't
        # populate `turn_rewards` (e.g. ablations with `kernel_eval_enabled=False`
        # or non-KernelAgentLoop agents).
        turn_rewards = self._get_extra_field(non_tensor, "turn_rewards")
        turn_results = self._get_extra_field(non_tensor, "turn_results")
        if turn_rewards and isinstance(turn_rewards, list):
            return self._build_response_from_turn_rewards(
                turn_rewards=turn_rewards,
                turn_results=turn_results or [],
            )

        # ---- Legacy single-call fallback ----
        # Decode the response. `attention_mask` covers prompt+response;
        # response-side mask is the trailing `response_length` slice.
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = int(data_item.batch["attention_mask"][-response_length:].sum().item())
        valid_response_ids = response_ids[:valid_response_length]

        response_str = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True),
        )

        # Pull kernel-RL-specific fields out of non_tensor_batch.
        ground_truth = non_tensor.get("reward_model", {}).get("ground_truth", "")
        if not ground_truth:
            ground_truth = non_tensor.get("ground_truth", non_tensor.get("reference_code", ""))

        extra_info = non_tensor.get("extra_info", {}) or {}
        entry_point = extra_info.get("entry_point", non_tensor.get("entry_point", "Model"))
        uuid = extra_info.get("uuid", non_tensor.get("uuid", non_tensor.get("uid", "")))

        # Inner per-trajectory call (sync, runs the HTTP roundtrip to KernelGym).
        per_traj = await self.loop.run_in_executor(
            None,
            lambda: self._inner(
                response_ids=valid_response_ids.tolist(),
                response_str=response_str,
                ground_truth=str(ground_truth),
                entry_point=str(entry_point),
                uuid=str(uuid),
                return_dict=True,
                response_length=response_length,
            ),
        )

        # The reward loop convention: scalar `reward_score` + a flat
        # `reward_extra_info` dict that the trainer aggregates and logs.
        reward_tensor = per_traj["reward_tensor"]
        if reward_tensor.numel() == 0:
            reward_score = 0.0
        else:
            # Per-trajectory rm_scores tensor has the reward placed at the
            # last valid token; the scalar is the max (or just the sum since
            # the rest are zero).
            reward_score = float(reward_tensor.sum().item())

        return {
            "reward_score": reward_score,
            "reward_extra_info": dict(per_traj.get("reward_extra_info", {})),
        }

    @staticmethod
    def _unwrap_object_field(value):
        """Object dtype np.arrays from `extra_fields` may arrive wrapped in
        a 0-d ndarray after DataProto indexing. Peel that layer so callers
        can treat the field as the underlying Python object."""
        import numpy as _np

        if isinstance(value, _np.ndarray):
            if value.ndim == 0:
                value = value.item()
            elif value.dtype == object and value.size == 1:
                value = value[0]
        return value

    def _get_extra_field(self, non_tensor, key):
        """Look up an agent-loop `extra_fields` entry on a single-row
        non_tensor batch.

        Two layouts are supported:

        1. Top-level: `non_tensor[key]` directly carries the value. This is
           the layout the trainer's batch-reward path uses (every
           `extra_fields` entry is split into its own top-level non-tensor
           column by `AgentLoopWorker._postprocess` at
           `verl/experimental/agent_loop/agent_loop.py:989-995`).
        2. Bundled under `tool_extra_fields`: the entire `output.extra_fields`
           dict is packed as a single length-1 object array under the key
           `tool_extra_fields` by `AgentLoopWorker._compute_score`
           (`verl/experimental/agent_loop/agent_loop.py:877-881`). That is
           how the reward-loop worker receives per-trajectory data when the
           agent loop streams reward computation alongside rollout.

        Returns ``None`` when the key is absent in both layouts."""
        value = self._unwrap_object_field(non_tensor.get(key))
        if value is not None:
            return value
        tool_extra = self._unwrap_object_field(non_tensor.get("tool_extra_fields"))
        if isinstance(tool_extra, dict):
            return tool_extra.get(key)
        return None

    def _build_response_from_turn_rewards(
        self,
        *,
        turn_rewards: list,
        turn_results: list,
    ) -> dict[str, Any]:
        """Assemble the RewardLoopWorker response from per-turn rewards
        produced by the agent loop, without calling KernelGym again.

        - `reward_score`: scalar sum of per-turn rewards (the trainer's
          default placement of this scalar at the last valid token is
          then overwritten by
          `DrKernelFullyAsyncTrainerImpl._fit_compute_reward`, which
          redistributes per-turn rewards onto each `response_mask=1`
          span's last token).
        - `reward_extra_info`: pass through (a) the per-turn list itself
          so the trainer can do the span redistribution; (b) the
          last-turn's summary metrics (`correctness`, `speedup`, etc.) so
          existing `_fit_collect_metrics` aggregations keep working.
        """
        clean_rewards = [float(r) for r in turn_rewards]
        reward_score = float(sum(clean_rewards))

        # Last turn's metrics — matches the previous "score whole trajectory"
        # semantics for logging purposes. Per-turn versions can be added
        # later under e.g. `kernel/per_turn/...` if we want them in TB.
        last_result = turn_results[-1] if turn_results and isinstance(turn_results[-1], dict) else {}
        last_speedup = last_result.get("speedup", 0.0)
        if last_speedup is None:
            last_speedup = 0.0
        last_correctness = bool(last_result.get("correctness", False))
        last_compiled = bool(last_result.get("compiled", False))
        last_success = bool(last_result.get("success", False))
        last_status = str(last_result.get("status", "unknown"))
        last_decoy = bool(last_result.get("decoy_kernel", False))
        try:
            speedup_eps = float(self._inner.reward_config.speedup_eps)
        except Exception:
            speedup_eps = 0.01
        is_speedup_positive = float(last_speedup) >= (1.0 + speedup_eps)

        # Fast@<X>/last — last-turn-only Fast@X indicator (a per-trajectory
        # binary). Emitted under `Fast@<X>/last` so the trainer's bare
        # `kernel/Fast@<X>` tag is free for the per-(traj, turn) mean
        # written by `_fit_collect_metrics` (matching `kernel/correctness/rate`
        # etc.). The auto-aggregation loop picks this up by the `Fast@`
        # prefix and produces `kernel/Fast@<X>/last`.
        thresholds = self._inner.fast_at_thresholds
        try:
            last_turn_fast_at_raw = compute_fast_at_indicators(last_correctness, float(last_speedup), thresholds)
        except Exception:
            last_turn_fast_at_raw = {}
        last_turn_fast_at = {f"{k}/last": v for k, v in last_turn_fast_at_raw.items()}

        # Walk every turn once to compute both:
        #   - `Fast@<X>/best`: 1 iff ANY turn in this trajectory hit Fast@X.
        #     Backsliding-aware — captures models that found a fast kernel
        #     mid-revision but later regressed. Auto-aggregated to
        #     `kernel/Fast@<X>/best` (training) and
        #     `val-aux/<src>/Fast@<X>/best/mean@N` (validation) by the
        #     trainer's `FAST_AT_KEY_PREFIX` loop.
        #   - Per-(traj, turn) means for this trajectory: `correctness/turn_avg`,
        #     `compilation/turn_avg`, `is_speedup_positive/turn_avg`,
        #     `speedup/turn_avg`, `Fast@<X>/turn_avg`. These give validation
        #     an apples-to-apples counterpart to training's flat per-(traj,
        #     turn) `kernel/correctness/rate` family (see
        #     `DrKernelFullyAsyncTrainerImpl._collect_per_turn_kernel_stats`).
        #     `process_validation_metrics` averages each across rollouts/prompts
        #     into `val-aux/<src>/<key>/turn_avg/mean@N`.
        fast_at_best = {f"{k}/best": 0.0 for k in last_turn_fast_at_raw}
        fast_at_turn_count: dict[str, float] = {k: 0.0 for k in last_turn_fast_at_raw}
        sum_correct = 0.0
        sum_compiled = 0.0
        sum_speedup_pos = 0.0
        sum_speedup = 0.0
        counted_turns = 0
        for result in turn_results:
            if not isinstance(result, dict):
                continue
            counted_turns += 1
            t_correctness = bool(result.get("correctness", False))
            sum_correct += 1.0 if t_correctness else 0.0
            sum_compiled += 1.0 if bool(result.get("compiled", False)) else 0.0
            t_speedup = result.get("speedup", 0.0)
            if t_speedup is None:
                t_speedup = 0.0
            try:
                t_speedup_f = float(t_speedup)
            except (TypeError, ValueError):
                t_speedup_f = 0.0
            sum_speedup += t_speedup_f
            sum_speedup_pos += 1.0 if t_speedup_f >= (1.0 + speedup_eps) else 0.0
            try:
                turn_fast_at = compute_fast_at_indicators(t_correctness, t_speedup_f, thresholds)
            except Exception:
                continue
            for key, value in turn_fast_at.items():
                if value > 0.5:
                    fast_at_best[f"{key}/best"] = 1.0
                fast_at_turn_count[key] = fast_at_turn_count.get(key, 0.0) + float(value)

        # Use counted_turns (turns that produced a dict result) as the
        # denominator. Falls back to 1 to avoid div-by-zero on degenerate
        # trajectories — value is then 0.0 either way since the sums are 0.
        denom = counted_turns or 1
        per_turn_avg = {
            "correctness/turn_avg": sum_correct / denom,
            "compilation/turn_avg": sum_compiled / denom,
            "is_speedup_positive/turn_avg": sum_speedup_pos / denom,
            "speedup/turn_avg": sum_speedup / denom,
            "reward/turn_avg": (sum(clean_rewards) / denom) if clean_rewards else 0.0,
        }
        fast_at_turn_avg = {f"{k}/turn_avg": v / denom for k, v in fast_at_turn_count.items()}

        reward_extra_info: dict[str, Any] = {
            "correctness": last_correctness,
            "performance": float(last_speedup),
            "is_speedup_positive": bool(is_speedup_positive),
            "is_decoy_kernel": last_decoy,
            "compilation": last_compiled,
            "success": last_success,
            "status": last_status,
            "error": str(last_result.get("error", "") or ""),
            "num_turns": len(clean_rewards),
        }

        # Last-turn debug telemetry (mirrors what the legacy single-call
        # path used to surface). Keeps the existing
        # `val-aux/<src>/num_custom_kernel/mean@1` etc. TB tags alive after
        # the fast-path switch — same semantics (last-turn-only value),
        # different physical source (per-turn KernelGym result instead of
        # one call on the concatenated multi-turn response).
        for telemetry_key in (
            "num_custom_kernel",
            "num_total_kernels",
            "custom_kernel_cuda_time_in_profiling_us",
            "total_kernel_run_time_in_profiling_us",
        ):
            value = last_result.get(telemetry_key, 0)
            try:
                reward_extra_info[telemetry_key] = float(value or 0)
            except (TypeError, ValueError):
                reward_extra_info[telemetry_key] = 0.0
        num_total = reward_extra_info["num_total_kernels"]
        num_custom = reward_extra_info["num_custom_kernel"]
        reward_extra_info["num_coverage"] = float(num_custom) / num_total if num_total > 0 else 0.0
        total_run_time = reward_extra_info["total_kernel_run_time_in_profiling_us"]
        custom_run_time = reward_extra_info["custom_kernel_cuda_time_in_profiling_us"]
        reward_extra_info["time_coverage"] = float(custom_run_time) / total_run_time if total_run_time > 0 else 0.0

        # Do NOT re-emit `turn_rewards` here — it already lives in
        # `non_tensor_batch["turn_rewards"]` via the agent loop's
        # `extra_fields` (see `agent_loop.py:981-995`). The trainer's
        # `_redistribute_per_turn_rewards` reads it from that path.
        reward_extra_info.update(last_turn_fast_at)
        reward_extra_info.update(fast_at_best)
        reward_extra_info.update(per_turn_avg)
        reward_extra_info.update(fast_at_turn_avg)
        return {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info,
        }


# Alias so verl's `load_extern_object(module, object_name="kernel_async")`
# resolves. Quirk in verl: with `reward_loop_source: importlib`, the loader
# uses `reward_manager_cfg.name` (i.e. `reward_model.reward_manager`, which
# we set to `"kernel_async"` in the yaml) as the **class name** to look up
# in the module — not the user-facing `reward_loop_class_name`. This alias
# preserves the conventional `reward_model.reward_manager=kernel_async`
# spelling without forcing the user to special-case the importlib path.
kernel_async = KernelAsyncRewardManager
