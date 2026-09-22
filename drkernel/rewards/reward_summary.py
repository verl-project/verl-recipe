# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""Pure reward-summary computation, factored for the rollout-side agent loop.

The methods here mirror what `recipe.drkernel.rewards.reward_client.KernelRewardClient`
does in `calculate_reward_*`, `compute_coverage_reward`, and
`_merge_reward_result` — that is, the second half of the original
DR.Kernel reward pipeline (raw KernelGym `/results` -> reward summary
-> merged dict that becomes `env_state` in the original
`drkernel/KernelGYM/.../vllm_async_engine.py:1747-1769`).

Why a separate class instead of reusing `KernelRewardClient` directly:
- `KernelRewardClient.__init__` spins up a Ray actor pool
  (`_HybridHttpWorker` + token-bucket rate limiter) for batched HTTP
  fan-out. The rollout-side multi-turn agent loop is itself inside a
  Ray actor (`AgentLoopWorker`) and only needs one trajectory's worth
  of summary computation per turn — pulling in the actor pool would
  waste resources and create cross-actor ownership puzzles.
- This module exposes only the pure, deterministic part of the reward
  path so the agent loop's between-turn feedback can be byte-identical
  to what the original engine puts in `env_state`.

Keep in sync with `KernelRewardClient`: the bodies of
`calculate_reward_like_kernel`, `compute_coverage_reward`,
`calculate_reward_weighted`, `calculate_reward_speedup`,
`_get_reward_func`, and `_merge_reward_result` are copied from
`reward_client.py` verbatim. If those methods change in the reward
path, mirror the changes here too.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class KernelRewardSummarizer:
    """Stateless (modulo `reward_config`) reward-summary computer.

    Constructed from the same `reward_config` block the reward path
    feeds into `KernelRewardClient`. Pulling out only the fields the
    `calculate_reward_*` methods read keeps this module independent of
    the Ray-actor side of the reward client.
    """

    def __init__(self, reward_config: Any):
        self.reward_config = reward_config
        self.reward_func_name = reward_config.reward_func_name
        self.init_correct_weight = float(reward_config.init_correct_weight)
        self.init_performance_weight = float(reward_config.init_performance_weight)
        self.speedup_eps = float(reward_config.speedup_eps)
        self.penalty_score = float(reward_config.reward_policy.penalties.penalty_score)
        self.speedup_reward_upper_bound = float(reward_config.speedup_reward_upper_bound)
        self.speedup_reward_lower_bound = float(reward_config.speedup_reward_lower_bound)

    def _get_reward_func(self):
        try:
            func = getattr(self, str(self.reward_func_name), None)
            if callable(func):
                return func
        except Exception:
            pass
        try:
            print(
                f"[RewardSummarizer] invalid reward_func_name={self.reward_func_name}, fallback to calculate_reward_like_kernel"  # noqa: E501
            )
        except Exception:
            pass
        return self.calculate_reward_like_kernel

    def calculate_reward_like_kernel(self, result: dict[str, Any]) -> dict[str, Any]:
        if result.get("status") != "completed":
            error_message = result.get("error_message", "Task failed")
            if error_message == "Task failed":
                error_message = result.get("error", "Task failed")
            return {
                "reward": -1.0,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": error_message,
            }
        if result.get("decoy_kernel", False):
            return {
                "reward": -1.0,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": -1.0,
            }
        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0)
        compiled = result.get("compiled", False)

        penalties = self.reward_config.reward_policy.penalties
        compilation_fail_penalty = float(penalties.get("compilation_fail", -0.5))
        correctness_fail_penalty = float(penalties.get("correctness_fail", -0.3))
        perf_degrade_penalty = float(penalties.get("perf_degrade", -0.1))

        if not compiled:
            reward = compilation_fail_penalty
        elif not correctness:
            reward = correctness_fail_penalty
        else:
            if speedup >= 3.0:
                reward = 1.0
            elif speedup >= 2.0:
                reward = 0.8
            elif speedup >= 1.5:
                reward = 0.6
            elif speedup >= 1.2:
                reward = 0.4
            elif speedup >= 1.0:
                reward = 0.2
            else:
                reward = perf_degrade_penalty
        return {
            "reward": reward,
            "speedup": speedup,
            "success": compiled and correctness,
            "correctness": correctness,
            "compiled": compiled,
            "score": reward,
        }

    def compute_coverage_reward(self, result: dict[str, Any]) -> dict[str, Any]:
        metadata = result.get("metadata") or {}

        def _get_field(*keys: str, default: int = 0) -> int:
            for k in keys:
                if k in metadata:
                    return metadata.get(k) or default
                if k in result:
                    return result.get(k) or default
            return default

        num_custom_kernel = _get_field("num_custom_kernels", "num_custom_kernel")
        num_total_kernels = _get_field("num_total_kernels", "num_total_kernel", "num_total_kernels")
        custom_kernel_cuda_time_in_profiling_us = _get_field("custom_kernel_cuda_time_in_profiling_us")
        total_kernel_run_time_in_profiling_us = _get_field("total_kernel_run_time_in_profiling_us")

        num_coverage = 0
        if num_total_kernels > 0:
            num_coverage = num_custom_kernel / num_total_kernels

        time_coverage = 0
        if total_kernel_run_time_in_profiling_us > 0:
            time_coverage = custom_kernel_cuda_time_in_profiling_us / total_kernel_run_time_in_profiling_us

        if self.reward_config.coverage_reward.reward_type == "time_coverage":
            coverage = time_coverage
        elif self.reward_config.coverage_reward.reward_type == "number_coverage":
            coverage = num_coverage
        else:
            raise ValueError(f"Invalid reward type: {self.reward_config.coverage_reward.reward_type}")

        return {
            "coverage": coverage,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "custom_kernel_cuda_time_in_profiling_us": custom_kernel_cuda_time_in_profiling_us,
            "total_kernel_run_time_in_profiling_us": total_kernel_run_time_in_profiling_us,
        }

    def calculate_reward_weighted(self, result: dict[str, Any]) -> dict[str, Any]:
        penalty_score = self.penalty_score

        if result.get("status") != "completed":
            error_message = result.get("error_message", "Task failed")
            if error_message == "Task failed":
                error_message = result.get("error", "Task failed")

            return_result = {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": error_message,
            }
            for key in result.keys():
                if key not in return_result:
                    return_result[key] = result[key]
            return return_result

        if result.get("decoy_kernel", False):
            return {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": penalty_score,
            }

        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0)
        compiled = result.get("compiled", False)
        profiling = result.get("profiling", None)

        if speedup is None:
            speedup = 0.0

        is_speedup_positive = speedup >= (1 + self.speedup_eps)

        reward = self.init_correct_weight * correctness + self.init_performance_weight * is_speedup_positive

        num_custom_kernel = 0
        num_total_kernels = 0
        custom_kernel_cuda_time_in_profiling_us = 0
        total_kernel_run_time_in_profiling_us = 0
        final_reward = reward
        if correctness:
            coverage_dict = self.compute_coverage_reward(result)
            coverage = coverage_dict["coverage"]
            num_custom_kernel = coverage_dict["num_custom_kernel"]
            num_total_kernels = coverage_dict["num_total_kernels"]
            custom_kernel_cuda_time_in_profiling_us = coverage_dict["custom_kernel_cuda_time_in_profiling_us"]
            total_kernel_run_time_in_profiling_us = coverage_dict["total_kernel_run_time_in_profiling_us"]
            if self.reward_config.coverage_reward.enable:
                final_reward += self.reward_config.coverage_reward.weight * coverage

        return {
            "reward": final_reward,
            "speedup": speedup,
            "success": compiled and correctness,
            "correctness": correctness,
            "compiled": compiled,
            "score": final_reward,
            "profiling": profiling,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "custom_kernel_cuda_time_in_profiling_us": custom_kernel_cuda_time_in_profiling_us,
            "total_kernel_run_time_in_profiling_us": total_kernel_run_time_in_profiling_us,
        }

    def calculate_reward_speedup(self, result: dict[str, Any]) -> dict[str, Any]:
        penalty_score = self.penalty_score

        if result.get("status") != "completed":
            error_message = result.get("error_message", "Task failed")
            if error_message == "Task failed":
                error_message = result.get("error", "Task failed")

            return_result = {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": error_message,
            }
            for key in result.keys():
                if key not in return_result:
                    return_result[key] = result[key]
            return return_result

        if result.get("decoy_kernel", False):
            return {
                "reward": penalty_score,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "decoy_kernel": True,
                "error": "Reward hacking: Decoy kernel detected",
                "score": penalty_score,
            }

        correctness = result.get("correctness", False)
        speedup = result.get("speedup", 0.0)
        compiled = result.get("compiled", False)
        profiling = result.get("profiling", None)

        if speedup is None:
            speedup = 0.0

        reward_speedup = speedup
        if speedup > self.speedup_reward_upper_bound:
            reward_speedup = self.speedup_reward_upper_bound

        if reward_speedup < self.speedup_reward_lower_bound:
            reward_speedup = 0.0

        reward = self.init_correct_weight * correctness + self.init_performance_weight * reward_speedup

        num_custom_kernel = 0
        num_total_kernels = 0
        custom_kernel_cuda_time_in_profiling_us = 0
        total_kernel_run_time_in_profiling_us = 0
        final_reward = reward
        if correctness:
            coverage_dict = self.compute_coverage_reward(result)
            coverage = coverage_dict["coverage"]
            num_custom_kernel = coverage_dict["num_custom_kernel"]
            num_total_kernels = coverage_dict["num_total_kernels"]
            custom_kernel_cuda_time_in_profiling_us = coverage_dict["custom_kernel_cuda_time_in_profiling_us"]
            total_kernel_run_time_in_profiling_us = coverage_dict["total_kernel_run_time_in_profiling_us"]
            if self.reward_config.coverage_reward.enable:
                final_reward += self.reward_config.coverage_reward.weight * coverage

        return {
            "reward": final_reward,
            "speedup": speedup,
            "success": compiled and correctness,
            "correctness": correctness,
            "compiled": compiled,
            "score": final_reward,
            "profiling": profiling,
            "num_custom_kernel": num_custom_kernel,
            "num_total_kernels": num_total_kernels,
            "custom_kernel_cuda_time_in_profiling_us": custom_kernel_cuda_time_in_profiling_us,
            "total_kernel_run_time_in_profiling_us": total_kernel_run_time_in_profiling_us,
        }

    def _merge_reward_result(self, raw_result: dict[str, Any], reward_summary: dict[str, Any]) -> dict[str, Any]:
        """Merge raw KernelServer response payload with derived reward summary."""
        merged: dict[str, Any] = {}
        if raw_result:
            merged.update(raw_result)
        if reward_summary:
            merged.update(reward_summary)
        return merged

    def summarize(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        """Convenience: pick the configured reward func, compute the summary,
        merge with the raw result, and return the merged dict.

        Mirrors `KernelRewardClient.compute_batch_rewards` lines
        ``reward_func = self._get_reward_func(); reward_summary =
        reward_func(data); merged[orig_idx] = self._merge_reward_result(
        data, reward_summary)`` — i.e. exactly what the original
        DR.Kernel `env_state` ends up containing.
        """
        try:
            reward_func = self._get_reward_func()
            reward_summary = reward_func(raw_result)
            return self._merge_reward_result(raw_result, reward_summary)
        except Exception as exc:
            logger.warning("[KernelRewardSummarizer] summarize failed (%s); returning raw result", exc)
            return raw_result
