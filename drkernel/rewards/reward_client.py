# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""
Hybrid Kernel reward client (composed implementation):
- External API matches KernelServer (/evaluate submit, /status poll, /results fetch).
- Concurrency and rate limiting use the sandbox fusion Ray worker pool + global token bucket.
- Does not inherit Enhanced; it directly reuses the core request/poll/reward logic to keep behavior aligned,
  leaving observability to be added later if needed.

Two-level timeout design:
1. task_timeout (in payload["timeout"]): Server-side execution limit for kernel evaluation
2. task_timeout_in_client: Client-side polling timeout including queue wait time
Invariant: task_timeout_in_client >= task_timeout (client waits longer due to queuing)
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from typing import Any, Optional
from uuid import uuid4

import httpx
import ray

logger = logging.getLogger(__name__)


@ray.remote
class _HybridHttpWorker:
    def __init__(self, server_url: str, rate_limit: int, default_timeout: int, acquire_timeout: int) -> None:
        self.server_url = server_url
        # print(f"[DEBUG] Default timeout: {default_timeout}")
        self.default_timeout = int(default_timeout)
        self.acquire_timeout = int(acquire_timeout)
        self._limits = httpx.Limits(max_keepalive_connections=64, max_connections=128, keepalive_expiry=30.0)
        self._client = httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=self.default_timeout, write=10.0, pool=5.0),
            limits=self._limits,
            headers={"Content-Type": "application/json"},
        )
        # Rate-limiter dropped (see `submit_and_poll` for context). Per-actor
        # concurrency is still bounded by Ray's `max_concurrency=N` on this
        # actor, and KernelGym backpressures via Redis if its server-side
        # worker pool is saturated.
        self._rate_limit_worker = None

    def _backoff(self, attempt: int, base: int = 2, cap: int = 30) -> float:
        return min(base**attempt, cap)

    def get_token_in_use(self) -> int:
        # Token bucket disabled — return sentinel so the heartbeat that
        # logs `tokens_in_use=...` keeps a stable shape.
        return -1

    def submit_and_poll(
        self, task_data: dict[str, Any], client_timeout: int, max_retries: Optional[int]
    ) -> dict[str, Any]:
        """Submit task and poll for results.

        Args:
            task_data: Task payload including server-side timeout in task_data["timeout"]
            client_timeout: Client-side total timeout including queue wait + execution time
            max_retries: Max retry attempts for submission failures
        """
        start_ts = time.time()
        # Rate-limiter dropped — concurrency is bounded per-actor by Ray's
        # `max_concurrency=N` on this `_HybridHttpWorker`, and KernelGym
        # backpressures via Redis when its server-side pool is saturated.
        try:
            # Submit with limited retries: 429/503/timeout/connect errors.
            attempt = 0
            unlimited = max_retries is None or max_retries == -1
            while unlimited or attempt < (max_retries or 0):
                try:
                    # Log once on first attempt to help debug "server did not receive request".
                    if attempt == 0:
                        print(
                            f"[HybridWorker] POST /evaluate task_id={task_data.get('task_id', '')} url={self.server_url}"  # noqa: E501
                        )
                    resp = self._client.post(f"{self.server_url}/evaluate", json=task_data)
                    # Log status code to help diagnose non-200 responses.
                    try:
                        print(
                            f"[HybridWorker] POST /evaluate resp={resp.status_code} task_id={task_data.get('task_id', '')}"  # noqa: E501
                        )
                    except Exception:
                        pass
                    if resp.status_code == 200:
                        break
                    if resp.status_code in (429, 503):
                        time.sleep(self._backoff(attempt, base=2 if resp.status_code == 429 else 5))
                        attempt += 1
                        continue
                    resp.raise_for_status()
                except (httpx.TimeoutException, httpx.ConnectError) as e:
                    if unlimited or attempt < (max_retries or 0) - 1:
                        time.sleep(self._backoff(attempt))
                        attempt += 1
                        continue
                    return {"status": "failed", "error_message": str(e)}
                except Exception as e:
                    return {"status": "failed", "error_message": str(e)}

            # Poll status at a fixed 1s interval.
            task_id = task_data.get("task_id", "")
            last_status = None
            while time.time() - start_ts < client_timeout:
                try:
                    s = self._client.get(f"{self.server_url}/status/{task_id}")
                    if s.status_code == 200:
                        data = s.json()
                        status = data.get("status", "unknown")
                        if status != last_status:
                            last_status = status
                            try:
                                print(f"[HybridWorker] STATUS task_id={task_id} -> {status}")
                            except Exception:
                                pass
                        if status in ("completed", "failed", "timeout", "cancelled"):
                            if status == "completed":
                                r = self._client.get(f"{self.server_url}/results/{task_id}")
                                if r.status_code == 200:
                                    result = r.json()
                                    result["status"] = status
                                    return result
                                return {
                                    "status": status,
                                    "error_message": f"Failed to fetch results: HTTP {r.status_code}",
                                }
                            return {"status": status, "error_message": data.get("error_message", f"Task {status}")}
                except Exception:
                    pass
                time.sleep(1.0)

            return {"status": "timeout", "error_message": f"Task timeout after {client_timeout}s (client-side)"}
        finally:
            # No need to release here (already released during submission).
            pass


class KernelRewardClient:
    def __init__(self, *, reward_config: Any) -> None:
        # Allow passing a wrapper config object.
        if hasattr(reward_config, "reward_model"):
            reward_config = reward_config.reward_model

        # Read required fields from reward_config.
        self.server_url = str(reward_config.server_url)
        self.timeout = float(reward_config.timeout)
        # task_timeout_in_client: client-side timeout including queue wait (should >= task_timeout)
        self.task_timeout_in_client = int(getattr(reward_config, "task_timeout_in_client", self.timeout))
        self.max_retries = reward_config.max_retries
        self.rate_limit = int(reward_config.rate_limit)
        if self.rate_limit <= 0:
            self.rate_limit = 1
        # Use max_concurrent as worker concurrency.
        self.num_workers = int(reward_config.max_concurrent)
        self.task_counter = 0
        self.acquire_timeout = int(reward_config.acquire_timeout)

        # Reward policy (aligned with KernelRewardClient); use defaults if not set.
        self.reward_config = reward_config

        # Ray worker (persistent httpx.Client + global token bucket).
        self._worker = _HybridHttpWorker.options(max_concurrency=self.num_workers).remote(
            self.server_url, self.rate_limit, int(self.timeout), self.acquire_timeout
        )
        # Rate-limiter dropped (see `_HybridHttpWorker.__init__`). Heartbeat
        # logging treats `_rate_limit_worker is None` as "rate limit
        # disabled" and reports `tokens_in_use=-1` accordingly.
        self._rate_limit_worker = None

        # Reward function weights and parameters.
        self.reward_func_name = reward_config.reward_func_name
        self.init_correct_weight = float(reward_config.init_correct_weight)
        self.init_performance_weight = float(reward_config.init_performance_weight)
        self.speedup_eps = float(reward_config.speedup_eps)
        self.penalty_score = float(reward_config.reward_policy.penalties.penalty_score)
        self.speedup_reward_upper_bound = float(reward_config.speedup_reward_upper_bound)
        self.speedup_reward_lower_bound = float(reward_config.speedup_reward_lower_bound)

    def _get_reward_func(self):
        """Select reward function based on config; default to calculate_reward_like_kernel."""
        try:
            func = getattr(self, str(self.reward_func_name), None)
            if callable(func):
                return func
        except Exception:
            pass
        try:
            print(
                f"[HybridClient] invalid reward_func_name={self.reward_func_name}, fallback to calculate_reward_like_kernel"  # noqa: E501
            )
        except Exception:
            pass
        return self.calculate_reward_like_kernel

    def _next_task_id(self, prefix: str) -> str:
        try:
            self.task_counter += 1
        except Exception:
            # Fallback: still guarantee uniqueness.
            self.task_counter = int(time.time() * 1000) % 1000000
        return f"{prefix}_{self.task_counter:06d}_{uuid4().hex[:8]}"

    def _preflight_validate(self, reference_code: str, kernel_code: str, entry_point: str) -> tuple[bool, str]:
        """Minimal preflight: verify entry point exists to avoid meaningless requests."""
        try:
            ref_required = f"class {entry_point}"
            ker_required = f"class {entry_point}New"
            ref_ok = ref_required in (reference_code or "")
            ker_ok = ker_required in (kernel_code or "")
            if ref_ok and ker_ok:
                return True, ""
            missing = []
            if not ref_ok:
                missing.append(ref_required)
            if not ker_ok:
                missing.append(ker_required)
            return False, ", ".join(missing)
        except Exception as e:
            logger.debug(f"preflight skipped due to error: {e}")
            return True, ""

    def calculate_reward_like_kernel(self, result: dict[str, Any]) -> dict[str, Any]:
        if result.get("status") != "completed":
            error_message = result.get("error_message", "Task failed")
            if error_message == "Task failed":
                error_message = result.get("error", "Task failed")
            print(f"[HybridClient] calculate_reward_like_kernel error_message: {error_message}")
            print(f"[HybridClient] Task failed result: {result}")
            return {
                "reward": -1.0,
                "speedup": 0.0,
                "success": False,
                "correctness": False,
                "compiled": False,
                "error": error_message,
            }
        # Server returned a decoy kernel; force -1 and carry the marker.
        if result.get("decoy_kernel", False):
            try:
                print("[HybridClient] decoy_kernel detected; forcing reward -1")
            except Exception:
                pass
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
        # Some server versions put coverage fields in metadata, possibly with plural names; normalize here.
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

        # Only log keys once when all fields are missing to aid debugging.
        if (
            not num_custom_kernel
            and not num_total_kernels
            and "num_custom_kernel" not in result
            and "num_total_kernels" not in result
            and "num_custom_kernels" not in metadata
            and "num_total_kernels" not in metadata
        ):
            try:
                print(f"[HybridClient] coverage fields missing, fallback to 0: keys={list(result.keys())}")
            except Exception:
                pass

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
            print(f"[HybridClient] calculate_reward_like_kernel error_message: {error_message}")
            print(f"[HybridClient] Task failed result: {result}")

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
        # Server returned a decoy kernel; force penalty and carry the marker.
        # TODO Temporary disable decoy kernel detection
        if result.get("decoy_kernel", False):
            try:
                print("[HybridClient] decoy_kernel detected; forcing reward -1")
            except Exception:
                pass
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
        # In fact, profiling is always None here since it is actually inside metadata
        profiling = result.get("profiling", None)

        if speedup is None:
            speedup = 0.0

        is_speedup_positive = speedup >= (1 + self.speedup_eps)  # ignore too small speedup

        reward = self.init_correct_weight * correctness + self.init_performance_weight * is_speedup_positive

        num_custom_kernel = 0
        num_total_kernels = 0
        custom_kernel_cuda_time_in_profiling_us = 0
        total_kernel_run_time_in_profiling_us = 0
        # if self.reward_config.coverage_reward.enable and correctness:
        final_reward = reward
        if correctness:
            coverage_dict = self.compute_coverage_reward(result)
            coverage = coverage_dict["coverage"]
            num_custom_kernel = coverage_dict["num_custom_kernel"]
            num_total_kernels = coverage_dict["num_total_kernels"]
            custom_kernel_cuda_time_in_profiling_us = coverage_dict["custom_kernel_cuda_time_in_profiling_us"]
            total_kernel_run_time_in_profiling_us = coverage_dict["total_kernel_run_time_in_profiling_us"]
            print(f"[DEBUG] coverage: {coverage}")
            print(f"[DEBUG] speedup: {speedup}")
            print(f"[DEBUG] num_custom_kernel: {num_custom_kernel}")
            print(f"[DEBUG] num_total_kernels: {num_total_kernels}")
            print(f"[DEBUG] custom_kernel_cuda_time_in_profiling_us: {custom_kernel_cuda_time_in_profiling_us}")
            print(f"[DEBUG] total_kernel_run_time_in_profiling_us: {total_kernel_run_time_in_profiling_us}")
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
            print(f"[HybridClient] calculate_reward_like_kernel error_message: {error_message}")
            print(f"[HybridClient] Task failed result: {result}")

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
        # Server returned a decoy kernel; force penalty and carry the marker.
        # TODO Temporary disable decoy kernel detection
        if result.get("decoy_kernel", False):
            try:
                print("[HybridClient] decoy_kernel detected; forcing reward -1")
            except Exception:
                pass
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
        # In fact, profiling is always None here since it is actually inside metadata
        profiling = result.get("profiling", None)

        if speedup is None:
            speedup = 0.0

        # is_speedup_positive = speedup >= (1 + self.speedup_eps) # ignore too small speedup

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

        # if self.reward_config.coverage_reward.enable and correctness:
        final_reward = reward
        if correctness:
            coverage_dict = self.compute_coverage_reward(result)
            coverage = coverage_dict["coverage"]
            num_custom_kernel = coverage_dict["num_custom_kernel"]
            num_total_kernels = coverage_dict["num_total_kernels"]
            custom_kernel_cuda_time_in_profiling_us = coverage_dict["custom_kernel_cuda_time_in_profiling_us"]
            total_kernel_run_time_in_profiling_us = coverage_dict["total_kernel_run_time_in_profiling_us"]

            print(f"[DEBUG] coverage: {coverage}")
            print(f"[DEBUG] speedup: {reward_speedup}")
            print(f"[DEBUG] num_custom_kernel: {num_custom_kernel}")
            print(f"[DEBUG] num_total_kernels: {num_total_kernels}")
            print(f"[DEBUG] custom_kernel_cuda_time_in_profiling_us: {custom_kernel_cuda_time_in_profiling_us}")
            print(f"[DEBUG] total_kernel_run_time_in_profiling_us: {total_kernel_run_time_in_profiling_us}")

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

    async def compute_batch_rewards(
        self,
        tasks: list[dict[str, Any]],
        *,
        use_reference_cache: Optional[bool] = None,
        is_valid: Optional[bool] = None,
        task_timeout: Optional[int] = None,
        task_timeout_in_client: Optional[int] = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        penalty_score = self.penalty_score
        if not tasks:
            return []
        # print(f"[DEBUG] Task timeout: {task_timeout or self.timeout}")
        effective_timeout = int(task_timeout or self.timeout)
        effective_timeout_in_client = int(task_timeout_in_client or self.task_timeout_in_client)

        # Validate timeout invariant: client timeout should be >= server timeout
        if effective_timeout_in_client < effective_timeout:
            print(
                f"[WARNING] task_timeout_in_client ({effective_timeout_in_client}s) < task_timeout ({effective_timeout}s)"  # noqa: E501
            )
            print("[WARNING] Adjusting task_timeout_in_client to match task_timeout to respect timeout invariant")
            effective_timeout_in_client = effective_timeout
        obj_refs: list[ray.ObjectRef] = []
        index_map: list[int] = []  # worker submission order -> original index
        prefilled: dict[int, dict[str, Any]] = {}
        submitted: int = 0
        skipped: int = 0
        # Map obj_ref index -> task metadata for heartbeat tracking of pending tasks.
        idx_to_task_info: dict[int, dict[str, Any]] = {}

        for idx, task in enumerate(tasks):
            kcode = task.get("kernel_code", "")
            ep = task.get("entry_point", "Model")
            ok, missing = self._preflight_validate(task.get("reference_code", ""), kcode, ep)
            if not ok:
                try:
                    print(f"[HybridClient] preflight failed(idx={idx}): missing {missing} entry_point={ep}")
                except Exception:
                    pass
                prefilled[idx] = {
                    "reward": penalty_score,
                    "speedup": 0.0,
                    "success": False,
                    "correctness": False,
                    "compiled": False,
                    "error": f"Client validation failed: missing {missing}",
                }
                continue

            # Per-task timeout handling: fall back to batch default when explicitly None.
            # Server-side task execution timeout.
            per_task_timeout_raw = task.get("task_timeout", None)
            per_task_timeout = effective_timeout if per_task_timeout_raw is None else int(per_task_timeout_raw)

            # Client-side task execution timeout.
            # Client-side timeout should be >= server-side timeout.
            # Because queuing is involved, client timeout should be >= server timeout.
            per_task_timeout_in_client_raw = task.get("task_timeout_in_client", None)
            per_task_timeout_in_client = (
                effective_timeout_in_client
                if per_task_timeout_in_client_raw is None
                else int(per_task_timeout_in_client_raw)
            )

            # Validate per-task timeout invariant
            if per_task_timeout_in_client < per_task_timeout:
                per_task_timeout_in_client = per_task_timeout
            # print(f"[DEBUG] Per task timeout: {per_task_timeout}")

            # Randomly log one task (~5%).
            try:
                if random.random() < 0.05:

                    def _clip2(s: Optional[str], n: int = 600) -> str:
                        try:
                            return (s or "")[:n]
                        except Exception:
                            return str(s)[:n]

                    print(
                        f"[HybridClient] DEBUG(entry_point={ep})\n[ref]\n{task.get('reference_code', '')}\n[kernel]\n{kcode}"  # noqa: E501
                    )
            except Exception:
                pass

            payload = {
                "task_id": task.get("task_id") or self._next_task_id("parallel_task"),
                "reference_code": task.get("reference_code", ""),
                "kernel_code": kcode,
                "backend": "triton",
                "num_correct_trials": task.get("num_correct_trials", 5),
                "num_perf_trials": task.get("num_perf_trials", 100),
                "timeout": per_task_timeout,
                "priority": "normal",
                "entry_point": ep,
                "is_valid": task.get("is_valid", is_valid),
                "verbose_errors": task.get("verbose_errors", True),
                "enable_profiling": task.get("enable_profiling", True),
                "detect_decoy_kernel": task.get("detect_decoy_kernel", True),
                "reference_backend": task.get("reference_backend", None),
                # KernelGym evaluation toolkit. Must be set explicitly: the
                # server ignores its DEFAULT_TOOLKIT env for /evaluate
                # (workflow controller pre-fills toolkit="kernelbench").
                # "sandbox_v3" enables AST decoy validation.
                "toolkit": task.get("toolkit") or "kernelbench",
            }

            # enforce detect decoy kernel if validate
            if payload["is_valid"]:
                print(f"Enforce detect decoy kernel if validate: {payload['detect_decoy_kernel']}")
                payload["detect_decoy_kernel"] = True

            ucache = task.get("use_reference_cache", use_reference_cache)
            if ucache:
                payload["use_reference_cache"] = True
                if task.get("uuid"):
                    payload["uuid"] = task["uuid"]

            # Record task metadata for heartbeat tracking.
            obj_ref_idx = len(obj_refs)  # Index of the obj_ref about to be appended to the list.
            idx_to_task_info[obj_ref_idx] = {
                "task_id": payload["task_id"],
                "entry_point": ep,
                "uuid": payload.get("uuid", ""),
                "orig_idx": idx,
            }

            obj_refs.append(
                self._worker.submit_and_poll.remote(
                    payload,
                    per_task_timeout_in_client,
                    self.max_retries,
                )
            )
            index_map.append(idx)
            submitted += 1

        try:
            skipped = len(prefilled)
            print(f"[HybridClient] batch submitted={submitted} skipped={skipped}")
        except Exception:
            pass

        # Heartbeat: report progress and token-bucket level every 60s.
        pending = set(range(len(obj_refs)))
        results: list[tuple[int, dict[str, Any]]] = []
        start_ts = time.time()
        # Track remaining refs to avoid reprocessing completed tasks and looping.
        remaining_refs: list[ray.ObjectRef] = list(obj_refs)
        ref_to_idx = {ref: i for i, ref in enumerate(obj_refs)}

        while remaining_refs:
            done, remaining_refs = await asyncio.to_thread(ray.wait, remaining_refs, num_returns=1, timeout=60)
            if done:
                ref = done[0]
                idx = ref_to_idx.get(ref, None)
                try:
                    res = await asyncio.to_thread(ray.get, ref)
                except Exception as e:
                    res = {"status": "failed", "error_message": str(e)}
                if idx is not None and idx in pending:
                    results.append((idx, res))
                    pending.discard(idx)
            else:
                elapsed = time.time() - start_ts
                # Rate-limiter dropped — heartbeat reports the sentinel.
                in_use = -1

                # Collect detailed info for pending tasks for logging.
                pending_tasks_info = []
                for p_idx in sorted(list(pending))[:10]:  # Log at most the first 10 pending tasks.
                    if p_idx in idx_to_task_info:
                        info = idx_to_task_info[p_idx]
                        pending_tasks_info.append(
                            f"task_id={info['task_id']} entry={info['entry_point']} uuid={info['uuid'][:8] if info['uuid'] else 'N/A'}"  # noqa: E501
                        )

                pending_summary = "; ".join(pending_tasks_info) if pending_tasks_info else "N/A"
                if len(pending) > 10:
                    pending_summary += f" ... (+{len(pending) - 10} more)"

                print(
                    f"[BatchHeartbeat] hybrid: completed={len(results)}/{len(obj_refs)}, pending={len(pending)}, elapsed={elapsed:.1f}s tokens_in_use={in_use}/{self.rate_limit}"  # noqa: E501
                )
                print(f"[BatchHeartbeat] pending_tasks: {pending_summary}")
        # Merge back to original order.
        merged: list[Optional[dict[str, Any]]] = [None] * len(tasks)
        # Fill prefilled first.
        for i, v in prefilled.items():
            merged[i] = v
        # Then fill worker results.
        for idx_in_obj, data in results:
            orig_idx = index_map[idx_in_obj]
            reward_func = self._get_reward_func()
            reward_summary = reward_func(data)
            merged[orig_idx] = self._merge_reward_result(data, reward_summary)
        # Fallback for any missing entries.
        for i, v in enumerate(merged):
            if v is None:
                merged[i] = {
                    "reward": penalty_score,
                    "speedup": 0.0,
                    "success": False,
                    "correctness": False,
                    "compiled": False,
                    "error": "Unknown error",
                    "num_custom_kernel": 0,
                    "num_total_kernels": 0,
                    "custom_kernel_cuda_time_in_profiling_us": 0,
                    "total_kernel_run_time_in_profiling_us": 0,
                }
        return merged  # type: ignore[return-value]
