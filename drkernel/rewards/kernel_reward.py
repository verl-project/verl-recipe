# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Kernel 奖励函数实现
与 KernelServer 集成，评估内核代码的质量和性能
"""

import asyncio
import logging
import re
import threading
from typing import Any

from recipe.drkernel.rewards.reward_client import KernelRewardClient

# 全局客户端实例与其配置，复用连接且在配置变更时重建
_global_client = None
_global_client_cfg = {}


def _run_coro_in_new_loop(coro):
    """Run an async coroutine to completion in a worker thread with a
    fresh event loop.

    `compute_kernel_reward_batch` is sync but its body needs to await the
    async `KernelRewardClient.compute_batch_rewards`. Running the coroutine
    in a dedicated thread with a private loop avoids interaction with any
    caller-owned loop (e.g. when invoked from inside a Ray async actor's
    executor).
    """
    box: dict[str, Any] = {}

    def _runner():
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            box["result"] = loop.run_until_complete(coro)
        except BaseException as exc:  # noqa: BLE001
            box["exc"] = exc
        finally:
            try:
                loop.close()
            finally:
                asyncio.set_event_loop(None)

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join()
    if "exc" in box:
        raise box["exc"]
    return box.get("result")


def extract_reference_code(solution_str: str) -> str:
    """
    从解决方案字符串中提取参考代码

    Args:
        solution_str: 包含提示和响应的完整字符串

    Returns:
        提取的参考代码
    """
    # 查找参考实现标记
    patterns = [
        r"# Reference Implementation\s*\n(.*?)(?=# Your Task|# Generate|$)",
        r"```python\s*# Reference\s*\n(.*?)```",
        r"# PyTorch Reference:\s*\n(.*?)(?=# Task|# Generate|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()

    # 如果没有找到特定标记，尝试提取第一个 Python 代码块
    code_block_match = re.search(r"```python\s*\n(.*?)```", solution_str, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1).strip()

    # 回退到整个字符串
    return solution_str


def extract_kernel_code(solution_str: str) -> str:
    """
    从解决方案字符串中提取内核代码

    Args:
        solution_str: 包含提示和响应的完整字符串

    Returns:
        提取的内核代码
    """
    # 查找内核实现标记
    patterns = [
        r"```triton[ \t]*\r?\n(.*?)```",
        r"# Kernel Implementation\s*\n(.*?)(?=# End|$)",
        r"```python\s*# Kernel\s*\n(.*?)```",
        r"# Your implementation:\s*\n(.*?)(?=# End|$)",
        r"# Generated kernel:\s*\n(.*?)(?=# End|$)",
    ]

    for pattern in patterns:
        match = re.search(pattern, solution_str, re.DOTALL)
        if match:
            return match.group(1).strip()

    # 如果没有找到特定标记，尝试提取最后一个代码块
    code_blocks = re.findall(r"```(?:\w+)?\s*\n?(.*?)```", solution_str, re.DOTALL)
    if code_blocks:
        return code_blocks[-1].strip()

    # 回退：假设整个响应就是内核代码
    return solution_str


def _build_tasks(solution_strs, ground_truths, entry_points, uuids, is_valid, reward_config):
    """Build the per-task payload list. Pure data assembly — reads
    `reward_config` fields once so the async/sync wrappers stay slim."""
    try:
        task_timeout = getattr(reward_config, "task_timeout", None)
        task_timeout_in_client = getattr(reward_config, "task_timeout_in_client", None)
    except Exception:
        task_timeout = None
        task_timeout_in_client = None

    num_perf_trials = reward_config.num_perf_trials
    num_correct_trials = reward_config.num_correct_trials
    enable_profiling = reward_config.enable_profiling
    verbose_errors = reward_config.verbose_errors
    detect_decoy_kernel = reward_config.detect_decoy_kernel
    reference_backend = reward_config.reference_backend
    # Optional for backward compatibility with older trainer YAMLs.
    toolkit = getattr(reward_config, "toolkit", "kernelbench")

    tasks = []
    for i, solution_str in enumerate(solution_strs):
        tasks.append(
            {
                "reference_code": ground_truths[i],
                "kernel_code": extract_kernel_code(solution_str),
                "entry_point": entry_points[i],
                "use_reference_cache": False,
                "uuid": uuids[i] if uuids is not None else "",
                "is_valid": is_valid,
                "task_timeout": task_timeout,
                "task_timeout_in_client": task_timeout_in_client,
                "num_correct_trials": num_correct_trials,
                "num_perf_trials": num_perf_trials,
                "enable_profiling": enable_profiling,
                "verbose_errors": verbose_errors,
                "detect_decoy_kernel": detect_decoy_kernel,
                "reference_backend": reference_backend,
                "toolkit": toolkit,
            }
        )
    return tasks, task_timeout, task_timeout_in_client


def _get_or_init_client(reward_config):
    """Lazily build (and cache) a KernelRewardClient keyed on the config."""
    server_url = getattr(reward_config, "server_url", None)
    if not server_url:
        raise ValueError("server_url is required and cannot be None or empty")

    global _global_client, _global_client_cfg
    if _global_client is None or _global_client_cfg is not reward_config:
        _global_client = KernelRewardClient(reward_config=reward_config)
        _global_client_cfg = reward_config
    return _global_client


async def _compute_batch_rewards_async(solution_strs, ground_truths, entry_points, **kwargs):
    """Pure-async batch reward computation — the canonical path.

    The HTTP fan-out stays inside `client.compute_batch_rewards`, which is
    itself async. The sync wrapper `compute_kernel_reward_batch` bridges
    sync→async via `_run_coro_in_new_loop`.
    """
    reward_config = kwargs.get("reward_config", None)
    if hasattr(reward_config, "reward_model"):
        reward_config = reward_config.reward_model
    if reward_config is None:
        raise ValueError("reward_config is required")

    uuids = kwargs.get("uuids", None)
    is_valid = kwargs.get("is_valid", False)

    tasks, task_timeout, task_timeout_in_client = _build_tasks(
        solution_strs, ground_truths, entry_points, uuids, is_valid, reward_config
    )

    client = _get_or_init_client(reward_config)
    return await client.compute_batch_rewards(
        tasks,
        use_reference_cache=False,
        is_valid=is_valid,
        task_timeout=task_timeout,
        task_timeout_in_client=task_timeout_in_client,
    )


def compute_kernel_reward_batch(solution_strs: list, ground_truths: list, entry_points: str, **kwargs) -> list:
    """Synchronous batch reward computation — DR.Kernel-faithful entry point.

    Wired into the kernel-RL recipe via `drkernel_kernel_trainer_native.yaml`
    as `custom_reward_function.name`. The DR.Kernel-style `kernel_async`
    reward manager (`recipe.drkernel.workers.reward_manager.kernel_async`)
    calls this with a one-element batch per trajectory.
    """
    reward_config = kwargs.get("reward_config", None)
    if hasattr(reward_config, "reward_model"):
        reward_config = reward_config.reward_model
    try:
        return _run_coro_in_new_loop(_compute_batch_rewards_async(solution_strs, ground_truths, entry_points, **kwargs))
    except Exception as e:
        logging.error(f"Error in compute_kernel_reward_batch: {e}")
        return [
            {
                "score": reward_config.reward_policy.penalties.penalty_score,
                "reward": reward_config.reward_policy.penalties.penalty_score,
                "correctness": False,
                "success": False,
                "error": str(e),
            }
            for _ in solution_strs
        ]
