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
Kernel 奖励管理器，专门用于 kernel code RL 训练
复用 laser 的架构，集成 KernelServer 进行性能评估
"""

import logging

import torch
from recipe.drkernel.rewards.fast_at import compute_fast_at_indicators, resolve_thresholds

from verl.utils.reward_score import default_compute_score


# @register("kernel")
class AsyncKernelRewardManager:
    """Kernel 奖励管理器，集成 KernelServer 进行内核性能评估"""

    def __init__(
        self, tokenizer, num_examine=5, compute_score=None, reward_fn_key="data_source", reward_config=None, **kwargs
    ) -> None:
        """
        初始化 KernelRewardManager

        Args:
            tokenizer: 分词器
            num_examine: 打印到控制台的样本数量
            compute_score: 自定义评分函数
            reward_fn_key: 用于识别数据源的键
            reward_config: Hydra/OmegaConf 下的 reward_model 配置（唯一客户端配置载体）
            **kwargs: 其他参数
        """

        if hasattr(reward_config, "reward_model"):
            reward_config = reward_config.reward_model

        self.reward_config = reward_config
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.is_valid = kwargs.get("is_valid", False)
        self.server_url = self.reward_config.server_url
        self.reward_policy = self.reward_config.reward_policy
        self.task_timeout = self.reward_config.task_timeout
        self.print_status = getattr(self.reward_config, "print_status", False)

        # 验证 server_url 不为空
        if not self.server_url:
            raise ValueError("server_url is required for KernelRewardManager")

        self.reward_weights = self.reward_config.reward_weights

        # Speedup thresholds for the per-trajectory Fast@X indicators.
        # Override from the launcher via:
        #   reward_model.reward_kwargs.fast_at_thresholds=[0.5,1.0,1.5]
        self.fast_at_thresholds = resolve_thresholds(getattr(self.reward_config, "fast_at_thresholds", None))

        self.logger = logging.getLogger(__name__)

        # 打印配置信息（全部来源于 reward_config）
        self.logger.info(f"KernelRewardManager initialized with server: {self.server_url}")
        self.logger.info(f"Reward weights: {self.reward_weights}")
        try:
            enhanced = self.reward_config.enhanced
            use_sandbox_rate_limit = self.reward_config.use_sandbox_rate_limit
            rate_limit = self.reward_config.rate_limit
            timeout = self.reward_config.timeout
            max_concurrent = self.reward_config.max_concurrent
            print(
                f"[RewardManager] cfg enhanced={enhanced} use_sandbox_rate_limit={use_sandbox_rate_limit} rate_limit={rate_limit} timeout={timeout} max_concurrent={max_concurrent}"  # noqa: E501
            )
        except Exception:
            pass

    def execute_env(self, response_str: str, ground_truth: str, entry_point: str, uuid: str, response_ids: list[int]):
        """
        Execute the environment and return the result
        We split it since we hope to re-evaluate when the speedup value is anomaly large.
        """

        try:
            # 准备批量计算的参数
            solution_strs = [response_str]
            ground_truths = [ground_truth]
            entry_points = [entry_point]
            uuids = [uuid]

            # 调用评分函数
            if callable(self.compute_score):
                # 检查是否支持批量处理（更稳健地识别 partial 包裹的真实函数）
                is_batch = False
                func_name = ""
                # 直接标记优先
                if getattr(self.compute_score, "_is_batch", False):
                    is_batch = True
                # 尝试从 partial 的 raw_fn 中获取标记或名称
                underlying_func = None
                if hasattr(self.compute_score, "func"):
                    # functools.partial(func, *args, **kwargs) 中的 func
                    underlying_func = self.compute_score.func
                    if getattr(underlying_func, "_is_batch", False):
                        is_batch = True
                # 对于 _call_with_kwargs 这类包装，raw_fn 通常在 partial.args[0]
                if hasattr(self.compute_score, "args") and self.compute_score.args:
                    possible_raw_fn = self.compute_score.args[0]
                    if callable(possible_raw_fn):
                        underlying_func = possible_raw_fn
                        if getattr(underlying_func, "_is_batch", False):
                            is_batch = True
                # 名称兜底判断
                if hasattr(self.compute_score, "__name__"):
                    func_name = self.compute_score.__name__
                elif underlying_func is not None and hasattr(underlying_func, "__name__"):
                    func_name = underlying_func.__name__
                if "batch" in func_name.lower():
                    is_batch = True

                # 仅传递必要控制参数：reward_config 与 is_valid
                safe_kwargs = {"reward_config": self.reward_config, "is_valid": self.is_valid}

                if is_batch:
                    results = self.compute_score(solution_strs, ground_truths, entry_points, uuids=uuids, **safe_kwargs)
                else:
                    # 单个处理
                    results = []
                    for i, (solution_str, ground_truth, entry_point) in enumerate(
                        zip(solution_strs, ground_truths, entry_points, strict=False)
                    ):
                        uuid_val = uuids[i] if i < len(uuids) else None
                        single_kwargs = {**safe_kwargs, "entry_point": entry_point, "uuid": uuid_val}
                        result = self.compute_score(
                            solution_str=solution_str, ground_truth=ground_truth, **single_kwargs
                        )
                        results.append(result)
            else:
                # 使用默认评分函数
                results = []
                for i, (solution_str, ground_truth, entry_point) in enumerate(
                    zip(solution_strs, ground_truths, entry_points, strict=False)
                ):
                    uuid_val = uuids[i] if i < len(uuids) else None
                    result = default_compute_score(
                        solution_str=solution_str,
                        ground_truth=ground_truth,
                        entry_point=entry_point,
                        uuid=uuid_val,
                        is_valid=self.is_valid,
                    )
                    results.append(result)

        except Exception as e:
            self.logger.error(f"Error in reward computation: {e}")
            results = [
                {
                    "score": self.reward_config.reward_policy.penalties.penalty_score,
                    "reward": self.reward_config.reward_policy.penalties.penalty_score,
                    "correctness": False,
                    "success": False,
                    "compiled": False,
                    "error": str(e),
                    "num_custom_kernel": 0,
                    "num_total_kernels": 0,
                    "custom_kernel_cuda_time_in_profiling_us": 0,
                    "total_kernel_run_time_in_profiling_us": 0,
                }
                for _ in range(len(response_ids))
            ]

        if len(results) != 1:
            raise ValueError(f"The length of results should be 1, but got {len(results)}")

        return results

    # def __call__(self, data: DataProto, return_dict: bool = False, **kwargs):
    def __call__(
        self,
        response_ids: list[int],
        response_str: str,
        ground_truth: str,
        entry_point: str,
        uuid: str,
        return_dict: bool = True,
        return_full_state: bool = False,
        **kwargs,
    ):
        """
        Async reward manager for kernel code RL training

        Only pass necessary data to the reward manager to keep efficient in async mode.

        Args:
            response_ids: Response token ids
            response_str: Response string
            ground_truth: Ground truth
            entry_point: Entry point
            uuid: UUID
            return_dict: Whether to return a dictionary
            **kwargs: Additional keyword arguments for the reward function
        Returns:
            Reward tensor or a dictionary containing reward information
        """
        # 如果已经有 rm_scores，直接返回
        # if "rm_scores" in data.batch.keys():
        #     if return_dict:
        #         return {"reward_tensor": data.batch["rm_scores"]}
        #     else:
        #         return data.batch["rm_scores"]

        # 初始化返回张量，长度与响应截断长度一致，避免在后续裁剪时丢失分数
        max_response_length = kwargs.get("response_length")
        valid_response_length = len(response_ids)
        if max_response_length is not None:
            valid_response_length = min(valid_response_length, int(max_response_length))
        valid_response_length = max(valid_response_length, 1)

        reward_tensor = torch.zeros(valid_response_length, dtype=torch.float32)
        # reward_extra_info = defaultdict(list)
        reward_extra_info = {}

        # 性能指标张量
        correctness_tensor = torch.zeros(1, dtype=torch.float32)
        performance_tensor = torch.zeros(1, dtype=torch.float32)
        compilation_tensor = torch.zeros(1, dtype=torch.float32)

        print(f"[DEBUG] entry point in reward manager: {entry_point}")

        # 使用计算函数进行评估

        results = self.execute_env(response_str, ground_truth, entry_point, uuid, response_ids)

        speedup = results[0].get("speedup", 0.0)

        if speedup is None:
            speedup = 0.0

        if speedup > self.reward_config.speedup_reward_upper_bound:
            print("[DEBUG] speedup is anomaly large, re-execute the environment")
            results = self.execute_env(response_str, ground_truth, entry_point, uuid, response_ids)
            speedup = results[0].get("speedup", 0.0)

        results = results[0]

        score = results.get("score", results.get("reward", 0.0))
        num_custom_kernel = results.get("num_custom_kernel", 0)
        num_total_kernels = results.get("num_total_kernels", 0)
        custom_kernel_cuda_time_in_profiling_us = results.get("custom_kernel_cuda_time_in_profiling_us", 0)
        total_kernel_run_time_in_profiling_us = results.get("total_kernel_run_time_in_profiling_us", 0)
        correctness = results.get("correctness", False)
        success = results.get("success", False)
        compiled = results.get("compiled", False)
        speedup = results.get("speedup", 0.0)
        if speedup is None:
            speedup = 0.0
        status = results.get("status", "unknown")
        err_msg = results.get("error") or ""
        is_speedup_positive = speedup >= 1.0 + self.reward_config.speedup_eps
        is_decoy_kernel = results.get("decoy_kernel", False)

        target_index = valid_response_length - 1
        reward_tensor[target_index] = score
        correctness_tensor[0] = float(correctness)
        performance_tensor[0] = speedup
        compilation_tensor[0] = float(compiled)

        reward_extra_info["correctness"] = correctness
        reward_extra_info["performance"] = speedup
        reward_extra_info["is_speedup_positive"] = is_speedup_positive
        reward_extra_info["is_decoy_kernel"] = is_decoy_kernel
        reward_extra_info["compilation"] = compiled
        reward_extra_info["success"] = success
        reward_extra_info["status"] = status
        reward_extra_info["error"] = err_msg

        print(f"[DEBUG] num_custom_kernel in reward manager: {num_custom_kernel}")
        print(f"[DEBUG] num_total_kernels in reward manager: {num_total_kernels}")
        print(
            f"[DEBUG] custom_kernel_cuda_time_in_profiling_us in reward manager: {custom_kernel_cuda_time_in_profiling_us}"  # noqa: E501
        )
        print(
            f"[DEBUG] total_kernel_run_time_in_profiling_us in reward manager: {total_kernel_run_time_in_profiling_us}"
        )
        # new features
        reward_extra_info["num_custom_kernel"] = num_custom_kernel
        reward_extra_info["num_total_kernels"] = num_total_kernels
        num_coverage = 0
        if num_total_kernels > 0:
            num_coverage = num_custom_kernel / num_total_kernels
        reward_extra_info["num_coverage"] = float(f"{num_coverage:.2f}")
        reward_extra_info["custom_kernel_cuda_time_in_profiling_us"] = custom_kernel_cuda_time_in_profiling_us
        reward_extra_info["total_kernel_run_time_in_profiling_us"] = total_kernel_run_time_in_profiling_us
        time_coverage = 0
        if total_kernel_run_time_in_profiling_us > 0:
            time_coverage = custom_kernel_cuda_time_in_profiling_us / total_kernel_run_time_in_profiling_us
        reward_extra_info["time_coverage"] = float(f"{time_coverage:.2f}")

        # Fast@X indicators — fraction of kernels that pass correctness AND achieve
        # >= X speedup over Torch. The mean of these binary values across a batch
        # (training) or across rollouts/prompts (validation, via
        # `process_validation_metrics`) recovers the Fast@X metric.
        reward_extra_info.update(compute_fast_at_indicators(correctness, speedup, self.fast_at_thresholds))

        # reward_extra_info["correctness"].append(correctness)
        # reward_extra_info["performance"].append(speedup)
        # reward_extra_info["is_speedup_positive"].append(is_speedup_positive)
        # reward_extra_info["is_decoy_kernel"].append(is_decoy_kernel)
        # reward_extra_info["compilation"].append(compiled)
        # reward_extra_info["success"].append(success)
        # reward_extra_info.setdefault("status", []).append(status)
        # reward_extra_info.setdefault("error", []).append(err_msg or "")

        if self.print_status:
            self.logger.info(
                f"[KernelEvalStatus] idx={0} status={status} compiled={compiled} correct={correctness} speedup={speedup} uuid={uuid} entry={entry_point} error={err_msg}"  # noqa: E501
            )

        if return_dict:
            reward_extra_info["correctness_tensor"] = correctness_tensor
            reward_extra_info["performance_tensor"] = performance_tensor
            reward_extra_info["compilation_tensor"] = compilation_tensor

            return_dict = {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }

            if return_full_state:
                return_dict["env_state"] = results

            return return_dict
        else:
            if return_full_state:
                return reward_tensor, reward_extra_info, results
            else:
                return reward_tensor, reward_extra_info
