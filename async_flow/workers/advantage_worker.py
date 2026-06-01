#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import DictConfig
from recipe.async_flow.utils.transfer_queue.tq_client import get_transferqueue_client
from recipe.async_flow.workers.base_async_worker import AsyncWorkerMixin
from recipe.async_flow.workers.data_dispatch_strategy import EngineBackend
from tensordict import TensorDict

from verl import DataProto
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage


@ray.remote
class AdvantageWorker(AsyncWorkerMixin):
    """Advantage Worker - 从 TQ 获取 reward，计算 advantages，写回 TQ。

    继承 AsyncWorkerMixin 获得 TQ 循环能力。
    """

    # TQ 交互配置
    CONSUMER_NAME = "advantage"
    INPUT_COLUMNS = (
        "prompt",
        "prompt_length",
        "responses",
        "response_length",
        "model_version",
        "labels",
        "input_ids",
        "attention_mask",
        "position_ids",
        "prompt_uid",
        "raw_prompt",
        "data_source",
        "rm_scores",
    )
    OUTPUT_COLUMNS = ("reward", "advantage", "token_level_scores", "returns")

    def __init__(
        self,
        config: DictConfig,
    ) -> None:
        """初始化 Reward Adv Worker。

        Args:
            config: 完整配置（DictConfig 格式）
            scoring_fn: 自定义 scoring function
        """
        # 提取 reward_model 配置用于父类初始化
        self.config = config

        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        engine_backend = EngineBackend.NONE
        topic = config.async_flow.get("experience_topic", "experience")
        experience_count = config.async_flow.reward_experience_count
        n_samples = config.actor_rollout_ref.rollout.n
        assert experience_count % n_samples == 0, (
            f"Advantage worker {experience_count=} must be divisible by {n_samples=}"
        )
        self.EXTRA_FETCH_KWARGS = {"get_n_samples": True}
        self.init_async_worker(
            tq_client=get_transferqueue_client(),
            topic=topic,
            experience_count=experience_count,
            engine_backend=engine_backend,
            dispatch_strategy_kwargs={},
        )

        # Rollout data logging configuration
        self.rollout_data_dir = config.trainer.get("rollout_data_dir", None)
        if self.rollout_data_dir:
            from verl.utils import hf_tokenizer

            trust_remote_code = config.data.get("trust_remote_code", False)
            self.tokenizer = hf_tokenizer(config.actor_rollout_ref.model.path, trust_remote_code=trust_remote_code)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """初始化模型。
        优势计算无需初始化模型。
        """
        self.logger.info("Advantage worker skipping model initialization")

    def process_batch(self, payload: dict[str, Any], indexes: list[int]) -> dict[str, Any]:
        """处理一个批次：计算 rewards 和 advantages（按 group 计算 baseline）。"""
        reward_metrics = {}
        batch = self._convert_tensor_to_dataproto(payload)  # tqbridge 需要 BatchMeta 暂时不支持

        reward_tensor = batch.batch["rm_scores"]
        reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
        reward_extra_infos_dict = (
            {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
        )

        batch, reward_metrics = self.compute_advantage(batch, reward_tensor, reward_extra_infos_dict, reward_metrics)

        # Async log rollout data if enabled
        if self.rollout_data_dir:
            # Get step from model_version in payload
            model_version_list = payload.get("model_version", [0] * len(batch))
            step = int(model_version_list[0]) if isinstance(model_version_list[0], (int, torch.Tensor)) else 0
            # TODO: the _log_rollout_data dump operation need to be implemented asynchronously
            self._log_rollout_data(batch, reward_extra_infos_dict, self.rollout_data_dir, step)

        # 返回数据（将被写入 TQ）
        return {
            "reward": list(torch.unbind(batch.batch["token_level_rewards"].float(), dim=0)),
            "advantage": list(torch.unbind(batch.batch["advantages"].float(), dim=0)),
            "token_level_scores": list(torch.unbind(batch.batch["token_level_scores"].float(), dim=0)),
            "returns": list(torch.unbind(batch.batch["returns"].float(), dim=0)),
        }

    def _convert_tensor_to_dataproto(self, payload: dict[str, Any]) -> DataProto:
        """
        将从 TQ 中取出的 payload 转换成 DataProto
        """

        def _get_attribute(column_name):
            if isinstance(payload[column_name], torch.Tensor):
                return payload[column_name]
            else:
                return torch.stack(payload[column_name])

        # 先创建 TensorDict
        batch_size = len(payload["input_ids"])
        batch = TensorDict(
            {
                "input_ids": _get_attribute("input_ids"),
                "attention_mask": _get_attribute("attention_mask"),
                "position_ids": _get_attribute("position_ids"),
                "prompts": _get_attribute("prompt"),
                "responses": _get_attribute("responses"),
                "rm_scores": _get_attribute("rm_scores"),
            },
            batch_size=torch.Size([batch_size]),
        )

        # 再创建 non_tensor_batch，需要 str_labels 和 uid
        reward_model = np.array([{"ground_truth": label} for label in payload["labels"]], dtype=object)
        non_tensor_batch = {
            "data_source": np.array(payload["data_source"]),
            "uid": np.array(payload["prompt_uid"]),
            "reward_model": reward_model,
            "raw_prompt": np.array(payload["raw_prompt"]),
        }
        batch = DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info={})

        return batch

    def compute_advantage(
        self, batch, reward_tensor, reward_extra_infos_dict, reward_metrics
    ) -> tuple[DataProto, dict]:
        # 计算 advantages（按 group_size 分组，使用同一 prompt 的样本求均值作为 baseline）
        rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
        bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

        # we combine with rule-based rm
        reward_extra_infos_dict: dict[str, list]
        batch.batch["token_level_scores"] = reward_tensor

        if reward_extra_infos_dict:
            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

        # compute rewards. apply_kl_penalty if available
        if self.config.algorithm.use_kl_in_reward:
            batch, kl_metrics = apply_kl_penalty(
                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
            )
            reward_metrics.update(kl_metrics)
        else:
            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

        # Compute rollout correction: IS weights, rejection sampling, and metrics
        # Only runs in decoupled mode (computes once per batch using stable π_old)
        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
        if (
            rollout_corr_config is not None
            and "rollout_log_probs" in batch.batch
            and not bypass_recomputing_logprobs  # Only in decoupled mode
        ):
            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

            # Compute IS weights, apply rejection sampling, compute metrics
            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
            # IS and off-policy metrics already have rollout_corr/ prefix
            reward_metrics.update(is_metrics)
        # 使用 VeRL 原生的功能计算advantage
        norm_adv_by_std_in_grpo = self.config.algorithm.get(
            "norm_adv_by_std_in_grpo", True
        )  # GRPO adv normalization factor

        batch = compute_advantage(
            batch,
            adv_estimator=self.config.algorithm.adv_estimator,
            gamma=self.config.algorithm.gamma,
            lam=self.config.algorithm.lam,
            num_repeat=self.config.actor_rollout_ref.rollout.n,
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            config=self.config.algorithm,
        )

        return batch, reward_metrics

    def get_experience_step(self) -> int:
        """获取worker的mini step。"""
        return 1

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path, step):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        filename = os.path.join(dump_path, f"{step}_{timestamp}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [step] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        self.logger.info(f"[RewardAdvWorker] Dumped generations to {filename}")

    def _log_rollout_data(self, batch: DataProto, reward_extra_infos_dict: dict, rollout_data_dir: str, step: int):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """

        inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
        scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
        sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

        reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
        if "request_id" in batch.non_tensor_batch:
            reward_extra_infos_dict.setdefault(
                "request_id",
                batch.non_tensor_batch["request_id"].tolist(),
            )
        try:
            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
                step=step,
            )
        except Exception as e:
            self.logger.error(f"[RewardAdvWorker] log worker failed: {e}")
