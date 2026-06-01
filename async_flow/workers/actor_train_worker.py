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
import time
from typing import Any

import ray
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from recipe.async_flow.utils.metric.prometheus import marked_timer
from recipe.async_flow.utils.transfer_queue.tq_client import get_transferqueue_client
from recipe.async_flow.workers.base_async_worker import AsyncWorkerMixin
from recipe.async_flow.workers.data_dispatch_strategy import EngineBackend
from tensordict import TensorDict

from verl import DataProto
from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.ppo.metric_utils import compute_data_metrics
from verl.utils import tensordict_utils
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.py_functional import rename_dict
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import left_right_2_no_padding


@ray.remote
class ActorTrainWorker(ActorRolloutRefWorker, AsyncWorkerMixin):
    """Actor Train Worker - 从 TQ 获取训练数据，执行模型更新.

    继承 ActorRolloutRefWorker 获得训练能力，继承 AsyncWorkerMixin 获得 TQ 循环能力.

    支持：
    1. 异步训练更新
    2. 版本管理和权重同步
    3. 训练指标收集
    """

    # TQ 交互配置
    CONSUMER_NAME = "actor_train"
    INPUT_COLUMNS = (
        "prompt",
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
        "model_version",
        "old_logprobs",
        "reference_logprobs",
        "reward",
        "advantage",
        "token_level_scores",
        "returns",
    )
    OUTPUT_COLUMNS = ("metrics",)

    def __init__(self, config: DictConfig, role: str = "actor", flow_control_queue=None, **kwargs) -> None:
        """初始化 Actor Train Worker.

        Args:
            config: 所有配置内容
            role: worker 角色，默认为 "actor"
        """
        assert role == "actor", "ActorTrainWorker must have role 'actor'"
        # 调用父类初始化
        ActorRolloutRefWorker.__init__(self, config=config.actor_rollout_ref, role=role, **kwargs)

        # metrics
        self._metrics: list[dict] = []
        self._print_version_metrics = True
        self._last_processed_indexes = None

        self.staleness = config.async_flow.staleness
        self.n_samples = config.actor_rollout_ref.rollout.n
        self.calculate_entropy = config.actor_rollout_ref.actor.entropy_coeff != 0.0
        self.shuffle = config.actor_rollout_ref.actor.shuffle
        self.seed = config.actor_rollout_ref.actor.data_loader_seed

        self.INPUT_COLUMNS = list(self.INPUT_COLUMNS)
        if config.actor_rollout_ref.rollout.calculate_log_probs:
            self.INPUT_COLUMNS.append("rollout_log_probs")

        # 版本管理
        ppo_mini_batch_size = config.actor_rollout_ref.actor.ppo_mini_batch_size
        self.mini_batch_size = ppo_mini_batch_size * self.n_samples
        train_batch_size = config.data.train_batch_size * self.n_samples
        assert train_batch_size % self.mini_batch_size == 0, "ppo_train_batch_size must be multiple of mini_batch_size"
        self.optimizer_step = train_batch_size // self.mini_batch_size  # 参数更新步数
        self._mini_batch_iter = 0  # 步数迭代器 [0 - self.optimizer_step]
        self._current_version = 0
        self.weight_updated = True
        # 推理流量控制
        self.flow_control_queue = flow_control_queue

        self.topic = config.async_flow.get("experience_topic", "experience")
        self.EXTRA_FETCH_KWARGS = {"get_n_samples": True}
        self.init_async_worker(
            tq_client=get_transferqueue_client(),
            topic=self.topic,
            experience_count=self.mini_batch_size,  # Not real value here, should be divided by dp size
            engine_backend=EngineBackend.FSDP,
            dispatch_strategy_kwargs={"rank": self.rank, "world_size": self.world_size},
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_ckpt_engine(self):
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
        backend = checkpoint_engine_config.backend
        bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
        engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
        checkpoint_engine = CheckpointEngineRegistry.new(
            backend, is_master=(torch.distributed.get_rank() == 0), bucket_size=bucket_size, **engine_kwargs
        )
        from recipe.async_flow.utils.async_flow_checkpoint_engine import AsyncFlowCheckpointEngineWithCache

        self.checkpoint_engine = AsyncFlowCheckpointEngineWithCache(non_cache_engine=checkpoint_engine)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        """Execute checkpoint engine method.

        Args:
            method (str): Checkpoint engine method name.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        """
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    def _build_fetch_params(self) -> dict[str, Any]:
        """覆写参数 experience_count"""
        mini_batch_size_per_gpu = int(self._experience_count / self.actor.engine.get_data_parallel_size())
        params = {
            "consumer": self.CONSUMER_NAME,
            "experience_columns": list(self.INPUT_COLUMNS),
            "experience_count": mini_batch_size_per_gpu,
            "allowed_staleness": self.staleness,
            "latest_version": self._current_version,
            "get_n_samples": True,
        }
        params.update(self.EXTRA_FETCH_KWARGS)
        return params

    def on_process_begin(self) -> None:
        """等待参数更新"""
        update_param_wait_time = 0
        with marked_timer("wait_update", {}):
            while self._mini_batch_iter == 0 and not self.weight_updated:
                time.sleep(1)
                update_param_wait_time += 1
        self.logger.debug(
            f"[Trainer] wait update time:{update_param_wait_time}, {self._mini_batch_iter=}, {self.weight_updated=}"
        )

    def process_batch(self, payload: dict[str, Any], indexes: list[int]) -> dict[str, Any]:
        """处理一个批次：执行训练更新。

        1. 从 payload 提取训练数据
        2. 调用父类 update_actor 执行训练
        3. 返回训练指标
        """
        # 准备训练数据
        train_data = self._prepare_training_data(payload)
        # 执行训练
        torch.distributed.barrier()
        output_proto = self._do_train_step(train_data)
        self._mini_batch_iter += 1

        # 收集并汇总指标（复用 verl 原生的指标格式）
        metrics = tensordict_utils.get(output_proto, "metrics")
        metrics = rename_dict(metrics, "actor/")
        input_batch = DataProto.from_tensordict(tensor_dict=train_data, meta_info=None)
        metrics.update(compute_data_metrics(input_batch, use_critic=False))

        if self.config.rollout.calculate_log_probs:
            metrics.update(self._compute_debug_metrics(input_batch))

        self._last_processed_indexes = indexes
        self._tq_client.delete_experience(indexes=indexes, topic=self.topic)

        summary = {
            "version": float(self._current_version),
            **metrics,
        }
        self._metrics.append(summary)

        return {}

    def on_process_end(self) -> None:
        """更新版本并同步权重到 Worker"""
        if self._mini_batch_iter == self.optimizer_step:
            dist.barrier()
            self.weight_updated = False
            self._current_version += 1
            self._mini_batch_iter = 0
        self._release_flow_control(len(self._last_processed_indexes))
        self._last_processed_indexes = None

    def _release_flow_control(self, num_samples: int):
        assert num_samples % self.n_samples == 0, (
            f"train num_samples={num_samples} must be multiple of n_samples={self.n_samples}"
        )
        prompts_to_release = num_samples // self.n_samples
        self.flow_control_queue.get_nowait_batch(prompts_to_release)
        self.logger.info(
            f"[TRAIN] Released {prompts_to_release} capacity tokens, remaining={self.flow_control_queue.qsize()}"
        )

    def _prepare_training_data(self, payload: dict[str, Any]) -> TensorDict:
        """准备训练数据为 TensorDict 适配engine worker"""
        from torch.nn.utils.rnn import pad_sequence

        prompts = torch.stack(payload["prompt"]).long()
        input_ids = torch.stack(payload["input_ids"]).long()
        attention_mask = torch.stack(payload["attention_mask"]).long()
        position_ids = torch.stack(payload["position_ids"])
        responses = torch.stack(payload["responses"]).long()
        response_mask = torch.stack(payload["response_mask"]).long()
        response_length = responses.size(-1)
        global_token_num = torch.sum(attention_mask, dim=-1).tolist()
        old_logprobs = torch.stack(payload["old_logprobs"]).float()
        ref_logprobs = torch.stack(payload["reference_logprobs"]).float()
        token_level_rewards = torch.stack(payload["reward"]).float()
        token_level_scores = torch.stack(payload["token_level_scores"]).float()
        returns = torch.stack(payload["returns"]).float()

        # Pad advantages
        advantages_list = payload["advantage"]  # List[torch.Tensor]
        advantages_padded = pad_sequence(
            [adv.flatten() if isinstance(adv, torch.Tensor) else torch.tensor([adv]) for adv in advantages_list],
            batch_first=True,
            padding_value=0.0,
        ).float()
        # 扩展 advantages 到 response 长度
        if advantages_padded.dim() == 1:
            advantages_padded = advantages_padded.unsqueeze(-1)
        if advantages_padded.size(-1) != response_length:
            if advantages_padded.size(-1) > response_length:
                advantages_padded = advantages_padded[:, :response_length]
            else:
                advantages_padded = advantages_padded.expand(-1, response_length)

        # 创建 TensorDict
        batch_dict = {
            "input_ids": input_ids,
            "prompts": prompts,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "responses": responses,
            "response_mask": response_mask,
            "old_log_probs": old_logprobs,
            "ref_log_prob": ref_logprobs,
            "advantages": advantages_padded,
            "token_level_rewards": token_level_rewards,
            "token_level_scores": token_level_scores,
            "returns": returns,
        }

        rollout_log_probs_list = payload.get("rollout_log_probs", None)
        if rollout_log_probs_list:
            batch_dict["rollout_log_probs"] = torch.stack(rollout_log_probs_list).float()

        data = DataProto.from_single_dict(batch_dict, meta_info={"global_token_num": global_token_num})
        input_data = data.to_tensordict()
        input_data = left_right_2_no_padding(input_data)

        return input_data

    def _do_train_step(self, data: TensorDict) -> TensorDict:
        """执行一步训练。"""
        temperature = self.config.rollout.temperature
        tensordict_utils.assign_non_tensor(
            data,
            calculate_entropy=self.calculate_entropy,
            temperature=temperature,
            global_batch_size=self.mini_batch_size,
            mini_batch_size=self.mini_batch_size,
            seed=self.seed,
            dataloader_kwargs={"shuffle": self.shuffle},
        )
        # 调用父类的 train_mini_batch 方法
        torch.distributed.barrier()
        output_proto = self.update_actor(data)
        return output_proto

    def _compute_debug_metrics(self, data: DataProto) -> dict[str, Any]:
        """
        compute debug metrics between rollout and actor log probs,
        including pearson_correlation_coefficient and log_prob_diff.
        """
        rollout_log_probs = data.batch["rollout_log_probs"]
        if rollout_log_probs is None:
            return {}

        from verl.utils.debug.metrics import calculate_debug_metrics

        return calculate_debug_metrics(data)

    def get_experience_step(self) -> int:
        """获取worker的mini step。"""
        return self.optimizer_step

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_current_version(self) -> int:
        """获取当前模型版本。"""
        return self._current_version

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_metrics(self) -> list[dict]:
        """获取当前worker的Metrics。"""
        current_metrics = self._metrics
        self._metrics = []
        return current_metrics

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, version_id):
        self.logger.debug(f"[ActorTrainWorker] start send weights:{version_id=}, {self.rank=}!!!!!!!!!")
        torch.distributed.barrier()
        per_tensor_param, _ = self.actor.engine.get_per_tensor_param()
        await self.checkpoint_engine.send_weights(per_tensor_param)
        self._current_version = version_id
        self.weight_updated = True
        torch.distributed.barrier()
        self.logger.debug(f"[ActorTrainWorker] Finish send weights {version_id=}, {self.rank=}!!!!!!!!!")
