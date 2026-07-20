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
import copy
import time
from collections import deque
from typing import Any

import ray
import torch
from omegaconf import DictConfig
from recipe.async_flow.utils.metric.prometheus import marked_timer
from recipe.async_flow.utils.transfer_queue.tq_client import get_transferqueue_client
from recipe.async_flow.workers.base_async_worker import AsyncWorkerMixin
from recipe.async_flow.workers.data_dispatch_strategy import EngineBackend
from tensordict import TensorDict
from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

from verl import DataProto
from verl.checkpoint_engine.base import CheckpointEngineRegistry
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import tensordict_utils
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@ray.remote(concurrency_groups={"fwd_weight_update": 1})
class ActorForwardWorker(ActorRolloutRefWorker, AsyncWorkerMixin):
    """Actor Forward Worker - 从 TQ 获取 responses，计算 old logprobs，写回 TQ。

    继承 ActorRolloutRefWorker 获得前向计算能力，继承 AsyncWorkerMixin 获得 TQ 循环能力。
    """

    # TQ 交互配置
    CONSUMER_NAME = "actor_forward"
    INPUT_COLUMNS = (
        "prompt",
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
        "model_version",
    )
    OUTPUT_COLUMNS = ("old_logprobs", "model_version")

    def __init__(self, config: DictConfig, role: str = "actor", **kwargs) -> None:
        """初始化 Actor Forward Worker。

        Args:
            config: actor_rollout_ref 子配置（DictConfig 格式）
            role: worker 角色，默认为 "actor"
            async_flow_config: async_flow 配置
        """
        assert role == "actor", "ActorForwardWorker must have role 'actor'"
        local_config = copy.deepcopy(config.actor_rollout_ref)
        local_config.actor.fsdp_config.forward_only = True
        ActorRolloutRefWorker.__init__(self, config=local_config, role=role, **kwargs)

        self._active_version = -1
        self._pending_version = -1

        # 模型权重更新变量
        self._paused = False
        checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
        self.ckpt_backend = checkpoint_engine_config.backend
        if self.ckpt_backend == "flexfetch":
            from recipe.async_flow.utils.checkpoint_engine.flexfetch_checkpoint_engine import (  # noqa: F401
                FlexFetchCheckpointEngine,
            )
        self.staleness = config.async_flow.staleness
        self.temp_queue = deque(maxlen=self.staleness + 1)
        self.ckpt_queue = deque(maxlen=self.staleness + 1)

        train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        topic = config.async_flow.get("experience_topic", "experience")
        experience_count = config.async_flow.fwd_experience_count
        worker_count = config.async_resources.ref_fwd.nnodes * config.async_resources.ref_fwd.n_gpus_per_node
        assert train_batch_size % experience_count == 0
        self.experience_step = train_batch_size // (experience_count * worker_count)

        self.EXTRA_FETCH_KWARGS = {"get_n_samples": False}
        self._state_dict = {}
        self.init_async_worker(
            tq_client=get_transferqueue_client(),
            topic=topic,
            experience_count=experience_count,
            engine_backend=EngineBackend.FSDP,
            dispatch_strategy_kwargs={
                "rank": self.rank,
                "world_size": self.world_size,
                "query_usable_count_fn": self._query_usable_count,
                "needed_count_fn": lambda: int(self._experience_count) * self.world_size,
            },
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_ckpt_engine(self):
        if self.ckpt_backend != "flexfetch":
            checkpoint_engine_config = omega_conf_to_dataclass(self.config.rollout.checkpoint_engine)
            backend = checkpoint_engine_config.backend
            bucket_size = checkpoint_engine_config.update_weights_bucket_megabytes << 20
            engine_kwargs = checkpoint_engine_config.engine_kwargs.get(backend, {})
            self.checkpoint_engine = CheckpointEngineRegistry.new(backend, bucket_size=bucket_size, **engine_kwargs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE, blocking=False)
    def execute_checkpoint_engine(self, method: str, *args, **kwargs):
        return getattr(self.checkpoint_engine, method)(*args, **kwargs)

    async def _query_usable_count(self) -> int:
        """异步查询 TQ 当前 (topic, consumer) 视角下的全局可用样本数。

        get_data_usable_set 只查询不消费，可以反复调用。
        通过 ray 的 asyncio 接口获取，不阻塞 event loop。
        """
        # TODO: 切换到tq_client的接口，而不是通过manager调用
        count, _ = await self._tq_client.manager.get_data_usable_set.remote(
            self._topic,
            self.CONSUMER_NAME,
            list(self.INPUT_COLUMNS),
        )
        return int(count)

    def on_process_begin(self) -> None:
        paused_time = 0
        while self._paused:
            time.sleep(1)
            paused_time += 1
            self.logger.debug(f"[FWD] {self.rank=} waiting for weight update ...")

    def process_batch(self, payload: dict[str, Any], indexes: list[int]) -> dict[str, Any]:
        """处理一个批次：计算 actor log probs。"""
        prompt = torch.stack(payload["prompt"]).long()
        input_ids = torch.stack(payload["input_ids"]).long()
        attention_mask = torch.stack(payload["attention_mask"]).long()
        position_ids = torch.stack(payload["position_ids"])
        responses = torch.stack(payload["responses"]).long()
        response_mask = torch.stack(payload["response_mask"]).long()
        global_token_num = torch.sum(attention_mask, dim=-1).tolist()

        data = DataProto.from_single_dict(
            {
                "prompts": prompt,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
            },
            meta_info={"global_token_num": global_token_num},
        )
        input_data = data.to_tensordict()
        input_data = left_right_2_no_padding(input_data)

        if self._active_version != self._pending_version:
            self.logger.debug(f"[FWD] {self.rank=} loading weights {self._pending_version=}")
            with marked_timer("load_weight", {}):
                model = self.actor.engine.module
                options = StateDictOptions(full_state_dict=True, broadcast_from_rank0=False, strict=False)
                load_result = set_model_state_dict(model=model, model_state_dict=self._state_dict, options=options)
            self._active_version = self._pending_version
            if isinstance(self._state_dict, dict):
                self._state_dict.clear()
            self._state_dict = {}
            self.logger.info(f"[FWD] finishes updating weights with version {self._active_version=} {load_result=}")

        output_proto = self._do_compute_log_prob(input_data)
        self.logger.debug(f"[FWD] {self.rank=} finish computing")

        # 提取 log probs（返回 tensor 列表）
        logprobs_unpad = tensordict_utils.get(output_proto, "log_probs")
        logprobs_tensor = no_padding_2_padding(logprobs_unpad, input_data)
        old_logprobs = list(torch.unbind(logprobs_tensor.cpu().float(), dim=0))
        # 返回数据（将被写入 TQ，全部为 tensor 格式）
        return {"old_logprobs": old_logprobs}

    def _do_compute_log_prob(self, data: TensorDict) -> TensorDict:
        """执行 log prob 计算。"""
        self.logger.debug(f"[FWD] {self.rank=} start computing")
        temperature = self.config.rollout.temperature
        tensordict_utils.assign_non_tensor(data, calculate_entropy=True, compute_loss=False, temperature=temperature)
        # 调用父类的 compute_log_prob
        output_proto = self.compute_log_prob(data)
        return output_proto

    def get_experience_step(self) -> int:
        """获取worker的mini step。"""
        return self.experience_step

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_current_version(self) -> int:
        """获取当前模型版本。"""
        return self._active_version

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def pause(self):
        with self.lock:
            self._paused = True
        self.logger.info(f"[FWD] model paused: {self._paused}, {self.rank=}")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, version_id):
        self.logger.info(f"[FWDWorker] starts to receive weights (with version {version_id}) at {self.rank=}")
        with marked_timer("update_weights", {}):
            if self.ckpt_backend == "flexfetch":
                weights_gen = self.checkpoint_engine.receive_weights(version_id=version_id)
            else:
                weights_gen = self.checkpoint_engine.receive_weights()
            async for name, tensor in weights_gen:
                self._state_dict[name] = tensor.detach().cpu()
        self.logger.info(f"[FWDWorker]  finishes updating weights with version {self._active_version=} .")
        self._pending_version = version_id
        self._paused = False
        self.logger.info(f"[FWDWorker] finish receiving weights (with version {version_id}) at {self.rank=}")
