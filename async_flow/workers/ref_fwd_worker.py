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
from typing import Any

import ray
import torch
from omegaconf import DictConfig
from recipe.async_flow.utils.transfer_queue.tq_client import get_transferqueue_client
from recipe.async_flow.workers.base_async_worker import AsyncWorkerMixin
from recipe.async_flow.workers.data_dispatch_strategy import EngineBackend

from verl import DataProto
from verl.utils import tensordict_utils
from verl.workers.engine_workers import ActorRolloutRefWorker
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


@ray.remote
class ReferenceForwardWorker(ActorRolloutRefWorker, AsyncWorkerMixin):
    """
    Reference Forward Worker - 从 TQ 获取 responses，计算 reference logprobs，写回 TQ。
    """

    # TQ 交互配置
    CONSUMER_NAME = "ref_forward"
    INPUT_COLUMNS = (
        "prompt",
        "input_ids",
        "attention_mask",
        "position_ids",
        "responses",
        "response_mask",
        "model_version",
    )
    OUTPUT_COLUMNS = ("reference_logprobs",)

    def __init__(self, config: DictConfig, role: str = "ref", **kwargs) -> None:
        """初始化 Reference Forward Worker。

        Args:
            config: actor_rollout_ref 子配置（DictConfig 格式）
            role: worker 角色，必须为 "ref"
            async_flow_config: async_flow 配置
        """
        assert role == "ref", "ReferenceForwardWorker must have role 'ref'"
        ActorRolloutRefWorker.__init__(self, config=config.actor_rollout_ref, role=role, **kwargs)

        train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        topic = config.async_flow.get("experience_topic", "experience")
        experience_count = config.async_flow.ref_experience_count
        worker_count = config.async_resources.ref_fwd.nnodes * config.async_resources.ref_fwd.n_gpus_per_node
        assert train_batch_size % experience_count == 0
        self.experience_step = train_batch_size // (experience_count * worker_count)  # 参数更新步数
        self.EXTRA_FETCH_KWARGS = {"get_n_samples": False}
        self.init_async_worker(
            tq_client=get_transferqueue_client(),
            topic=topic,
            experience_count=experience_count,
            engine_backend=EngineBackend.FSDP,
            dispatch_strategy_kwargs={"rank": self.rank, "world_size": self.world_size},
        )

    def process_batch(self, payload: dict[str, Any], indexes: list[int]) -> dict[str, Any]:
        """处理一个批次：计算 reference log probs。"""
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

        # 调用父类计算 reference log probs
        """执行 reference log prob 计算。"""
        temperature = self.config.rollout.temperature
        tensordict_utils.assign_non_tensor(
            input_data,
            temperature=temperature,
            calculate_entropy=False,
            compute_loss=False,
        )
        # 调用父类的 compute_ref_log_prob
        output_proto = self.compute_ref_log_prob(input_data)

        # 提取 log probs（返回 tensor 列表）
        """从 DataProto 中提取 log probs，返回 tensor 列表。"""
        logprobs_unpad = tensordict_utils.get(output_proto, "log_probs")
        logprobs_tensor = no_padding_2_padding(logprobs_unpad, input_data)
        ref_logprobs = list(torch.unbind(logprobs_tensor.cpu().float(), dim=0))
        # 返回数据（将被写入 TQ，全部为 tensor 格式）
        return {"reference_logprobs": ref_logprobs}

    def get_experience_step(self) -> int:
        """获取worker的mini step。"""
        return self.experience_step
