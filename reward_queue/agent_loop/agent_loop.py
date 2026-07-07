# Copyright 2026 Huawei Technologies Co., Ltd. and/or its affiliates
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
AgentLoopWorkerForRewardQueue extends AgentLoopWorker for reward queue workflow.
"""

import logging
import os
from typing import Any

import hydra
import numpy as np
import torch
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopOutput,
    AgentLoopWorker,
    DictConfigWrap,
    ToolListWrap,
    _agent_loop_registry,
    _InternalAgentLoopOutput,
    get_trajectory_info,
    rollout_trace_attr,
)
from verl.protocol import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class AgentLoopWorkerForRewardQueue(AgentLoopWorker):
    async def _run_reward_agent_loop(
        self,
        sampling_params: dict[str, Any],
        trajectory: dict[str, Any],
        *,
        agent_name: str,
        trace: bool = True,
        **kwargs,
    ) -> _InternalAgentLoopOutput:
        with rollout_trace_attr(
            step=trajectory["step"],
            sample_index=trajectory["sample_index"],
            rollout_n=trajectory["rollout_n"],
            validate=trajectory["validate"],
            name="agent_loop",
            trace=trace,
        ):
            assert agent_name in _agent_loop_registry, (
                f"Agent loop {agent_name} not registered, registered agent loops: {_agent_loop_registry.keys()}"
            )

            agent_loop_config = _agent_loop_registry[agent_name]
            agent_loop = hydra.utils.instantiate(
                config=agent_loop_config,
                trainer_config=DictConfigWrap(config=self.config),
                server_manager=self.llm_client,
                tokenizer=self.tokenizer,
                processor=self.processor,
                dataset_cls=self.dataset_cls,
                data_config=DictConfigWrap(self.config.data),
                tools=ToolListWrap(self.tools),
            )

            import time as _time

            inference_start_timestamp = _time.time()
            output: AgentLoopOutput = await agent_loop.run(sampling_params, **kwargs)
            inference_end_timestamp = _time.time()

            output.extra_fields["inference_start_timestamp"] = inference_start_timestamp
            output.extra_fields["inference_end_timestamp"] = inference_end_timestamp
            output.extra_fields["inference_duration"] = inference_end_timestamp - inference_start_timestamp

            return output

    def _padding_postprocess(self, output, validate, **kwargs) -> _InternalAgentLoopOutput:
        output.extra_fields["raw_prompt"] = kwargs.get("raw_prompt", None)

        self.tokenizer.padding_side = "left"
        prompt_output = self.tokenizer.pad(
            {"input_ids": output.prompt_ids},
            padding="max_length",
            max_length=self.rollout_config.prompt_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if prompt_output["input_ids"].dim() == 1:
            prompt_output["input_ids"] = prompt_output["input_ids"].unsqueeze(0)
            prompt_output["attention_mask"] = prompt_output["attention_mask"].unsqueeze(0)

        self.tokenizer.padding_side = "right"
        response_output = self.tokenizer.pad(
            {"input_ids": output.response_ids},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=True,
        )
        if response_output["input_ids"].dim() == 1:
            response_output["input_ids"] = response_output["input_ids"].unsqueeze(0)
            response_output["attention_mask"] = response_output["attention_mask"].unsqueeze(0)

        response_mask_output = self.tokenizer.pad(
            {"input_ids": output.response_mask},
            padding="max_length",
            max_length=self.rollout_config.response_length,
            return_tensors="pt",
            return_attention_mask=False,
        )
        if response_mask_output["input_ids"].dim() == 1:
            response_mask_output["input_ids"] = response_mask_output["input_ids"].unsqueeze(0)

        response_logprobs = None
        if output.response_logprobs is not None:
            pad_size = self.rollout_config.response_length - len(output.response_logprobs)
            response_logprobs = torch.tensor(output.response_logprobs + [0.0] * pad_size).unsqueeze(0)

        response_mask = response_mask_output["input_ids"] * response_output["attention_mask"]
        attention_mask = torch.cat([prompt_output["attention_mask"], response_output["attention_mask"]], dim=1)
        input_ids = torch.cat([prompt_output["input_ids"], response_output["input_ids"]], dim=1)

        routed_experts = None
        if output.routed_experts is not None:
            total_length = input_ids.shape[1]
            length, layer_num, topk_num = output.routed_experts.shape
            if isinstance(output.routed_experts, np.ndarray):
                routed_experts_array = output.routed_experts
                if not routed_experts_array.flags.writeable:
                    routed_experts_array = routed_experts_array.copy()
                experts_tensor = torch.from_numpy(routed_experts_array)
            elif isinstance(output.routed_experts, torch.Tensor):
                experts_tensor = output.routed_experts
            else:
                raise TypeError(f"Unsupported type for routed_experts: {type(output.routed_experts)}")
            routed_experts = torch.zeros(1, total_length, layer_num, topk_num, dtype=experts_tensor.dtype)
            start_pos = prompt_output["input_ids"].shape[1] - len(output.prompt_ids)
            end_pos = min(start_pos + length, total_length)
            routed_experts[:, start_pos:end_pos] = experts_tensor.unsqueeze(0)

        multi_modal_inputs = self._compute_multi_modal_inputs(output, input_ids)
        position_ids = self._compute_position_ids(input_ids, attention_mask, multi_modal_inputs)

        return _InternalAgentLoopOutput(
            prompt_ids=prompt_output["input_ids"],
            response_ids=response_output["input_ids"],
            input_ids=input_ids,
            position_ids=position_ids,
            response_mask=response_mask,
            attention_mask=attention_mask,
            response_logprobs=response_logprobs,
            routed_experts=routed_experts,
            multi_modal_inputs=multi_modal_inputs,
            multi_modal_data=output.multi_modal_data,
            teacher_logprobs=None,
            teacher_ids=None,
            reward_score=None,
            num_turns=output.num_turns,
            metrics=output.metrics,
            extra_fields=output.extra_fields,
        )

    def _build_reward_input(self, padded_output: _InternalAgentLoopOutput, kwargs: dict, validate: bool) -> DataProto:
        prompts = padded_output.prompt_ids
        responses = padded_output.response_ids
        attention_mask = padded_output.attention_mask
        input_ids = padded_output.input_ids
        position_ids = padded_output.position_ids

        batch = TensorDict(
            {
                "prompts": prompts,
                "responses": responses,
                "attention_mask": attention_mask,
                "input_ids": input_ids,
                "position_ids": position_ids,
            },
            batch_size=1,
        )

        meta_info = {"validate": validate}
        non_tensor_dict = {k: np.array([v]) for k, v in kwargs.items()}
        non_tensor_dict["__num_turns__"] = np.array([padded_output.num_turns])
        non_tensor_dict["tool_extra_fields"] = np.array([padded_output.extra_fields], dtype=object)

        return DataProto(batch=batch, non_tensor_batch=non_tensor_dict, meta_info=meta_info)

    async def _impl_generate_single_for_reward_queue(
        self, batch: DataProto
    ) -> tuple[DataProto, DataProto, float, float]:
        config = self.rollout_config
        sampling_params = dict(
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repetition_penalty=1.0,
            logprobs=config.calculate_log_probs,
        )

        if batch.meta_info.get("validate", False):
            sampling_params["top_p"] = config.val_kwargs.top_p
            sampling_params["top_k"] = config.val_kwargs.top_k
            sampling_params["temperature"] = config.val_kwargs.temperature

        if "agent_name" not in batch.non_tensor_batch:
            default_agent_loop = config.agent.default_agent_loop
            batch.non_tensor_batch["agent_name"] = np.array([default_agent_loop] * len(batch), dtype=object)

        if "index" in batch.non_tensor_batch:
            index_val = batch.non_tensor_batch["index"]
            index = [index_val[0] if isinstance(index_val, (list, np.ndarray)) else index_val]
        else:
            index = [0]

        trajectory_info = await get_trajectory_info(
            batch.meta_info.get("global_steps", -1), index, batch.meta_info.get("validate", False)
        )

        kwargs = {
            k: (v[0] if isinstance(v, (list, np.ndarray)) and len(v) > 0 else v)
            for k, v in batch.non_tensor_batch.items()
        }
        validate = batch.meta_info.get("validate", False)

        output = await self._run_reward_agent_loop(sampling_params, trajectory_info[0], **kwargs)

        padded_output = self._padding_postprocess(output, validate=validate, **kwargs)
        reward_input = self._build_reward_input(padded_output, kwargs, validate=validate)

        batch_td = TensorDict(
            {
                "prompts": padded_output.prompt_ids,
                "responses": padded_output.response_ids,
                "response_mask": padded_output.response_mask,
                "input_ids": padded_output.input_ids,
                "attention_mask": padded_output.attention_mask,
                "position_ids": padded_output.position_ids,
            },
            batch_size=1,
        )
        if padded_output.response_logprobs is not None:
            batch_td["rollout_log_probs"] = padded_output.response_logprobs
        if padded_output.routed_experts is not None:
            batch_td["routed_experts"] = padded_output.routed_experts

        non_tensor_batch = {}
        non_tensor_batch["__num_turns__"] = np.array([padded_output.num_turns], dtype=np.int32)

        default_extra_keys = {
            "turn_scores",
            "tool_rewards",
            "min_global_steps",
            "max_global_steps",
            "extras",
            "inference_start_timestamp",
            "inference_end_timestamp",
            "inference_duration",
        }
        all_keys = set(padded_output.extra_fields.keys()) | default_extra_keys
        for key in all_keys:
            val = padded_output.extra_fields.get(key)
            if isinstance(val, (list, tuple, dict)):
                arr = np.empty(1, dtype=object)
                arr[0] = val
            else:
                arr = np.array([val], dtype=object)
            non_tensor_batch[key] = arr

        for k, v in batch.non_tensor_batch.items():
            if isinstance(v, (list, np.ndarray)):
                non_tensor_batch[k] = np.array([v[0]], dtype=object) if len(v) == 1 else v[:1]
            else:
                non_tensor_batch[k] = np.array([v], dtype=object)

        if padded_output.multi_modal_inputs is not None:
            non_tensor_batch["multi_modal_inputs"] = np.array([padded_output.multi_modal_inputs], dtype=object)

        metrics = [padded_output.metrics.model_dump()]
        meta_info = {"metrics": metrics}

        padded_dp = DataProto(batch=batch_td, non_tensor_batch=non_tensor_batch, meta_info=meta_info)

        inf_start = padded_output.extra_fields.get("inference_start_timestamp", 0.0)
        inf_end = padded_output.extra_fields.get("inference_end_timestamp", 0.0)

        return padded_dp, reward_input, inf_start, inf_end

    async def generate_sequences(self, batch: DataProto) -> tuple[DataProto, DataProto, float, float] | DataProto:
        if batch.meta_info.get("validate", False):
            return await super().generate_sequences(batch)
        return await self._impl_generate_single_for_reward_queue(batch)
