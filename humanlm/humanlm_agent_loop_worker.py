# Copyright 2026 HUMANLM team and/or its affiliates
# Copyright 2026 Bytedance Ltd. and/or its affiliates

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

import ray
import numpy as np
from tensordict import TensorDict
from verl.protocol import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, AgentLoopManager


class HumanLMAgentLoopWorker(AgentLoopWorker):

    async def _compute_score(self, output, prompts, responses, attention_mask, input_ids, position_ids, kwargs):
        if output.reward_score is None and self.reward_loop_worker_handles is not None:
            extra_info = kwargs.get("extra_info", {})
            index = extra_info.get("index", 0)
            state_name = extra_info.get("state_name", "response")
            key = f"{state_name}:{index}"
            worker_idx = hash(key) % len(self.reward_loop_worker_handles)
            selected = self.reward_loop_worker_handles[worker_idx]

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
            non_tensor_batch = {
                **{k: np.array([v]) for k, v in kwargs.items()},
                "__num_turns__": np.array([output.num_turns]),
                "tool_extra_fields": np.array([output.extra_fields], dtype=object),
            }
            data = DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
            # global_steps was injected into kwargs by generate_sequences, pass it via meta_info
            data.meta_info["global_steps"] = kwargs.get("global_steps", 0)
            result = await selected.compute_score.remote(data)
            output.reward_score = result["reward_score"]
            output.extra_fields["reward_extra_info"] = result["reward_extra_info"]


class HumanLMAgentLoopManager(AgentLoopManager):

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(HumanLMAgentLoopWorker)
        super().__init__(*args, **kwargs)

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # Inject global_steps into non_tensor_batch so it flows through kwargs to _compute_score
        global_steps = prompts.meta_info.get("global_steps", 0)
        batch_size = len(prompts)
        prompts.non_tensor_batch["global_steps"] = np.array([global_steps] * batch_size)

        # Now call parent which will chunk and dispatch to workers
        return super().generate_sequences(prompts)