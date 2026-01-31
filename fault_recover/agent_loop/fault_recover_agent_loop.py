# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import logging
import os
from typing import Any, Optional
from uuid import uuid4

import ray
from omegaconf import DictConfig

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker, AsyncLLMServerManager
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class FaultRecoverAsyncLLMServerManager(AsyncLLMServerManager):
    """
    A class to manage multiple OpenAI compatible LLM servers. This class provides
    - Load balance: least requests load balancing
    - Sticky session: send multi-turn chat completions to same server for automatic prefix caching
    """

    @rollout_trace_op
    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        global_id: int = None,
    ) -> TokenOutput:
        """Generate tokens from prompt ids.

        Args:
            request_id (str): request id for sticky session.
            prompt_ids (List[int]): List of prompt token ids.
            sampling_params (Dict[str, Any]): Sampling parameters for the chat completion.
            global_id: Global batch id of req.

        Returns:
            TokenOutput: token output
        """
        server = self._choose_server(request_id)
        new_request_id = uuid4().hex
        tokens_queue = None
        if global_id is not None:
            from recipe.fault_recover.fault_manager import get_tokens_queue

            tokens_queue = get_tokens_queue()

        if tokens_queue is not None:
            await tokens_queue.put.remote((new_request_id, global_id))

        output = await server.generate.remote(
            request_id=new_request_id,  # use new request_id for each turn
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
        )

        if tokens_queue is not None:
            await tokens_queue.put.remote(
                {
                    new_request_id: {
                        "log_probs": output.log_probs,
                        "routed_experts": output.routed_experts,
                        "num_preempted": output.num_preempted,
                    }
                }
            )

        return output


class FaultRecoverAgentLoopWorker(AgentLoopWorker):
    """Agent loop worker takes a batch of messages and run each message in an agent loop."""

    def __init__(
        self,
        config: DictConfig,
        server_handles: list[ray.actor.ActorHandle],
        reward_router_address: str = None,
    ):
        super().__init__(config, server_handles, reward_router_address)
        self.server_manager = FaultRecoverAsyncLLMServerManager(config, server_handles)


class FaultRecoverAgentLoopManager(AgentLoopManager):
    """Agent loop manager that manages a group of agent loop workers."""

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        rm_resource_pool: RayResourcePool = None,
    ):
        """Initialize agent loop manager.

        Args:
            config (DictConfig): trainer config.
            worker_group (RayWorkerGroup): ActorRolloutRef worker group for hybrid mode; None for standalone mode.
            rollout_resource_pool (RayResourcePool): Resource pool for actor rollout (Colocate or Standalone mode).
            rm_resource_pool (RayResourcePool): Resource pool for reward model (Standalone mode).
        """
        self.config = config
        self.worker_group = worker_group
        self.reward_model_manager = None
        self.reward_router_address = None
        if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
            from verl.experimental.reward_loop import RewardModelManager

            self.reward_model_manager = RewardModelManager(config.reward_model, rm_resource_pool)
            self.reward_router_address = self.reward_model_manager.get_router_address()

        # for recipe to change
        if not hasattr(self, "rollout_replica_class"):
            from recipe.fault_recover.vllm_rollout.vllm_async_server import FaultRecovervLLMReplica

            self.rollout_replica_class = FaultRecovervLLMReplica
        if not hasattr(self, "agent_loop_workers_class"):
            self.agent_loop_workers_class = ray.remote(FaultRecoverAgentLoopWorker)

        self._initialize_llm_servers(rollout_resource_pool)
        self._init_agent_loop_workers()
