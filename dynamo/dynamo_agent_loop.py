# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Dynamo-specific AgentLoopManager.

Dynamo exposes one logical rollout endpoint through a master dynamo.frontend.
The worker-level routing happens inside Dynamo's KV router, not in verl's
GlobalRequestLoadBalancer. This module keeps the AgentLoop execution model but
replaces the generic server manager with a direct Dynamo server manager.
"""

from __future__ import annotations

from typing import Any, Optional
from uuid import uuid4

import ray

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from verl.experimental.teacher_loop import MultiTeacherModelManager
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.replica import DiffusionOutput, TokenOutput


class DynamoServerManager:
    """Direct manager for the shared Dynamo frontend actor.

    Unlike AsyncLLMServerManager, this class intentionally does not acquire a
    server from GlobalRequestLoadBalancer. Dynamo owns routing behind its
    frontend, so verl should only call the single shared Dynamo actor.
    """

    def __init__(self, servers: list[tuple[str, ray.actor.ActorHandle]]):
        if len(servers) != 1:
            raise ValueError(f"DynamoServerManager expects exactly one shared server, got {len(servers)}")
        self.server_address, self.server = servers[0]

    async def generate(
        self,
        request_id,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        **kwargs: Any,
    ) -> TokenOutput | DiffusionOutput:
        return await self.server.generate.remote(
            request_id=request_id or uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            **kwargs,
        )


class DynamoAgentLoopWorker(AgentLoopWorker):
    """AgentLoopWorker that talks directly to the shared Dynamo actor."""

    def __init__(self, config, servers, load_balancer_handle=None, *args, **kwargs):
        self.server_manager = DynamoServerManager(servers)
        super().__init__(config, servers, load_balancer_handle, *args, **kwargs)


class DynamoAgentLoopManager(AgentLoopManager):
    """AgentLoopManager that bypasses verl-side server load balancing."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(DynamoAgentLoopWorker)
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(
        cls,
        config,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
        reward_loop_worker_handles: list[ray.actor.ActorHandle] = None,
        teacher_model_manager: MultiTeacherModelManager = None,
    ):
        instance = cls(
            config,
            worker_group,
            rollout_resource_pool,
            teacher_model_manager,
            reward_loop_worker_handles,
        )
        await instance._initialize_llm_servers()
        # Deliberately skip _init_global_load_balancer(); Dynamo frontend/KV
        # router owns worker routing for the shared pool.
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        self.agent_loop_workers = []
        num_workers = self.rollout_config.agent.num_workers
        servers = list(zip(self.server_addresses, self.server_handles, strict=True))

        if len(servers) != 1:
            raise RuntimeError(f"DynamoAgentLoopManager expects one shared frontend, got {len(servers)}")

        if self.distillation_enabled:
            raise NotImplementedError("DynamoAgentLoopManager does not yet support distillation teacher routing")

        node_ids = [node["NodeID"] for node in ray.nodes() if node["Alive"] and node["Resources"].get("CPU", 0) > 0]
        for i in range(num_workers):
            node_id = node_ids[i % len(node_ids)]
            self.agent_loop_workers.append(
                self.agent_loop_workers_class.options(
                    name=f"dynamo_agent_loop_worker_{i}_{uuid4().hex[:8]}",
                    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
                        node_id=node_id, soft=True
                    ),
                ).remote(
                    self.config,
                    servers,
                    None,
                    None,
                    None,
                    self.reward_loop_worker_handles,
                )
            )


__all__ = ["DynamoAgentLoopManager", "DynamoAgentLoopWorker", "DynamoServerManager"]
