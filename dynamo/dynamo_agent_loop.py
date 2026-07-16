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
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.llm_server import LLMServerManager
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.utils import update_prometheus_config


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
    ) -> TokenOutput:
        return await self.server.generate.remote(
            request_id=request_id or uuid4().hex,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
            **kwargs,
        )


class DynamoLLMServerManager(LLMServerManager):
    """LLM server manager that launches Dynamo through its shared worker pool."""

    async def _initialize_llm_servers(self, start_rank: int = 0):
        if self.worker_group is None:
            raise ValueError("Dynamo rollout requires hybrid mode with an actor rollout worker group")

        from recipe.dynamo.dynamo_async_server import DynamoReplica

        replica = DynamoReplica(
            replica_rank=start_rank,
            config=self.rollout_config,
            model_config=self.model_config,
            gpus_per_node=self.rollout_config.n_gpus_per_node,
        )
        await replica.init_hybrid_worker_pool(self.worker_group)

        self.rollout_replicas = [replica]
        self.server_handles = [replica._server_handle]
        self.server_addresses = [replica._server_address]
        print(f"DynamoLLMServerManager: {self.server_addresses}")

        if self.rollout_config.prometheus.enable:
            if self.rollout_config.disable_log_stats:
                raise ValueError("PROMETHEUS needs disable_log_stats==False, but it is currently True.")
            update_prometheus_config(self.rollout_config.prometheus, self.server_addresses, self.rollout_config.name)


class DynamoAgentLoopWorker(AgentLoopWorker):
    """Compatibility wrapper for Dynamo agent loop workers."""


class DynamoAgentLoopManager(AgentLoopManager):
    """AgentLoopManager compatible with the current verl LLMServerClient API."""

    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(DynamoAgentLoopWorker)
        super().__init__(*args, **kwargs)

    @classmethod
    @auto_await
    async def create(cls, *args, **kwargs):
        instance = cls(*args, **kwargs)
        await instance._init_agent_loop_workers()
        return instance

    async def _init_agent_loop_workers(self):
        await super()._init_agent_loop_workers()


__all__ = ["DynamoAgentLoopManager", "DynamoAgentLoopWorker", "DynamoLLMServerManager", "DynamoServerManager"]
