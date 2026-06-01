# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import asyncio

from omegaconf import DictConfig
from recipe.partial_rollout.vllm_rollout.vllm_async_server import PartialRolloutvLLMReplica

from verl.experimental.fully_async_policy.fully_async_rollouter import FullyAsyncLLMServerClient
from verl.single_controller.ray.base import RayResourcePool, RayWorkerGroup
from verl.utils.ray_utils import auto_await
from verl.workers.rollout.llm_server import LLMServerClient, LLMServerManager


class PartialRolloutLLMServerManager(LLMServerManager):
    """LLMServerManager that:

    1. Swaps the replica class to `PartialRolloutvLLMReplica` so each server exposes
       the Python-side `_resume_event`-gated `cancel`/`resume`. Setting
       `self.rollout_replica_class` before `super().__init__` short-circuits
       upstream's default lookup at `LLMServerManager.__init__`.
    2. Overrides `get_client` to force every caller — including
       `RayPPOTrainer.init_workers`, which calls `get_client()` with no arg
       (defaults to plain `LLMServerClient`) — onto the retry-on-abort
       `FullyAsyncLLMServerClient`. The retry loop is gated on
       `config.async_training.partial_rollout=True` set in the run scripts via
       `+async_training.partial_rollout=True`.

    Installed via a monkey-patch in `PartialRolloutRayPPOTrainer.init_workers` because
    upstream has no FQN config knob for `LLMServerManager`.
    """

    def __init__(
        self,
        config: DictConfig,
        worker_group: RayWorkerGroup = None,
        rollout_resource_pool: RayResourcePool = None,
    ):
        self.rollout_replica_class = PartialRolloutvLLMReplica
        super().__init__(config, worker_group, rollout_resource_pool)

    def get_client(self, client_cls=LLMServerClient, **kwargs) -> LLMServerClient:
        # Ignore the requested client_cls; partial-rollout requires FullyAsyncLLMServerClient
        # for the abort-then-retry loop that hides cancel/resume from AgentLoop.
        return super().get_client(client_cls=FullyAsyncLLMServerClient, **kwargs)

    async def cancel(self):
        await asyncio.gather(*[replica.cancel() for replica in self.rollout_replicas])

    async def resume(self):
        await asyncio.gather(*[replica.resume() for replica in self.rollout_replicas])

    @auto_await
    async def set_global_steps(self, global_steps: int):
        # Prime each rollout server's `self.global_steps` before generation.
        # `checkpoint_manager.update_weights(global_steps=...)` would normally set
        # it, but its dispatch is `blocking=False`, so the trainer's first
        # generate can race the initial sync — when the server still has
        # `self.global_steps=None`, `FullyAsyncLLMServerClient` propagates None
        # into `TokenOutput.extra_fields["min/max_global_steps"]` and
        # `assemble_batch_from_rollout_samples` then crashes on `abs(None - None)`.
        # Calling this explicitly before generate eliminates the race.
        await asyncio.gather(*[replica.set_global_steps(global_steps) for replica in self.rollout_replicas])
