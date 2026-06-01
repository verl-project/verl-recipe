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
from typing import Any, Optional

import ray

from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica


@ray.remote
class PartialRolloutvLLMHttpServer(vLLMHttpServer):
    """vLLM HTTP server with a Python-side resume-gate.

    Why this exists: vLLM <0.12 has no `pause_generation` primitive — its
    `abort_all_requests` only signals abort on currently-tracked requests,
    leaving the engine free to accept new ones immediately after. PartialRollout's
    workers retry aborted `client.generate(...)` calls inside
    `FullyLLMServerClient`, so the moment `cancel()` returns, fresh
    requests start hitting the engine again — and the next
    `sleep_replicas` → `update_weights` → wake_up sequence races those
    requests' kernels, surfacing as `CUDA error: an illegal memory access`.

    `_resume_event` is the gate: every `generate(...)` first awaits the
    event before reaching vLLM. While `cancel()` holds the gate cleared,
    new requests hang at the entry instead of touching the engine and
    instead of retry-storming through `FullyLLMServerClient` (which would
    just bounce off the gate each iteration). `cancel()` then drives
    `abort_all_requests(reset_prefix_cache=False)` in a loop until any
    in-flight requests that beat the gate have drained. `resume()` sets
    the gate; all queued callers proceed at once.

    TODO: drop this whole wrapper (and the `rollout_replica_class` swap in
    `PartialRolloutLLMServerManager`) once verl's minimum vLLM is ≥0.12 — upstream's
    `pause_generation(wait_for_inflight_requests=False)` provides the same
    gate-then-drain semantics at the engine layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inflight: int = 0
        # Set == open (generate proceeds), cleared == closed (generate hangs
        # until resume sets it). Initial state open so startup generates
        # immediately.
        self._resume_event: asyncio.Event = asyncio.Event()
        self._resume_event.set()

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
        audio_data: Optional[list[Any]] = None,
        mm_processor_kwargs: Optional[dict[str, Any]] = None,
        priority: int = 0,
    ) -> TokenOutput:
        await self._resume_event.wait()
        self.inflight += 1
        try:
            return await vLLMHttpServer.generate(
                self,
                prompt_ids,
                sampling_params,
                request_id,
                image_data=image_data,
                video_data=video_data,
                audio_data=audio_data,
                mm_processor_kwargs=mm_processor_kwargs,
                priority=priority,
            )
        finally:
            self.inflight -= 1

    async def cancel(self):
        self._resume_event.clear()
        while self.inflight:
            # `reset_prefix_cache=False`: clearing the prefix cache leaves
            # the engine in a state that corrupts the subsequent
            # update_weights wake_up. The prefix cache holds prompt tokens,
            # not weight-dependent activations, so keeping it is safe.
            await self.abort_all_requests(reset_prefix_cache=False)
            await asyncio.sleep(0)

    async def resume(self):
        self._resume_event.set()


class PartialRolloutvLLMReplica(vLLMReplica):
    """vLLM replica using `PartialRolloutvLLMHttpServer` and fanning `cancel`/`resume`/
    `set_global_steps` out to every server."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = PartialRolloutvLLMHttpServer

    async def cancel(self):
        await asyncio.gather(*[server.cancel.remote() for server in self.servers])

    async def resume(self):
        await asyncio.gather(*[server.resume.remote() for server in self.servers])

    async def set_global_steps(self, global_steps: int):
        await asyncio.gather(*[server.set_global_steps.remote(global_steps) for server in self.servers])
