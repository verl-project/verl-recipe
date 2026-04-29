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
import asyncio
import logging
import os
from typing import Any, Optional

import ray

from verl.single_controller.ray import RayClassWithInitArgs
from verl.workers.rollout.replica import TokenOutput
from verl.workers.rollout.vllm_rollout.vllm_async_server import vLLMHttpServer, vLLMReplica

logger = logging.getLogger(__name__)


def _patch_expandable_segments():
    """Disable expandable_segments in PYTORCH_NPU_ALLOC_CONF to avoid vllm_ascend assertion error."""
    conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
    if "expandable_segments:True" in conf:
        os.environ["PYTORCH_NPU_ALLOC_CONF"] = conf.replace("expandable_segments:True", "expandable_segments:False")
        logger.info("Server process: patched PYTORCH_NPU_ALLOC_CONF to disable expandable_segments")


class LegacyAsyncFlowHttpServer(vLLMHttpServer):
    """vLLMHttpServer with model_version tracking for async flow."""

    def __init__(self, *args, **kwargs):
        _patch_expandable_segments()
        super().__init__(*args, **kwargs)
        self.model_version: int = 0
        self._pause_cond = asyncio.Condition()
        self._paused = False
        self.logger = logging.getLogger(__name__)
        self.logger.info("Using legacy vllm api")

    def get_model_version(self) -> int:
        return self.model_version

    def set_model_version(self, version: int):
        self.model_version = version

    async def generate(
        self,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        request_id: str,
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Generate sequence with token-in-token-out."""
        async with self._pause_cond:
            await self._pause_cond.wait_for(lambda: not self._paused)

        return await super().generate(prompt_ids, sampling_params, request_id, image_data, video_data)

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
        timeout_s: float = 600.0,
    ) -> None:
        async with self._pause_cond:
            if self._paused:
                return
            self._paused = True

        if not wait_for_inflight_requests:
            request_ids = list(self.engine.output_processor.request_states.keys())
            if request_ids:
                await self.abort_all_requests(reset_prefix_cache=False)

        # Drain inflight before clearing cache / returning (matches "block until truly paused")
        await asyncio.wait_for(self.wait_for_requests_to_drain(), timeout=timeout_s)

        if clear_cache:
            try:
                await self.clear_kv_cache()
            except Exception as e:
                self.logger.warning(f"clear_kv_cache failed during pause: {e}")

    async def resume_generation(self) -> None:
        """Resume generation after pause_generation."""
        async with self._pause_cond:
            self._paused = False
            self._pause_cond.notify_all()  # Wake up all waiting requests

    async def is_paused(self) -> bool:
        """Return whether the server is currently paused."""
        async with self._pause_cond:
            return self._paused


class AsyncFlowHttpServer(vLLMHttpServer):
    """vLLMHttpServer with model_version tracking for async flow."""

    def __init__(self, *args, **kwargs):
        _patch_expandable_segments()
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Using new vllm api")

    def get_model_version(self) -> int:
        return self.global_steps

    def set_model_version(self, version: int):
        self.global_steps = version

    async def pause_generation(
        self,
        *,
        wait_for_inflight_requests: bool = False,
        clear_cache: bool = True,
        timeout_s: float = 600.0,
    ) -> None:
        self.logger.debug("begin pausing")
        await asyncio.wait_for(
            self.engine.pause_generation(
                wait_for_inflight_requests=wait_for_inflight_requests, clear_cache=clear_cache
            ),
            timeout=timeout_s,
        )
        self.logger.debug("end pausing")

    async def resume_generation(self) -> None:
        self.logger.debug("begin resuming")
        await self.engine.resume_generation()
        self.logger.debug("end resuming")

    async def is_paused(self) -> bool:
        return await self.engine.is_paused()


class AsyncFlowReplica(vLLMReplica):
    """vLLMReplica that uses AsyncFlowHttpServer with model_version tracking."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import vllm
        from packaging.version import Version

        vllm_version = Version(vllm.__version__)
        if vllm_version < Version("0.11"):
            self.server_class = ray.remote(LegacyAsyncFlowHttpServer)
        else:
            self.server_class = ray.remote(AsyncFlowHttpServer)

    def get_ray_class_with_init_args(self) -> RayClassWithInitArgs:
        from recipe.async_flow.workers.rollout_worker import AsyncFlowCheckpointEngineWorker

        worker_dict_cls = RayClassWithInitArgs(
            cls=ray.remote(AsyncFlowCheckpointEngineWorker),
            rollout_config=self.config,
            model_config=self.model_config,
            replica_rank=self.replica_rank,
        )
        return worker_dict_cls

    async def pause(
        self,
        *,
        wait_for_inflight_requests: bool = True,
        clear_cache: bool = True,
        timeout_s: float = 900.0,
    ) -> dict[str, Any]:
        await self._server_handle.pause_generation.remote(
            wait_for_inflight_requests=wait_for_inflight_requests,
            clear_cache=clear_cache,
            timeout_s=timeout_s,
        )
        return {"status": "paused", "is_paused": True}

    async def resume(self) -> dict[str, Any]:
        await self._server_handle.resume_generation.remote()
        return {"status": "resumed", "is_paused": False}

    async def is_paused(self) -> dict[str, Any]:
        paused = await self._server_handle.is_paused.remote()
        return {"is_paused": bool(paused)}
