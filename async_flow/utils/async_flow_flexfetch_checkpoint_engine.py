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
import threading

import ray

# 触发 backend 注册：导入该模块会执行
# `@CheckpointEngineRegistry.register("flexfetch")` 装饰器，
# 否则下方 CheckpointEngineRegistry.new/get("flexfetch") 会因未注册而报错。
from recipe.async_flow.utils.checkpoint_engine.flexfetch_checkpoint_engine import (  # noqa: F401
    FlexFetchCheckpointEngine,
)

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineManager,
    CheckpointEngineRegistry,
    CheckpointEngineWorker,
)
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.utils.distributed import initialize_global_process_group_ray
from verl.utils.profiler import marked_timer
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.rollout import BaseRollout, get_rollout_class
from verl.workers.rollout.replica import RolloutReplica

log_level = os.getenv("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


class AsyncFlowCheckpointEngineWorker(CheckpointEngineWorker, Worker):
    """CheckpointEngineWorker adapted for AsyncFlow."""

    def __init__(
        self,
        rollout_config: RolloutConfig,
        model_config: HFModelConfig,
        server_adapter: BaseRollout = None,
        *args,
        **kwargs,
    ) -> None:
        Worker.__init__(self)
        self.rollout_config = rollout_config
        self.model_config = model_config

        self.server_adapter: BaseRollout = server_adapter
        backend = self.rollout_config.checkpoint_engine.backend
        bucket_size = self.rollout_config.checkpoint_engine.update_weights_bucket_megabytes << 20
        engine_kwargs = self.rollout_config.checkpoint_engine.engine_kwargs.get(backend, {})
        if backend == "flexfetch":
            engine_kwargs["is_trainer"] = False
            engine_kwargs["is_master"] = int(os.environ["RANK"]) == 0

        self.checkpoint_engine: CheckpointEngine = CheckpointEngineRegistry.new(
            backend, bucket_size=bucket_size, **engine_kwargs
        )

        self.extra_rollout_args = args
        self.extra_rollout_kwargs = kwargs
        if self.server_adapter is None:
            self.server_adapter = get_rollout_class(self.rollout_config.name, self.rollout_config.mode)(
                *self.extra_rollout_args,
                config=self.rollout_config,
                model_config=self.model_config,
                device_mesh=None,
                **self.extra_rollout_kwargs,
            )
        # sglang and trt-llm need device_mesh for internal communication
        initialize_global_process_group_ray(timeout_second=None, backend="cpu:gloo")
        # ── cluster trace auto-install (driver orchestration) ────────────
        if os.environ.get("VERL_CLUSTER_TRACE"):
            from recipe.async_flow.utils.cluster_trace.trace_logger import install

            install(role="ckpt_engine", rank=0)
        # ─────────────────────────────────────────────────────────────────

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, version_id: int = None):
        logger.debug(f"[DEBUG] [AsyncFlowCheckpointEngineWorker] start recv weights {version_id} at {self.rank=}")
        weights = self.checkpoint_engine.receive_weights(version_id=version_id)
        logger.debug(f"[DEBUG] [AsyncFlowCheckpointEngineWorker] start load weights {version_id}")
        with marked_timer("sync_replica_receive_and_update", {}):
            await self.server_adapter.update_weights(weights, global_steps=version_id)
        logger.debug(f"[DEBUG] [AsyncFlowCheckpointEngineWorker] load weights {version_id} done")


_worker_with_cache_class = ray.remote(AsyncFlowCheckpointEngineWorker)


class AsyncFlowCheckpointEngineManager(CheckpointEngineManager):
    def __init__(
        self,
        config: CheckpointEngineConfig,
        trainer_wg: RayWorkerGroup,
        replicas: list[RolloutReplica] = None,
        fwd_wg: RayWorkerGroup = None,
    ) -> None:
        super().__init__(config, trainer_wg, replicas)
        if trainer_wg is None or replicas is None or fwd_wg is None:
            raise ValueError("trainer_wg rollout and fwd_worker_group should not be None")
        self.trainer_wg = trainer_wg
        self.replicas = replicas
        self.fwd_wg = fwd_wg
        self.backend_cls = CheckpointEngineRegistry.get(config.backend)
        self.replica_worker_slice = []
        self._loop = None
        self._loop_thread = None
        self._ensure_loop_running()
        self.is_first_sync = True

    def _ensure_loop_running(self):
        if self._loop is not None and self._loop.is_running():
            return

        self._loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(self._loop)
            self._loop.run_forever()

        self._loop_thread = threading.Thread(target=run_loop, daemon=True)
        self._loop_thread.start()
        logger.info("Ckpt engine mgr Background event loop started")

    def update_weights_rollout_fwd(self, version_id, wait_for_inflight=True):
        logger.debug(f"[update_weights_rollout_fwd] version_id={version_id}")

        trainer_wg = self.trainer_wg
        fwd_wg = self.fwd_wg

        ray.get(trainer_wg.execute_checkpoint_engine(["launch_server"] * trainer_wg.world_size))
        ray.get(trainer_wg.update_weights(version_id))

        async def _receive_and_update_replica(replica, rollout_wg, version_id):
            with marked_timer("sync_replica_pause", {}):
                pause_status = await replica.pause(wait_for_inflight_requests=wait_for_inflight)
            logger.debug(f"replica pause_status={pause_status}")
            # build_process_group 放入线程, 避免阻塞事件循环导致各 replica 被串行化
            with marked_timer("sync_replica_build_process_group", {}):
                await asyncio.to_thread(self.build_process_group, rollout_wg)

            # 显式 await update_weights/finalize (blocking=False 返回 ObjectRef),
            # 保证权重真正加载完成后再 resume, 否则会在 stale/partial 权重上恢复生成
            with marked_timer("sync_replica_update_weights", {}):
                await asyncio.to_thread(ray.get, rollout_wg.update_weights(version_id))
            with marked_timer("sync_replica_finalize", {}):
                await asyncio.to_thread(
                    ray.get, rollout_wg.execute_checkpoint_engine(["finalize"] * rollout_wg.world_size)
                )

            with marked_timer("sync_replica_resume", {}):
                resume_status = await replica.resume()
            logger.debug(f"replica resume_status={resume_status}")

        async def _update_fwd(fwd_wg, version_id):
            with marked_timer("sync_fwd_pause", {}):
                await asyncio.to_thread(ray.get, fwd_wg.pause())

            with marked_timer("sync_fwd_build_process_group", {}):
                await asyncio.to_thread(self.build_process_group, fwd_wg)
            with marked_timer("sync_fwd_update_weights", {}):
                await asyncio.to_thread(ray.get, fwd_wg.update_weights(version_id))
            with marked_timer("sync_fwd_finalize", {}):
                await asyncio.to_thread(ray.get, fwd_wg.execute_checkpoint_engine(["finalize"] * fwd_wg.world_size))

        def _log_future_exception(fut):
            try:
                fut.result()
            except Exception:
                logger.exception(f"Async weight update failed for version_id={version_id}")

        for replica in self.replicas:
            logger.debug("replica RayWorkerGroup begin")
            rollout_wg = RayWorkerGroup(
                worker_handles=replica.workers,
                worker_names=[None] * len(replica.workers),
                ray_cls_with_init=RayClassWithInitArgs(cls=_worker_with_cache_class),
            )
            logger.debug("replica RayWorkerGroup end")
            fut = asyncio.run_coroutine_threadsafe(
                _receive_and_update_replica(replica, rollout_wg, version_id), self._loop
            )
            fut.add_done_callback(_log_future_exception)

        fwd_fut = asyncio.run_coroutine_threadsafe(_update_fwd(fwd_wg, version_id), self._loop)
        fwd_fut.add_done_callback(_log_future_exception)
        ray.get(trainer_wg.execute_checkpoint_engine(["finalize"] * trainer_wg.world_size))
        logger.debug(f"[update_weights_rollout_fwd] Updates submitted for version_id={version_id}")

    def stop_server(self):
        ray.get(self.trainer_wg.execute_checkpoint_engine(["stop_server"] * self.trainer_wg.world_size))
