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
import concurrent.futures
import logging
import os
import threading
import time

import ray

from verl.checkpoint_engine.base import (
    CheckpointEngineManager,
    CheckpointEngineRegistry,
    CheckpointEngineWorker,
)
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.utils.profiler import marked_timer
from verl.workers.config import CheckpointEngineConfig
from verl.workers.rollout.replica import RolloutReplica

log_level = os.getenv("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


def build_topology(trainer_world_size: int, rollout_world_size: int, fwd_world_size: int, metadata: list[dict]):
    """
    Build topology for Trainer, Rollout, and Fwd workers.
    Rank assignment:
    - Trainer (Rank 0): Master, sends weights.
    - Trainer (Rank -1): Passive, does nothing.
    - Rollout (Rank 1 ~ M): Receives weights.
    - Fwd (Rank M+1 ~ M+K): Receives weights.
    """

    comm_world_size = 1 + rollout_world_size + fwd_world_size

    # 1. Trainer 配置
    # metadata[0] 是 Trainer Rank 0 的元数据（包含 IP/Port），广播给所有人
    trainer_kwargs = {
        "rank": [0] + [-1] * (trainer_world_size - 1),
        "world_size": [comm_world_size] * trainer_world_size,
        "master_metadata": [metadata[0]] * trainer_world_size,
    }

    # 2. Rollout 配置
    # Rank 范围: [1, rollout_world_size]
    rollout_kwargs = {
        "rank": list(range(1, rollout_world_size + 1)),
        "world_size": [comm_world_size] * rollout_world_size,
        "master_metadata": [metadata[0]] * rollout_world_size,
    }

    # 3. Fwd 配置
    # Rank 范围: [rollout_world_size + 1, rollout_world_size + fwd_world_size]
    fwd_start_rank = rollout_world_size + 1
    fwd_kwargs = {
        "rank": list(range(fwd_start_rank, fwd_start_rank + fwd_world_size)),
        "world_size": [comm_world_size] * fwd_world_size,
        "master_metadata": [metadata[0]] * fwd_world_size,
    }

    return trainer_kwargs, rollout_kwargs, fwd_kwargs


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
        self.rollout_wg = self._init_rollout_wg()
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

    def _init_rollout_wg(self):
        all_workers = []
        for replica in self.replicas:
            start = len(all_workers)
            all_workers.extend(replica.workers)
            self.replica_worker_slice.append((start, len(all_workers)))
        rollout_wg = RayWorkerGroup(
            worker_handles=all_workers, ray_cls_with_init=RayClassWithInitArgs(cls=CheckpointEngineWorker)
        )
        return rollout_wg

    def build_process_group(self, rollout: RayWorkerGroup, fwd: RayWorkerGroup, metadata):
        """Build process group for trainer, rollout replicas and fwd ."""
        trainer = self.trainer_wg

        trainer_kwargs, rollout_kwargs, fwd_kwargs = build_topology(
            trainer.world_size, rollout.world_size, fwd.world_size, metadata
        )

        for k, v in trainer_kwargs.items():
            assert len(v) == trainer.world_size, f"trainer_kwargs[{k}] length error"
        for k, v in rollout_kwargs.items():
            assert len(v) == rollout.world_size, f"rollout_kwargs[{k}] length error"
        for k, v in fwd_kwargs.items():
            assert len(v) == fwd.world_size, f"fwd_kwargs[{k}] length error"

        trainer_kwargs["method"] = ["init_process_group"] * trainer.world_size
        rollout_kwargs["method"] = ["init_process_group"] * rollout.world_size
        fwd_kwargs["method"] = ["init_process_group"] * fwd.world_size

        ray.get(
            trainer.execute_checkpoint_engine(**trainer_kwargs)
            + rollout.execute_checkpoint_engine(**rollout_kwargs)
            + fwd.execute_checkpoint_engine(**fwd_kwargs)
        )

    def update_weights_rollout_fwd(self, version_id, wait_for_inflight=True):
        logger.debug(f"[update_weights_rollout_fwd] version_id={version_id}")

        trainer_wg = self.trainer_wg
        rollout_wg = self.rollout_wg
        fwd_wg = self.fwd_wg

        with marked_timer("sync_fwd_pause", {}):
            pause_status = fwd_wg.pause()
        logger.debug(f"FWD paused: {pause_status}")

        if not self.is_first_sync:
            start = time.perf_counter()
            logger.debug(f"AgentLoop All Rollout replicas start pausing,time {time.perf_counter()}")
            with marked_timer("sync_replica_pause", {}):
                self.pause_all_replicas(wait_for_inflight)
            logger.info(f"AgentLoop All Rollout replicas paused, used time: {time.perf_counter() - start}")
        self.is_first_sync = False

        with marked_timer("sync_prepare", {}):
            metadata = ray.get(
                trainer_wg.execute_checkpoint_engine(["prepare"] * trainer_wg.world_size)
                + rollout_wg.execute_checkpoint_engine(["prepare"] * rollout_wg.world_size)
                + fwd_wg.execute_checkpoint_engine(["prepare"] * fwd_wg.world_size)
            )

        with marked_timer("sync_build_process_group", {}):
            self.build_process_group(rollout_wg, fwd_wg, metadata)

        # 4. update weights of all workers
        with marked_timer("sync_update_weights", {}):
            ray.get(
                trainer_wg.update_weights(version_id)
                + rollout_wg.update_weights(version_id)
                + fwd_wg.update_weights(version_id)
            )

        with marked_timer("sync_finalize", {}):
            ray.get(
                trainer_wg.execute_checkpoint_engine(["finalize"] * trainer_wg.world_size)
                + rollout_wg.execute_checkpoint_engine(["finalize"] * rollout_wg.world_size)
                + fwd_wg.execute_checkpoint_engine(["finalize"] * fwd_wg.world_size)
            )

        with marked_timer("sync_notify_replica_update", {}):
            self.resume_all_replica()

    def pause_all_replicas(self, wait_for_inflight=True):
        """同步等待所有 replica 暂停"""
        futures = [
            asyncio.run_coroutine_threadsafe(replica.pause(wait_for_inflight_requests=wait_for_inflight), self._loop)
            for replica in self.replicas
        ]
        if futures:
            concurrent.futures.wait(futures)
            for fut in futures:
                fut.result()

    def resume_all_replica(self):
        futures = [asyncio.run_coroutine_threadsafe(replica.resume(), self._loop) for replica in self.replicas]
        if futures:
            concurrent.futures.wait(futures)
            for fut in futures:
                fut.result()
