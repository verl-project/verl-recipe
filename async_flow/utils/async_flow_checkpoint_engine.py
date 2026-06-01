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
from enum import Enum
from typing import Any, Generator

import ray
import torch

from verl.checkpoint_engine.base import (
    CheckpointEngine,
    CheckpointEngineManager,
    CheckpointEngineWithCache,
    CheckpointEngineWorker,
)
from verl.single_controller.base.decorator import Dispatch, register
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.workers.config import CheckpointEngineConfig, HFModelConfig, RolloutConfig
from verl.workers.rollout.replica import RolloutReplica

log_level = os.getenv("LOGGING_LEVEL", logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True)
logger = logging.getLogger(__name__)


class AsyncFlowCheckpointEngineWithCache(CheckpointEngineWithCache):
    def __init__(
        self,
        non_cache_engine: CheckpointEngine,
        multi_version=False,
        capacity=40,  # GB
        role=None,
        worker_backend=None,
    ) -> None:
        # 调用父类初始化
        self.non_cache_engine = non_cache_engine
        self.host_param_buffer = {}  # {version_id: {name, tensor}}
        self.ckpt_size = -1
        self.version_order = []
        self.capacity = capacity
        self.multi_version = multi_version
        # role 用于指示 receive 端的 ckpt engine 是属于 fwd 还是 rollout worker, 为 None 表明是 trainer 发送端
        self.role = role
        assert self.role in [ReceiverRole.Rollout, ReceiverRole.ActorFwd] or self.role is None
        # 只有 fwd 端需要区分 worker_backend, 当前 fsdp 只在 rank0 存储一份完整权重
        self.worker_backend = worker_backend
        assert self.worker_backend in ["fsdp", "megatron"] or self.worker_backend is None

    def prepare(self) -> dict[str, Any]:
        return self.non_cache_engine.prepare()

    @classmethod
    def build_topology(
        cls, trainer_world_size: int, rollout_world_size: int, fwd_world_size: int, metadata: list[dict]
    ):
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

    def init_process_group(self, **kwargs):
        return self.non_cache_engine.init_process_group(**kwargs)

    def finalize(self):
        # only finalize the communication module
        return self.non_cache_engine.finalize()

    def _check_capacity_and_pop(self, cur_version):
        if not self.multi_version:
            return

        if cur_version in self.version_order:
            raise RuntimeError(f"version {cur_version} has received")

        self.version_order.append(cur_version)

        if len(self.version_order) * self.ckpt_size > self.capacity:
            # exceed the capacity if next ckpt is cached, pop oldest version
            oldest_version = self.version_order.pop(0)
            self.host_param_buffer.pop(oldest_version)

    async def send_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], version_id=None):
        logger.debug("[AsyncFlowCheckpointEngineWithCache.send_weights] start ...")
        await self.non_cache_engine.send_weights(weights)
        logger.debug("[AsyncFlowCheckpointEngineWithCache.send_weights] finish ...")

    async def receive_weights(self, version_id):
        logger.debug("[DEBUG] [AsyncFlowCheckpointEngineWithCache] start recv weights !!!!!!!!!")
        if self.multi_version:
            self._check_capacity_and_pop(version_id)
        else:
            self.version_order = [version_id]
            version_id = 0

        count_flag = True if self.ckpt_size < 0 else False
        if version_id not in self.host_param_buffer:
            self.host_param_buffer[version_id] = {}

        # 只有 fwd worker 需要对参数接收进行优化, 在不同worker后端实现的优化方式不同, 当前只实现了fsdp后端下的优化
        should_receive = True
        if self.role == ReceiverRole.ActorFwd and self.worker_backend == "fsdp":
            if torch.distributed.get_rank() != 0:
                should_receive = False

        async for name, param in self.non_cache_engine.receive_weights():
            if should_receive:
                cpu_param = param.cpu()
                self.host_param_buffer[version_id][name] = cpu_param
                if count_flag:
                    self.ckpt_size += cpu_param.nbytes
            else:
                # fwd+fsdp 的非 rank 0 节点会走到这里, 其他情况不变
                # host_param_buffer[version_id] 依然维持上面初始化的空字典 {}
                pass

    def get_weights(self, version_id=None) -> tuple[Generator[tuple[str, torch.Tensor], None, None], int]:
        logger.debug(f"[AsyncFlowCheckpointEngineWithCache.get_weights] version_id={version_id}")
        if not self.multi_version:
            cur_version = self.version_order[-1]
            version_id = 0  # actual key is 0

        def weights_generator():
            yield from self.host_param_buffer[version_id].items()

        return weights_generator(), cur_version


class ReceiverRole(Enum):
    """Checkpoint Engine 接收端角色。"""

    Rollout = 0
    ActorFwd = 1
    RefFwd = 2


class AsyncFlowCheckpointEngineWorker(CheckpointEngineWorker):
    """CheckpointEngineWorker adapted for AsyncFlow."""

    def __init__(
        self,
        rollout_config: RolloutConfig,
        model_config: HFModelConfig,
        server_adapter=None,
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            rollout_config=rollout_config,
            model_config=model_config,
            server_adapter=server_adapter,
            **kwargs,
        )

        self.checkpoint_engine = AsyncFlowCheckpointEngineWithCache(
            non_cache_engine=self.checkpoint_engine, role=ReceiverRole.Rollout
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def load_weights(self, version_id):
        """Load cached weights from host buffer and push to vLLM server via ZMQ/IPC."""
        logger.info(f"[AsyncFlowCheckpointEngineWorker] start load weights {version_id}")

        weights_gen, _ = self.checkpoint_engine.get_weights(version_id)
        await self.server_adapter.update_weights(weights_gen, global_steps=version_id)

        logger.info(f"[AsyncFlowCheckpointEngineWorker] Weights version {version_id} loaded")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    async def update_weights(self, version_id):
        """Receive weights from trainer into host cache (does NOT push to vLLM server)."""
        logger.info(f"[AsyncFlowCheckpointEngineWorker] start recv weights {version_id}")
        await self.checkpoint_engine.receive_weights(version_id)


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
        self.backend_cls = AsyncFlowCheckpointEngineWithCache
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
            worker_handles=all_workers, ray_cls_with_init=RayClassWithInitArgs(cls=_worker_with_cache_class)
        )
        return rollout_wg

    def build_process_group(self, rollout: RayWorkerGroup, fwd: RayWorkerGroup, metadata):
        """Build process group for trainer, rollout replicas and fwd ."""
        trainer = self.trainer_wg

        trainer_kwargs, rollout_kwargs, fwd_kwargs = self.backend_cls.build_topology(
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

    def update_weights_rollout_fwd(self, version_id):
        logger.debug(f"[update_weights_rollout_fwd] version_id={version_id}")

        trainer_wg = self.trainer_wg
        rollout_wg = self.rollout_wg
        fwd_wg = self.fwd_wg

        pause_status = fwd_wg.pause()
        logger.debug(f"FWD paused: {pause_status}")

        if not self.is_first_sync:
            start = time.perf_counter()
            logger.debug(f"AgentLoop All Rollout replicas start pausing,time {time.perf_counter()}")
            self.pause_all_replicas()
            logger.info(f"AgentLoop All Rollout replicas paused, used time: {time.perf_counter() - start}")
        self.is_first_sync = False

        metadata = ray.get(
            trainer_wg.execute_checkpoint_engine(["prepare"] * trainer_wg.world_size)
            + rollout_wg.execute_checkpoint_engine(["prepare"] * rollout_wg.world_size)
            + fwd_wg.execute_checkpoint_engine(["prepare"] * fwd_wg.world_size)
        )

        self.build_process_group(rollout_wg, fwd_wg, metadata)

        # 4. update weights of all workers
        ray.get(
            trainer_wg.update_weights(version_id)
            + rollout_wg.update_weights(version_id)
            + fwd_wg.update_weights(version_id)
        )

        ray.get(
            trainer_wg.execute_checkpoint_engine(["finalize"] * trainer_wg.world_size)
            + rollout_wg.execute_checkpoint_engine(["finalize"] * rollout_wg.world_size)
            + fwd_wg.execute_checkpoint_engine(["finalize"] * fwd_wg.world_size)
        )

        self.notify_replica_update(version_id)

    def pause_all_replicas(self):
        """同步等待所有 replica 暂停"""
        futures = [
            asyncio.run_coroutine_threadsafe(replica.pause(wait_for_inflight_requests=True), self._loop)
            for replica in self.replicas
        ]
        if futures:
            concurrent.futures.wait(futures)
            for fut in futures:
                fut.result()

    def notify_replica_update(self, version_id):
        logger.debug(f"[notify_replica_update] version_id={version_id}, num_replicas={len(self.replicas)}")

        async def _receive_and_update_replica(replica, ckpt_workers, version_id):
            await asyncio.gather(*[w.load_weights.remote(version_id) for w in ckpt_workers])
            await replica.resume()

        futures = []

        for replica, (start, end) in zip(self.replicas, self.replica_worker_slice, strict=False):
            fut = asyncio.run_coroutine_threadsafe(
                _receive_and_update_replica(replica, self.rollout_wg.workers[start:end], version_id), self._loop
            )
            futures.append(fut)

        if futures:
            concurrent.futures.wait(futures, return_when=concurrent.futures.ALL_COMPLETED)

        for fut in futures:
            try:
                fut.result()
            except Exception as e:
                logger.exception(f"[notify_replica_update] replica update failed version_id={version_id}")
                raise e

        logger.debug(f"[notify_replica_update] All replicas updated for version_id={version_id}")
