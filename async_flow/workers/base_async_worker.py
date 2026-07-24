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
import time
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from recipe.async_flow.utils.metric.prometheus import marked_timer
from recipe.async_flow.utils.transfer_queue.tq_client import TransferQueueClient
from recipe.async_flow.workers.data_dispatch_strategy import (
    EngineBackend,
    create_dispatch_strategy,
)

from verl.single_controller.base.decorator import Dispatch, register
from verl.utils.py_functional import append_to_dict


class AsyncWorkerMixin(ABC):
    """异步 Worker Mixin，提供 TransferQueue 交互循环逻辑。

    子类必须定义以下类属性：
        CONSUMER_NAME: TQ consumer 名称
        INPUT_COLUMNS: 从 TQ 读取的列名
        OUTPUT_COLUMNS: 写入 TQ 的列名
    子类可选以下类属性：
        EXTRA_FETCH_KWARGS：给 get_experience_async 增加参数（比如 get_n_samples、train 的 staleness/version 等）
        EXTRA_PUT_KWARGS：给 put_experience_async 增加参数
    子类必须实现以下方法：
        process_batch(): 处理一个批次的数据
    """

    # 子类必须定义的类属性
    CONSUMER_NAME: str = ""
    INPUT_COLUMNS: tuple[str, ...] = ()
    OUTPUT_COLUMNS: tuple[str, ...] = ()
    # 子类可选的类属性，用于扩展
    EXTRA_FETCH_KWARGS: dict[str, Any] = {}
    EXTRA_PUT_KWARGS: dict[str, Any] = {}

    def init_async_worker(
        self,
        tq_client: TransferQueueClient,
        topic: str = "experience",
        experience_count: int = 2,
        engine_backend: EngineBackend = EngineBackend.NONE,
        dispatch_strategy_kwargs: Optional[dict[str, Any]] = None,
    ) -> None:
        """初始化异步 Worker。

        Args:
            tq_client: TensorDockClient 实例（必需）
            topic: TQ topic 名称
            experience_count: 每次从TD取数大小
            engine_backend: 并行后端类型，决定 dispatch/collect 行为
            dispatch_strategy_kwargs: 并行策略的额外参数（如 rank, size, group 等）
        """
        if tq_client is None:
            raise ValueError("tq_client is required")

        self._tq_client = tq_client
        self._topic = topic
        self._experience_count = experience_count
        self._loop_running = False
        self._loop_thread: Optional[threading.Thread] = None
        self._last_error: Optional[str] = None
        self._stats = {"processed_batches": 0, "processed_samples": 0, "errors": 0}

        self.lock = threading.Lock()

        # metrics
        self._timing_metrics = {}
        # 初始化并行策略
        self._engine_backend = engine_backend
        strategy_kwargs = dispatch_strategy_kwargs or {}
        self._dispatch_strategy = create_dispatch_strategy(engine_backend, **strategy_kwargs)
        self.EXTRA_FETCH_KWARGS = dict(getattr(self, "EXTRA_FETCH_KWARGS", {}) or {})
        self.EXTRA_PUT_KWARGS = dict(getattr(self, "EXTRA_PUT_KWARGS", {}) or {})

        self.put_data_counter = 0

        # ── cluster trace auto-install ───────────────────────────────────
        if os.environ.get("VERL_CLUSTER_TRACE"):
            from recipe.async_flow.utils.cluster_trace.trace_logger import _get_role, install

            rank = int(os.environ.get("RANK", 0))
            role = _get_role(type(self).__name__)
            install(role=role, rank=rank)
        # ────────────────────────────────────────────────────────────────

    @abstractmethod
    def process_batch(self, payload: dict[str, Any], indexes: list[int]) -> dict[str, Any]:
        """处理一个批次的数据。

        子类实现此方法：
        1. 从 payload 中提取数据
        2. 调用计算逻辑
        3. 返回处理结果（TQ 只支持 tensor 格式）

        Args:
            payload: 从 TQ 获取的数据字典
            indexes: 数据索引列表

        Returns:
            处理后的数据字典，将被写入 TQ；返回空字典表示不写回
        """
        pass

    def on_process_begin(self) -> None:  # noqa: B027
        """获取数据前的逻辑。"""
        pass

    def on_process_end(self) -> None:  # noqa: B027
        """处理完数据后的逻辑。"""
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_async_loop(self) -> None:
        """启动异步处理循环。"""
        if self._loop_running:
            self.logger.warning(f"[{self.CONSUMER_NAME}] Loop is already running")
            return
        self._last_error = None
        self._loop_running = True
        self._loop_thread = threading.Thread(target=self._run_loop, daemon=True)
        self._loop_thread.start()
        self.logger.info(f"[{self.CONSUMER_NAME}] Loop thread started")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_async_loop(self) -> None:
        """停止异步处理循环。"""
        self._loop_running = False
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=5.0)
            self._loop_thread = None
        self.logger.info(f"[{self.CONSUMER_NAME}] Loop stopped")

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_stats(self) -> dict[str, Any]:
        """获取统计信息。"""
        return {
            "consumer_name": self.CONSUMER_NAME,
            "running": self._loop_running,
            "errors": self._stats["errors"],
            "last_error": self._last_error,
            **self._stats,
        }

    @property
    def logger(self) -> logging.Logger:
        if not hasattr(self, "_logger") or self._logger is None:
            log_level = os.getenv("LOGGING_LEVEL", logging.INFO)
            logging.basicConfig(
                level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", force=True
            )
            self._logger = logging.getLogger(f"async_worker.{self.CONSUMER_NAME}")
        return self._logger

    def _run_loop(self) -> None:
        """在新线程中运行异步循环（设置设备上下文）。"""
        try:
            self._setup_device_context()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._async_loop())
            finally:
                loop.close()
        except Exception as e:
            self._loop_running = False
            self._last_error = f"{type(e).__name__}: {str(e)}"
            self._stats["errors"] += 1
            self.logger.exception(f"[{self.CONSUMER_NAME}] Fatal error in _run_loop: {e}")

    def _setup_device_context(self) -> None:
        """设置线程的设备上下文（NPU/CUDA 设备上下文是线程本地的）。"""
        from verl.utils.device import get_device_name

        device_name = get_device_name()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if device_name == "npu":
            import torch_npu

            torch_npu.npu.set_device(local_rank)
        elif device_name == "cuda":
            torch.cuda.set_device(local_rank)

    def _build_fetch_params(self) -> dict[str, Any]:
        params = {
            "consumer": self.CONSUMER_NAME,
            "experience_columns": list(self.INPUT_COLUMNS),
            "experience_count": self._experience_count,
        }
        params.update(self.EXTRA_FETCH_KWARGS)
        return params

    def _build_put_params(
        self, data: dict[str, Any], indexes: list[int], version: Optional[int] = None
    ) -> dict[str, Any]:
        params = {
            "data_dict": data,
            "indexes": indexes,
        }
        params.update(self.EXTRA_PUT_KWARGS)
        return params

    async def _async_loop(self) -> None:
        """异步处理主循环：从 TQ 获取数据 -> 处理 -> 写回 TQ。"""
        self._loop_running = True
        self.logger.info(f"[{self.CONSUMER_NAME}] Starting async loop ...")

        e2e_start = time.perf_counter()
        while self._loop_running:
            try:
                # metrics define
                timing_raw = {}
                e2e_time = f"e2e_max_{self.CONSUMER_NAME}"
                wait_time = f"wait_max_{self.CONSUMER_NAME}"
                compute_time = f"compute_max_{self.CONSUMER_NAME}"
                with marked_timer(e2e_time, timing_raw):
                    time.sleep(0.1)
                    with self.lock:
                        self.on_process_begin()
                        with marked_timer(wait_time, timing_raw):
                            payload, indexes = await self._dispatch_strategy.dispatch_data(
                                tq_client=self._tq_client,
                                fetch_params=self._build_fetch_params(),
                            )
                            if payload is None or indexes is None or not indexes:
                                continue
                        timing_raw[wait_time] = time.perf_counter() - e2e_start
                        self.logger.debug(
                            f"[{self.CONSUMER_NAME}] fetch data cost time: {timing_raw[wait_time]} "
                            f"fetch count:{len(indexes)}"
                        )

                        if getattr(self, "_dist_barrier", False):
                            barrier_time = f"max_barrier_{self.CONSUMER_NAME}"
                            with marked_timer(barrier_time, timing_raw):
                                torch.distributed.barrier()

                        # 处理批次
                        with marked_timer(compute_time, timing_raw):
                            output_data = self.process_batch(payload, indexes)
                        self.logger.debug(f"[{self.CONSUMER_NAME}] batch compute cost time: {timing_raw[compute_time]}")

                        # 使用并行策略进行 collect
                        if output_data:
                            version = output_data.pop("_version", None)
                            with marked_timer("put_data", timing_raw):
                                await self._dispatch_strategy.collect_data(
                                    tq_client=self._tq_client,
                                    put_params=self._build_put_params(output_data, indexes, version),
                                )
                                self.put_data_counter += len(indexes)
                            self.logger.debug(
                                f"[{self.CONSUMER_NAME}] put data cost time: {timing_raw['put_data']}"
                                f"put count:{len(indexes)}, total put count:{self.put_data_counter}"
                            )
                timing_raw[e2e_time] = time.perf_counter() - e2e_start
                self._collect_timing_metrics(payload, timing_raw)
                # 更新统计
                self._stats["processed_batches"] += 1
                self._stats["processed_samples"] += len(indexes)

                self.on_process_end()
                e2e_start = time.perf_counter()
            except Exception as e:
                self._loop_running = False
                self._last_error = f"{type(e).__name__}: {str(e)}"
                self._stats["errors"] += 1
                self.logger.exception(f"[{self.CONSUMER_NAME}] Fatal error: {e}")
                break

        self.logger.info(
            f"[{self.CONSUMER_NAME}] Async loop stopped (processed {self._stats['processed_samples']} samples)"
        )

    def get_consumer_name(self) -> str:
        return self.CONSUMER_NAME

    def get_input_columns(self) -> list[str]:
        return list(self.INPUT_COLUMNS)

    def get_output_columns(self) -> list[str]:
        return list(self.OUTPUT_COLUMNS)

    @abstractmethod
    def get_experience_step(self) -> int:
        """子类实现此方法获取消费轮次"""
        pass

    def _collect_timing_metrics(self, payload, timing_raw) -> None:
        mini_step_metrics: dict[str, Any] = {}

        e2e_time = f"e2e_max_{self.CONSUMER_NAME}"
        wait_time = f"wait_max_{self.CONSUMER_NAME}"
        compute_time = f"compute_max_{self.CONSUMER_NAME}"
        local_num_tokens = torch.cat(payload["attention_mask"]).sum()

        mini_step_metrics["total_num_tokens"] = float(local_num_tokens)
        mini_step_metrics[e2e_time] = timing_raw[e2e_time]
        mini_step_metrics[wait_time] = timing_raw[wait_time]
        mini_step_metrics[compute_time] = timing_raw[compute_time]

        # 默认预留版本
        current_version = "None"
        if getattr(self, "_print_version_metrics", False):
            current_version = self.get_current_version()
        experience_step = self.get_experience_step()
        summary = {
            "consumer_name": self.CONSUMER_NAME,
            "version": current_version,
            "experience_step": experience_step,
            **mini_step_metrics,
        }
        append_to_dict(self._timing_metrics, summary)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_timing_metrics(self) -> dict:
        """获取当前worker的Metrics。"""
        current_metrics = self._timing_metrics
        self._timing_metrics = {}
        return current_metrics
