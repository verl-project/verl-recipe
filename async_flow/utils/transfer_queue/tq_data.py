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
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

import ray
import zmq
import zmq.asyncio
from recipe.async_flow.utils.transfer_queue.tq_config import GROUP_SHARED_COLUMNS, ZMQ_HWM
from recipe.async_flow.utils.transfer_queue.tq_structures import ExperienceTable
from recipe.async_flow.utils.transfer_queue.tq_utils import setup_logger
from torch import Tensor


@ray.remote
class TransferQueueShard:
    def __init__(self, shard_id: int, port: Optional[int] = None):
        """
        Initialize a data shard.
        - shard_id: ID of this shard (for logging).
        - port: TCP port to bind the ZMQ Router server on.
        """

        self.logger = setup_logger(f"Shard_{shard_id}")
        self.logger.info(f"Shard {shard_id} started")

        self.shard_id = shard_id
        # 存储不同 Topic 的 ExperienceTable
        self.tables: dict[str, Any] = {}

        # Set up ZMQ Router socket
        self.zmq_ctx = zmq.asyncio.Context()
        self.router = self.zmq_ctx.socket(zmq.ROUTER)

        # HWM 配置，防止突发流量爆内存
        self.router.setsockopt(zmq.RCVHWM, ZMQ_HWM)
        self.router.setsockopt(zmq.SNDHWM, ZMQ_HWM)

        # 禁用 LINGER，防止 Actor 销毁时卡顿
        self.router.setsockopt(zmq.LINGER, 0)

        # Bind port
        if port is None:
            port = self.router.bind_to_random_port("tcp://0.0.0.0")
        else:
            self.router.bind(f"tcp://0.0.0.0:{port}")

        self.endpoint = f"tcp://{ray.util.get_node_ip_address()}:{port}"
        self.logger.info(f"DATA Shard[{shard_id}]: Started at {self.endpoint}")
        self.manager = ray.get_actor("TransferQueueManager")

        # Start background loop
        self._running = True
        self._loop_task = asyncio.get_event_loop().create_task(self._serve_loop())

        self._cleaner_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tq_cleaner")

    def add_experience_table(self, topic: str, n_samples_per_prompt: int, experience_columns: list[str]) -> None:
        """
        Create an ExperienceTable for a topic (auto-scaling).
        Define logical distribution only; storage grows dynamically.
        """
        if topic in self.tables:
            raise ValueError(f"Topic '{topic}' already exists")
        self.tables[topic] = ExperienceTable(
            n_samples_per_prompt=n_samples_per_prompt,
            experience_columns=experience_columns,
        )

    def remove_experience_table(self, topic: str) -> None:
        """
        Remove the storage table for the given topic.
        """
        if topic in self.tables:
            table = self.tables.pop(topic)
            self._cleaner_pool.submit(self._background_clear, table, topic)
            self.logger.info(f"Shard: Scheduled cleanup for topic '{topic}'")
            return

    def _background_clear(self, table, topic_name):
        try:
            table.clear()
        except Exception as e:
            self.logger.error(f"Error cleaning topic {topic_name}: {e}")
        finally:
            del table

    def clear_experience_table(self, topic: str):
        if topic not in self.tables:
            raise ValueError(f"Unknown topic '{topic}'. Known topics: {list(self.tables.keys())}")

        old_table = self.tables[topic]

        new_table = ExperienceTable(
            n_samples_per_prompt=old_table.n_samples_per_prompt, experience_columns=old_table.experience_columns
        )

        self.tables[topic] = new_table

        # 后台销毁旧表 (Drop)
        self._cleaner_pool.submit(self._background_clear, old_table, topic)

        self.logger.info(f"DATA Shard[{self.shard_id}]: Cleared topic '{topic}' (Async swap).")

    def prune_experience_table(self, topic: str, indexes: list[int] = None, check_n_samples: bool = False):
        """Prune stored experiences by index for a single topic in this shard."""
        if topic not in self.tables:
            raise ValueError(f"Unknown topic '{topic}. Known topics: {list(self.tables.keys())}'")
        if not indexes:
            raise ValueError("Indexes must be provided for prune_topic_by_indexes")
        tbl = self.tables[topic]

        if check_n_samples:
            # Check if all samples for a single prompt are pruned together.
            n_samples = tbl.n_samples_per_prompt
            groups = {}
            for idx in indexes:
                groups.setdefault(idx // n_samples, []).append(idx)

            incomplete = []
            for group_sub_indexes in groups.values():
                if len(set(group_sub_indexes)) != n_samples:
                    incomplete.extend(group_sub_indexes)
            if incomplete:
                raise ValueError(
                    f"Shard {self.shard_id} received indexes: {sorted(set(incomplete))}, "
                    "which are incomplete for prompt groups"
                )
            absence = [group_id for group_id in groups.keys() if group_id not in tbl.owned_groups]
            if absence:
                raise ValueError(
                    f"Shard {self.shard_id} received group_ids: {sorted(set(absence))}, which are not in the shard"
                )

        tbl.prune(indexes)
        self.logger.info(f"DATA Shard[{self.shard_id}]: Pruned indexes{indexes} on topic '{topic}'.")

    async def _serve_loop(self):
        """
        核心事件循环：持续监听并分发任务。
        """
        while self._running:
            try:
                # copy=False 减少内存搬运
                frames = await self.router.recv_multipart(copy=False)

                # 创建并发任务处理消息，不阻塞接收循环
                asyncio.create_task(self._process_message(frames))
            except zmq.ZMQError as e:
                self.logger.error(f"ZMQ Error in serve loop: {e}")
                await asyncio.sleep(0.1)
            except Exception as e:
                self.logger.exception(f"Unexpected Error in serve loop: {e}")
                await asyncio.sleep(0.1)

    async def _process_message(self, frames: list[zmq.Frame]):
        """
        协议解析与路由分发
        Frames 结构: [Identity, Command, Header, Payload...]
        """
        # 基础协议长度校验
        if len(frames) < 3:
            self.logger.error(f"Malformed request, frames len: {len(frames)}")
            return

        identity = frames[0]
        command = frames[1].bytes

        try:
            if command == b"PUT":
                # PUT: [Identity, b'PUT', Header, Payload_1, Payload_2...]
                await self._handle_put(identity, frames[2], frames[3:])
            elif command == b"GET":
                # GET: [Identity, b'GET', Header]
                await self._handle_get(identity, frames[2])
            else:
                self.logger.warning(f"Unknown command: {command}")
                await self.router.send_multipart([identity, b"ERROR: Unknown command"])

        except Exception as e:
            # 捕获所有处理层面的异常，返回给客户端
            self.logger.exception(f"Error processing {command}")
            error_msg = f"ERROR: {str(e)}".encode()
            try:
                await self.router.send_multipart([identity, error_msg])
            except Exception as send_err:
                self.logger.error(f"Failed to send error response: {send_err}")

    async def _handle_put(self, identity: zmq.Frame, header_frame: zmq.Frame, payload_frames: list[zmq.Frame]):
        """
        处理写入请求。
        """
        header = pickle.loads(header_frame.bytes)

        topic = header.get("topic")
        if not topic:
            raise ValueError("Missing 'topic' in header")

        if topic not in self.tables:
            raise ValueError(f"Table is not established for topic:{topic}")

        table = self.tables[topic]
        table.put_batch(
            global_ids=header["indexes"],
            col_order=header["order"],
            col_inputs_meta=header,
            payload_frames=payload_frames,
        )
        reported_indexes = header["indexes"]
        self.logger.debug(f"Putting {len(reported_indexes)} indexes in {topic}")

        # 3. Notify manager of updated columns
        # 需要按列分别处理：共享列扩展索引，非共享列保持原索引
        try:
            await self._update_data_status_by_column(topic, reported_indexes, header["order"], header["columns"])
        except Exception as e:
            self.logger.error(f"Shard [{self.shard_id}]: update_data_status failed: {e}")

        await self.router.send_multipart([identity, b"ACK"])

    async def _update_data_status_by_column(self, topic: str, indexes: list[int], col_order: list[str], col_info: dict):
        """
        按列分别更新数据状态：共享列扩展索引，非共享列保持原索引

        Args:
            topic: Topic名称
            indexes: 原始索引列表
            col_order: 列顺序
            col_info: 列信息字典，包含 ref_multiplier
        """

        # 按列分组：共享列和非共享列需要不同的索引
        shared_columns = []
        non_shared_columns = []

        for col_name in col_order:
            ref_multiplier = col_info[col_name].get("ref_multiplier", 1)
            if ref_multiplier > 1 and col_name in GROUP_SHARED_COLUMNS:
                shared_columns.append(col_name)
            else:
                non_shared_columns.append(col_name)

        # 更新非共享列状态（使用原始索引）
        if non_shared_columns:
            try:
                await self.manager.update_data_status.remote(topic, indexes, non_shared_columns)
            except Exception as e:
                self.logger.error(f"Failed to update status for non-shared columns {non_shared_columns}: {e}")

        # 更新共享列状态（扩展索引为整个group）
        if shared_columns:
            n_samples = self.tables[topic].n_samples_per_prompt
            extended_indexes = set()

            for idx in indexes:
                group_id = idx // n_samples
                # 为整个 group 的所有索引建立映射
                for offset in range(n_samples):
                    extended_indexes.add(group_id * n_samples + offset)

            extended_indexes = sorted(extended_indexes)

            try:
                await self.manager.update_data_status.remote(topic, extended_indexes, shared_columns)
            except Exception as e:
                self.logger.error(f"Failed to update status for shared columns {shared_columns}: {e}")

            self.logger.debug(
                f"Updated shared columns {shared_columns} status: indexes={indexes} -> extended={extended_indexes}"
            )

    async def _handle_get(self, identity: zmq.Frame, header_frame: zmq.Frame):
        """
        处理读取请求。
        """
        header = pickle.loads(header_frame.bytes)

        # Topic 校验
        topic = header.get("topic")
        if topic not in self.tables:
            raise ValueError(f"Topic '{topic}' not found on shard {self.shard_id}")

        table = self.tables[topic]

        result_meta, result_payloads = table.get_batch(
            target_global_idxs=header["indexes"], target_cols=header["experience_columns"]
        )

        # 响应格式: [Identity, MetaBytes, Payload_1, Payload_2...]
        meta_bytes = pickle.dumps(result_meta)

        send_frames = [identity, meta_bytes]
        send_frames.extend(result_payloads)

        await self.router.send_multipart(send_frames)

    def get_endpoint(self):
        return self.endpoint

    def get_values(self, topic: str, column: str, global_indexes: list[int]) -> list[Tensor]:
        """
        Return stored values for the given column at the specified global indexes.
        No validation, rely on upper layer correctness.
        """
        tbl = self.tables[topic]
        values = [
            tbl.data[column][idx // tbl.n_samples_per_prompt][idx % tbl.n_samples_per_prompt] for idx in global_indexes
        ]
        return values

    def update_owned_groups(self, topic: str, group_ids: list[int]):
        if topic not in self.tables:
            raise ValueError(f"Unknown topic '{topic}'")
        self.tables[topic].owned_groups.update(group_ids)

    def get_all_columns_storage_bytes(self, topic: str) -> dict[str, int]:
        """
        返回所有列的实际存储字节数
        计算方法：对每个 MemorySegment，取其中所有 entry 的最大 (offset + length)
        """
        if topic not in self.tables:
            raise ValueError(f"Unknown topic '{topic}'")
        table = self.tables[topic]

        result = {}
        for column, col_index in table.indices.items():
            segments_bytes = {}  # segment_id -> max_end_offset
            for _, (segment, offset, length) in col_index.items():
                seg_id = id(segment)
                max_end = offset + length
                if seg_id not in segments_bytes:
                    segments_bytes[seg_id] = 0
                segments_bytes[seg_id] = max(segments_bytes[seg_id], max_end)
            result[column] = sum(segments_bytes.values())

        return result

    def reset_all(self):
        topics = list(self.tables.keys())

        for topic in topics:
            self.remove_experience_table(topic)

        self.logger.info(f"DATA Shard[{self.shard_id}]: Dropped ALL topics/tables ({len(topics)} topics).")

    async def async_shutdown(self):
        """
        停止服务：
        1. 停止接收循环
        2. 取消挂起的异步任务
        3. 销毁 ZMQ 上下文和 Socket
        """
        self.logger.info(f"DATA Shard[{self.shard_id}]: Shutting down...")

        # 停止标志位
        self._running = False

        # 取消后台循环任务
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()
            try:
                # 等待任务取消完成，确保不会在资源销毁后尝试访问 Socket
                await self._loop_task
            except asyncio.CancelledError:
                self.logger.info("Serve loop cancelled successfully.")
            except Exception as e:
                self.logger.warning(f"Error during loop cancellation: {e}")

        # 清理 ZMQ 资源
        # LINGER=0 确保关闭时不等待未发送消息，立即释放
        if not self.router.closed:
            self.router.close(linger=0)

        if not self.zmq_ctx.closed:
            self.zmq_ctx.destroy(linger=0)

        self.logger.info(f"DATA Shard[{self.shard_id}]: Shutdown complete.")
