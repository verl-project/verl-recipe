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
import pickle
import threading
import time
from collections import defaultdict
from typing import Any, Optional

import numpy as np
import ray
import torch
import zmq
import zmq.asyncio
from recipe.async_flow.utils.transfer_queue.tq_config import (
    DEFAULT_TOPIC,
    GROUP_SHARED_COLUMNS,
    MAX_CONCURRENT_GETS,
    NUM_SAMPLE_PER_SEGMENT,
    PADDED_COLUMNS,
    ZMQ_HWM,
)
from recipe.async_flow.utils.transfer_queue.tq_sampler import (
    BaseSampler,
)
from recipe.async_flow.utils.transfer_queue.tq_utils import (
    deserialize_column_from_frame,
    deserialize_column_pickle_from_frame,
    get_numpy_dtype,
    serialize_batch,
    serialize_batch_pickle,
    torch_to_numpy,
)


class TransferQueueClient:
    def __init__(
        self,
        manager_handle,
        logger=None,
        micro_batch_size=NUM_SAMPLE_PER_SEGMENT,
        max_concurrent_gets=MAX_CONCURRENT_GETS,
    ):
        """
        Initialize the client interface.
        - manager_handle: A Ray actor handle for Manager.
        """
        self.manager = manager_handle
        self.zmq_context = zmq.Context.instance()
        self.async_ctx = zmq.asyncio.Context()
        self._local = threading.local()
        self.socket_cache = {}
        self.num_sample_per_segment = micro_batch_size
        self.max_concurrent_gets = max_concurrent_gets
        self.get_semaphore = asyncio.Semaphore(self.max_concurrent_gets)

        if logger is None:
            self.logger = logging.getLogger("TransferQueueClient")
        else:
            self.logger = logger

    def add_topic(
        self,
        prompts_num: int,
        n_samples_per_prompt: int,
        experience_columns: list[str],
        experience_consumers: list[str],
        topic: str = DEFAULT_TOPIC,
    ) -> None:
        """
        Register a topic on the manager and provision per-shard storage.
        - Pure schema setup; no data is inserted here.
        - Raises if the topic already exists.
        """
        if not topic:
            raise ValueError("Topic must be non-empty")
        ray.get(
            self.manager.add_topic.remote(
                topic,
                prompts_num,
                n_samples_per_prompt,
                experience_columns,
                experience_consumers,
            )
        )
        self.logger.info(
            f"Client: Created topic '{topic}' with n_samples_per_prompt={n_samples_per_prompt}, "
            f"experience_columns={experience_columns}, experience_consumers={experience_consumers}"
        )

    def delete_topic(self, topic: str = DEFAULT_TOPIC) -> None:
        """
        Delete the specified topic, including tables on the manager and all shards.
        """
        if topic is None:
            topic = DEFAULT_TOPIC
        ray.get(self.manager.delete_topic.remote(topic))

    def reset_all(self):
        """Fully reset back to post-__init__ state: drop all topics/tables and reset manager state."""
        # NOTE: this removes schemas entirely; add_topic must be called again afterwards.
        ray.get(self.manager.reset_all.remote())
        self.logger.info("Client: Fully reset TransferQueue to post-__init__ state (ALL topics dropped).")

    def _get_async_socket(self, endpoint: str) -> zmq.asyncio.Socket:
        """Get or create a ZMQ DEALER socket for the specific endpoint."""
        if not hasattr(self._local, "socket_cache"):
            self._local.socket_cache = {}

        if endpoint not in self._local.socket_cache:
            sock = self.async_ctx.socket(zmq.DEALER)

            sock.setsockopt(zmq.SNDHWM, ZMQ_HWM)
            sock.setsockopt(zmq.RCVHWM, ZMQ_HWM)

            # 设置 Linger 为 0，确保关闭时不阻塞
            sock.setsockopt(zmq.LINGER, 0)

            sock.connect(endpoint)
            self._local.socket_cache[endpoint] = sock

        return self._local.socket_cache[endpoint]

    def _invalidate_socket(self, endpoint: str):
        if not hasattr(self._local, "socket_cache"):
            return

        if endpoint in self._local.socket_cache:
            sock = self._local.socket_cache[endpoint]
            try:
                sock.close(linger=0)
            except Exception:
                pass
            del self._local.socket_cache[endpoint]

    def put_experience(
        self,
        data_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        indexes: Optional[list[int]] = None,
        version: int = None,
        unpad_pairs: list[tuple[str, str, str]] = None,
        topic: str = None,
        save_dtype: str = None,
        needs_expansion: bool = False,
    ) -> list[int]:
        return asyncio.run(
            self.put_experience_async(data_dict, indexes, version, unpad_pairs, topic, save_dtype, needs_expansion)
        )

    async def put_experience_async(
        self,
        data_dict: dict[str, torch.Tensor | list[torch.Tensor]],
        indexes: Optional[list[int]] = None,
        version: int = None,
        unpad_pairs: list[tuple[str, str, str]] = None,
        topic: str = None,
        save_dtype: str = None,
        needs_expansion: bool = False,
    ) -> list[int]:
        """
        将数据写入TransferQueue，支持共享列分离发送和unpad处理

        Args:
            data_dict: 数据字典 {列名: 数据}
            indexes: 已分配的索引索引（场景a），None则自动分配（场景b/c/d）
            version: 版本号
            unpad_pairs: unpad配对列表，如 [('prompt', 'prompt_length', 'right')]
            needs_expansion: 扩展模式标志（场景a: True, 其他: False）
        """
        start_time = time.perf_counter()
        if topic is None:
            topic = DEFAULT_TOPIC
        n_samples_per_prompt = ray.get(self.manager.get_n_samples_per_prompt.remote(topic))

        # 处理 unpad 映射关系: col_name -> (len_col_name, mode)
        unpad_map = {}
        if unpad_pairs:
            for col, len_col, mode in unpad_pairs:
                unpad_map[col] = (len_col, mode)

        # Step 1: 准备indexes 并与 batch_size核对
        batch_size = self._infer_batch_size(data_dict, unpad_pairs)
        if batch_size == 1:
            data_dict = self._normalize_single_item_data(data_dict)
        if indexes is None:
            uuids = self._extract_uuids(data_dict)
            indexes = ray.get(self.manager.allocate_indexes.remote(topic, batch_size, needs_expansion, uuids))
        batch_size_to_put = batch_size * n_samples_per_prompt if needs_expansion else batch_size
        if batch_size_to_put != len(indexes):
            raise ValueError(
                f"batch_size_to_put:{batch_size_to_put} doesn't match with length of indexes:{len(indexes)}"
            )

        # Step 2: 切分数据（共享列 vs 非共享列）
        shared_dict, non_shared_dict, groups_to_mark = self._filter_and_split_data(
            data_dict, indexes, topic, needs_expansion
        )

        # Step 3: 并行发送任务
        tasks = []

        # 任务1: 发送共享列（如果有）
        if shared_dict:
            # 计算共享列的目标索引：每个group的第一个索引
            shared_indexes = [group_id * n_samples_per_prompt for group_id in sorted(groups_to_mark)]

            try:
                shared_route_map = await self.manager.get_targets_for_put.remote(topic, shared_indexes)
            except Exception as e:
                self.logger.error(f"Routing failed for shared columns: {e}")
                return []

            for endpoint, global_idxs in shared_route_map.items():
                shared_items = [
                    (i, global_idx) for i, global_idx in enumerate(shared_indexes) if global_idx in global_idxs
                ]
                tasks.append(
                    self._process_node_data(endpoint, shared_items, shared_dict, {}, topic, save_dtype, is_shared=True)
                )

        # 任务2: 发送非共享列（如果有）
        if non_shared_dict:
            try:
                non_shared_route_map = await self.manager.get_targets_for_put.remote(topic, indexes)
            except Exception as e:
                self.logger.error(f"Routing failed for non-shared columns: {e}")
                return []

            global_idx_to_local_map = {global_idx: idx for idx, global_idx in enumerate(indexes)}
            endpoint_groups = defaultdict(list)
            routed_global_idxs = set()

            for endpoint, global_idxs in non_shared_route_map.items():
                for global_idx in global_idxs:
                    local_idx = global_idx_to_local_map[global_idx]
                    endpoint_groups[endpoint].append((local_idx, global_idx))
                    routed_global_idxs.add(global_idx)

            # 检查是否有 ID 未找到路由
            if len(routed_global_idxs) < len(indexes):
                missing_global_idxs = set(indexes) - routed_global_idxs
                for missing_global_idx in missing_global_idxs:
                    self.logger.warning(f"No route found for index {missing_global_idx}")

            for endpoint, items in endpoint_groups.items():
                tasks.append(
                    self._process_node_data(endpoint, items, non_shared_dict, {}, topic, save_dtype, is_shared=False)
                )

        # 如果没有数据要发送
        if not tasks:
            self.logger.warning(f"No data to send for indexes {indexes}")
            return []

        # Step 4: 并行执行所有发送任务
        results = await asyncio.gather(*tasks)

        # Step 5: 写入成功后，标记共享列状态
        for group_id in groups_to_mark:
            for col_name in GROUP_SHARED_COLUMNS:
                if col_name in shared_dict:
                    ray.get(self.manager.mark_column_written_in_group.remote(topic, group_id, col_name))

        # Step 6: 记录版本
        if version is not None:
            ray.get(self.manager.record_versions.remote(topic, version, indexes))

        # Step 7: 记录日志
        total_latency = (time.perf_counter() - start_time) * 1000
        total_bytes = sum(r["bytes"] for r in results)
        total_mb = total_bytes / (1024 * 1024)

        self.logger.info(
            f"[TransferQueue Put] | Total Latency: {total_latency:.2f} ms | Total Volume: {total_mb:.2f} MB | "
            f"Routed_experts exists : {'routed_experts' in data_dict} | Data_size: {len(indexes)} | "
            f"data_dict.keys : {list(data_dict.keys())} | Indexes: {indexes} | "
        )

        if self.logger.isEnabledFor(logging.DEBUG):
            for res in results:
                self.logger.debug(
                    f"  -> Node: {res['endpoint']} | Cnt: {res['count']} | "
                    f"Sz: {res['bytes'] / 1024:.2f} KB | Lat: {res['latency']:.2f} ms"
                )
        return indexes

    def _extract_uuids(self, data_dict: dict[str, Any]) -> list[str]:
        """提取UUID列的值（如果有）"""
        uuid_col = [k for k in data_dict if k.endswith(("_uuid", "_uid"))]
        if len(uuid_col) > 1:
            raise ValueError(f"Expected at most 1 UUID column, found {len(uuid_col)}: {uuid_col}")
        if not uuid_col:
            return []
        raw_list = data_dict[uuid_col[0]]
        uuids = [str(x) for x in raw_list] if raw_list else []
        return uuids

    def _infer_batch_size(
        self, data_dict: dict[str, Any], unpad_pairs: Optional[list[tuple[str, str, str]]] = None
    ) -> int:
        """
        推断数据批量大小

        优先级：
        1. List 的长度（最明确）
        2. unpad_pairs 中 pair[0] 的 2D+ tensor 的 shape[0]
        3. 默认为 1（包括 UUID、2D tensor、其他所有情况）

        Args:
            data_dict: 输入数据字典
            unpad_pairs: unpad 配对列表
        """
        # 1. 优先从 List 长度推断
        for value in data_dict.values():
            if isinstance(value, list) and len(value) > 0:
                return len(value)

        # 2. 从 padded_columns 中的 2D+ tensor 推断
        padded_columns = PADDED_COLUMNS.copy()
        if unpad_pairs:
            padded_columns.update(pair[0] for pair in unpad_pairs)
            # padded_columns.update(col_name for pair in unpad_pairs for col_name in pair)
        for col_name, value in data_dict.items():
            if col_name in padded_columns and isinstance(value, torch.Tensor) and value.ndim >= 2:
                return value.shape[0]

        # 3. 默认为 1（包括 UUID、2D tensor 等）
        self.logger.debug(
            f"No list or padded 2D tensor found, inferring batch_size=1. data_dict keys: {list(data_dict.keys())}"
        )
        return 1

    def _normalize_single_item_data(
        self, data_dict: dict[str, torch.Tensor | list[torch.Tensor]]
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        """
        Normalize single-item input to batch format.

        当 batch_size==1 时，将非 list 的值封装成 [value]

        Args:
            data_dict: 输入数据字典

        Returns:
            Normalized data_dict
        """
        normalized = {}
        for key, value in data_dict.items():
            if isinstance(value, list):
                normalized[key] = value
            else:
                if isinstance(value, torch.Tensor) and value.ndim == 0:
                    # 0D 标量 → 转换为 1D 再包装
                    normalized[key] = [value.unsqueeze(0)]
                else:
                    normalized[key] = [value]
        return normalized

    def _filter_and_split_data(
        self, data_dict: dict, indexes: list[int], topic: str, needs_expansion: bool = False
    ) -> tuple[dict[str, list], dict[str, list], set[int]]:
        """
        将data_dict切分为共享列和非共享列

        Args:
            needs_expansion: 是否为扩展模式（场景a: True, 场景b/c/d: False）
                - True: data_dict[i] 对应 group_i 的所有索引
                - False: data_dict[i] 属于 indexes[i] 所在的组

        返回：
            - shared_dict: 共享列字典（每个group取一个数据）
            - non_shared_dict: 非共享列字典（完整数据）
            - groups_to_mark: 需要标记写入状态的group集合
        """
        # 前置检查：如果没有共享列，直接返回原data_dict
        if not any(col_name in GROUP_SHARED_COLUMNS for col_name in data_dict.keys()):
            return {}, data_dict, set()

        # 根据模式确定唯一group列表
        n_samples = ray.get(self.manager.get_n_samples_per_prompt.remote(topic))
        if needs_expansion:
            # 扩展模式（场景a）：data_dict[i] 就是 group i 的数据
            data_len = len(next(iter(data_dict.values())))
            unique_groups = set(range(data_len))
        else:
            # 正常模式（场景b/c/d）：data_dict[i] 属于 indexes[i] 所在的组
            idx_to_group = {i: idx // n_samples for i, idx in enumerate(indexes)}
            unique_groups = set(idx_to_group.values())

        # 收集需要发送共享列的[列, 组]组合
        groups_by_col: dict[str, set[int]] = {}
        for col_name in data_dict.keys():
            if col_name in GROUP_SHARED_COLUMNS:
                for group_id in unique_groups:
                    if not ray.get(self.manager.is_column_written_in_group.remote(topic, group_id, col_name)):
                        if col_name not in groups_by_col:
                            groups_by_col[col_name] = set()
                        groups_by_col[col_name].add(group_id)

        # 如果没有需要发送的任何共享列，返回空shared_dict，过滤共享列
        if not groups_by_col:
            filtered_non_shared = {}
            for col_name, col_data in data_dict.items():
                if col_name not in GROUP_SHARED_COLUMNS:
                    filtered_non_shared[col_name] = col_data
            return {}, filtered_non_shared, set()

        # 收集所有需要标记的组
        groups_to_mark = set()
        for groups in groups_by_col.values():
            groups_to_mark.update(groups)

        # 切分数据字典
        shared_dict = {}
        non_shared_dict = {}

        for col_name, col_data in data_dict.items():
            if col_name in GROUP_SHARED_COLUMNS:
                # 该列是否在 groups_by_col 中（需要发送）
                groups_for_this_col = groups_by_col.get(col_name)
                if not groups_for_this_col:
                    # 该列不需要发送，跳过
                    continue
                # 共享列：只为需要发送的组提供数据
                shared_dict[col_name] = []
                if needs_expansion:
                    for group_id in sorted(groups_for_this_col):
                        if group_id < len(col_data):
                            shared_dict[col_name].append(col_data[group_id])
                else:
                    # 正常模式：构建 group -> [local_idx] 映射
                    group_to_row_indices: dict[int, list[int]] = {}
                    for local_idx, group_id in enumerate(indexes):
                        group_id = idx_to_group[local_idx]
                        group_to_row_indices.setdefault(group_id, []).append(local_idx)

                    for group_id in sorted(groups_for_this_col):
                        if group_id in group_to_row_indices:
                            first_row_idx = group_to_row_indices[group_id][0]
                            shared_dict[col_name].append(col_data[first_row_idx])
            else:
                # 非共享列：保留完整数据
                non_shared_dict[col_name] = col_data

        return shared_dict, non_shared_dict, groups_to_mark

    async def _process_node_data(
        self,
        endpoint: str,
        items: list[tuple[int, int]],
        data_dict: dict[str, Any],
        unpad_map: dict[str, tuple[str, str]],
        topic: str,
        save_dtype: str = None,
        is_shared: bool = False,
    ) -> dict[str, Any]:
        """
        通用发送方法，支持共享列和非共享列

        Args:
            endpoint: 目标endpoint
            items: (local_idx, global_id) 列表
            data_dict: 数据字典
            unpad_map: unpad映射
            topic: topic名称
            save_dtype: 保存数据类型
            is_shared: 是否是共享列

        Returns:
            统计信息字典
        """
        node_start = time.perf_counter()
        total_sent_bytes = 0
        async with self.get_semaphore:
            lock = self._get_endpoint_lock(endpoint)
            async with lock:
                try:
                    sock = self._get_async_socket(endpoint)
                    recv_futures = []

                    for i in range(0, len(items), self.num_sample_per_segment):
                        batch_items = items[i : i + self.num_sample_per_segment]

                        start_local_idx = batch_items[0][0]
                        batch_len = len(batch_items)
                        end_local_idx = start_local_idx + batch_len
                        slice_obj = slice(start_local_idx, end_local_idx)

                        global_ids_batch = [x[1] for x in batch_items]

                        col_order = []
                        header = {"topic": topic, "indexes": global_ids_batch, "columns": {}, "order": []}
                        payload_frames = []

                        for col_name, raw_data in data_dict.items():
                            # Determine encoding based on data type
                            use_pickle_encoding = False
                            if isinstance(raw_data, torch.Tensor):
                                # Tensor → raw encoding
                                batch_data = torch_to_numpy(raw_data[slice_obj])
                                use_pickle_encoding = False
                            elif isinstance(raw_data, list):
                                # Check if list contains tensors or other types
                                if len(raw_data) > 0:
                                    sample = (
                                        raw_data[start_local_idx] if start_local_idx < len(raw_data) else raw_data[0]
                                    )
                                    if isinstance(sample, (torch.Tensor, np.ndarray)):
                                        # List[Tensor] → raw encoding
                                        batch_data = [
                                            torch_to_numpy(raw_data[idx])
                                            for idx in range(start_local_idx, end_local_idx)
                                        ]
                                        use_pickle_encoding = False
                                    else:
                                        # List[Any] → pickle encoding
                                        use_pickle_encoding = True
                                else:
                                    # Empty list → pickle encoding
                                    use_pickle_encoding = True
                            else:
                                # Other types → pickle encoding
                                use_pickle_encoding = True

                            # Serialize based on encoding
                            if use_pickle_encoding:
                                # Pickle encoding doesn't support shape preservation
                                shapes = None
                                # Prepare data as List[Any] for pickle
                                batch_objects = []
                                for idx in range(start_local_idx, end_local_idx):
                                    if isinstance(raw_data, torch.Tensor):
                                        batch_objects.append(
                                            raw_data[idx].item() if raw_data[idx].numel() == 1 else raw_data[idx]
                                        )
                                    elif isinstance(raw_data, list):
                                        batch_objects.append(raw_data[idx])
                                    else:
                                        batch_objects.append(raw_data)
                                final_buffer, lengths, dtype_str = serialize_batch_pickle(batch_objects)
                            else:
                                # Original raw encoding path
                                # 准备 Unpad 长度信息
                                batch_lens = None
                                pad_side = None

                                if col_name in unpad_map:
                                    len_col_name, pad_side = unpad_map[col_name]
                                    full_len_col = data_dict.get(len_col_name)

                                    if full_len_col is not None:
                                        if isinstance(full_len_col, torch.Tensor):
                                            batch_lens = torch_to_numpy(full_len_col[slice_obj])
                                        else:
                                            batch_lens = np.array(full_len_col[slice_obj])

                                        if save_dtype:
                                            batch_lens = batch_lens.astype(get_numpy_dtype(save_dtype))

                                final_buffer, lengths, dtype_str, shapes = serialize_batch(
                                    batch_data, batch_lens, pad_side, save_dtype
                                )

                            # 添加 ref_multiplier
                            n_samples_per_prompt = ray.get(self.manager.get_n_samples_per_prompt.remote(topic))
                            ref_multiplier = n_samples_per_prompt if is_shared else 1

                            col_order.append(col_name)
                            header["columns"][col_name] = {
                                "dtype": dtype_str,
                                "lengths": lengths,
                                "shapes": shapes,
                                "ref_multiplier": ref_multiplier,
                                "encoding": "pickle" if use_pickle_encoding else "raw",
                            }
                            payload_frames.append(final_buffer)

                        # 发送
                        header["order"] = col_order
                        header_bytes = pickle.dumps(header)
                        await sock.send_multipart([b"PUT", header_bytes] + payload_frames, copy=False)
                        total_sent_bytes += len(header_bytes) + sum(
                            b.nbytes if hasattr(b, "nbytes") else len(b) for b in payload_frames
                        )
                        recv_futures.append(sock.recv())

                    if recv_futures:
                        responses = await asyncio.gather(*recv_futures)
                        for idx, resp in enumerate(responses):
                            if not resp == b"ACK":
                                error_msg = f"Batch {idx} on {endpoint} failed. Expected b'ACK', got: {resp}"
                                self.logger.error(error_msg)
                                raise RuntimeError(error_msg)

                except Exception as e:
                    self.logger.error(f"Put error on {endpoint}: {e}")
                    self._invalidate_socket(endpoint)
                    raise e

        return {
            "endpoint": endpoint,
            "count": len(items),
            "bytes": total_sent_bytes,
            "latency": (time.perf_counter() - node_start) * 1000,
        }

    def get_experience(
        self,
        consumer: str,
        experience_columns: list[str],
        experience_count: int = None,
        indexes: list[int] = None,
        get_n_samples: bool = False,
        allowed_staleness: int = None,
        latest_version: int = None,
        topic: str = DEFAULT_TOPIC,
        sampler_func: Optional[BaseSampler] = None,
        copy: bool = False,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor] | list[Any]], list[int]]:
        """
        获取 experience 接口，传入 consumer、experience_columns、experience_count 获取指定数量的 experience
        封装 get_experience_async，异步获取，同步返回
        Args:
            consumer (str):
            experience_columns (List[str]):
            experience_count (int) = None:
            indexes (List[int]): = None,
            get_n_samples (bool): = False,
            topic (str): 不传入则默认为 DEFAULT_TOPIC,
            copy (bool): 内存策略控制。
                - False(默认): 执行 Zero-copy。Tensor 与 Frame 共享底层内存。
                  注意：生成的 Tensor 可能为只读；修改 Tensor 会直接影响底层 Buffer
                - True: 执行深拷贝。Tensor 拥有独立内存，可安全写入，不受 Frame 生命周期影响。
        Returns:
            Dict[str, Union[torch.Tensor, List[torch.Tensor]]]: key 为 columns,
            value 为该 columns 的 tensor 构成的 list。
            List[int]: 获取到的 index 列表
        """
        return asyncio.run(
            self.get_experience_async(
                consumer,
                experience_columns,
                experience_count,
                indexes,
                get_n_samples,
                allowed_staleness,
                latest_version,
                topic,
                sampler_func,
                copy,
            )
        )

    async def get_experience_async(
        self,
        consumer: str,
        experience_columns: list[str],
        experience_count: int = None,
        indexes: list[int] = None,
        get_n_samples: bool = False,
        allowed_staleness: int = None,
        latest_version: int = None,
        topic: str = DEFAULT_TOPIC,
        sampler_func: Optional[BaseSampler] = None,
        copy: bool = False,
    ) -> tuple[dict[str, torch.Tensor | list[torch.Tensor] | list[Any]], list[int]]:
        start_time = time.perf_counter()

        # 1. Validation & Defaulting
        if topic is None:
            topic = DEFAULT_TOPIC
        if experience_count is None and indexes is None:
            raise ValueError("Either experience_count or indexes must be provided")

        # 2. RPC: Ask Manager for Shard Allocation
        # Manager 返回的 shard_map 结构: {endpoint_url: [global_index_1, global_index_2, ...]}
        try:
            if indexes is not None:
                shard_map = await self.manager.allocate_shard_for_indexes.remote(
                    topic, consumer, experience_columns, indexes, allowed_staleness, latest_version
                )
            else:
                shard_map = await self.manager.allocate_shard_and_indexes.remote(
                    topic,
                    consumer,
                    experience_columns,
                    experience_count,
                    get_n_samples,
                    allowed_staleness,
                    latest_version,
                    sampler_func,
                )
        except Exception:
            # self.logger.error(f"Failed to allocate shards: {e}")
            return None, None

        if not shard_map:
            # self.logger.error("Get experience returned no data from manager.")
            return None, None

        # 创建异步抓取任务
        tasks = []
        for endpoint, target_indexes in shard_map.items():
            tasks.append(self._fetch_one_shard(endpoint, topic, experience_columns, target_indexes, copy))

        # 等待所有分片返回
        # results 类型: List[Optional[Dict]]
        shard_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 汇总字节数
        total_bytes = 0
        for result in shard_results:
            if result is not None and not isinstance(result, Exception):
                total_bytes += result.get("bytes", 0)
        total_mb = total_bytes / (1024 * 1024)

        # 聚合
        id_to_data_map = {}
        total_items = 0
        successful_shards = 0

        for res in shard_results:
            if isinstance(res, Exception) or res is None:
                continue
            successful_shards += 1
            returned_ids = res["indexes"]
            shard_data = res["data"]

            count = len(returned_ids)
            total_items += count
            # 遍历存入字典, 不产生搬运
            for i in range(count):
                global_idx = returned_ids[i]
                sample_pack = {col: shard_data[col][i] for col in experience_columns if col in shard_data}
                id_to_data_map[global_idx] = sample_pack

        if total_items == 0:
            return None, None

        # 根据输入的 indexes 顺序生成最终列表
        final_indexes = []
        final_columns = {col: [] for col in experience_columns}

        # 未提供 index 时, 根据返回结果排序构建最终列表
        if indexes is None:
            target_iterator = sorted(id_to_data_map.keys())
        else:
            target_iterator = indexes

        for target_id in target_iterator:
            if target_id in id_to_data_map:
                final_indexes.append(target_id)
                sample = id_to_data_map[target_id]
                for col in experience_columns:
                    final_columns[col].append(sample[col])
            else:
                continue

        # 6. Logging
        end_time = time.perf_counter()
        total_latency = (end_time - start_time) * 1000

        self.accumulate_timing("get", float(total_latency / 1000.0))

        self.logger.info(
            f"[TransferQueue Get] | Total Latency: {total_latency:.2f} ms | "
            f"Total Volume: {total_mb:.2f} MB | Shards: {successful_shards} | "
            f"Routed_experts exists : {'routed_experts' in experience_columns} | "
            f"Data_size: {len(final_indexes)} | Total_items: {total_items} | "
            f"{experience_columns=} | Indexes: {final_indexes} | "
        )

        return final_columns, final_indexes

    async def _fetch_one_shard(
        self,
        endpoint: str,
        topic: str,
        columns: list[str],
        indexes: list[int],
        copy=False,
    ) -> dict[str, Any]:
        # 全局并发控制
        async with self.get_semaphore:
            # 获取该 Endpoint 的专属锁，防止并发写入导致 ZMQ 状态机混乱或串包
            lock = self._get_endpoint_lock(endpoint)
            async with lock:
                try:
                    sock = self._get_async_socket(endpoint)
                    req_header = {"topic": topic, "experience_columns": columns, "indexes": indexes}

                    await sock.send_multipart([b"GET", pickle.dumps(req_header)])

                    reply = await sock.recv_multipart(copy=False)

                    # 计算接收的总字节数（网络传输实际字节数）
                    received_bytes = sum(
                        frame.nbytes if hasattr(frame, "nbytes") else len(frame.bytes) for frame in reply
                    )

                    if not reply:
                        raise RuntimeError("Empty reply received")

                    meta_frame = reply[0]
                    if meta_frame.bytes.startswith(b"ERROR:"):
                        raise RuntimeError(f"Remote Error: {meta_frame.bytes.decode()}")

                    meta = pickle.loads(meta_frame.bytes)
                    returned_indexes = meta["indexes"]
                    col_meta_map = meta["columns"]

                    # 反序列化
                    result_data = {}
                    if len(reply) - 1 != len(columns):
                        raise RuntimeError(f"Column mismatch from {endpoint}. Exp {len(columns)}, Got {len(reply) - 1}")

                    for i, col_name in enumerate(meta["order"]):
                        if i + 1 >= len(reply):
                            break

                        frame = reply[i + 1]
                        col_info = col_meta_map.get(col_name)

                        if col_info:
                            encoding = col_info.get("encoding", "raw")  # Default to raw for backward compatibility

                            if encoding == "pickle":
                                objects = deserialize_column_pickle_from_frame(
                                    frame,
                                    col_info["lengths"],  # byte lengths for pickle
                                )
                                result_data[col_name] = objects
                            else:  # 'raw' encoding
                                tensors = deserialize_column_from_frame(
                                    frame, col_info["dtype"], col_info["lengths"], copy, col_info["shapes"]
                                )
                                result_data[col_name] = tensors
                        else:
                            result_data[col_name] = []

                    return {"data": result_data, "indexes": returned_indexes, "bytes": received_bytes}

                except Exception as e:
                    self.logger.error(f"Error fetching shard {endpoint}: {e}")
                    self._invalidate_socket(endpoint)
                    return None

    # def _get_endpoint_lock(self, endpoint: str) -> asyncio.Lock:
    #     """
    #     获取针对特定 Endpoint 的协程锁，确保复用 Socket 时的 Send-Recv 原子性。
    #     """
    #     if not hasattr(self, "_endpoint_locks"):
    #         self._endpoint_locks = {}

    #     if endpoint not in self._endpoint_locks:
    #         self._endpoint_locks[endpoint] = asyncio.Lock()
    #     return self._endpoint_locks[endpoint]

    def _get_endpoint_lock(self, endpoint: str) -> asyncio.Lock:
        if not hasattr(self, "_endpoint_locks"):
            self._endpoint_locks = {}

        existing_lock = self._endpoint_locks.get(endpoint)

        if existing_lock is not None:
            try:
                # 尝试获取当前循环
                current_loop = asyncio.get_running_loop()
                # 重点：不要直接调用 lock._get_loop()，
                # 而是直接尝试 acquire。如果 loop 不对，这里会报错。
                # 或者我们直接访问内部属性（虽然不推荐但安全）：
                if existing_lock._loop is current_loop:
                    return existing_lock
            except (RuntimeError, AttributeError):
                # 如果报错了，说明 loop 变了，我们需要创建一个新的
                pass

        # 走到这里说明：要么没锁，要么锁不可用
        new_lock = asyncio.Lock()
        self._endpoint_locks[endpoint] = new_lock
        return new_lock

    def _get_socket(self, endpoint: str) -> zmq.Socket:
        """Get or create a Dealer socket connected to the given endpoint."""
        # Use thread-local sockets because ZeroMQ sockets are not thread-safe.
        if not hasattr(self._local, "sockets"):
            self._local.sockets = {}
        tl_sockets: dict[str, zmq.Socket] = self._local.sockets
        if endpoint in tl_sockets:
            return tl_sockets[endpoint]
        sock = self.zmq_context.socket(zmq.DEALER)
        sock.connect(endpoint)
        tl_sockets[endpoint] = sock
        return sock

    def delete_experience(
        self,
        indexes: list[int] = None,
        versions: list[int] = None,
        latest_version: int = None,
        allowed_staleness: int = None,
        delete_all: bool = False,
        topic: str = DEFAULT_TOPIC,
    ):
        return ray.get(
            self.manager.delete_experience.remote(
                indexes, versions, latest_version, allowed_staleness, delete_all, topic
            )
        )

    def create_timing_item(self, name: str) -> None:
        """Create (ensure) a timing item on the manager."""
        ray.get(self.manager.create_timing_item.remote(name))

    def accumulate_timing(self, name: str, seconds: float) -> None:
        """Accumulate elapsed seconds into a timing item on the manager."""
        ray.get(self.manager.accumulate_timing.remote(name, float(seconds)))

    def get_timing(self, name: str) -> float:
        """Return a single timing item (seconds)."""
        return ray.get(self.manager.get_timing.remote(name))

    def get_timings(self) -> dict[str, float]:
        """Return all timing items (a dict of name -> seconds)."""
        return ray.get(self.manager.get_timings.remote())

    def reset_timings(self) -> None:
        """Reset all timing items to zero."""
        ray.get(self.manager.reset_timings.remote())

    def get_data_ready_set(
        self, topic: str, experience_columns: list[str], indexes: list[int] = None, get_n_samples: bool = False
    ):
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.get_data_ready_set.remote(topic, experience_columns, indexes, get_n_samples))

    def get_data_consumed_set(self, topic: str, consumer: str, indexes: list[int] = None, get_n_samples: bool = False):
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.get_data_consumed_set.remote(topic, consumer, indexes, get_n_samples))

    def get_data_usable_set(
        self,
        topic: str,
        consumer: str,
        experience_columns: list[str],
        allowed_staleness: int = None,
        latest_version: int = None,
        indexes: list[int] = None,
        get_n_samples: bool = False,
    ):
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(
            self.manager.get_data_usable_set.remote(
                topic, consumer, experience_columns, allowed_staleness, latest_version, indexes, get_n_samples
            )
        )

    def get_allocation_for_new_groups(self, topic: str, num_new_groups: int) -> list[list[int]]:
        """
        为新组分配索引。

        Args:
            topic: Topic名称
            num_new_groups: 需要的新组数量

        Returns:
            List[List[int]]: 分配的组ID列表，按分片分组
        """
        if topic is None:
            topic = DEFAULT_TOPIC
        return ray.get(self.manager.get_allocation_for_new_groups.remote(topic, num_new_groups))


def get_transferqueue_client(name: str = "TransferQueueManager") -> TransferQueueClient:
    """
    Get a new Client instance connected to the named TransferQueueManager actor.
    This uses Ray's global name registry to locate the manager.
    """
    if name is None:
        name = "TransferQueueManager"
    mgr = ray.get_actor(name)
    return TransferQueueClient(mgr)
