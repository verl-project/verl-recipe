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
import threading
from typing import Any

import numpy as np
import zmq
from recipe.async_flow.utils.transfer_queue.tq_utils import setup_logger


class MemorySegment:
    """
    物理存储单元：持有 zmq.Frame 并管理引用计数
    """

    __slots__ = ("_frame", "_buffer", "ref_count")

    def __init__(self, frame: zmq.Frame, initial_ref: int):
        self._frame = frame
        # 创建 memoryview，实现零拷贝访问 C++ 底层内存
        self._buffer = memoryview(frame)
        self.ref_count = initial_ref

    @property
    def buffer(self) -> memoryview:
        return self._buffer

    def release(self, count: int = 1) -> bool:
        """
        减少引用计数。
        返回 True 表示资源已被彻底释放（引用计数归零）。
        """
        self.ref_count -= count
        if self.ref_count <= 0:
            # 显式解除引用，允许 GC 回收底层的 zmq.Frame
            self._frame = None
            self._buffer = None
            return True
        return False


class ExperienceTable:
    """
    高性能经验回放表：支持列式存储、批量写入和零拷贝拼接读取。
    """

    def __init__(self, n_samples_per_prompt: int, experience_columns: list[str]):
        if n_samples_per_prompt <= 0:
            raise ValueError("n_samples_per_prompt must be a positive integer")
        self.n_samples_per_prompt = n_samples_per_prompt
        if not experience_columns:
            raise ValueError("experience_columns must be provided")
        self.experience_columns = experience_columns
        self.logger = setup_logger("ExperienceTable")
        self.logger.debug(f"ExperienceTable initialized successfully, n_samples_per_prompt={self.n_samples_per_prompt}")

        # Column Name -> Global ID -> (MemorySegment, byte_offset, byte_length)
        self.indices: dict[str, dict[int, tuple[MemorySegment, int, int]]] = {}
        self.col_metas: dict[str, tuple[int, str, str]] = {}  # (item_size, dtype_str, encoding)
        # Column Name -> Global ID -> shape tuple
        self.col_shapes: dict[str, dict[int, tuple[int, ...]]] = {}
        self._lock = threading.Lock()
        self.owned_groups = set()

    @staticmethod
    def _get_item_size(dtype_str: str) -> int:
        return np.dtype(dtype_str).itemsize

    def put_batch(
        self,
        global_ids: list[int],
        col_order: list[str],
        col_inputs_meta: dict[str, Any],
        payload_frames: list[zmq.Frame],
    ):
        batch_size = len(global_ids)
        if batch_size == 0:
            return

        cols_info = col_inputs_meta["columns"]
        validated_metas = []

        if len(payload_frames) < len(col_order):
            missing_col = col_order[len(payload_frames)]
            raise ValueError(f"Missing frame for column {missing_col}")

        # 校验收到结果是否正确
        for i, col_name in enumerate(col_order):
            frame = payload_frames[i]
            col_info = cols_info[col_name]
            ref_multiplier = col_info.get("ref_multiplier", 1)  # 从header读取
            initial_ref_count = batch_size * ref_multiplier
            elem_lengths = col_info["lengths"]
            dtype_str = col_info["dtype"]

            encoding = col_info.get("encoding", "raw")  # Default for backward compatibility
            shapes = col_info.get("shapes")  # Get shapes info

            if col_name in self.col_metas:
                existing_item_size, existing_dtype, existing_encoding = self.col_metas[col_name]
                # Validate encoding consistency
                if existing_encoding != encoding:
                    raise ValueError(
                        f"Encoding mismatch for column '{col_name}': existing={existing_encoding}, new={encoding}"
                    )
                if encoding == "raw" and existing_dtype != dtype_str:
                    raise ValueError(
                        f"Dtype mismatch for column '{col_name}': existing={existing_dtype}, new={dtype_str}"
                    )
                item_size = existing_item_size
            else:
                if encoding == "raw":
                    item_size = np.dtype(dtype_str).itemsize
                else:
                    item_size = 1  # Not meaningful for pickle
                self.col_metas[col_name] = (item_size, dtype_str, encoding)

            # Calculate expected bytes based on encoding
            if encoding == "pickle":
                expected_bytes = sum(elem_lengths)  # lengths are byte lengths
            else:
                total_elements = sum(elem_lengths)  # lengths are element counts
                expected_bytes = total_elements * item_size

            if expected_bytes != len(frame.bytes):
                raise ValueError(
                    f"Size Mismatch '{col_name}': Meta {expected_bytes} bytes != Frame {len(frame.bytes)} bytes"
                )

            if len(elem_lengths) != batch_size:
                raise ValueError(f"Length mismatch for col '{col_name}'")

            validated_metas.append(
                (
                    col_name,
                    frame,
                    item_size,
                    elem_lengths,
                    dtype_str,
                    ref_multiplier,
                    initial_ref_count,
                    encoding,
                    shapes,
                )
            )

        with self._lock:
            for (
                col_name,
                frame,
                item_size,
                elem_lengths,
                dtype_str,
                ref_multiplier,
                initial_ref_count,
                encoding,
                shapes,
            ) in validated_metas:
                if col_name not in self.col_metas:
                    self.col_metas[col_name] = (item_size, dtype_str, encoding)

                if col_name not in self.indices:
                    self.indices[col_name] = {}

                col_index = self.indices[col_name]

                # Initialize shapes dict for this column if shapes provided
                if shapes is not None and col_name not in self.col_shapes:
                    self.col_shapes[col_name] = {}

                # 初始引用计数 = 样本数 * ref_multiplier
                segment = MemorySegment(frame, initial_ref=initial_ref_count)
                current_byte_offset = 0

                # 扩展 global_ids 用于建立索引映射
                if ref_multiplier > 1:
                    # 共享列：扩展 global_ids 覆盖整个group
                    n_samples = self.n_samples_per_prompt
                    data_idx = 0

                    for global_idx in global_ids:
                        group_id = global_idx // n_samples
                        if data_idx < len(elem_lengths):
                            n_elems_or_bytes = elem_lengths[data_idx]
                            # Calculate byte_len based on encoding
                            if encoding == "pickle":
                                byte_len = n_elems_or_bytes  # Already byte length
                            else:
                                byte_len = n_elems_or_bytes * item_size
                            entry = (segment, current_byte_offset, byte_len)

                            # Save shape for this data item
                            if shapes is not None and data_idx < len(shapes):
                                shape = shapes[data_idx]
                                # All indices in the group share the same shape
                                for group_offset in range(n_samples):
                                    actual_global_idx = group_id * n_samples + group_offset
                                    self.col_shapes[col_name][actual_global_idx] = shape

                            # 为整个group的每个索引建立相同指向
                            for group_offset in range(n_samples):
                                actual_global_idx = group_id * n_samples + group_offset
                                if actual_global_idx in col_index:
                                    old_segment = col_index[actual_global_idx][0]
                                    old_segment.release()
                                col_index[actual_global_idx] = entry
                            current_byte_offset += byte_len
                            data_idx += 1
                else:
                    # 非共享列：正常一一映射
                    for idx_in_batch, (global_idx, n_elems_or_bytes) in enumerate(
                        zip(global_ids, elem_lengths, strict=False)
                    ):
                        # Calculate byte_len based on encoding
                        if encoding == "pickle":
                            byte_len = n_elems_or_bytes  # Already byte length
                        else:
                            byte_len = n_elems_or_bytes * item_size
                        entry = (segment, current_byte_offset, byte_len)

                        # Save shape for this data item
                        if shapes is not None and idx_in_batch < len(shapes):
                            self.col_shapes[col_name][global_idx] = shapes[idx_in_batch]

                        if global_idx in col_index:
                            old_segment = col_index[global_idx][0]
                            old_segment.release()
                        col_index[global_idx] = entry
                        current_byte_offset += byte_len

    def get_batch(self, target_global_idxs: list[int], target_cols: list[str]) -> tuple[dict[str, Any], list[bytes]]:
        result_meta = {
            "indexes": target_global_idxs,
            "columns": {col: {"lengths": []} for col in target_cols},
            "order": target_cols,
        }

        result_frames = []

        with self._lock:
            for col_name in target_cols:
                col_static_meta = self.col_metas.get(col_name)
                if not col_static_meta:
                    raise KeyError(f"Column {col_name} not found.")

                item_size, dtype, encoding = col_static_meta
                result_meta["columns"][col_name]["dtype"] = dtype
                result_meta["columns"][col_name]["encoding"] = encoding

                col_index = self.indices.get(col_name)
                if col_index is None:
                    raise KeyError(f"Column index {col_name} is empty or missing.")

                # Get shapes for this column if available
                col_shapes = self.col_shapes.get(col_name)
                shapes_list = []

                total_bytes = 0
                batch_entries = []
                meta_lengths_append = result_meta["columns"][col_name]["lengths"].append

                for global_idx in target_global_idxs:
                    entry = col_index.get(global_idx)
                    if entry is None:
                        raise KeyError(f"Global ID {global_idx} missing in column {col_name}.")
                    batch_entries.append(entry)

                    byte_len = entry[2]
                    total_bytes += byte_len

                    # Store appropriate length based on encoding
                    if encoding == "pickle":
                        meta_lengths_append(byte_len)  # Already byte length
                    else:
                        meta_lengths_append(byte_len // item_size)  # Convert to element count

                    # Collect shape for this index
                    if col_shapes is not None:
                        shapes_list.append(col_shapes.get(global_idx))
                    else:
                        shapes_list.append(None)

                # Always add shapes to result meta (shapes is a required field)
                result_meta["columns"][col_name]["shapes"] = shapes_list

                dest_buffer = bytearray(total_bytes)
                cursor = 0

                for seg, off, byte_len in batch_entries:
                    dest_buffer[cursor : cursor + byte_len] = seg.buffer[off : off + byte_len]
                    cursor += byte_len

                result_frames.append(dest_buffer)

        return result_meta, result_frames

    def prune(self, global_idxs_to_remove: list[int]):
        """
        清理逻辑：遍历所有列，尝试移除指定的 global_idx。
        """
        with self._lock:
            for col_name, col_index in self.indices.items():
                for global_idx in global_idxs_to_remove:
                    if global_idx in col_index:
                        entry = col_index.pop(global_idx)
                        segment = entry[0]
                        segment.release()
            # Also clean up shapes
            for col_name, col_shapes in self.col_shapes.items():
                for global_idx in global_idxs_to_remove:
                    col_shapes.pop(global_idx, None)
            for global_idx in global_idxs_to_remove:
                group_id = global_idx // self.n_samples_per_prompt
                self.owned_groups.discard(group_id)

    def clear(self):
        """
        1. 清理整个 ExperienceTable
        2. 显式释放所有 MemorySegment 的引用计数 (从而释放 zmq.frame)
        3. 重置元数据
        """
        with self._lock:
            for col_index in self.indices.values():
                for entry in col_index.values():
                    # entry: (segment, offset, length)
                    segment = entry[0]
                    segment.release()

            self.indices.clear()
            self.col_metas.clear()
            self.col_shapes.clear()
            self.owned_groups.clear()
