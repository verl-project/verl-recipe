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
import logging
import pickle
import sys
import warnings
from typing import Any, Optional

import ml_dtypes
import numpy as np
import torch
import zmq

_NP_DTYPE_MAP = {
    # --- 标准类型 ---
    "float32": np.dtype("float32"),
    "float64": np.dtype("float64"),
    "float16": np.dtype("float16"),
    "int32": np.dtype("int32"),
    "int64": np.dtype("int64"),
    "int16": np.dtype("int16"),
    "int8": np.dtype("int8"),
    "uint8": np.dtype("uint8"),
    "bool": np.dtype("bool"),
    # --- 别名 ---
    "single": np.dtype("float32"),
    "double": np.dtype("float64"),
    "long": np.dtype("int64"),
    "int": np.dtype("int32"),
    "half": np.dtype("float16"),
    # --- ml_dtypes 类型 (BFloat16) ---
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "bf16": np.dtype(ml_dtypes.bfloat16),  # 常用别名
    # --- ml_dtypes 类型 (Float8) ---
    # 默认 float8 指向 e4m3fn (推理常用)
    "float8": np.dtype(ml_dtypes.float8_e4m3fn),
    "fp8": np.dtype(ml_dtypes.float8_e4m3fn),
    "float8_e4m3fn": np.dtype(ml_dtypes.float8_e4m3fn),
    "fp8_e4m3": np.dtype(ml_dtypes.float8_e4m3fn),
    "float8_e5m2": np.dtype(ml_dtypes.float8_e5m2),
    "fp8_e5m2": np.dtype(ml_dtypes.float8_e5m2),
}


def get_numpy_dtype(dtype_input) -> np.dtype:
    """
    强制返回 np.dtype 实例。
    """
    # 1. 如果已经是 dtype 对象，直接返回
    if isinstance(dtype_input, np.dtype):
        return dtype_input

    # 2. 查表或转换
    if isinstance(dtype_input, str):
        if t := _NP_DTYPE_MAP.get(dtype_input):
            return t

    # 3. 兜底尝试 (比如传入 float 类型，或表中没有的字符串)
    try:
        return np.dtype(dtype_input)
    except Exception as err:
        raise ValueError(f"Unsupported dtype: {dtype_input}") from err


def torch_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    智能转换 PyTorch Tensor 为 NumPy，处理 bf16/fp8/量化及内存不连续的特殊情况。
    """
    # 1. 量化类型 (qint8等): 必须提取底层整数表示，否则无法转换
    if tensor.is_quantized:
        tensor = tensor.int_repr()

    # 2. 基础清理: 去除梯度
    tensor = tensor.detach()

    # 3. 内存连续性检查: view() 操作要求内存连续。如果由切片(slicing)生成，必须先整理内存。
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()

    # 4. BFloat16 (2字节): 伪装成 int16 传输，零拷贝还原
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.int16).cpu().numpy().view(ml_dtypes.bfloat16)

    # 5. Float8 (1字节): 伪装成 int8 传输，零拷贝还原
    # 使用 hasattr 兼容旧版 PyTorch
    elif hasattr(torch, "float8_e4m3fn") and tensor.dtype == torch.float8_e4m3fn:
        return tensor.view(torch.int8).cpu().numpy().view(ml_dtypes.float8_e4m3fn)

    elif hasattr(torch, "float8_e5m2") and tensor.dtype == torch.float8_e5m2:
        return tensor.view(torch.int8).cpu().numpy().view(ml_dtypes.float8_e5m2)

    # 6. 常规兜底 (float32, int64, complex32等)
    # 注: PyTorch complex32 会在此自动转换为 NumPy complex64
    return tensor.cpu().numpy()


def serialize_batch(
    batch_data: np.ndarray | list[np.ndarray],
    batch_lens: Optional[np.ndarray] = None,
    pad_mode: Optional[str] = None,
    save_dtype: str = None,
) -> tuple[np.ndarray, list[int], str, list[tuple[int, ...]]]:
    # 取第一个元素判断类型
    sample = batch_data[0]
    source_dtype = sample.dtype

    if save_dtype is not None:
        target_dtype = get_numpy_dtype(save_dtype)
    else:
        target_dtype = source_dtype

    # 判断是否为 2D 矩阵
    is_2d_matrix = isinstance(batch_data, np.ndarray) and batch_data.ndim == 2

    # 输入是 Tensor 转换来的整块 Numpy，直接操作 Numpy
    if is_2d_matrix:
        rows, cols = batch_data.shape
        # 生成 shapes：每个 row 的 shape 为 (cols,)
        shapes = [(cols,) for _ in range(rows)]

        # 不需要 Unpad
        if batch_lens is None:
            # 尝试 Zero-copy View
            if source_dtype == target_dtype and batch_data.flags["C_CONTIGUOUS"]:
                final_buffer = batch_data.ravel()
            else:
                final_buffer = batch_data.astype(target_dtype).ravel()

            # 长度固定为列长
            length_list = [int(cols)] * rows
        # 需要 Unpad
        else:
            batch_lens = np.asarray(batch_lens).reshape(-1)
            # 校验 pad 输入
            if len(batch_lens) != rows:
                raise ValueError(f"Batch lens length ({len(batch_lens)}) must match number of rows ({rows})")
            # 利用 Broadcasting 生成布尔掩码
            col_indices = np.arange(cols).reshape(1, -1)
            if pad_mode == "right_pad":
                mask = col_indices < batch_lens[:, None]
            elif pad_mode == "left_pad":
                mask = col_indices >= (cols - batch_lens[:, None])
            else:
                raise ValueError("pad_mode must be 'right_pad' or 'left_pad'")

            #  Masking 直接提取有效数据并 Flatten
            valid_data = batch_data[mask]

            if source_dtype != target_dtype:
                final_buffer = valid_data.astype(target_dtype)
            else:
                final_buffer = valid_data
            # Meta 计算
            length_list = [int(ln) for ln in batch_lens]
            # 对于 unpad 的情况，每个 row 的 shape 为 (length,)
            shapes = [(int(ln),) for ln in batch_lens]
    # 输入为 List[Tensor]
    else:
        # 收集每个 tensor 的原始形状
        shapes = [tuple(arr.shape) for arr in batch_data]

        if batch_lens is None:
            # 等长数组无 unpad 或变长数组时，直接获取需要的长度
            effective_lens = [x.size for x in batch_data]
        else:
            # 需要 unpad
            effective_lens = batch_lens

        # 一次性分配内存
        total_elements = sum(effective_lens)
        final_buffer = np.empty(total_elements, dtype=target_dtype)

        length_list = []
        curr_offset = 0

        for i, arr in enumerate(batch_data):
            length = int(effective_lens[i])
            length_list.append(length)
            flat_src = arr.ravel()
            src_len = flat_src.size

            # 仅当源数据长度与目标长度不一致时处理
            if batch_lens is not None and src_len != length:
                if pad_mode == "right_pad":
                    flat_src = flat_src[:length]
                elif pad_mode == "left_pad":
                    flat_src = flat_src[-length:]
                else:
                    raise ValueError(f"Length mismatch ({src_len} vs {length}) but valid pad_mode not provided.")
            # 写入 Buffer
            final_buffer[curr_offset : curr_offset + length] = flat_src
            curr_offset += length

    # 特殊类型处理：转换为标准整数类型发送（兼容 buffer protocol）
    # 注意：target_dtype.name 保持原始类型，用于元数据；只有 final_buffer 的内存 view 改变
    if target_dtype.name in ("bfloat16", "bf16"):
        final_buffer = final_buffer.view(np.int16)
    elif target_dtype.name in ("float8", "fp8", "float8_e4m3fn", "fp8_e4m3", "float8_e5m2", "fp8_e5m2"):
        final_buffer = final_buffer.view(np.int8)

    return final_buffer, length_list, target_dtype.name, shapes


def deserialize_column_from_frame(
    frame: zmq.Frame, dtype_str: str, lengths: list[int], copy: bool = False, shapes: list[tuple[int, ...]] = None
) -> list[torch.Tensor]:
    if not lengths:
        return []

    # 1. 特殊类型处理：读为标准整数类型，避免 buffer protocol 问题，转 torch 后再转回原始类型
    if dtype_str in ("bfloat16", "bf16"):
        # 先读为 int16（支持 buffer）；零拷贝操作：int16 和 bf16 都是 2 字节
        src_view = np.frombuffer(frame, dtype=np.int16)
    elif dtype_str in ("float8", "fp8", "float8_e4m3fn", "fp8_e4m3", "float8_e5m2", "fp8_e5m2"):
        # 先读为 int8（支持 buffer）；零拷贝操作：int8 和 bf8 都是 1 字节
        src_view = np.frombuffer(frame, dtype=np.int8)
    else:
        # 其他类型直接读取
        src_view = np.frombuffer(frame, dtype=np.dtype(dtype_str))

    if src_view.size != sum(lengths):
        raise ValueError(f"Data size mismatch. Meta sums to {sum(lengths)}, Frame has {src_view.size}")

    # 2. 根据copy参数选择深拷贝或零拷贝模式处理数据
    if copy:
        # 触发一次 copy, list 内 Tensor 变为非只读
        slab_np = src_view.copy()
        # 将 ZMQ Buffer 拷入 PyTorch 连续内存
        slab = torch.from_numpy(slab_np)
    else:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, message=".*not writable.*")
            slab = torch.from_numpy(src_view)

    # 3. 特殊类型：用 PyTorch 原生方式转换回原始类型
    if dtype_str in ("bfloat16", "bf16"):
        slab = slab.to(torch.bfloat16)
    elif dtype_str in ("float8", "fp8", "float8_e4m3fn", "fp8_e4m3"):
        slab = slab.to(getattr(torch, "float8_e4m3fn", torch.float16))
    elif dtype_str in ("float8_e5m2", "fp8_e5m2"):
        slab = slab.to(getattr(torch, "float8_e5m2", torch.float16))

    # 切分 flat tensors
    flat_tensors = list(torch.split(slab, lengths))

    # 4. 根据形状恢复原始维度
    if shapes is not None:
        return [flat.view(shape) for flat, shape in zip(flat_tensors, shapes, strict=False)]
    else:
        return flat_tensors


def serialize_batch_pickle(batch_data: list[Any], pickle_protocol: int = None) -> tuple[bytes, list[int], str]:
    """
    Serialize arbitrary Python objects using pickle.

    Args:
        batch_data: List of Python objects to serialize (all same type per column)
        pickle_protocol: Pickle protocol version (default: pickle.HIGHEST_PROTOCOL)

    Returns:
        Tuple[bytes, List[int], str]: (concatenated pickle blobs, byte_lengths, "pickle")
    """
    if not batch_data:
        return b"", [], "pickle"

    if pickle_protocol is None:
        pickle_protocol = pickle.HIGHEST_PROTOCOL

    pickle_blobs = []
    byte_lengths = []

    for obj in batch_data:
        blob = pickle.dumps(obj, protocol=pickle_protocol)
        pickle_blobs.append(blob)
        byte_lengths.append(len(blob))

    final_buffer = b"".join(pickle_blobs)

    return final_buffer, byte_lengths, "pickle"


def deserialize_column_pickle_from_frame(frame: zmq.Frame, byte_lengths: list[int]) -> list[Any]:
    """
    Deserialize pickled Python objects from a frame.

    Args:
        frame: zmq.Frame containing concatenated pickle blobs
        byte_lengths: Byte length of each pickle blob

    Returns:
        List[Any]: Deserialized Python objects

    Raises:
        ValueError: If frame size doesn't match expected byte_lengths sum
        pickle.UnpicklingError: If pickle deserialization fails
    """

    if not byte_lengths:
        return []

    total_bytes = sum(byte_lengths)
    if total_bytes != len(frame.bytes):
        raise ValueError(f"Pickle data size mismatch. Meta sums to {total_bytes}, Frame has {len(frame.bytes)}")

    results = []
    cursor = 0

    for byte_len in byte_lengths:
        blob = frame.bytes[cursor : cursor + byte_len]
        obj = pickle.loads(blob)
        results.append(obj)
        cursor += byte_len

    return results


def get_no_pad_length(prompts: torch.Tensor, pad_id: Optional[int] = None) -> list[torch.Tensor]:
    """
    Compute the actual (unpadded) length of each prompt in the batch.

    Args:
        prompts: Tensor of shape (batch_size, seq_len).
        pad_id: Optional padding token ID. If provided, lengths are computed
                up to the first occurrence of pad_id; otherwise full length.

    Returns:
        A list of 1D tensor containing the length (number of non-pad tokens) of the corresponding prompt.
    """
    lengths: list[torch.Tensor] = []
    batch_size, seq_len = prompts.size()

    for i in range(batch_size):
        tokens = prompts[i]
        if pad_id is not None:
            # find first pad token
            pads = torch.nonzero(tokens == pad_id, as_tuple=False)
            if pads.numel() > 0:
                length = pads[0].item()
            else:
                length = seq_len
        else:
            length = seq_len

        lengths.append(torch.tensor([length]))

    return lengths


def setup_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免 Ray Worker 重用导致 Handler 堆积
    if not logger.handlers:
        # 显式输出到 stdout，Ray Log Monitor 会自动捕获这个流
        sh = logging.StreamHandler(stream=sys.stdout)
        formatter = logging.Formatter("%(asctime)s - (%(name)s) - %(levelname)s - %(message)s")
        sh.setFormatter(formatter)
        logger.addHandler(sh)
        logger.propagate = False
    return logger
