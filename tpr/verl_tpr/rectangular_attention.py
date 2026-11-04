# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""A local NPU adapter for right-down rectangular causal attention.

This module deliberately bypasses MindSpeed's ``DotProductAttention.forward``
configuration path.  It does not read or mutate the process-wide
``sparse_mode`` setting; only this call uses CANN sparse mode 3.
"""

from __future__ import annotations

import math
from functools import cache

import torch
from torch import Tensor

_RIGHT_DOWN_CAUSAL_MODE = 3
_CUSTOM_MASK_MODE = 0
_TND_LAYOUT = "TND"
_DEFAULT_PRE_TOKENS = 2**31 - 1
_COMPRESSED_CAUSAL_MASK_SIZE = 2048


@cache
def _compressed_causal_mask(device: torch.device) -> Tensor:
    """Build the compressed bool mask expected by CANN sparse modes 2/3."""

    return torch.triu(
        torch.ones(
            (_COMPRESSED_CAUSAL_MASK_SIZE, _COMPRESSED_CAUSAL_MASK_SIZE),
            dtype=torch.bool,
            device=device,
        ),
        diagonal=1,
    )


def _validate_inputs(query: Tensor, key: Tensor, value: Tensor, dropout_p: float) -> tuple[int, int, int, int]:
    tensors = {"query": query, "key": key, "value": value}
    for name, tensor in tensors.items():
        if tensor.ndim != 4:
            raise ValueError(f"{name} must have shape [sequence, batch, heads, head_dim], got {tensor.shape}")

    query_length, query_batch_size, query_heads, head_dim = query.shape
    kv_length, key_batch_size, key_heads, key_head_dim = key.shape

    if value.shape != key.shape:
        raise ValueError(f"key and value must have identical shapes, got key={key.shape}, value={value.shape}")
    if query_batch_size != 1 or key_batch_size != 1:
        raise ValueError(f"rectangular attention requires batch size 1, got Q={query_batch_size}, KV={key_batch_size}")
    if query_length <= 0 or kv_length <= 0:
        raise ValueError(f"sequence lengths must be positive, got Q={query_length}, KV={kv_length}")
    if query_length > kv_length:
        raise ValueError(f"query length cannot exceed KV length, got Q={query_length}, KV={kv_length}")
    if query_heads <= 0 or key_heads <= 0 or head_dim <= 0:
        raise ValueError(f"head counts and head_dim must be positive, got Q={query.shape}, KV={key.shape}")
    if key_head_dim != head_dim:
        raise ValueError(f"Q and KV head dimensions must match, got Q={head_dim}, KV={key_head_dim}")
    if query_heads % key_heads != 0:
        raise ValueError(f"query head count must be divisible by KV head count, got Q={query_heads}, KV={key_heads}")
    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError(f"Q, K, and V dtypes must match, got Q={query.dtype}, K={key.dtype}, V={value.dtype}")
    if query.device != key.device or query.device != value.device:
        raise ValueError(f"Q, K, and V devices must match, got Q={query.device}, K={key.device}, V={value.device}")
    if not query.is_floating_point():
        raise ValueError(f"Q, K, and V must use a floating-point dtype, got {query.dtype}")
    if dropout_p != 0.0:
        raise ValueError(f"rectangular attention MVP requires dropout_p=0, got {dropout_p}")

    return query_length, kv_length, query_heads, head_dim


def _validate_attention_mask(
    attention_mask: Tensor | None,
    *,
    query_length: int,
    kv_length: int,
    device: torch.device,
) -> None:
    if attention_mask is None:
        return
    if not isinstance(attention_mask, Tensor):
        raise TypeError(f"attention_mask must be a torch.Tensor or None, got {type(attention_mask).__name__}")
    if attention_mask.dtype != torch.bool:
        raise ValueError(f"attention_mask must use torch.bool, got {attention_mask.dtype}")
    if attention_mask.device != device:
        raise ValueError(f"attention_mask must be on {device}, got {attention_mask.device}")
    expected_shape = (query_length, kv_length)
    if tuple(attention_mask.shape) != expected_shape:
        raise ValueError(f"attention_mask must have shape {expected_shape}, got {tuple(attention_mask.shape)}")


def _attention_mask_parameters(
    attention_mask: Tensor | None,
    *,
    device: torch.device,
) -> tuple[Tensor, int, int]:
    if attention_mask is None:
        return _compressed_causal_mask(device), _RIGHT_DOWN_CAUSAL_MODE, 0
    # sparse mode 0 applies pre/next_tockens in addition to atten_mask. Keep
    # that window fully open so the physical causal/padding mask is the sole
    # visibility contract for rectangular Q/KV layouts.
    return attention_mask.contiguous(), _CUSTOM_MASK_MODE, _DEFAULT_PRE_TOKENS


def rectangular_causal_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    softmax_scale: float | None = None,
    dropout_p: float = 0.0,
    pre_tokens: int = _DEFAULT_PRE_TOKENS,
    inner_precise: int = 0,
    attention_mask: Tensor | None = None,
) -> Tensor:
    """Run differentiable rectangular causal attention on equal-length NPU sequences.

    Args:
        query: Post-RoPE query in Megatron layout ``[Sq, 1, Hq, D]``.
        key: Post-RoPE key, including the external prefix, in layout
            ``[Skv, 1, Hkv, D]``.
        value: Value, including the external prefix, with the same shape as
            ``key``.
        softmax_scale: Attention score scale. Defaults to ``1 / sqrt(D)``.
        dropout_p: Attention dropout. The MVP only supports zero.
        pre_tokens: CANN's preceding-token window. It defaults to an effectively
            unbounded value so sparse mode 3 supplies the causal boundary.
        inner_precise: Forwarded unchanged to ``npu_fusion_attention``.
        attention_mask: Optional physical ``[Sq, Skv]`` bool mask. ``True``
            entries are excluded. Supplying it selects CANN's standard custom
            mask mode instead of the compressed right-down causal mask.

    Returns:
        Attention context in Megatron layout ``[Sq, 1, Hq * D]``.
    """

    query_length, kv_length, query_heads, head_dim = _validate_inputs(query, key, value, dropout_p)
    _validate_attention_mask(
        attention_mask,
        query_length=query_length,
        kv_length=kv_length,
        device=query.device,
    )

    if query.device.type != "npu":
        raise RuntimeError(f"rectangular_causal_attention requires an NPU tensor, got device {query.device}")
    if pre_tokens < 0:
        raise ValueError(f"pre_tokens must be non-negative, got {pre_tokens}")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if not math.isfinite(softmax_scale) or softmax_scale <= 0:
        raise ValueError(f"softmax_scale must be finite and positive, got {softmax_scale}")

    try:
        import torch_npu
    except ImportError as exc:  # pragma: no cover - only reachable in a misconfigured NPU runtime
        raise RuntimeError("torch_npu is required for rectangular_causal_attention") from exc

    query_tnd = query.squeeze(1).contiguous()
    key_tnd = key.squeeze(1).contiguous()
    value_tnd = value.squeeze(1).contiguous()
    attention_mask, sparse_mode, next_tokens = _attention_mask_parameters(
        attention_mask,
        device=query.device,
    )

    output = torch_npu.npu_fusion_attention(
        query_tnd,
        key_tnd,
        value_tnd,
        query_heads,
        _TND_LAYOUT,
        pse=None,
        padding_mask=None,
        atten_mask=attention_mask,
        scale=softmax_scale,
        pre_tockens=pre_tokens,
        next_tockens=next_tokens,
        keep_prob=1.0,
        inner_precise=inner_precise,
        sparse_mode=sparse_mode,
        actual_seq_qlen=[query_length],
        actual_seq_kvlen=[kv_length],
    )[0]

    expected_shape = (query_length, query_heads, head_dim)
    if output.shape != expected_shape:
        raise RuntimeError(f"unexpected CANN attention output shape: expected {expected_shape}, got {output.shape}")
    return output.reshape(query_length, 1, query_heads * head_dim)
