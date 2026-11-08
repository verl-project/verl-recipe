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

"""Public TPR Ring CP execution primitives; see ring_layout/ring_attention."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import torch.distributed as dist
from torch import Tensor

# Expose the custom autograd implementation to internal callers.
from .ring_attention import (
    _RingTPRAttention,
)
from .ring_layout import (
    RingLocalKVBlock,
    SequenceShard,
    _normalize_inputs,
    _prepare_ring_layout,
    _RingStaticLayout,
    make_ring_sequence_shard,
    resolve_cp_group,
)

if TYPE_CHECKING:
    from .state import KVStack, PastKVAnchors


def ring_cp_attention(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    prefix_blocks: Sequence[RingLocalKVBlock] = (),
    current_shard: SequenceShard,
    cp_group: Any,
    softmax_scale: float | None = None,
    _layout: _RingStaticLayout | None = None,
) -> Tensor:
    """Run blockwise TPR attention over MindSpeed causal Ring shards."""

    blocks, config = _normalize_inputs(
        query,
        current_key,
        current_value,
        prefix_blocks=prefix_blocks,
        current_shard=current_shard,
        cp_group=cp_group,
        softmax_scale=softmax_scale,
        _layout=_layout,
    )
    flat_kv = []
    for block in blocks:
        flat_kv.extend((block.key, block.value))
    flat_kv.extend((current_key, current_value))
    return _RingTPRAttention.apply(query, *flat_kv, config)


class RingCPAttentionBackend:
    """Bind one TPR Segment and ordered Prefix blocks to MindSpeed Ring CP."""

    def __init__(
        self,
        *,
        global_prefix_length: int,
        current_shard: SequenceShard,
        prefix_blocks_by_layer: Mapping[int, Sequence[RingLocalKVBlock]],
        cp_group: Any,
    ) -> None:
        if (
            not isinstance(global_prefix_length, int)
            or isinstance(global_prefix_length, bool)
            or global_prefix_length < 0
        ):
            raise ValueError("global_prefix_length must be a non-negative integer")
        normalized = {layer: tuple(blocks) for layer, blocks in sorted(prefix_blocks_by_layer.items())}
        if not normalized:
            raise ValueError("prefix_blocks_by_layer must contain every TPR layer")
        path_signature = None
        for layer, blocks in normalized.items():
            if not isinstance(layer, int) or isinstance(layer, bool) or layer <= 0:
                raise ValueError(f"layer numbers must be positive integers, got {layer!r}")
            signature = tuple(block.segment_id for block in blocks)
            if path_signature is None:
                path_signature = signature
            elif signature != path_signature:
                raise ValueError("Prefix path must be identical across Ring attention layers")
            if sum(block.shard.global_length for block in blocks) != global_prefix_length:
                raise ValueError(f"layer {layer} Prefix blocks do not cover the Prefix length")
        first_blocks = next(iter(normalized.values()))
        layout = _prepare_ring_layout(first_blocks, current_shard, cp_group)
        for layer, blocks in normalized.items():
            if tuple((block.segment_id, block.shard) for block in blocks) != layout.prefix_signature:
                raise ValueError(f"layer {layer} Ring Prefix placement differs from other layers")
        self._ring_layout = layout
        self._global_prefix_length = global_prefix_length
        self._current_shard = current_shard
        self._prefix_blocks_by_layer = MappingProxyType(normalized)
        self._cp_group = cp_group
        self._parallel_size = layout.cp_size

    @property
    def global_prefix_length(self) -> int:
        return self._global_prefix_length

    @property
    def global_suffix_length(self) -> int:
        return self._current_shard.global_length

    @property
    def local_suffix_length(self) -> int:
        return self._current_shard.local_length

    @property
    def parallel_size(self) -> int:
        return self._parallel_size

    def attention(
        self,
        layer_number: int,
        query: Tensor,
        new_key: Tensor,
        new_value: Tensor,
        *,
        softmax_scale: float | None,
    ) -> Tensor:
        try:
            blocks = self._prefix_blocks_by_layer[layer_number]
        except KeyError as exc:
            raise KeyError(f"Ring CP context has no layer {layer_number}") from exc
        return ring_cp_attention(
            query,
            new_key,
            new_value,
            prefix_blocks=blocks,
            current_shard=self._current_shard,
            cp_group=self._cp_group,
            softmax_scale=softmax_scale,
            _layout=self._ring_layout,
        )


def _prefix_blocks_by_layer(
    kv_stack: KVStack,
    anchors: PastKVAnchors | None,
    expected_layer_numbers: tuple[int, ...],
) -> dict[int, tuple[RingLocalKVBlock, ...]]:
    """Select one ordered ancestor path; preserve original or anchored KV leaves."""
    sources = (
        tuple((segment_id, kv_stack.get(segment_id).kv) for segment_id in kv_stack.segment_ids)
        if anchors is None
        else tuple((entry.segment_id, entry) for entry in anchors.entries)
    )
    return {
        layer: tuple(
            RingLocalKVBlock(segment_id, state.shard, *state.key_values[layer]) for segment_id, state in sources
        )
        for layer in expected_layer_numbers
    }


# Production CP policy: only the Prefix-aware Ring implementation is admitted.
class RingCPBackend:
    def __init__(
        self,
        group: Any,
        *,
        parallel_size: int | None = None,
        parallel_rank: int | None = None,
        sp_tp_size: int = 1,
    ):
        if (parallel_size is None) != (parallel_rank is None):
            raise ValueError("parallel_size and parallel_rank must be provided together")
        if parallel_size is None:
            parallel_size, parallel_rank = resolve_cp_group(group)
        if parallel_size <= 1 or not 0 <= parallel_rank < parallel_size:
            raise ValueError(f"invalid Ring CP coordinate {parallel_rank}/{parallel_size}")
        self._cp_group = group
        self._parallel_size = parallel_size
        self._parallel_rank = parallel_rank
        if not isinstance(sp_tp_size, int) or isinstance(sp_tp_size, bool) or sp_tp_size < 1:
            raise ValueError(f"sp_tp_size must be a positive integer, got {sp_tp_size!r}")
        self._sp_tp_size = sp_tp_size

    @property
    def sp_tp_size(self) -> int:
        return self._sp_tp_size

    @property
    def parallel_size(self) -> int:
        return self._parallel_size

    @property
    def parallel_rank(self) -> int:
        return self._parallel_rank

    def validate_segment_length(self, global_length: int) -> None:
        self.make_sequence_shard(global_length)

    def make_sequence_shard(self, global_length: int) -> SequenceShard:
        return make_ring_sequence_shard(
            global_length,
            cp_rank=self.parallel_rank,
            cp_size=self.parallel_size,
            sp_tp_size=self.sp_tp_size,
        )

    def make_attention_backend(
        self,
        kv_stack: KVStack,
        *,
        expected_layer_numbers: tuple[int, ...],
        current_shard: SequenceShard,
        past_anchors: PastKVAnchors | None,
    ) -> RingCPAttentionBackend:
        return RingCPAttentionBackend(
            global_prefix_length=(
                kv_stack.prefix_length if past_anchors is None else past_anchors.global_prefix_length
            ),
            current_shard=current_shard,
            prefix_blocks_by_layer=_prefix_blocks_by_layer(
                kv_stack,
                past_anchors,
                expected_layer_numbers,
            ),
            cp_group=self._cp_group,
        )


@dataclass(frozen=True, slots=True)
class TPREngineCPRuntime:
    group: Any | None
    ring: RingCPBackend | None
    size: int
    rank: int

    @property
    def enabled(self) -> bool:
        return self.size > 1


def resolve_engine_cp_runtime(engine: Any, model_config: Any) -> TPREngineCPRuntime:
    configured_size = engine.engine_config.context_parallel_size
    if not isinstance(configured_size, int) or isinstance(configured_size, bool) or configured_size <= 0:
        raise ValueError(f"context_parallel_size must be positive, got {configured_size!r}")
    if configured_size == 1:
        return TPREngineCPRuntime(group=None, ring=None, size=1, rank=0)
    if getattr(engine.engine_config, "dynamic_context_parallel", False):
        raise NotImplementedError("Dynamic CP is not supported by TPR")
    from megatron.core import parallel_state

    if not dist.is_initialized() or not parallel_state.model_parallel_is_initialized():
        raise RuntimeError("TPR CP requires initialized torch.distributed and Megatron groups")
    group = parallel_state.get_context_parallel_group()
    size, rank = resolve_cp_group(group)
    if size != configured_size:
        raise RuntimeError(f"TPR CP topology mismatch: configured {configured_size}, actual {size}")
    sp_tp_size = (
        getattr(model_config, "tensor_model_parallel_size", 1)
        if getattr(model_config, "sequence_parallel", False)
        else 1
    )
    return TPREngineCPRuntime(
        group=group,
        ring=RingCPBackend(
            group,
            parallel_size=size,
            parallel_rank=rank,
            sp_tp_size=sp_tp_size,
        ),
        size=size,
        rank=rank,
    )


def aggregate_cp_loss(
    loss_sum: Tensor,
    normalized_loss: Tensor,
    runtime: TPREngineCPRuntime,
) -> tuple[Tensor, Tensor]:
    """Aggregate CP losses, sharing the collective when both inputs are identical."""
    global_loss_sum = loss_sum.detach().clone()
    if runtime.enabled:
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM, group=runtime.group)
    if loss_sum is normalized_loss:
        return global_loss_sum, global_loss_sum.clone()

    global_normalized_loss = normalized_loss.detach().clone()
    if runtime.enabled:
        dist.all_reduce(global_normalized_loss, op=dist.ReduceOp.SUM, group=runtime.group)
    return global_loss_sum, global_normalized_loss
