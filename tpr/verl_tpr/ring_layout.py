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

"""Ring zigzag SequenceShard layout, padding and shape validation."""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

SequenceRange = tuple[int, int]


def _validate_integer(name: str, value: int, *, minimum: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}")


def _validate_parallel_coordinates(cp_rank: int, cp_size: int) -> None:
    _validate_integer("cp_size", cp_size, minimum=1)
    _validate_integer("cp_rank", cp_rank, minimum=0)
    if cp_rank >= cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}")


def _normalize_dimension(tensor: Tensor, dim: int) -> int:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor).__name__}")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError(f"dim must be an integer, got {dim!r}")
    normalized = dim if dim >= 0 else tensor.ndim + dim
    if normalized < 0 or normalized >= tensor.ndim:
        raise IndexError(f"dimension {dim} is out of range for tensor with {tensor.ndim} dimensions")
    return normalized


def _validate_global_offset(global_offset: int, global_length: int) -> None:
    _validate_integer("global_offset", global_offset, minimum=0)
    if global_offset >= global_length:
        raise ValueError(f"global_offset must be in [0, {global_length}), got {global_offset}")


@dataclass(frozen=True, slots=True)
class SequenceShard:
    """A Segment's logical length and ordered physical ranges on one CP rank.

    CP1 and Ring use the same metadata. Tail padding stays in the physical
    ranges so every rank has equal-shaped tensors; logical ranges exclude it.
    """

    global_length: int
    ranges: tuple[SequenceRange, ...]
    cp_rank: int = 0
    cp_size: int = 1
    padded_length: int | None = None

    def __post_init__(self) -> None:
        _validate_integer("global_length", self.global_length, minimum=1)
        _validate_parallel_coordinates(self.cp_rank, self.cp_size)
        padded_length = self.global_length if self.padded_length is None else self.padded_length
        _validate_integer("padded_length", padded_length, minimum=self.global_length)
        ranges = tuple(tuple(item) for item in self.ranges)
        if not ranges:
            raise ValueError("ranges must contain at least one non-empty range")
        previous_end = 0
        for index, item in enumerate(ranges):
            if len(item) != 2:
                raise ValueError(f"ranges[{index}] must be a (start, end) pair, got {item!r}")
            start, end = item
            _validate_integer(f"ranges[{index}].start", start, minimum=0)
            _validate_integer(f"ranges[{index}].end", end, minimum=1)
            if start >= end or end > padded_length:
                raise ValueError(f"ranges[{index}] must be non-empty and within [0, {padded_length})")
            if index and start < previous_end:
                raise ValueError("ranges must be ordered and non-overlapping")
            previous_end = end
        object.__setattr__(self, "ranges", ranges)
        object.__setattr__(self, "padded_length", padded_length)

    @classmethod
    def full(cls, global_length: int) -> SequenceShard:
        return cls(global_length, ((0, global_length),))

    @property
    def global_ranges(self) -> tuple[SequenceRange, ...]:
        return tuple((start, min(end, self.global_length)) for start, end in self.ranges if start < self.global_length)

    @property
    def local_length(self) -> int:
        return sum(end - start for start, end in self.ranges)

    @property
    def valid_local_length(self) -> int:
        return sum(end - start for start, end in self.global_ranges)

    @property
    def physical_global_ranges(self) -> tuple[SequenceRange, ...]:
        return self.ranges

    @property
    def is_full(self) -> bool:
        return self.valid_local_length == self.global_length

    def select(self, tensor: Tensor, *, dim: int = 0) -> Tensor:
        dim = _normalize_dimension(tensor, dim)
        if tensor.shape[dim] != self.global_length:
            raise ValueError(f"tensor sequence dimension must be {self.global_length}, got {tensor.shape[dim]}")
        chunks = []
        for start, end in self.ranges:
            valid_start, valid_end = min(start, self.global_length), min(end, self.global_length)
            chunk = tensor.narrow(dim, valid_start, valid_end - valid_start)
            padding = end - start - (valid_end - valid_start)
            if padding:
                shape = list(tensor.shape)
                shape[dim] = padding
                # Retain an autograd edge even on ranks that own only padding.
                chunk = torch.cat((chunk, tensor.new_zeros(shape)), dim=dim)
            chunks.append(chunk)
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=dim)

    def global_indices(self, *, device: torch.device | str | None = None) -> Tensor:
        return self.physical_global_indices(device=device).clamp_max(self.global_length - 1)

    def physical_global_indices(self, *, device: torch.device | str | None = None) -> Tensor:
        chunks = tuple(torch.arange(start, end, dtype=torch.long, device=device) for start, end in self.ranges)
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)

    def owns(self, global_offset: int) -> bool:
        return self.global_to_local(global_offset) is not None

    def global_to_local(self, global_offset: int) -> int | None:
        _validate_global_offset(global_offset, self.global_length)
        local_start = 0
        for start, end in self.ranges:
            if start <= global_offset < end:
                return local_start + global_offset - start
            local_start += end - start
        return None

    def local_to_global(self, local_offset: int) -> int:
        _validate_integer("local_offset", local_offset, minimum=0)
        if local_offset >= self.local_length:
            raise ValueError(f"local_offset must be in [0, {self.local_length}), got {local_offset}")
        for start, end in self.ranges:
            if local_offset < end - start:
                global_offset = start + local_offset
                if global_offset >= self.global_length:
                    raise ValueError(f"local offset {local_offset} refers to a padding token")
                return global_offset
            local_offset -= end - start
        raise AssertionError("validated local offset was not mapped")


def round_up_sequence_length(logical_length: int, alignment: int) -> int:
    """Return the smallest positive multiple of ``alignment`` covering a Segment."""

    _validate_integer("logical_length", logical_length, minimum=1)
    _validate_integer("alignment", alignment, minimum=1)
    return ((logical_length + alignment - 1) // alignment) * alignment


def make_cp1_sp_sequence_shard(
    global_length: int,
    *,
    sp_tp_size: int,
) -> SequenceShard:
    """Return a contiguous CP1 shard padded for native Megatron TP+SP.

    CP1 owns the complete logical Segment. When sequence parallelism is
    enabled, its physical sequence must be divisible by TP so Megatron can
    scatter the sequence dimension evenly across TP ranks. Padding is kept
    as a trailing physical range and excluded by the logical length.
    """

    _validate_integer("global_length", global_length, minimum=1)
    _validate_integer("sp_tp_size", sp_tp_size, minimum=1)
    padded_length = round_up_sequence_length(global_length, sp_tp_size)
    return SequenceShard(
        global_length,
        ((0, padded_length),),
        cp_rank=0,
        cp_size=1,
        padded_length=padded_length,
    )


def iter_valid_sequence_slices(
    shard: SequenceShard,
) -> tuple[tuple[SequenceRange, slice], ...]:
    """Map each non-empty logical range to its slice in the local physical tensor."""

    result: list[tuple[SequenceRange, slice]] = []
    local_start = 0
    for physical_start, physical_end in shard.physical_global_ranges:
        physical_length = physical_end - physical_start
        valid_end = min(physical_end, shard.global_length)
        if physical_start < valid_end:
            valid_length = valid_end - physical_start
            result.append(((physical_start, valid_end), slice(local_start, local_start + valid_length)))
        local_start += physical_length
    if sum(item[0][1] - item[0][0] for item in result) != shard.valid_local_length:
        raise RuntimeError("valid sequence slices do not match shard validity metadata")
    return tuple(result)


def resolve_cp_group(group: Any) -> tuple[int, int]:
    """Resolve an existing distributed CP group without depending on AllGather."""
    if not dist.is_available() or not dist.is_initialized():
        if group is not None:
            raise RuntimeError("a CP process group requires torch.distributed initialization")
        return 1, 0
    return dist.get_world_size(group), dist.get_rank(group)


_group_world_size_and_rank = resolve_cp_group

_TND_LAYOUT = "TND"
_RIGHT_DOWN_CAUSAL_MODE = 3
_FULL_ATTENTION_MODE = 0
_MAX_TOKENS = 2**31 - 1


class RingBlockKind(str, Enum):
    """Visibility of one Current query range against one KV range."""

    FULL = "full"
    CAUSAL = "causal"
    SKIP = "skip"


@dataclass(frozen=True, slots=True)
class RingLocalKVBlock:
    """One Prefix segment's local zigzag KV shard for a single layer."""

    segment_id: int
    shard: SequenceShard
    key: Tensor
    value: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.segment_id, int) or isinstance(self.segment_id, bool) or self.segment_id < 0:
            raise ValueError(f"segment_id must be a non-negative integer, got {self.segment_id!r}")


@dataclass(frozen=True, slots=True)
class _RingAttentionConfig:
    cp_group: Any
    global_ranks: tuple[int, ...]
    cp_size: int
    cp_rank: int
    segment_lengths: tuple[int, ...]
    segment_padded_lengths: tuple[int, ...]
    current_shard: SequenceShard
    query_heads: int
    head_dim: int
    softmax_scale: float
    coalesce_prefix_full: bool = False
    coalesce_prefix_query: bool = False
    coalesce_prefix_padded: bool = False


@dataclass(frozen=True, slots=True)
class _RingStaticLayout:
    """Validated once per Segment; independent of per-layer K/V tensors."""

    cp_group: Any
    cp_size: int
    cp_rank: int
    global_ranks: tuple[int, ...]
    current_shard: SequenceShard
    prefix_signature: tuple[tuple[int, SequenceShard], ...]
    segment_lengths: tuple[int, ...]
    segment_padded_lengths: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _RingRangeSlice:
    """One physical Ring chunk plus its clipped logical validity range."""

    logical_range: SequenceRange
    physical_range: SequenceRange
    local_slice: slice
    tensor: Tensor

    @property
    def physical_length(self) -> int:
        return self.physical_range[1] - self.physical_range[0]

    @property
    def valid_length(self) -> int:
        return self.logical_range[1] - self.logical_range[0]


def make_ring_sequence_shard(
    global_length: int,
    *,
    cp_rank: int,
    cp_size: int,
    padded_length: int | None = None,
    sp_tp_size: int = 1,
) -> SequenceShard:
    """Return a Zigzag Ring shard with optional native TP/SP alignment.

    `sp_tp_size=1` keeps the existing Ring CP layout when sequence parallelism
    is disabled. With SP, every CP-local physical segment must also be
    divisible by the tensor-parallel size.
    """

    if not isinstance(global_length, int) or isinstance(global_length, bool) or global_length <= 0:
        raise ValueError(f"global_length must be a positive integer, got {global_length!r}")
    if not isinstance(cp_size, int) or isinstance(cp_size, bool) or cp_size <= 1:
        raise ValueError(f"Ring CP requires cp_size greater than one, got {cp_size!r}")
    if not isinstance(cp_rank, int) or isinstance(cp_rank, bool) or not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank!r}")
    _validate_integer("sp_tp_size", sp_tp_size, minimum=1)
    chunk_count = 2 * cp_size
    # Zigzag Ring needs two equal chunks per CP rank. Megatron native SP
    # further partitions each CP-local segment over TP, so both constraints
    # apply to the same physical length.
    alignment = math.lcm(chunk_count, cp_size * sp_tp_size)
    if padded_length is None:
        padded_length = round_up_sequence_length(global_length, alignment)
    if padded_length < global_length or padded_length % alignment != 0:
        raise ValueError(
            f"padded_length must cover {global_length} and be divisible by {alignment}, got {padded_length}"
        )
    chunk_length = padded_length // chunk_count
    first_start = cp_rank * chunk_length
    second_start = (chunk_count - cp_rank - 1) * chunk_length
    return SequenceShard(
        global_length,
        (
            (first_start, first_start + chunk_length),
            (second_start, second_start + chunk_length),
        ),
        cp_rank=cp_rank,
        cp_size=cp_size,
        padded_length=padded_length,
    )


def classify_ring_block(
    query_range: SequenceRange,
    kv_range: SequenceRange,
    *,
    is_prefix: bool,
) -> RingBlockKind:
    """Classify FULL/CAUSAL/SKIP using logical Segment-local positions."""

    if is_prefix:
        return RingBlockKind.FULL
    query_start, query_end = query_range
    kv_start, kv_end = kv_range
    if kv_end <= query_start:
        return RingBlockKind.FULL
    if kv_start >= query_end:
        return RingBlockKind.SKIP
    if query_range != kv_range:
        raise ValueError(
            "partially overlapping Current query/KV ranges cannot use the aligned Ring schedule: "
            f"Q={query_range}, KV={kv_range}"
        )
    return RingBlockKind.CAUSAL


def _load_mindspeed_ring_primitives():
    try:
        from mindspeed.core.context_parallel.utils import RingP2P, forward_update_without_fused
    except ImportError as exc:  # pragma: no cover - requires the server MindSpeed runtime
        raise RuntimeError("MindSpeed RingP2P and online-softmax utilities are required") from exc
    return RingP2P, forward_update_without_fused


def _process_group_global_ranks(process_group: Any) -> tuple[int, ...]:
    cp_size, cp_rank = _group_world_size_and_rank(process_group)
    if hasattr(dist, "get_process_group_ranks"):
        global_ranks = tuple(dist.get_process_group_ranks(process_group))
    elif hasattr(dist, "get_global_rank"):
        global_ranks = tuple(dist.get_global_rank(process_group, rank) for rank in range(cp_size))
    elif process_group is dist.group.WORLD:
        global_ranks = tuple(range(cp_size))
    else:  # pragma: no cover - only relevant to obsolete torch.distributed versions
        raise RuntimeError("cannot resolve global ranks for the CP process group")
    if len(global_ranks) != cp_size:
        raise RuntimeError(f"CP group exposes {len(global_ranks)} global ranks, expected {cp_size}")
    if dist.get_rank() != global_ranks[cp_rank]:
        raise RuntimeError("CP process-group rank order does not match the current global rank")
    return global_ranks


def _validate_ring_shard(
    shard: SequenceShard,
    *,
    cp_size: int,
    cp_rank: int,
    name: str,
) -> None:
    if not isinstance(shard, SequenceShard):
        raise TypeError(f"{name} shard must implement SequenceShard, got {type(shard).__name__}")
    expected = make_ring_sequence_shard(
        shard.global_length,
        cp_rank=cp_rank,
        cp_size=cp_size,
        padded_length=shard.padded_length,
    )
    if (shard.ranges, shard.cp_rank, shard.cp_size) != (expected.ranges, cp_rank, cp_size):
        raise ValueError(f"{name} shard must use the MindSpeed causal Ring layout")


def _validate_kv_pair(
    key: Tensor,
    value: Tensor,
    *,
    shard: SequenceShard,
    name: str,
) -> None:
    if not isinstance(key, Tensor) or not isinstance(value, Tensor):
        raise TypeError(f"{name} K/V must be torch tensors")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"{name} K/V must have shape [sequence, batch, heads, head_dim]")
    if key.shape != value.shape:
        raise ValueError(f"{name} K/V shapes must match")
    if key.shape[0] != shard.local_length or key.shape[1] != 1:
        raise ValueError(f"{name} K/V must start with [{shard.local_length}, 1], got {tuple(key.shape)}")
    if key.dtype != value.dtype or key.device != value.device:
        raise ValueError(f"{name} K/V dtype and device must match")
    if not key.is_floating_point():
        raise ValueError(f"{name} K/V must be floating point")


def _prepare_ring_layout(
    prefix_blocks: Sequence[RingLocalKVBlock],
    current_shard: SequenceShard,
    cp_group: Any,
) -> _RingStaticLayout:
    """Validate immutable CP placement once when creating a Segment context."""
    cp_size, cp_rank = _group_world_size_and_rank(cp_group)
    _validate_ring_shard(current_shard, cp_size=cp_size, cp_rank=cp_rank, name="current")
    seen: set[int] = set()
    signature = []
    for index, block in enumerate(prefix_blocks):
        if not isinstance(block, RingLocalKVBlock):
            raise TypeError(f"prefix_blocks[{index}] must be RingLocalKVBlock")
        if block.segment_id in seen:
            raise ValueError(f"duplicate Prefix segment_id {block.segment_id}")
        seen.add(block.segment_id)
        _validate_ring_shard(
            block.shard,
            cp_size=cp_size,
            cp_rank=cp_rank,
            name=f"Prefix block {block.segment_id}",
        )
        signature.append((block.segment_id, block.shard))
    shards = tuple(shard for _, shard in signature) + (current_shard,)
    return _RingStaticLayout(
        cp_group=cp_group,
        cp_size=cp_size,
        cp_rank=cp_rank,
        global_ranks=_process_group_global_ranks(cp_group),
        current_shard=current_shard,
        prefix_signature=tuple(signature),
        segment_lengths=tuple(shard.global_length for shard in shards),
        segment_padded_lengths=tuple(shard.padded_length for shard in shards),
    )


def ring_merge_settings() -> tuple[bool, bool, bool]:
    """Enable Prefix QKV Merge and physical-padding merge by default.

    Explicit environment overrides are retained for controlled ablations and
    rollback; normal TPR execution requires no additional merge switches.
    """
    return tuple(
        os.getenv(name, "1") == "1"
        for name in (
            "TPR_RING_COALESCE_PREFIX_FULL",
            "TPR_RING_COALESCE_PREFIX_QUERY",
            "TPR_RING_COALESCE_PREFIX_PADDED",
        )
    )


def _normalize_inputs(
    query: Tensor,
    current_key: Tensor,
    current_value: Tensor,
    *,
    prefix_blocks: Sequence[RingLocalKVBlock],
    current_shard: SequenceShard,
    cp_group: Any,
    softmax_scale: float | None,
    _layout: _RingStaticLayout | None = None,
) -> tuple[tuple[RingLocalKVBlock, ...], _RingAttentionConfig]:
    if _layout is None:
        cp_size, cp_rank = _group_world_size_and_rank(cp_group)
        _validate_ring_shard(current_shard, cp_size=cp_size, cp_rank=cp_rank, name="current")
    else:
        if _layout.cp_group is not cp_group or _layout.current_shard != current_shard:
            raise ValueError("prepared Ring layout does not match this CP group or Current shard")
        cp_size, cp_rank = _layout.cp_size, _layout.cp_rank
    _validate_kv_pair(current_key, current_value, shard=current_shard, name="current")
    if not isinstance(query, Tensor) or query.ndim != 4:
        raise ValueError("query must have shape [sequence, batch, heads, head_dim]")
    if query.shape[0] != current_shard.local_length or query.shape[1] != 1:
        raise ValueError(f"query must start with [{current_shard.local_length}, 1], got {tuple(query.shape)}")
    if query.dtype != current_key.dtype or query.device != current_key.device:
        raise ValueError("query dtype/device must match Current KV")
    if query.shape[-1] != current_key.shape[-1]:
        raise ValueError("query and KV head dimensions must match")
    if query.shape[2] % current_key.shape[2] != 0:
        raise ValueError("query head count must be divisible by KV head count")

    blocks = tuple(prefix_blocks)
    seen: set[int] = set()
    for index, block in enumerate(blocks):
        if not isinstance(block, RingLocalKVBlock):
            raise TypeError(f"prefix_blocks[{index}] must be RingLocalKVBlock")
        if _layout is None:
            if block.segment_id in seen:
                raise ValueError(f"duplicate Prefix segment_id {block.segment_id}")
            seen.add(block.segment_id)
            _validate_ring_shard(
                block.shard,
                cp_size=cp_size,
                cp_rank=cp_rank,
                name=f"Prefix block {block.segment_id}",
            )
        _validate_kv_pair(
            block.key,
            block.value,
            shard=block.shard,
            name=f"Prefix block {block.segment_id}",
        )
        if block.key.shape[1:] != current_key.shape[1:]:
            raise ValueError(f"Prefix block {block.segment_id} shape must match Current KV")
        if block.key.dtype != current_key.dtype or block.key.device != current_key.device:
            raise ValueError(f"Prefix block {block.segment_id} dtype/device must match Current KV")

    if _layout is not None and (tuple((block.segment_id, block.shard) for block in blocks) != _layout.prefix_signature):
        raise ValueError("prepared Ring layout does not match Prefix segment placement")
    scale = query.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    if not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError(f"softmax_scale must be finite and positive, got {scale!r}")
    if _layout is None:
        global_ranks = _process_group_global_ranks(cp_group)
        shards = tuple(block.shard for block in blocks) + (current_shard,)
        lengths = tuple(shard.global_length for shard in shards)
        padded_lengths = tuple(shard.padded_length for shard in shards)
    else:
        global_ranks = _layout.global_ranks
        lengths = _layout.segment_lengths
        padded_lengths = _layout.segment_padded_lengths
    coalesce_full, coalesce_query, coalesce_padded = ring_merge_settings()
    config = _RingAttentionConfig(
        cp_group=cp_group,
        global_ranks=global_ranks,
        cp_size=cp_size,
        cp_rank=cp_rank,
        segment_lengths=lengths,
        segment_padded_lengths=padded_lengths,
        current_shard=current_shard,
        query_heads=query.shape[2],
        head_dim=query.shape[-1],
        softmax_scale=float(scale),
        coalesce_prefix_full=coalesce_full,
        coalesce_prefix_query=coalesce_query,
        coalesce_prefix_padded=coalesce_padded,
    )
    return blocks, config
