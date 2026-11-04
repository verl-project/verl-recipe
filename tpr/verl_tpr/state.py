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

"""Graph-free KV on the DFS path, with one gradient relay per segment."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import count
from types import MappingProxyType

import torch
from torch import Tensor

from verl_tpr.context import KVPair
from verl_tpr.context import _compact_kv_tensor as _compact_graph_free_tensor

from .ring import SequenceShard
from .tree import SegmentId, SegmentSpec, _validate_segment_id

_generations = count()


def _validate_layer_number(layer_number: int) -> None:
    if not isinstance(layer_number, int) or isinstance(layer_number, bool) or layer_number <= 0:
        raise ValueError(f"layer_number must be a positive integer, got {layer_number!r}")


def _normalize_local_kv(key_values: Mapping[int, KVPair], *, local_length: int) -> dict[int, KVPair]:
    if not isinstance(key_values, Mapping) or not key_values:
        raise ValueError("key_values must be a non-empty mapping")
    normalized, placement = {}, None
    for layer, pair in key_values.items():
        _validate_layer_number(layer)
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise TypeError(f"layer {layer} KV must be a (key, value) tuple")
        key, value = pair
        if not isinstance(key, Tensor) or not isinstance(value, Tensor):
            raise TypeError(f"layer {layer} K/V must be torch tensors")
        if key.ndim != 4 or key.shape != value.shape:
            raise ValueError(f"layer {layer} K/V shapes must match [sequence, batch, heads, head_dim]")
        if key.shape[0] != local_length:
            raise ValueError(f"layer {layer} KV local sequence length must be {local_length}, got {key.shape[0]}")
        if key.shape[1] != 1:
            raise ValueError(f"layer {layer} KV batch size must be 1, got {key.shape[1]}")
        if (key.device, key.dtype) != (value.device, value.dtype):
            raise ValueError(f"layer {layer} K/V device and dtype must match")
        if any(t.requires_grad or t.grad_fn is not None for t in pair):
            raise ValueError(f"layer {layer} cached K/V must be detached and graph-free")
        if placement is not None and placement != (key.device, key.dtype):
            raise ValueError("all KV layers in a prefix state must use the same device and dtype")
        placement = key.device, key.dtype
        normalized[layer] = tuple(_compact_graph_free_tensor(t) for t in pair)
    return dict(sorted(normalized.items()))


@dataclass(frozen=True, slots=True)
class KVPrefixAnchors:
    """Independent autograd leaves sharing one segment's compact KV storage."""

    segment_id: SegmentId
    shard: SequenceShard
    generation: int
    key_values: Mapping[int, KVPair]


@dataclass(frozen=True, slots=True)
class PastKVAnchors:
    """The same per-segment anchors serve both local and Ring attention.

    Local attention concatenates the leaves through autograd. Ring attention
    consumes each entry directly; both relay gradients through the same stack.
    """

    entries: tuple[KVPrefixAnchors, ...]
    stack_signature: tuple[SegmentId, ...]

    @property
    def prefix_length(self) -> int:
        return sum(entry.shard.global_length for entry in self.entries)

    global_prefix_length = prefix_length

    @property
    def key_values(self) -> Mapping[int, KVPair]:
        return _concatenate_kv(tuple((entry.shard, entry.key_values) for entry in self.entries))


def _concatenate_kv(entries: tuple[tuple[SequenceShard, Mapping[int, KVPair]], ...]) -> Mapping[int, KVPair]:
    if any(shard.cp_size != 1 for shard, _ in entries):
        raise RuntimeError("sharded KV must be consumed through a context-parallel attention backend")
    if len(entries) <= 1:
        return entries[0][1] if entries else MappingProxyType({})
    return MappingProxyType(
        {
            layer: tuple(torch.cat([kv[layer][index] for _, kv in entries], dim=0) for index in (0, 1))
            for layer in entries[0][1]
        }
    )


class KVPrefixState:
    """One segment's compact post-RoPE KV and delayed descendant gradients."""

    state_kind = "kv"

    def __init__(
        self,
        segment_id: SegmentId,
        sequence_length: int,
        key_values: Mapping[int, KVPair],
        *,
        shard: SequenceShard | None = None,
    ) -> None:
        _validate_segment_id(segment_id)
        if not isinstance(sequence_length, int) or isinstance(sequence_length, bool) or sequence_length <= 0:
            raise ValueError(f"sequence_length must be positive, got {sequence_length!r}")
        self.segment_id = segment_id
        self.shard = SequenceShard.full(sequence_length) if shard is None else shard
        if not isinstance(self.shard, SequenceShard):
            raise TypeError(f"shard must implement SequenceShard, got {type(self.shard).__name__}")
        if self.shard.global_length != sequence_length:
            raise ValueError("sequence_length must match shard global_length")
        self._key_values = _normalize_local_kv(key_values, local_length=self.local_length)
        self.layer_numbers = tuple(self._key_values)
        self._gradients: dict[int, KVPair] = {}
        self.released = False
        self._generation = next(_generations)

    @property
    def global_length(self) -> int:
        return self.shard.global_length

    @property
    def local_length(self) -> int:
        return self.shard.local_length

    @property
    def key_values(self) -> Mapping[int, KVPair]:
        self._ensure_live()
        return MappingProxyType(self._key_values)

    @property
    def gradients(self) -> Mapping[int, KVPair]:
        self._ensure_live()
        return MappingProxyType(self._gradients)

    def validate(self) -> None:
        self._ensure_live()
        normalized = _normalize_local_kv(self._key_values, local_length=self.local_length)
        if tuple(normalized) != self.layer_numbers or any(
            old is not new
            for layer, pair in normalized.items()
            for old, new in zip(self._key_values[layer], pair, strict=True)
        ):
            raise RuntimeError(f"segment {self.segment_id} KV storage or layers changed")

    def make_anchors(self) -> KVPrefixAnchors:
        leaves = MappingProxyType(
            {layer: tuple(t.detach().requires_grad_(True) for t in pair) for layer, pair in self.key_values.items()}
        )
        return KVPrefixAnchors(self.segment_id, self.shard, self._generation, leaves)

    def accumulate_layer_gradients(self, layer_number: int, key_grad: Tensor, value_grad: Tensor) -> None:
        pair = self.key_values[layer_number]
        gradients = key_grad, value_grad
        for tensor, gradient in zip(pair, gradients, strict=True):
            if gradient.shape != tensor.shape:
                raise ValueError(f"segment {self.segment_id} layer {layer_number} gradient shape mismatch")
            if (gradient.device, gradient.dtype) != (tensor.device, tensor.dtype):
                raise ValueError(f"segment {self.segment_id} layer {layer_number} gradient device or dtype mismatch")
        if layer_number not in self._gradients:
            self._gradients[layer_number] = tuple(
                t.detach().clone(memory_format=torch.contiguous_format) for t in gradients
            )
        else:
            for buffer, gradient in zip(self._gradients[layer_number], gradients, strict=True):
                buffer.add_(gradient.detach())

    def accumulate_anchor_gradients(self, anchors: KVPrefixAnchors, *, consume: bool = False) -> None:
        """Accumulate dKV, optionally taking ownership of executor-only grads."""
        self._ensure_live()
        if (anchors.segment_id, anchors.shard, anchors.generation) != (self.segment_id, self.shard, self._generation):
            raise RuntimeError(f"KV prefix anchors for segment {anchors.segment_id} are stale")
        if tuple(anchors.key_values) != self.layer_numbers:
            raise RuntimeError(f"KV prefix anchor layers must be {self.layer_numbers}")
        owned_storage = (
            {
                (tensor.device, tensor.untyped_storage().data_ptr())
                for mapping in (self._key_values, self._gradients)
                for pair in mapping.values()
                for tensor in pair
            }
            if consume
            else set()
        )
        for layer, (key, value) in anchors.key_values.items():
            if key.grad is None or value.grad is None:
                raise RuntimeError(f"layer {layer} KV prefix anchor gradient is missing")
            if consume and layer not in self._gradients:
                buffers = []
                for gradient in (key.grad, value.grad):
                    buffer = _compact_graph_free_tensor(gradient.detach())
                    # Custom backward functions may reuse a buffer for K/V or
                    # multiple layers. Future accumulation needs separate owners.
                    if (buffer.device, buffer.untyped_storage().data_ptr()) in owned_storage:
                        buffer = buffer.clone(memory_format=torch.contiguous_format)
                    owned_storage.add((buffer.device, buffer.untyped_storage().data_ptr()))
                    buffers.append(buffer)
                self._gradients[layer] = tuple(buffers)
            else:
                self.accumulate_layer_gradients(layer, key.grad, value.grad)
            if consume:
                key.grad = value.grad = None

    def consume_gradients(self) -> Mapping[int, KVPair]:
        self._ensure_live()
        if self._gradients and set(self._gradients) != set(self.layer_numbers):
            raise RuntimeError(f"segment {self.segment_id} gradient layers must be {self.layer_numbers}")
        gradients, self._gradients = self._gradients, {}
        self._generation = next(_generations)
        return MappingProxyType(gradients)

    def release(self) -> None:
        self._key_values.clear()
        self._gradients.clear()
        self.released = True
        self._generation = next(_generations)

    def _ensure_live(self) -> None:
        if self.released:
            raise RuntimeError(f"KV prefix state for segment {self.segment_id} has been released")


@dataclass(slots=True)
class KVStackEntry:
    segment: SegmentSpec
    kv: KVPrefixState

    @property
    def gradients(self) -> Mapping[int, KVPair]:
        return self.kv.gradients


class KVStack:
    """Own only the active root-to-parent path and its delayed dKV."""

    def __init__(self) -> None:
        self._entries: dict[SegmentId, KVStackEntry] = {}

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def segment_ids(self) -> tuple[SegmentId, ...]:
        return tuple(self._entries)

    @property
    def prefix_length(self) -> int:
        return sum(entry.segment.length for entry in self._entries.values())

    def top(self) -> KVStackEntry:
        if not self._entries:
            raise RuntimeError("KV stack is empty")
        return next(reversed(self._entries.values()))

    def get(self, segment_id: SegmentId) -> KVStackEntry:
        try:
            return self._entries[segment_id]
        except KeyError as exc:
            raise KeyError(f"segment {segment_id} is not on the KV stack") from exc

    def push(
        self,
        segment: SegmentSpec,
        key_values: Mapping[int, KVPair],
        *,
        shard: SequenceShard | None = None,
    ) -> KVStackEntry:
        if segment.segment_id in self._entries:
            raise RuntimeError(f"segment {segment.segment_id} is already on the KV stack")
        parent = self.top() if self._entries else None
        if segment.parent_id != (parent.segment.segment_id if parent else None):
            raise ValueError(f"cannot push segment {segment.segment_id}: wrong parent {segment.parent_id}")
        if segment.prefix_length != self.prefix_length or segment.position_start != self.prefix_length:
            raise ValueError(f"cannot push segment {segment.segment_id}: stack prefix length is {self.prefix_length}")
        kv = KVPrefixState(segment.segment_id, segment.length, key_values, shard=shard)
        if parent:
            if kv.layer_numbers != parent.kv.layer_numbers:
                raise ValueError(f"segment {segment.segment_id} KV layers must be {parent.kv.layer_numbers}")
            for layer, (key, _) in kv.key_values.items():
                previous = parent.kv.key_values[layer][0]
                if key.shape[1:] != previous.shape[1:]:
                    raise ValueError(f"layer {layer} KV non-sequence shape differs across segments")
                if (key.device, key.dtype) != (previous.device, previous.dtype):
                    raise ValueError(f"layer {layer} KV device or dtype differs across segments")
        entry = KVStackEntry(segment, kv)
        self._entries[segment.segment_id] = entry
        return entry

    def pop(self, segment_id: SegmentId) -> KVStackEntry:
        if self.top().segment.segment_id != segment_id:
            raise RuntimeError(f"cannot pop segment {segment_id}: stack top is {self.top().segment.segment_id}")
        return self._entries.pop(segment_id)

    def build_past_key_values(self) -> Mapping[int, KVPair]:
        return _concatenate_kv(tuple((entry.kv.shard, entry.kv.key_values) for entry in self._entries.values()))

    def build_past_anchors(self) -> PastKVAnchors:
        return PastKVAnchors(tuple(entry.kv.make_anchors() for entry in self._entries.values()), self.segment_ids)

    def accumulate_anchor_gradients(self, anchors: PastKVAnchors, *, consume: bool = False) -> None:
        if anchors.stack_signature != self.segment_ids:
            raise RuntimeError("stale Prefix anchor stack signature")
        if tuple(entry.segment_id for entry in anchors.entries) != self.segment_ids:
            raise RuntimeError("Prefix anchor entries do not match stack signature")
        for entry in anchors.entries:
            self.get(entry.segment_id).kv.accumulate_anchor_gradients(entry, consume=consume)

    def get_new_kv_gradients(self, segment_id: SegmentId) -> Mapping[int, KVPair]:
        return self.get(segment_id).gradients

    def assert_empty(self) -> None:
        if self._entries:
            raise RuntimeError(f"KV stack is not empty: {self.segment_ids}")

    def clear(self) -> None:
        """Release all path KV/dKV after an unrecoverable execution failure."""
        for entry in self._entries.values():
            entry.kv.release()
        self._entries.clear()
