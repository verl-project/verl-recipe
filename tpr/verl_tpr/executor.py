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

"""Push/pop execution of one TPR segment against a Megatron GPT model."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import Any, Protocol

import torch
from torch import Tensor, nn

from verl_tpr.context import KVPair, TPRAttentionContext, use_tpr_attention_context

from .ring import (
    RingCPBackend,
    SequenceShard,
    make_ring_sequence_shard,
    resolve_cp_group,
)
from .ring_layout import make_cp1_sp_sequence_shard
from .state import KVStack, PastKVAnchors
from .tree import SegmentId, SegmentPlan, SegmentSpec


class _RotaryEmbedding(Protocol):
    def __call__(
        self,
        max_seq_len: int,
        offset: int = 0,
        packed_seq: bool = False,
        cp_group: object | None = None,
    ) -> Tensor: ...


class _UnshardedContextParallelGroup:
    """Size-one sentinel that prevents RotaryEmbedding using its bound CP group."""

    @staticmethod
    def size() -> int:
        return 1


_UNSHARDED_CP_GROUP = _UnshardedContextParallelGroup()


@contextmanager
def disable_bound_context_parallel_sharding(rotary_embedding: _RotaryEmbedding):
    """Temporarily stop implicit CP slicing by a bound RotaryEmbedding group.

    GPTModel builds a rotary embedding in ``_preprocess`` even though TPR
    attention consumes the rank-local embedding stored in its execution
    context.  For padded, non-divisible sequence lengths that unused build can
    fail in MindSpeed before TPR attention is reached.  Override the embedding's
    bound group only for the model forward and restore it on every exit path.
    """

    if not hasattr(rotary_embedding, "cp_group"):
        yield
        return
    cp_group = rotary_embedding.cp_group
    rotary_embedding.cp_group = _UNSHARDED_CP_GROUP
    try:
        yield
    finally:
        rotary_embedding.cp_group = cp_group


def build_suffix_rotary_pos_emb(
    rotary_embedding: _RotaryEmbedding,
    *,
    prefix_length: int,
    suffix_length: int,
    disable_context_parallel_sharding: bool = False,
) -> Tensor:
    """Build standard 1D RoPE frequencies for positions ``[P, P+b)``.

    This delegates frequency generation to the model's existing Megatron
    ``RotaryEmbedding`` instance, preserving its rotary dimension, base,
    interpolation, interleaving, and scaling settings.  The MVP deliberately
    requests neither packed-sequence nor context-parallel sharding.
    """

    _validate_length("prefix_length", prefix_length, allow_zero=True)
    _validate_length("suffix_length", suffix_length, allow_zero=False)
    if not callable(rotary_embedding):
        raise TypeError("rotary_embedding must be callable")

    rotary_pos_emb = rotary_embedding(
        suffix_length,
        offset=prefix_length,
        packed_seq=False,
        cp_group=_UNSHARDED_CP_GROUP if disable_context_parallel_sharding else None,
    )
    if not isinstance(rotary_pos_emb, Tensor):
        raise TypeError(
            f"TPR RoPE MVP requires standard RotaryEmbedding to return one tensor; got {type(rotary_pos_emb).__name__}"
        )
    if rotary_pos_emb.ndim != 4:
        raise ValueError(
            f"suffix rotary embedding must have shape [suffix, 1, 1, rotary_dim], got {rotary_pos_emb.shape}"
        )
    if rotary_pos_emb.shape[0] != suffix_length:
        raise ValueError(
            f"suffix rotary embedding sequence length must be {suffix_length}, got {rotary_pos_emb.shape[0]}"
        )
    if rotary_pos_emb.shape[1] != 1 or rotary_pos_emb.shape[2] != 1:
        raise ValueError(f"TPR RoPE MVP requires singleton batch/head broadcast dimensions, got {rotary_pos_emb.shape}")
    if rotary_pos_emb.shape[3] <= 0:
        raise ValueError("suffix rotary embedding must have a positive rotary dimension")
    return rotary_pos_emb


def build_sharded_rotary_pos_emb(
    rotary_embedding: _RotaryEmbedding,
    *,
    position_start: int,
    shard: SequenceShard,
    disable_context_parallel_sharding: bool = False,
) -> Tensor:
    """Build RoPE in the local-token order described by ``shard``."""

    _validate_length("position_start", position_start, allow_zero=True)
    if not isinstance(shard, SequenceShard):
        raise TypeError(f"shard must implement SequenceShard, got {type(shard).__name__}")
    chunks = []
    for physical_start, physical_end in shard.physical_global_ranges:
        valid_end = min(physical_end, shard.global_length)
        valid_length = max(0, valid_end - physical_start)
        physical_length = physical_end - physical_start
        chunk = build_suffix_rotary_pos_emb(
            rotary_embedding,
            prefix_length=position_start + min(physical_start, shard.global_length - 1),
            suffix_length=max(valid_length, 1),
            disable_context_parallel_sharding=disable_context_parallel_sharding,
        )
        if valid_length < physical_length:
            padding = chunk[-1:].expand(physical_length - valid_length, *chunk.shape[1:])
            chunk = torch.cat((chunk if valid_length else chunk[:0], padding), dim=0)
        chunks.append(chunk)
    rotary_pos_emb = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)
    if rotary_pos_emb.shape[0] != shard.local_length:
        raise RuntimeError(
            f"sharded rotary embedding length must be {shard.local_length}, got {rotary_pos_emb.shape[0]}"
        )
    return rotary_pos_emb


def _validate_length(name: str, value: int, *, allow_zero: bool) -> None:
    lower_bound = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < lower_bound:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a {qualifier} integer, got {value!r}")


@dataclass(frozen=True, slots=True)
class PreparedReplayModel:
    """Batch-scoped static model facts and live stochastic-setting references.

    The model structure must not be replaced during a batch. Settings on the
    registered configs and Dropout modules are rechecked before execution.
    """

    model: nn.Module
    device: torch.device
    layer_numbers: tuple[int, ...]
    configs: tuple[Any, ...]
    dropouts: tuple[nn.Module, ...]

    def validate(self) -> None:
        _validate_replay_settings(self.configs, self.dropouts)


def prepare_replay_model(
    model: nn.Module,
    *,
    expected_layer_numbers: tuple[int, ...] | None = None,
) -> PreparedReplayModel:
    """Inspect the model once per PPO batch, rather than twice for every tree."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be torch.nn.Module, got {type(model).__name__}")
    if not hasattr(model, "rotary_pos_emb") or not callable(model.rotary_pos_emb):
        raise TypeError("model must expose a callable rotary_pos_emb")
    layers = tuple(_infer_layer_numbers(model) if expected_layer_numbers is None else expected_layer_numbers)
    if not layers or any(not isinstance(layer, int) or isinstance(layer, bool) or layer <= 0 for layer in layers):
        raise ValueError(f"expected_layer_numbers must contain positive integers, got {layers}")
    if len(set(layers)) != len(layers):
        raise ValueError(f"expected_layer_numbers contains duplicates: {layers}")
    configs, dropouts = _replay_settings(model)
    prepared = PreparedReplayModel(model, _model_device(model), tuple(sorted(layers)), configs, dropouts)
    prepared.validate()
    return prepared


def validate_replay_plan(
    plan: SegmentPlan,
    prepared_model: PreparedReplayModel,
    *,
    cp_size: int = 1,
    active_segment_ids: frozenset[SegmentId] | None = None,
) -> None:
    """Validate each Segment's physical sequence alignment before collectives.

    CP>1 uses joint Zigzag Ring / native SP alignment. CP1 owns the full
    Segment and uses contiguous trailing padding only when native SP is on.
    """
    config = getattr(prepared_model.model, "config", None)
    tp_size = getattr(config, "tensor_model_parallel_size", 1)
    if tp_size <= 1 or not getattr(config, "sequence_parallel", False):
        return
    for segment in plan.segments.values():
        if active_segment_ids is not None and segment.segment_id not in active_segment_ids:
            continue
        if cp_size > 1:
            physical_length = make_ring_sequence_shard(
                segment.length,
                cp_rank=0,
                cp_size=cp_size,
                sp_tp_size=tp_size,
            ).local_length
        else:
            physical_length = make_cp1_sp_sequence_shard(
                segment.length,
                sp_tp_size=tp_size,
            ).local_length
        if physical_length % tp_size:
            raise ValueError(
                f"segment {segment.segment_id} logical_length={segment.length} "
                f"has physical local_length={physical_length} not divisible by TP={tp_size}"
            )


class SegmentExecutor:
    """Execute graph-free Push and gradient-carrying Pop for a SegmentPlan.

    One iterative DFS owns the traversal and KV lifetime. The engine adapter
    owns gradient synchronization and optimizer steps across the forest.
    """

    def __init__(
        self,
        model: nn.Module,
        plan: SegmentPlan,
        *,
        expected_layer_numbers: tuple[int, ...] | None = None,
        kv_stack: KVStack | None = None,
        loss_scale_func: Callable[[Tensor], Tensor] | None = None,
        cp_group: Any | None = None,
        ring_runtime: RingCPBackend | None = None,
        segment_loss_fn: Callable[[SegmentSpec, Tensor], tuple[Tensor, dict[str, Any]]] | None = None,
        segment_loss_term_counts: Mapping[SegmentId, int] | None = None,
        active_segment_ids: frozenset[SegmentId] | None = None,
        prepared_model: PreparedReplayModel | None = None,
    ) -> None:
        if not isinstance(plan, SegmentPlan):
            raise TypeError(f"plan must be SegmentPlan, got {type(plan).__name__}")
        self._external_preparation = prepared_model is not None
        if prepared_model is None:
            prepared_model = prepare_replay_model(model, expected_layer_numbers=expected_layer_numbers)
        elif prepared_model.model is not model:
            raise ValueError("prepared_model belongs to a different model")
        elif (
            expected_layer_numbers is not None and tuple(sorted(expected_layer_numbers)) != prepared_model.layer_numbers
        ):
            raise ValueError("expected_layer_numbers differ from prepared_model")
        self.prepared_model = prepared_model

        if not callable(segment_loss_fn):
            raise TypeError("Production TPR requires a callable segment_loss_fn")
        counts = dict(segment_loss_term_counts or {})
        if not counts or any(segment_id not in plan.segments or count < 0 for segment_id, count in counts.items()):
            raise ValueError("invalid PPO segment objective counts")
        if active_segment_ids is not None:
            active_segment_ids = frozenset(active_segment_ids)
            if (
                not active_segment_ids.issubset(plan.segments)
                or any(
                    plan.get(segment_id).parent_id not in active_segment_ids
                    for segment_id in active_segment_ids
                    if segment_id != plan.root_id
                )
                or any(count and segment_id not in active_segment_ids for segment_id, count in counts.items())
            ):
                raise ValueError("active_segment_ids must contain objective owners and all their ancestors")
        self.active_segment_ids = active_segment_ids
        self.segment_loss_fn = segment_loss_fn
        self.segment_loss_term_counts = counts
        self.segment_loss_metrics: list[tuple[SegmentId, dict[str, Any]]] = []

        self.model = model
        self.plan = plan
        self.expected_layer_numbers = prepared_model.layer_numbers
        self.kv_stack = KVStack() if kv_stack is None else kv_stack
        self.loss_scale_func = loss_scale_func
        self.cp_group = cp_group
        config = getattr(prepared_model.model, "config", None)
        self.sp_tp_size = (
            getattr(config, "tensor_model_parallel_size", 1) if getattr(config, "sequence_parallel", False) else 1
        )
        if cp_group is None:
            if ring_runtime is not None:
                raise ValueError("ring_runtime requires a cp_group")
            self.cp_size, self.cp_rank = 1, 0
            self.ring_runtime = None
        else:
            self.cp_size, self.cp_rank = resolve_cp_group(cp_group)
            if self.cp_size <= 1:
                raise ValueError(f"cp_group must contain more than one rank, got {self.cp_size}")
            if ring_runtime is not None:
                if not isinstance(ring_runtime, RingCPBackend):
                    raise TypeError("ring_runtime must be a RingCPBackend")
                if (ring_runtime.parallel_size, ring_runtime.parallel_rank) != (self.cp_size, self.cp_rank):
                    raise ValueError("RingCPBackend coordinates do not match initialized CP group")
                if ring_runtime.sp_tp_size != self.sp_tp_size:
                    raise ValueError(
                        "RingCPBackend TP/SP alignment does not match the prepared model: "
                        f"expected {self.sp_tp_size}, got {ring_runtime.sp_tp_size}"
                    )
                self.ring_runtime = ring_runtime
            else:
                self.ring_runtime = RingCPBackend(
                    cp_group,
                    parallel_size=self.cp_size,
                    parallel_rank=self.cp_rank,
                    sp_tp_size=self.sp_tp_size,
                )
            for segment in plan.segments.values():
                try:
                    self.ring_runtime.validate_segment_length(segment.length)
                except ValueError as exc:
                    raise ValueError(f"segment {segment.segment_id} {exc}") from exc
        validate_replay_plan(plan, prepared_model, cp_size=self.cp_size, active_segment_ids=active_segment_ids)
        self._failed = False
        self._ran = False
        self._operation_depth = 0

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def cp_enabled(self) -> bool:
        return self.cp_group is not None

    def run(self) -> Tensor:
        """Execute each branch, then replay its ancestors in postorder."""
        self._ensure_healthy()
        if self._ran:
            raise RuntimeError("SegmentExecutor.run is single-use")
        losses = []
        with self._operation():
            self._ran = True
            self.kv_stack.assert_empty()
            if self.active_segment_ids == frozenset():
                return torch.zeros((), device=self.prepared_model.device)
            pending = [(self.plan.root_id, False)]
            while pending:
                segment_id, exiting = pending.pop()
                children = () if exiting else self._children_of(segment_id)
                if exiting:
                    losses.append(self.pop(segment_id))
                elif children:
                    self.push(segment_id)
                    pending.append((segment_id, True))
                    pending.extend((child.segment_id, False) for child in reversed(children))
                else:
                    losses.append(self.visit_leaf(segment_id))
            self.kv_stack.assert_empty()
            total = losses[0]
            for loss in losses[1:]:
                total = total + loss
            return total

    def push(self, segment_id: SegmentId) -> None:
        """Save graph-free per-layer Prefix KV without owning loss."""

        with self._operation():
            segment = self.plan.get(segment_id)
            context, logits = self._forward(
                segment,
                past_key_values={} if self.cp_enabled else self.kv_stack.build_past_key_values(),
                no_grad=True,
            )
            del logits
            self._assert_collected_layers(context)
            # The Push context already detaches and compacts each layer's KV.
            self.kv_stack.push(segment, context.new_key_values, shard=self._segment_shard(segment))

    def pop(self, segment_id: SegmentId) -> Tensor:
        """Recompute current KV and relay gradients from descendant Segments."""

        with self._operation():
            entry = self.kv_stack.top()
            if entry.segment.segment_id != segment_id:
                raise RuntimeError(f"cannot pop segment {segment_id}: stack top is {entry.segment.segment_id}")

            relayed_gradients = dict(entry.gradients)
            if relayed_gradients and tuple(relayed_gradients) != self.expected_layer_numbers:
                raise RuntimeError(
                    f"segment {segment_id} relayed KV gradient layers must be {self.expected_layer_numbers}, "
                    f"got {tuple(relayed_gradients)}"
                )
            popped_entry = self.kv_stack.pop(segment_id)
            segment = popped_entry.segment
            del entry
            # The popped segment's cached KV is dead before recompute: its
            # relayed gradients and metadata were copied above, while the
            # replay consumes only the remaining ancestor path as past KV.
            # Release it for CP=1 as well as CP>1 so cached current KV does not
            # overlap with the graph-connected KV recreated by this forward.
            popped_entry.kv.release()
            return self._backward_segment(segment, relayed_gradients)

    def visit_leaf(self, segment_id: SegmentId) -> Tensor:
        """Forward/backward one leaf without writing its new KV to the path stack."""

        with self._operation():
            segment = self.plan.get(segment_id)
            if self._children_of(segment_id):
                raise ValueError(f"segment {segment_id} is not a leaf")

            expected_parent = self.kv_stack.top().segment.segment_id if len(self.kv_stack) else None
            if segment.parent_id != expected_parent:
                raise RuntimeError(
                    f"cannot visit leaf {segment_id}: expected parent on stack {segment.parent_id}, "
                    f"got {expected_parent}"
                )
            if segment.prefix_length != self.kv_stack.prefix_length:
                raise RuntimeError(
                    f"cannot visit leaf {segment_id}: stack prefix length is {self.kv_stack.prefix_length}, "
                    f"got {segment.prefix_length}"
                )

            return self._backward_segment(segment, leaf=True)

    def _backward_segment(
        self,
        segment: SegmentSpec,
        relayed_gradients: Mapping[int, KVPair] | None = None,
        *,
        leaf: bool = False,
    ) -> Tensor:
        """Replay against independent ancestor anchors, then return their dKV."""
        relayed_gradients = relayed_gradients or {}
        anchors = self.kv_stack.build_past_anchors()
        context, logits = self._forward(
            segment,
            no_grad=False,
            past_key_values={} if self.cp_enabled else anchors.key_values,
            sharded_past_anchors=anchors if self.cp_enabled else None,
        )
        self._assert_collected_layers(context)
        normalized_loss = self._compute_loss(segment, logits)
        count = self._owned_loss_count(segment)
        roots, gradients = [], []
        # Empty CP objectives must still traverse Attention's collectives.
        # Frozen current KV, however, has no graph to relay through.
        if leaf or count or self.cp_enabled:
            loss_root = self._prepare_backward_loss(
                normalized_loss,
                logits=logits,
                has_owned_loss=count > 0,
            )
            if loss_root.requires_grad:
                roots.append(loss_root)
                gradients.append(None)
        for layer, gradient in relayed_gradients.items():
            for root, grad in zip(context.new_key_values[layer], gradient, strict=True):
                if root.requires_grad:
                    roots.append(root)
                    gradients.append(grad)
        if roots:
            torch.autograd.backward(roots, grad_tensors=gradients)
            self.kv_stack.accumulate_anchor_gradients(anchors, consume=True)
        return normalized_loss.detach()

    def _forward(
        self,
        segment: SegmentSpec,
        *,
        past_key_values,
        no_grad: bool,
        sharded_past_anchors: PastKVAnchors | None = None,
    ) -> tuple[TPRAttentionContext, Tensor]:
        device = self.prepared_model.device
        shard = self._segment_shard(segment)
        local_token_ids = shard.select(segment.token_ids)
        input_ids = local_token_ids.to(device=device).unsqueeze(0)
        position_ids = shard.global_indices(device=device).add(segment.position_start).unsqueeze(0)
        attention_backend = None
        if self.cp_enabled:
            if self.ring_runtime is None:
                raise RuntimeError("Ring CP execution is missing its runtime")
            attention_backend = self.ring_runtime.make_attention_backend(
                self.kv_stack,
                expected_layer_numbers=self.expected_layer_numbers,
                current_shard=shard,
                past_anchors=sharded_past_anchors,
            )
        context = TPRAttentionContext(
            prefix_length=segment.prefix_length,
            suffix_length=segment.length,
            past_key_values=past_key_values,
            suffix_rotary_pos_emb=build_sharded_rotary_pos_emb(
                self.model.rotary_pos_emb,
                position_start=segment.position_start,
                shard=shard,
                disable_context_parallel_sharding=self.cp_enabled,
            ),
            attention_backend=attention_backend,
            compact_new_kv=no_grad,
            local_physical_suffix_length=(shard.local_length if not self.cp_enabled and self.sp_tp_size > 1 else None),
        )
        rope_context = (
            disable_bound_context_parallel_sharding(self.model.rotary_pos_emb) if self.cp_enabled else nullcontext()
        )
        with torch.set_grad_enabled(not no_grad), rope_context, use_tpr_attention_context(context):
            logits = self.model(input_ids=input_ids, position_ids=position_ids, attention_mask=None)
        if not isinstance(logits, Tensor) or logits.ndim != 3:
            raise ValueError(f"model must return logits [batch, sequence, vocab], got {type(logits).__name__}")
        expected_shape = (1, shard.local_length)
        if tuple(logits.shape[:2]) != expected_shape:
            raise ValueError(f"model logits must start with shape {expected_shape}, got {tuple(logits.shape)}")
        return context, logits

    def _compute_loss(self, segment: SegmentSpec, logits: Tensor) -> Tensor:
        """Delegate all policy math to VERL's original PPO callback."""
        if not self._owned_loss_count(segment):
            return _connected_zero(logits)
        loss, metrics = self.segment_loss_fn(segment, logits)
        if not isinstance(loss, Tensor) or loss.numel() != 1 or not loss.requires_grad:
            raise TypeError("PPO objective must return a differentiable scalar Tensor")
        self.segment_loss_metrics.append((segment.segment_id, metrics))
        return loss

    def _owned_loss_count(self, segment: SegmentSpec) -> int:
        return self.segment_loss_term_counts.get(segment.segment_id, 0)

    def _prepare_backward_loss(
        self,
        normalized_loss: Tensor,
        *,
        logits: Tensor,
        has_owned_loss: bool,
    ) -> Tensor:
        backward_loss = normalized_loss * self.cp_size if has_owned_loss else _connected_zero(logits)
        if self.loss_scale_func is not None:
            backward_loss = self.loss_scale_func(backward_loss)
        if not isinstance(backward_loss, Tensor) or backward_loss.numel() != 1:
            raise TypeError("loss_scale_func must return a scalar tensor")
        return backward_loss

    def _segment_shard(self, segment: SegmentSpec) -> SequenceShard:
        if not self.cp_enabled:
            if self.sp_tp_size > 1:
                return make_cp1_sp_sequence_shard(
                    segment.length,
                    sp_tp_size=self.sp_tp_size,
                )
            return SequenceShard.full(segment.length)
        if self.ring_runtime is None:
            raise RuntimeError("CP execution is missing its backend")
        return self.ring_runtime.make_sequence_shard(segment.length)

    def _ensure_healthy(self) -> None:
        if self._failed:
            raise RuntimeError("SegmentExecutor is failed and cannot continue")

    def _children_of(self, segment_id: SegmentId) -> tuple[SegmentSpec, ...]:
        children = self.plan.children_of(segment_id)
        if self.active_segment_ids is None:
            return children
        return tuple(child for child in children if child.segment_id in self.active_segment_ids)

    @contextmanager
    def _operation(self):
        self._ensure_healthy()
        outermost = self._operation_depth == 0
        self._operation_depth += 1
        try:
            if outermost:
                if self._external_preparation:
                    self.prepared_model.validate()
                else:
                    # Standalone users may change modules between construction
                    # and execution; the adapter's batch-scoped token is stricter.
                    _assert_deterministic_replay(self.model)
            yield
        except BaseException:
            self._failed = True
            if outermost:
                self.kv_stack.clear()
            raise
        finally:
            self._operation_depth -= 1

    def _assert_collected_layers(self, context: TPRAttentionContext) -> None:
        context.assert_new_kv_layers(self.expected_layer_numbers)


def _model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:
        raise ValueError("model must contain at least one parameter") from exc


def _connected_zero(logits: Tensor) -> Tensor:
    # A zero-width view keeps the entire logits graph reachable (including CP
    # collectives), without casting/reducing the vocabulary or reading NaNs.
    return logits[..., :0].sum()


def _assert_deterministic_replay(model: nn.Module) -> None:
    """Push and Pop must see the same KV; RNG replay is not implemented."""
    _validate_replay_settings(*_replay_settings(model))


def _replay_settings(model: nn.Module) -> tuple[tuple[Any, ...], tuple[nn.Module, ...]]:
    configs, dropouts = {}, []
    for module in model.modules():
        config = getattr(module, "config", None)
        if config is not None:
            configs[id(config)] = config
        if isinstance(module, nn.modules.dropout._DropoutNd):
            dropouts.append(module)
    return tuple(configs.values()), tuple(dropouts)


def _validate_replay_settings(configs, dropouts) -> None:
    for config in configs:
        for name in ("hidden_dropout", "embedding_dropout", "attention_dropout"):
            value = getattr(config, name, 0.0)
            if value is not None and float(value) != 0.0:
                raise ValueError(f"TPR replay requires {name}=0; stochastic Push/Pop is unsupported")
    for module in dropouts:
        if module.training and module.p != 0:
            raise ValueError("TPR replay requires training Dropout p=0; stochastic Push/Pop is unsupported")


def _infer_layer_numbers(model: nn.Module) -> tuple[int, ...]:
    decoder = getattr(model, "decoder", None)
    layers = getattr(decoder, "layers", None)
    if layers is None:
        raise ValueError("cannot infer TPR layer numbers; pass expected_layer_numbers explicitly")
    result = []
    for layer in layers:
        attention = getattr(layer, "self_attention", None)
        if getattr(attention, "tpr_state_kind", "kv") != "kv":
            raise NotImplementedError("Production TPR only supports Dense KV attention")
        layer_number = getattr(attention, "layer_number", None)
        if layer_number is None:
            raise ValueError("cannot infer layer_number from model.decoder.layers")
        result.append(layer_number)
    return tuple(result)
