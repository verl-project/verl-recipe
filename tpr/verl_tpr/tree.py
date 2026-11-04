# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Token-LCP prefix sharing and native PPO query ownership."""

from __future__ import annotations

import hashlib
import json
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import torch
from torch import Tensor

SegmentId = int


def _validate_segment_id(value: int, *, field_name: str = "segment_id") -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer, got {value!r}")


@dataclass(frozen=True, slots=True)
class SegmentSpec:
    segment_id: SegmentId
    parent_id: SegmentId | None
    token_ids: Tensor
    position_start: int
    prefix_length: int

    def __post_init__(self) -> None:
        for name, value in (
            ("segment_id", self.segment_id),
            ("position_start", self.position_start),
            ("prefix_length", self.prefix_length),
        ):
            _validate_segment_id(value, field_name=name)
        if self.parent_id is not None:
            _validate_segment_id(self.parent_id, field_name="parent_id")
        ids = self.token_ids
        if (
            self.prefix_length != self.position_start
            or not isinstance(ids, Tensor)
            or ids.device.type != "cpu"
            or ids.dtype != torch.long
            or ids.ndim != 1
            or not ids.numel()
        ):
            raise ValueError("invalid SegmentSpec token range or token_ids")
        if bool((ids < 0).any()):
            raise ValueError("token_ids must be non-negative")
        object.__setattr__(self, "token_ids", ids.detach().clone())

    @property
    def length(self) -> int:
        return self.token_ids.numel()

    @property
    def position_end(self) -> int:
        return self.position_start + self.length


@dataclass(frozen=True, slots=True)
class SegmentPlan:
    segments: Mapping[SegmentId, SegmentSpec] | Sequence[SegmentSpec]
    root_id: SegmentId
    _children: Mapping[SegmentId, tuple[SegmentId, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _validate_segment_id(self.root_id, field_name="root_id")
        source = list(self.segments.values() if isinstance(self.segments, Mapping) else self.segments)
        if not all(isinstance(item, SegmentSpec) for item in source):
            raise TypeError("segments must contain SegmentSpec objects")
        segments = {item.segment_id: item for item in source}
        if len(segments) != len(source):
            raise ValueError("segment IDs must be unique")
        roots = [key for key, item in segments.items() if item.parent_id is None]
        if roots != [self.root_id]:
            raise ValueError("invalid SegmentPlan root")
        children = {key: [] for key in segments}
        for item in source:
            if item.parent_id is None:
                if item.position_start != 0:
                    raise ValueError("root segment must start at position zero")
                continue
            parent = segments.get(item.parent_id)
            if parent is None or item.position_start != parent.position_end:
                raise ValueError(f"segment {item.segment_id} must start at parent end")
            children[item.parent_id].append(item.segment_id)
        object.__setattr__(self, "segments", MappingProxyType(segments))
        object.__setattr__(self, "_children", MappingProxyType({key: tuple(value) for key, value in children.items()}))

    def get(self, segment_id: SegmentId) -> SegmentSpec:
        return self.segments[segment_id]

    def children_of(self, segment_id: SegmentId) -> tuple[SegmentSpec, ...]:
        return tuple(self.segments[key] for key in self._children[segment_id])

    def path_to(self, segment_id: SegmentId) -> tuple[SegmentSpec, ...]:
        path = []
        while segment_id is not None:
            item = self.get(segment_id)
            path.append(item)
            segment_id = item.parent_id
        return tuple(reversed(path))


@dataclass(frozen=True, slots=True)
class SegmentObjectiveRef:
    segment_id: int
    query_offset: int
    target_token_id: int
    sample_row: int
    response_offset: int


@dataclass(frozen=True, slots=True)
class TreeExecutionPlan:
    uid: str
    tree_index: int
    member_rows: tuple[int, ...]
    segment_plan: SegmentPlan
    objective_refs: tuple[SegmentObjectiveRef, ...]
    member_uids: tuple[str, ...] = ()
    active_segment_ids: frozenset[SegmentId] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        active = set()
        for ref in self.objective_refs:
            segment_id = ref.segment_id
            while segment_id is not None and segment_id not in active:
                active.add(segment_id)
                segment_id = self.segment_plan.get(segment_id).parent_id
        object.__setattr__(self, "active_segment_ids", frozenset(active))

    @property
    def key(self) -> str:
        return f"{self.uid}:{self.tree_index}"

    @property
    def logical_loss_tokens(self) -> int:
        return len(self.objective_refs)


@dataclass(frozen=True, slots=True)
class ForestExecutionPlan:
    trees: tuple[TreeExecutionPlan, ...]
    logical_loss_tokens: int

    def __post_init__(self) -> None:
        total = sum(tree.logical_loss_tokens for tree in self.trees)
        if total != self.logical_loss_tokens:
            raise ValueError("forest logical_loss_tokens disagrees with objective refs")

    @property
    def segment_count(self) -> int:
        return sum(len(tree.segment_plan.segments) for tree in self.trees)


def _rows(batch: Any, key: str) -> list[Tensor]:
    try:
        value = batch[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"batch is missing {key!r}") from exc
    values = list(value) if isinstance(value, (list, tuple)) else list(value.unbind())
    if not all(isinstance(item, Tensor) and item.ndim == 1 for item in values):
        raise ValueError(f"{key!r} must contain one-dimensional tensor rows")
    return values


@dataclass(slots=True)
class _TreeNode:
    end: int
    source_row: int
    terminal_rows: list[int] = field(default_factory=list)
    children: list[_TreeNode] = field(default_factory=list)


def _lcp(left: Tensor, right: Tensor) -> int:
    length = min(left.numel(), right.numel())
    if not length:
        return 0
    equal = left[:length] == right[:length]
    return length if bool(equal.all()) else int((~equal).to(torch.int32).argmax())


def _clusters(rows: list[tuple[int, Tensor, str]], minimum: int):
    ordered = sorted(rows, key=lambda item: (tuple(item[1].tolist()), item[0]))
    groups, start = [], 0
    for index in range(1, len(ordered)):
        if _lcp(ordered[index - 1][1], ordered[index][1]) < minimum:
            groups.append(ordered[start:index])
            start = index
    groups.append(ordered[start:])
    return sorted(groups, key=lambda group: min(item[0] for item in group))


def _build_lcp_tree(rows: list[tuple[int, Tensor, str]]) -> _TreeNode:
    """Build a compressed prefix tree from sorted rows and adjacent Token LCPs."""
    root = _TreeNode(0, rows[0][0])
    stack, previous = [root], None
    for row, tokens, _uid in rows:
        if previous is not None:
            shared = _lcp(previous, tokens)
            while stack[-1].end > shared:
                stack.pop()
            if stack[-1].end < shared:
                parent, child = stack[-1], stack[-1].children[-1]
                node = _TreeNode(shared, child.source_row, children=[child])
                parent.children[-1] = node
                stack.append(node)
        if tokens.numel() == stack[-1].end:
            stack[-1].terminal_rows.append(row)
        else:
            leaf = _TreeNode(tokens.numel(), row, [row])
            stack[-1].children.append(leaf)
            stack.append(leaf)
        previous = tokens
    if len(root.children) != 1:
        raise RuntimeError("Token-LCP cluster does not have one common root prefix")
    return root.children[0]


def _compress(
    node: _TreeNode,
    parent_id: int | None,
    start: int,
    inputs: list[Tensor],
    minimum: int,
    segments: list[SegmentSpec],
    paths: dict[int, tuple[int, ...]],
    path: tuple[int, ...],
) -> None:
    while not node.terminal_rows and len(node.children) == 1:
        node = node.children[0]
    branches = len(node.children) + int(bool(node.terminal_rows))
    if node.end - start < minimum and (branches > 1 or len(node.terminal_rows) > 1):
        for row in node.terminal_rows:
            sid = len(segments)
            segments.append(SegmentSpec(sid, parent_id, inputs[row][start : node.end], start, start))
            paths[row] = (*path, sid)
        for child in node.children:
            _compress(child, parent_id, start, inputs, minimum, segments, paths, path)
        return
    sid = len(segments)
    segments.append(SegmentSpec(sid, parent_id, inputs[node.source_row][start : node.end], start, start))
    current = (*path, sid)
    for row in node.terminal_rows:
        paths[row] = current
    for child in node.children:
        _compress(child, sid, node.end, inputs, minimum, segments, paths, current)


def build_tree_execution_plans(
    keys: list[str] | tuple[str, ...],
    batch: Any,
    *,
    require_loss_mask_alignment: bool = False,
    min_shared_prefix_tokens: int = 1024,
) -> ForestExecutionPlan:
    """Build a minibatch-wide forest using exact Token-ID prefixes."""
    if (
        not isinstance(min_shared_prefix_tokens, int)
        or isinstance(min_shared_prefix_tokens, bool)
        or min_shared_prefix_tokens < 1
    ):
        raise ValueError("min_shared_prefix_tokens must be a positive integer")
    inputs, masks = _rows(batch, "input_ids"), _rows(batch, "response_mask")
    if len(keys) != len(inputs) or len(keys) != len(masks) or len(set(keys)) != len(keys):
        raise ValueError("keys and batch rows must match uniquely")
    responses = _rows(batch, "responses") if "responses" in batch else None
    loss_masks = _rows(batch, "loss_mask") if require_loss_mask_alignment else None
    if any(values is not None and len(values) != len(keys) for values in (responses, loss_masks)):
        raise ValueError("optional batch field rows do not match keys")

    token_rows, selected, cpu_inputs = [], [], []
    for row, (key, tokens, mask) in enumerate(zip(keys, inputs, masks, strict=True)):
        if (
            not isinstance(key, str)
            or not tokens.numel()
            or tokens.dtype == torch.bool
            or torch.is_floating_point(tokens)
            or torch.is_complex(tokens)
            or bool((tokens < 0).any())
        ):
            raise ValueError(f"invalid trajectory key or input_ids[{row}]")
        fields = key.rsplit("_", 2) if isinstance(key, str) else []
        if len(fields) != 3 or not all(fields) or not fields[2].isdigit():
            raise ValueError(f"unexpected trajectory key format: {key!r}")
        cpu = tokens.detach().to(device="cpu", dtype=torch.long)
        cpu_inputs.append(cpu)
        if mask.numel() > tokens.numel():
            raise ValueError(f"response_mask[{row}] is longer than input_ids[{row}]")
        response_start = tokens.numel() - mask.numel()
        if responses is not None and (
            responses[row].numel() != mask.numel() or not torch.equal(tokens[response_start:], responses[row])
        ):
            raise ValueError(f"response is not an input_ids suffix on row {row}")
        offsets = torch.nonzero(mask.to(torch.bool), as_tuple=False).flatten().tolist()
        if offsets and response_start + offsets[0] == 0:
            raise ValueError("supervised response token has no preceding query token")
        if loss_masks is not None:
            loss = loss_masks[row]
            if loss.numel() == tokens.numel():
                if bool(loss[:response_start].to(bool).any()):
                    raise ValueError(f"row {row} loss_mask includes prompt tokens")
                loss = loss[response_start:]
            if loss.numel() != mask.numel() or not torch.equal(loss.to(bool), mask.to(bool)):
                raise ValueError(f"row {row} response_mask and loss_mask disagree")
        token_rows.append((row, cpu, fields[0]))
        selected.append(offsets)

    result = []
    for tree_index, cluster in enumerate(_clusters(token_rows, min_shared_prefix_tokens)):
        root = _build_lcp_tree(cluster)
        segments, paths = [], {}
        _compress(root, None, 0, cpu_inputs, min_shared_prefix_tokens, segments, paths, ())
        members = tuple(sorted(item[0] for item in cluster))
        uids = tuple(dict.fromkeys(token_rows[row][2] for row in members))
        plan, refs = SegmentPlan(segments, root_id=0), []
        for row in members:
            response_start = inputs[row].numel() - masks[row].numel()
            path = paths[row]
            ends = [plan.get(segment_id).position_end for segment_id in path]
            for offset in selected[row]:
                target = response_start + offset
                segment_id = path[bisect_right(ends, target - 1)]
                refs.append(
                    SegmentObjectiveRef(
                        segment_id,
                        target - 1 - plan.get(segment_id).position_start,
                        int(cpu_inputs[row][target]),
                        row,
                        offset,
                    )
                )
        result.append(TreeExecutionPlan("__minibatch__", tree_index, members, plan, tuple(refs), uids))
    return ForestExecutionPlan(tuple(result), sum(map(len, selected)))


def _write_record(hasher, record) -> None:
    hasher.update(json.dumps(record, separators=(",", ":")).encode())


def _write_plan(hasher, plan: SegmentPlan) -> None:
    _write_record(hasher, plan.root_id)
    for segment in plan.segments.values():
        _write_record(
            hasher,
            (
                segment.segment_id,
                segment.parent_id,
                segment.position_start,
                segment.length,
            ),
        )
        hasher.update(segment.token_ids.contiguous().numpy().tobytes())


def digest_segment_plan(plan: SegmentPlan) -> str:
    hasher = hashlib.sha256(b"TPR-PPO-tree-v3")
    _write_plan(hasher, plan)
    return hasher.hexdigest()


def digest_forest_plan(forest: ForestExecutionPlan) -> str:
    hasher = hashlib.sha256(b"TPR-PPO-tree-v3")
    _write_record(hasher, forest.logical_loss_tokens)
    for tree in forest.trees:
        _write_record(hasher, (tree.uid, tree.tree_index, tree.member_rows, tree.member_uids or (tree.uid,)))
        _write_plan(hasher, tree.segment_plan)
        for ref in tree.objective_refs:
            _write_record(
                hasher,
                (
                    ref.segment_id,
                    ref.query_offset,
                    ref.target_token_id,
                    ref.sample_row,
                    ref.response_offset,
                ),
            )
    return hasher.hexdigest()


def assert_tp_plan_agreement(digest: str, *, tp_size: int, cp_size: int = 1) -> None:
    """Reject divergent plans on the native TP or TP×CP group before forward."""
    if tp_size < 1 or cp_size < 1:
        raise ValueError("tp_size and cp_size must be positive")
    size = tp_size * cp_size
    if size == 1:
        return
    if len(digest) != 64:
        raise ValueError("SHA256 digest must be a 64-character hex string")
    import torch.distributed as dist
    from megatron.core import parallel_state

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("TPR distributed process group is not initialized")
    group = (
        parallel_state.get_tensor_and_context_parallel_group()
        if cp_size > 1
        else parallel_state.get_tensor_model_parallel_group()
    )
    if group is None or dist.get_world_size(group=group) != size:
        raise RuntimeError("TPR native TP/CP process group size mismatch")
    words = [int(digest[i : i + 8], 16) for i in range(0, 64, 8)]
    device = torch.device("cpu")
    if dist.get_backend(group) != "gloo":
        from verl.utils.device import get_device_id, get_device_name

        device = torch.device(get_device_name(), get_device_id())
    local = torch.tensor(words, dtype=torch.int64, device=device)
    signatures = [torch.empty_like(local) for _ in range(size)]
    dist.all_gather(signatures, local, group=group)
    if any(not torch.equal(value, local) for value in signatures):
        raise RuntimeError("TPR model-parallel tree plan mismatch")
