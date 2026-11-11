# Copyright 2026 Bytedance Ltd. and/or its affiliates
"""Consume TPR physical logits using VERL's existing PPO loss function.

The adapter only maps (segment query, logical sample) to token log-probabilities
and builds a compact *logical* TensorDict. It does not implement PPO, KL,
clipping, or objective normalization. This path currently supports additive
vanilla PPO with token-mean aggregation, not sequence-level objectives.
"""

from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor

from .ring import SequenceShard
from .tree import SegmentObjectiveRef, SegmentSpec, TreeExecutionPlan


def _rows(batch: Any, key: str) -> list[Tensor]:
    try:
        value = batch[key]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing PPO field {key!r}") from exc
    values = list(value) if isinstance(value, (tuple, list)) else list(value.unbind())
    if not all(isinstance(row, Tensor) and row.ndim == 1 for row in values):
        raise ValueError(f"PPO field {key!r} must have 1-D rows")
    return values


def _native_log_probs(logits: Tensor, labels: Tensor) -> Tensor:
    from verl.utils.megatron.tensor_parallel import vocab_parallel_log_probs_from_logits

    return vocab_parallel_log_probs_from_logits(logits, labels)


def _native_entropy(logits: Tensor) -> Tensor:
    from verl.utils.megatron.tensor_parallel import vocab_parallel_entropy

    return vocab_parallel_entropy(logits)


def _metadata(batch: Any, key: str, default=None):
    from verl.utils import tensordict_utils as tu

    value = tu.get_non_tensor_data(batch, key=key, default=default)
    # Keep plain Mapping metadata support, while matching tu.get's stack
    # unwrapping for VERL assign_non_tensor(list) / TensorDict batches.
    return value.tolist() if isinstance(value, tu.NonTensorStack) else value


def _selected_scaled_logits(
    logits: Tensor, positions: Tensor, temperatures: Tensor, query_indices: Tensor | None
) -> Tensor:
    selected = logits[0].index_select(0, positions)
    # Keep the accepted BF16 selected-logit arithmetic by default. Real
    # Qwen3-4B PPO replay showed that unconditional FP32 promotion changed
    # logprobs and full gradients; opt in only for precision diagnostics.
    if selected.dtype in (torch.float16, torch.bfloat16) and os.getenv("TPR_SELECTED_LOGITS_FP32", "0") == "1":
        selected = selected.float()
    if query_indices is not None:
        selected = selected.index_select(0, query_indices)
    return selected / temperatures.to(selected.dtype)[:, None]


@torch.no_grad()
def _logit_parts(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Diagnostic target logits and log-partition over the *global* TP vocab."""
    values = logits.float()
    if not torch.distributed.is_initialized():
        return values.gather(1, targets[:, None]).squeeze(1), torch.logsumexp(values, dim=-1)
    from megatron.core import parallel_state

    group = parallel_state.get_tensor_model_parallel_group()
    if group is None:
        raise RuntimeError("TPR logit-parts probe requires the native Megatron TP group")
    rank = torch.distributed.get_rank(group)
    local_targets = targets - rank * values.shape[-1]
    owned = (local_targets >= 0) & (local_targets < values.shape[-1])
    target_logits = values.gather(1, local_targets.clamp(0, values.shape[-1] - 1)[:, None]).squeeze(1)
    target_logits = target_logits.masked_fill(~owned, 0)
    torch.distributed.all_reduce(target_logits, group=group)
    maximum = values.amax(dim=-1)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
    exp_sum = (values - maximum[:, None]).exp().sum(dim=-1)
    torch.distributed.all_reduce(exp_sum, group=group)
    return target_logits, maximum + exp_sum.log()


class SegmentPPOObjectiveAdapter:
    """Bridge SegmentExecutor logits to an unmodified VERL PPO loss callable.

    Important: the fake one-token 'prompts' below are bookkeeping only for
    VERL's no_padding_2_padding response-shifting function. They are never
    fed to a model. This ensures the packed new_log_probs are shifted into
    the compact response layout exactly as in native forward_step.
    """

    def __init__(
        self,
        batch: Any,
        loss_function: Callable,
        *,
        log_prob_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        entropy_fn: Callable[[Tensor], Tensor] | None = None,
        calculate_entropy: bool | None = None,
        dp_group: Any = None,
        log_prob_chunk_size: int | None = None,
    ) -> None:
        if loss_function is None:
            raise ValueError("PPO objective requires a VERL loss_function")
        cfg = getattr(loss_function, "keywords", {}).get("config")
        if cfg is not None:
            mode = cfg.policy_loss.get("loss_mode", "vanilla")
            if mode != "vanilla" or cfg.loss_agg_mode != "token-mean":
                raise NotImplementedError(
                    "segment-local PPO currently requires vanilla policy loss and token-mean "
                    f"aggregation, got {mode!r} and {cfg.loss_agg_mode!r}"
                )
        self.batch = batch
        self.loss_function = loss_function
        self.log_prob_fn = log_prob_fn or _native_log_probs
        self.entropy_fn = entropy_fn or _native_entropy
        self.calculate_entropy = (
            bool(_metadata(batch, "calculate_entropy", default=False))
            if calculate_entropy is None
            else bool(calculate_entropy)
        )
        self.dp_group = dp_group
        if log_prob_chunk_size is not None and (
            not isinstance(log_prob_chunk_size, int)
            or isinstance(log_prob_chunk_size, bool)
            or log_prob_chunk_size <= 0
        ):
            raise ValueError(f"log_prob_chunk_size must be a positive integer or None, got {log_prob_chunk_size!r}")
        self.log_prob_chunk_size = log_prob_chunk_size
        # Optional numerical probe only; disabled for ordinary training to
        # avoid device synchronization and host-side token materialization.
        self.capture_log_probs = bool(_metadata(batch, "tpr_capture_log_probs", default=False))
        self.debug_new_log_probs: dict[tuple[int, int], float] = {}
        self.capture_logit_parts = bool(_metadata(batch, "tpr_capture_logit_parts", default=False))
        self.debug_logit_parts: dict[tuple[int, int], tuple[float, float]] = {}
        # The adapter belongs to one immutable PPO minibatch. Prepare metadata
        # once, not per segment or per logical token (which can synchronize NPU).
        self._temperature_values: tuple[float, ...] | None = None
        self._field_rows: dict[str, list[Tensor]] | None = None
        self._response_rows: list[Tensor] | None = None
        self._normalization = {
            key: _metadata(batch, key, default)
            for key, default in (("dp_size", 1), ("batch_num_tokens", None), ("global_batch_size", None))
        }

    def bind_tree(self, tree_plan: TreeExecutionPlan, *, shard_for_segment=None):
        """Bind an executable tree to SegmentExecutor's loss hook.

        Return (loss_callback, per_segment_ref_counts). A single adapter can
        serve every tree of the original PPO mini-batch; each call retains the
        original batch-level normalization metadata.
        """
        if not isinstance(tree_plan, TreeExecutionPlan):
            raise TypeError("tree_plan must be a TreeExecutionPlan")
        refs_by_segment: dict[int, list[SegmentObjectiveRef]] = {
            segment_id: [] for segment_id in tree_plan.segment_plan.segments
        }
        for ref in tree_plan.objective_refs:
            refs_by_segment[ref.segment_id].append(ref)

        if shard_for_segment is not None:
            for segment_id, refs in refs_by_segment.items():
                shard = shard_for_segment(tree_plan.segment_plan.get(segment_id))
                refs_by_segment[segment_id] = [ref for ref in refs if shard.owns(ref.query_offset)]
        counts = {segment_id: len(refs) for segment_id, refs in refs_by_segment.items()}

        def segment_loss_fn(segment: SegmentSpec, logits: Tensor):
            shard = None if shard_for_segment is None else shard_for_segment(segment)
            return self.compute_loss(segment, logits, refs_by_segment[segment.segment_id], shard=shard)

        return segment_loss_fn, counts

    def _temperatures(self, refs: Sequence[SegmentObjectiveRef]) -> list[float]:
        if self._temperature_values is None:
            temperature = _metadata(self.batch, "temperature", default=1.0)
            count = len(self.batch["response_mask"])
            if isinstance(temperature, Tensor):
                values = temperature.detach().flatten().cpu().tolist()
                temperature = values[0] if len(values) == 1 else values
            if isinstance(temperature, (float, int)):
                values = (float(temperature),) * count
            elif isinstance(temperature, (list, tuple)):
                values = tuple(float(value) for value in temperature)
                if len(values) != count:
                    raise ValueError("temperature must contain one value per PPO sample")
            else:
                raise TypeError(
                    f"unsupported per-sample temperature type: {type(temperature)!r}, value={temperature!r}"
                )
            if any(not 0 < value < float("inf") for value in values):
                raise ValueError("temperature must be finite and positive")
            self._temperature_values = tuple(values)
        if any(not 0 <= ref.sample_row < len(self._temperature_values) for ref in refs):
            raise ValueError("objective ref sample_row is outside PPO minibatch")
        return [self._temperature_values[ref.sample_row] for ref in refs]

    def _prepare_batch_rows(self) -> None:
        if self._field_rows is not None:
            return
        masks = self.batch["response_mask"]
        # Dense/nested masks use one device-to-host transfer for the minibatch;
        # list inputs use one per row, never one per segment or logical token.
        if isinstance(masks, Tensor):
            masks = masks.detach().to(device="cpu", dtype=torch.bool)
        response_rows = _rows({"response_mask": masks}, "response_mask")
        response_rows = [row.detach().to(device="cpu", dtype=torch.bool) for row in response_rows]
        field_rows = {
            key: _rows(self.batch, key)
            for key in ("old_log_probs", "advantages", "ref_log_prob", "rollout_is_weights")
            if key in ("old_log_probs", "advantages") or key in self.batch
        }
        for key, rows in field_rows.items():
            if len(rows) != len(response_rows) or any(
                row.numel() != mask.numel() for row, mask in zip(rows, response_rows, strict=True)
            ):
                raise ValueError(f"PPO field {key!r} must match response_mask rows")
        self._response_rows, self._field_rows = response_rows, field_rows

    def compute_loss(
        self,
        segment: SegmentSpec,
        logits: Tensor,
        refs: Sequence[SegmentObjectiveRef],
        *,
        shard: SequenceShard | None = None,
    ) -> tuple[Tensor, dict]:
        """Return a native-normalized scalar and its segment-local metrics.

        Caller performs backward, prefix-state gradient relay and finalization.
        Empty refs are intentionally rejected: no-loss segments should use the
        executor's connected zero/relay path instead of invoking PPO.
        """
        refs = tuple(refs)
        if not refs:
            raise ValueError("no objective refs: use a zero-gradient relay path for this segment")
        expected_length = segment.length if shard is None else shard.local_length
        if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[1] != expected_length:
            raise ValueError("expected physical logits [1, local_length, local_vocab]")
        if self._normalization["batch_num_tokens"] is None:
            raise ValueError("PPO global batch_num_tokens is missing; compute it before TPR routing")
        temperatures = self._temperatures(refs)

        # Reuse the exact physical (query,target,temperature) log-prob compute.
        # Crucially, repeated *logical* refs are not discarded from the loss.
        unique: dict[tuple[int, int, float], int] = {}
        gather_ids: list[int] = []
        for ref, temperature in zip(refs, temperatures, strict=True):
            if ref.segment_id != segment.segment_id:
                raise ValueError("objective ref belongs to another segment")
            if not 0 <= ref.query_offset < segment.length:
                raise ValueError("objective ref query_offset is outside segment")
            offset = ref.query_offset if shard is None else shard.global_to_local(ref.query_offset)
            if offset is None:
                raise ValueError("PPO ref is not owned by this CP rank")
            key = (offset, ref.target_token_id, temperature)
            gather_ids.append(unique.setdefault(key, len(unique)))

        # A physical query may have several targets/temperatures. Promote that
        # query only once, then expand in FP32 so its gradients accumulate before
        # the single cast back to BF16/FP16, as with Native Float16Module output.
        # FP32/FP64 keep the existing key/chunk order without a redundant gather.
        promote = logits.dtype in (torch.float16, torch.bfloat16)
        query_ids, query_offsets, query_groups, key_queries = {}, [], [], []
        for key_id, (offset, _, _) in enumerate(unique):
            query = query_ids.setdefault(offset if promote else key_id, len(query_ids))
            if query == len(query_groups):
                query_offsets.append(offset)
                query_groups.append([])
            query_groups[query].append(key_id)
            key_queries.append(query)
        positions = torch.tensor(query_offsets, device=logits.device, dtype=torch.long)
        query_indices = torch.tensor(key_queries, device=logits.device, dtype=torch.long) if promote else None
        targets = torch.tensor([key[1] for key in unique], device=logits.device, dtype=torch.long)
        temps = torch.tensor([key[2] for key in unique], device=logits.device, dtype=torch.float32)
        # PPO's native vocab-parallel logprob may save large BF16/FP32
        # [unique_queries, local_vocab] intermediates until backward.
        # With CP2 on 40K samples that can OOM even though every segment's
        # decoder activation fits. Optional token-chunk checkpointing leaves
        # PPO normalization and the *physical logits* unchanged, but
        # rematerializes selected logits / softmax on the backward pass.
        # All TP ranks owning this CP shard have identical refs/chunk order.
        use_chunked_lp = (
            self.log_prob_chunk_size is not None
            and len(unique) > self.log_prob_chunk_size
            and not self.calculate_entropy
            and not self.capture_logit_parts
        )
        if use_chunked_lp:
            from torch.utils.checkpoint import checkpoint

            def log_probs_for_chunk(full_logits, chunk_positions, chunk_targets, chunk_temps, chunk_queries):
                scaled = _selected_scaled_logits(full_logits, chunk_positions, chunk_temps, chunk_queries)
                return self.log_prob_fn(scaled, chunk_targets)

            # Never split a low-precision query across chunks: that would cast
            # each partial gradient before accumulating it. The configured size
            # is a soft upper bound when one query has more target/temp pairs.
            chunks, start, key_ids = [], 0, []
            for query, group in enumerate(query_groups):
                if key_ids and len(key_ids) + len(group) > self.log_prob_chunk_size:
                    chunks.append((start, query, key_ids))
                    start, key_ids = query, []
                key_ids.extend(group)
            chunks.append((start, len(query_groups), key_ids))
            outputs, restore = [], [0] * len(unique)
            output_index = 0
            for start, end, key_ids in chunks:
                for key_id in key_ids:
                    restore[key_id] = output_index
                    output_index += 1
                indices = torch.tensor(key_ids, device=logits.device, dtype=torch.long)
                outputs.append(
                    checkpoint(
                        log_probs_for_chunk,
                        logits,
                        positions[start:end],
                        targets.index_select(0, indices),
                        temps.index_select(0, indices),
                        None if query_indices is None else query_indices.index_select(0, indices) - start,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                )
            restore = torch.tensor(restore, device=logits.device, dtype=torch.long)
            unique_log_probs = torch.cat(outputs, dim=0).index_select(0, restore)
        else:
            scaled_logits = _selected_scaled_logits(logits, positions, temps, query_indices)
            # Native FP32 vocab-parallel CE overwrites its input. Entropy also
            # saves/mutates its logits in backward, so it needs independent
            # storage even when computed before CE.
            if self.calculate_entropy:
                unique_entropy = self.entropy_fn(scaled_logits.clone())
            if self.capture_logit_parts:
                target_logits, lse = _logit_parts(scaled_logits, targets)
                parts = list(
                    zip(
                        target_logits.cpu().tolist(),
                        lse.cpu().tolist(),
                        strict=True,
                    )
                )
            unique_log_probs = self.log_prob_fn(scaled_logits, targets)
        if unique_log_probs.ndim != 1 or unique_log_probs.numel() != len(unique):
            raise ValueError("log_prob_fn must return one scalar per unique query/target/temperature")
        gather = torch.tensor(gather_ids, device=logits.device, dtype=torch.long)
        log_probs = unique_log_probs.index_select(0, gather)
        if self.capture_logit_parts:
            for ref, part_index in zip(refs, gather_ids, strict=True):
                probe_key = (ref.sample_row, ref.response_offset)
                if probe_key in self.debug_logit_parts:
                    raise ValueError(f"duplicate logit-parts probe for logical token {probe_key}")
                self.debug_logit_parts[probe_key] = parts[part_index]
        if self.capture_log_probs:
            for ref, value in zip(refs, log_probs.detach().float().cpu().tolist(), strict=True):
                key = (ref.sample_row, ref.response_offset)
                if key in self.debug_new_log_probs:
                    raise ValueError(f"duplicate new_log_prob probe for logical token {key}")
                self.debug_new_log_probs[key] = value

        entropy = None
        if self.calculate_entropy:
            if unique_entropy.ndim != 1 or unique_entropy.numel() != len(unique):
                raise ValueError("entropy_fn must return one value per unique query")
            entropy = unique_entropy.index_select(0, gather)

        # Construct minimal logical microbatch in original sample order.
        # Each local row gets a 1-token pseudo prompt, followed by its refs.
        # Packed model output is [lp0, ..., lp(R-1), unused_dummy] per row;
        # no_padding_2_padding extracts the first R entries for prompt_len=1.
        from tensordict import TensorDict

        from verl.utils import tensordict_utils as tu

        by_row: dict[int, list[int]] = defaultdict(list)
        for i, ref in enumerate(refs):
            by_row[ref.sample_row].append(i)
        rows = sorted(by_row)
        for row in rows:
            by_row[row].sort(key=lambda i: refs[i].response_offset)
            offsets = [refs[i].response_offset for i in by_row[row]]
            if len(set(offsets)) != len(offsets):
                raise ValueError(f"duplicate logical token ownership on row {row}")

        self._prepare_batch_rows()
        response_rows, field_rows = self._response_rows, self._field_rows
        n = len(rows)
        max_r = max(len(by_row[row]) for row in rows)
        dev = logits.device
        output_fields: dict[str, Tensor] = {
            "prompts": torch.zeros((n, 1), dtype=torch.long, device=dev),
            "responses": torch.zeros((n, max_r), dtype=torch.long, device=dev),
            "attention_mask": torch.zeros((n, 1 + max_r), dtype=torch.long, device=dev),
            "response_mask": torch.zeros((n, max_r), dtype=torch.bool, device=dev),
        }
        for name in field_rows:
            output_fields[name] = torch.zeros((n, max_r), dtype=log_probs.dtype, device=dev)

        token_outputs = {"log_probs": log_probs}
        if entropy is not None:
            token_outputs["entropy"] = entropy
        packed = {key: [] for key in token_outputs}
        for local_row, row in enumerate(rows):
            selected = by_row[row]
            count = len(selected)
            offsets = [refs[i].response_offset for i in selected]
            if any(i < 0 or i >= response_rows[row].numel() for i in offsets):
                raise ValueError(f"response_offset out of bounds for row {row}")
            if not bool(response_rows[row][offsets].all()):
                raise ValueError(f"objective ref points to a masked token on row {row}")
            idx = torch.tensor(selected, dtype=torch.long, device=dev)
            output_fields["attention_mask"][local_row, : count + 1] = 1
            output_fields["response_mask"][local_row, :count] = True
            output_fields["responses"][local_row, :count] = torch.tensor(
                [refs[i].target_token_id for i in selected], device=dev
            )
            indices = torch.tensor(offsets, dtype=torch.long, device=dev)
            for key, source in field_rows.items():
                output_fields[key][local_row, :count] = (
                    source[row].index_select(0, indices.to(source[row].device)).to(dev)
                )
            for key, values in token_outputs.items():
                values = values.index_select(0, idx)
                packed[key].extend((values, values.new_zeros(1)))

        compact = TensorDict(output_fields, batch_size=[n], device=dev)
        tu.assign_non_tensor(compact, **self._normalization)
        model_output = {key: torch.cat(values, dim=0) for key, values in packed.items()}
        # Fail at the TPR/native-PPO boundary with the actual local shapes,
        # rather than inside no_padding_2_padding's opaque offsets assert.
        expected_packed = len(refs) + n
        for key, values in model_output.items():
            if values.shape[0] != expected_packed:
                packed_lengths = [len(by_row[row]) + 1 for row in rows]
                raise RuntimeError(
                    f"TPR compact PPO {key} pack mismatch: "
                    f"actual={values.shape[0]} expected={expected_packed} "
                    f"row_lengths={packed_lengths} refs={len(refs)} rows={rows} "
                    f"shape={tuple(values.shape)}"
                )
        loss, metrics = self.loss_function(
            model_output=model_output,
            data=compact,
            dp_group=self.dp_group,
        )
        if not isinstance(loss, Tensor) or loss.numel() != 1:
            raise TypeError("native VERL loss_function must return a scalar Tensor")
        return loss, metrics
