# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DR.Kernel rollouter override.

Subclasses `verl.experimental.fully_async_policy.fully_async_rollouter.FullyAsyncRollouter`
to do three things on the validation side:

  1. Rewrite `val-core/<data_source>/reward/mean@N` from per-trajectory
     sum-of-turn-rewards to per-(traj, turn) mean, so it lines up with
     training's `critic/rewards/mean`. See `_val_metrics_update` below.

  2. Emit naive Pass@N metrics for binary indicators when
     `actor_rollout_ref.rollout.val_kwargs.n > 1`:

         val-core/<src>/Pass@<N>/correctness
         val-core/<src>/Pass@<N>/Fast@<X>/last
         val-core/<src>/Pass@<N>/Fast@<X>/best
         val-core/<src>/Pass@<N>/compilation
         val-core/<src>/Pass@<N>/is_speedup_positive

     For each (data_source, uid), Pass@N(key) = 1 iff any of the N rollouts
     hit that indicator. Averaged across uids per data_source. This is the
     "kernel-RL Pass@N": at least one of N stochastic samples produces a
     correct (or Fast@X-meeting) kernel.

     Note: `val-aux/<src>/<key>/best@<N>/mean` is already emitted by the
     parent via `process_validation_metrics` for the same indicators —
     that is a *bootstrap* estimate of best-of-N. Pass@<N>/<key> is the
     direct per-prompt any-of-N indicator, mean-aggregated. The two
     converge as the number of prompts grows.

  3. Capture a per-prompt rollout table (prompt text, ground truth, the N
     rollout summaries and the Pass@N indicators) so `main_validate.py`
     can include it in the standalone `val_metrics_step*.json` dump.
     Exposed via `get_per_prompt_table()`.

The per-prompt table is populated by intercepting `_dump_generations` —
the parent's `_validate` calls it once after the full validation loop and
before `_val_metrics_update`, so the captured input/output/gt/score lists
line up row-for-row with the data passed to `_val_metrics_update`. The
capture is a no-op when `trainer.validation_data_dir` is unset (parent
skips `_dump_generations` in that case) — `rollouts[*].output` falls
back to `None` and the rest of the per-prompt table is still produced
from `reward_extra_infos_dict`.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import ray

from verl.experimental.fully_async_policy.fully_async_rollouter import (
    FullyAsyncRollouter as _RemoteFullyAsyncRollouter,
)

# `FullyAsyncRollouter` is `@ray.remote(num_cpus=10, max_concurrency=100)`-decorated.
# Unwrap to subclass; re-decorate at the bottom with the same spec.
_BaseFullyAsyncRollouter = _RemoteFullyAsyncRollouter.__ray_actor_class__


# Scalar binary indicators (0/1) for which we emit a Pass@N value. These
# are populated per-trajectory by
# `recipe/drkernel/workers/reward_manager/kernel_async_adapter.py::
# _build_response_from_turn_rewards`. Float-valued keys (e.g. `speedup`,
# `reward`, `*/turn_avg`) are intentionally excluded — Pass@N is a
# binary-indicator concept; for continuous quantities the existing
# `val-aux/.../best@N/mean` bootstrap is the right summary.
_SCALAR_BINARY_PASS_KEYS = (
    "correctness",
    "compilation",
    "is_speedup_positive",
)


def _is_binary_pass_at_n_key(key: str) -> bool:
    """True if `key` is a per-trajectory 0/1 indicator we report Pass@N for."""
    if key in _SCALAR_BINARY_PASS_KEYS:
        return True
    # Fast@<X>/last (last-turn) and Fast@<X>/best (any-turn-achieved) are
    # the two binary forms emitted by the kernel_async adapter; the
    # corresponding `/turn_avg` is a float in [0, 1] and skipped here.
    if key.startswith("Fast@") and (key.endswith("/last") or key.endswith("/best")):
        return True
    return False


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator (Chen et al., 2021 — "Evaluating Large Language
    Models Trained on Code"):

        pass@k = E[ 1 - C(n-c, k) / C(n, k) ] = 1 - prod_{i=n-c+1}^{n} (1 - k/i)

    where ``n`` = #rollouts drawn per prompt and ``c`` = #rollouts that hit the
    indicator. Computed in the numerically stable product form (matches
    ``recipe/drkernel/scripts/validation/runs/pass_at_k_from_recorded.py`` and
    ``scripts/compute_pass_at_k.py`` bit-for-bit). For k=1 this reduces to the
    per-sample success rate c/n; for k>=n it reduces to the naive any-of-n
    indicator (1 iff c>0), i.e. the existing ``Pass@N``."""
    n, c, k = int(n), int(c), min(int(k), int(n))
    if n <= 0 or k <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    prod = 1.0
    for i in range(n - c + 1, n + 1):
        prod *= 1.0 - k / i
    return 1.0 - prod


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean env var (1/true/yes/on). Used for the standalone
    validation opt-ins; defaults keep training-time validation unchanged."""
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "y", "on")


def _parse_int_list(text: str | None) -> list[int]:
    """Parse a comma-separated int list (e.g. "1,8" or "[1, 8]"). Empty/None
    yields []."""
    if not text:
        return []
    out: list[int] = []
    for tok in str(text).strip().strip("[]").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(float(tok)))
        except ValueError:
            continue
    return out


# Structural keys in an incremental-dump record (everything else is a
# per-rollout reward_extra_info field). Kept in sync with the writer in
# `_append_incremental_records` and the reader in `_seed_from_incremental`.
_INCR_STRUCT_KEYS = frozenset({"_resume_key", "data_source", "uid", "input", "output", "gts", "step"})


class DrKernelFullyAsyncRollouterImpl(_BaseFullyAsyncRollouter):
    """FullyAsyncRollouter with per-turn reward aggregation, naive Pass@N
    metrics for binary indicators, and a per-prompt rollout table dumped
    via `main_validate.py`."""

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Tee the row-wise sample lists onto `self` so `_val_metrics_update`
        can build the per-prompt rollout table. Forwards to the parent
        unchanged so the existing `validation_data_dir/<step>.jsonl` dump
        is preserved byte-for-byte."""
        self._captured_dump_inputs = list(inputs)
        self._captured_dump_outputs = list(outputs)
        self._captured_dump_gts = list(gts)
        self._captured_dump_scores = list(scores)
        return super()._dump_generations(
            inputs=inputs,
            outputs=outputs,
            gts=gts,
            scores=scores,
            reward_extra_infos_dict=reward_extra_infos_dict,
            dump_path=dump_path,
        )

    def _val_metrics_update(
        self,
        data_sources,
        sample_uids,
        reward_extra_infos_dict,
        sample_turns,
    ):
        """Rewrite ``reward_extra_infos_dict["reward"]`` from per-trajectory
        sum to per-(traj, turn) mean, delegate to the parent, then layer
        naive Pass@N metrics + per-prompt table on top.

        Reward denominator sources, in priority order:
          1. ``reward_extra_infos_dict["num_turns"]`` — emitted by
             `recipe/drkernel/workers/reward_manager/kernel_async_adapter.py::
             _build_response_from_turn_rewards` as ``len(turn_rewards)``.
             This is the assistant-turn count that produced the reward
             sum, so the division is consistent.
          2. ``sample_turns`` (concatenated) — falls back to the
             ``__num_turns__`` non-tensor key, which counts
             ``user_turns + assistant_turns + 1`` (see `KernelAgentLoop.run`).
             Not strictly equivalent to ``len(turn_rewards)``, but a
             reasonable approximation for callers that didn't go through
             the fast path (e.g. legacy single-call reward manager).

        Falls back to the parent untouched if neither denominator lines up
        with the reward length (defensive — keeps validation working even
        if something upstream changes the shape).
        """
        rewards = reward_extra_infos_dict.get("reward")
        if rewards:
            num_rewards = len(rewards)
            num_turns = reward_extra_infos_dict.get("num_turns")
            if num_turns is None or len(num_turns) != num_rewards:
                num_turns = None
                if sample_turns:
                    try:
                        candidate = np.concatenate(sample_turns)
                    except Exception:
                        candidate = None
                    if candidate is not None and len(candidate) == num_rewards:
                        num_turns = candidate
            if num_turns is not None and len(num_turns) == num_rewards:
                rewards_arr = np.asarray(rewards, dtype=np.float64)
                turns_arr = np.asarray(num_turns, dtype=np.float64)
                # Defensive: avoid div-by-zero on degenerate (no-turn)
                # trajectories — value is 0/1 = 0.0 either way.
                safe_turns = np.where(turns_arr > 0, turns_arr, 1.0)
                reward_extra_infos_dict["reward"] = (rewards_arr / safe_turns).tolist()

        metric_dict = super()._val_metrics_update(
            data_sources,
            sample_uids,
            reward_extra_infos_dict,
            sample_turns,
        )

        # ---- Pass@N + per-prompt rollout table ----
        try:
            n_rollouts = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        except Exception:
            n_rollouts = 1

        if n_rollouts > 1:
            pass_n_metrics, per_prompt = self._compute_pass_at_n_and_per_prompt(
                data_sources=data_sources,
                sample_uids=sample_uids,
                reward_extra_infos_dict=reward_extra_infos_dict,
                n_rollouts=n_rollouts,
            )
            metric_dict.update(pass_n_metrics)
            self._captured_per_prompt = per_prompt
        else:
            # N=1 — Pass@N degenerates to the per-prompt indicator mean
            # already in val-aux/<src>/<key>/mean@1. Skip emission and
            # clear any stale per-prompt table from a prior do_validate().
            self._captured_per_prompt = []

        return metric_dict

    def _compute_pass_at_n_and_per_prompt(
        self,
        *,
        data_sources,
        sample_uids,
        reward_extra_infos_dict: dict[str, list],
        n_rollouts: int,
    ) -> tuple[dict[str, float], list[dict[str, Any]]]:
        """Group rows by (data_source, uid), compute naive Pass@N for binary
        indicators and build the per-prompt rollout table.

        Row index `i` is consistent across `data_sources[i]`,
        `sample_uids[i]`, `reward_extra_infos_dict[<key>][i]`, and the
        captured `_dump_*[i]` lists — they are all extended in lockstep
        by the parent's `_validate` loop.

        Returns:
            pass_n_metrics: dict of TB-tag → mean indicator value, e.g.
                ``{"val-core/<src>/Pass@8/correctness": 0.62, ...}``.
            per_prompt: list[dict], one entry per (data_source, uid), each
                containing the prompt text, ground truth, per-rollout dicts
                and the Pass@N indicators for that prompt.
        """
        ds_list = data_sources.tolist() if isinstance(data_sources, np.ndarray) else list(data_sources)
        sample_uids = [str(u) for u in sample_uids]
        n_rows = len(sample_uids)

        # Captured by `_dump_generations` (None when validation_data_dir
        # is unset — per-rollout output/prompt fall back to None).
        cap_in = getattr(self, "_captured_dump_inputs", None)
        cap_out = getattr(self, "_captured_dump_outputs", None)
        cap_gt = getattr(self, "_captured_dump_gts", None)
        cap_score = getattr(self, "_captured_dump_scores", None)

        def _safe_get(lst, idx):
            if lst is None or idx >= len(lst):
                return None
            return lst[idx]

        # Group row indices by (data_source, uid), preserving the order
        # in which each unique key first appears so the dumped table is
        # deterministic and matches the val_dataloader iteration order.
        groups: dict[tuple[str, str], list[int]] = defaultdict(list)
        first_seen_order: list[tuple[str, str]] = []
        for i in range(n_rows):
            key = (ds_list[i], sample_uids[i])
            if key not in groups:
                first_seen_order.append(key)
            groups[key].append(i)

        binary_keys = [k for k in reward_extra_infos_dict.keys() if _is_binary_pass_at_n_key(k)]

        # Unbiased pass@k (Chen et al. 2021) is opt-in via VAL_PASS_AT_K (e.g.
        # "1,8"). Empty => emit only the naive Pass@N below, leaving training-time
        # validation metrics unchanged. k == n_rollouts coincides with the naive
        # any-of-N indicator, so listing N just re-emits the same tag/value.
        pass_at_ks = self._pass_at_k_list()

        per_source_pass: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        # per_source_passk[ds][k][bkey] -> list of per-prompt pass@k values.
        per_source_passk: dict[str, dict[int, dict[str, list[float]]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        per_prompt: list[dict[str, Any]] = []

        for group_key in first_seen_order:
            ds, uid = group_key
            row_idxs = groups[group_key]
            first = row_idxs[0]

            rollouts: list[dict[str, Any]] = []
            for ri in row_idxs:
                rollout: dict[str, Any] = {
                    "row_index": ri,
                    "output": _safe_get(cap_out, ri),
                    "score": _safe_get(cap_score, ri),
                }
                for k, v in reward_extra_infos_dict.items():
                    if isinstance(v, list) and ri < len(v):
                        rollout[k] = v[ri]
                rollouts.append(rollout)

            pass_indicators: dict[str, float] = {}
            # pass_at_k_per_prompt[str(k)][bkey] -> unbiased pass@k for this prompt.
            pass_at_k_per_prompt: dict[str, dict[str, float]] = {str(k): {} for k in pass_at_ks}
            for bkey in binary_keys:
                vals: list[float] = []
                bvals = reward_extra_infos_dict.get(bkey)
                if not isinstance(bvals, list):
                    continue
                for ri in row_idxs:
                    if ri >= len(bvals):
                        continue
                    try:
                        vals.append(float(bvals[ri]))
                    except (TypeError, ValueError):
                        continue
                if not vals:
                    continue
                # Naive Pass@N: 1 iff at least one rollout hit the
                # indicator. Threshold at 0.5 to tolerate stray 0.0/1.0
                # encodings as bools, ints, or floats.
                ind = 1.0 if max(vals) >= 0.5 else 0.0
                pass_indicators[bkey] = ind
                per_source_pass[ds][bkey].append(ind)

                # Unbiased pass@k from this prompt's n rollouts: c = #hits.
                if pass_at_ks:
                    n_this = len(vals)
                    c_this = sum(1 for v in vals if v >= 0.5)
                    for k in pass_at_ks:
                        pk = pass_at_k_unbiased(n_this, c_this, k)
                        pass_at_k_per_prompt[str(k)][bkey] = pk
                        per_source_passk[ds][k][bkey].append(pk)

            entry: dict[str, Any] = {
                "data_source": ds,
                "uid": uid,
                "n_rollouts": len(row_idxs),
                "prompt": _safe_get(cap_in, first),
                "ground_truth": _safe_get(cap_gt, first),
                "pass_at_n": pass_indicators,
                "rollouts": rollouts,
            }
            if pass_at_ks:
                entry["pass_at_k"] = pass_at_k_per_prompt
            per_prompt.append(entry)

        pass_n_metrics: dict[str, float] = {}
        for ds, key2vals in per_source_pass.items():
            for bkey, vals in key2vals.items():
                if not vals:
                    continue
                pass_n_metrics[f"val-core/{ds}/Pass@{n_rollouts}/{bkey}"] = float(np.mean(vals))

        # Mean unbiased pass@k over prompts, per data source. pass@1 is the
        # correct per-sample success rate (NOT the any-of-N Pass@N above).
        for ds, k2 in per_source_passk.items():
            for k, key2vals in k2.items():
                for bkey, vals in key2vals.items():
                    if not vals:
                        continue
                    pass_n_metrics[f"val-core/{ds}/Pass@{k}/{bkey}"] = float(np.mean(vals))

        return pass_n_metrics, per_prompt

    # ------------------------------------------------------------------
    # Standalone-validation extras (progress / incremental dump / resume).
    #
    # ------------------------------------------------------------------
    def _pass_at_k_list(self) -> list[int]:
        return _parse_int_list(os.environ.get("VAL_PASS_AT_K"))

    def _incremental_enabled(self) -> bool:
        return _env_flag("VAL_INCREMENTAL", False)

    def _resume_enabled(self) -> bool:
        # Resume from a prior partial run by default whenever incremental is on;
        # set VAL_RESUME=0 to start clean (truncate the dump first).
        return _env_flag("VAL_RESUME", True)

    def _validate(self, merged: bool = False):
        if not self._incremental_enabled():
            return super()._validate(merged=merged)
        return self._validate_incremental(merged=merged)

    def _incremental_dump_file(self) -> str | None:
        val_dir = self.config.trainer.get("validation_data_dir", None)
        if not val_dir:
            return None
        if not os.path.isabs(val_dir):
            val_dir = os.path.join(os.getcwd(), val_dir)
        os.makedirs(val_dir, exist_ok=True)
        return os.path.join(val_dir, f"{self.global_steps}.jsonl")

    @staticmethod
    def _resume_key_from_nt(nt: dict[str, Any]) -> str:
        ds = str(nt.get("data_source", "unknown"))
        extra = nt.get("extra_info", {})
        pid = None
        if isinstance(extra, dict):
            pid = extra.get("problem_id", extra.get("name"))
        if pid is not None and str(pid) != "":
            return f"{ds}::{pid}"
        rm = nt.get("reward_model", {})
        gt = rm.get("ground_truth", "") if isinstance(rm, dict) else ""
        h = hashlib.sha1(str(gt).encode("utf-8")).hexdigest()[:16]
        return f"{ds}::gt::{h}"

    def _validate_incremental(self, merged: bool = False):
        from verl import DataProto
        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
        from verl.trainer.ppo.reward import extract_reward

        n_rollouts = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)

        # Accumulators (identical roles to the parent `_validate`).
        data_source_lst: list = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs: list = []
        sample_outputs: list = []
        sample_gts: list = []
        sample_scores: list = []
        sample_turns: list = []
        sample_uids: list = []

        total_prompts = len(self.val_dataset)
        chunk_size = self.config.data.get("val_batch_size", None)
        dump_file = self._incremental_dump_file()

        # ---- resume: reload completed prompts from a prior partial run ----
        completed_keys: set[str] = set()
        if dump_file and self._resume_enabled() and os.path.exists(dump_file):
            completed_keys = self._seed_from_incremental(
                dump_file,
                data_source_lst=data_source_lst,
                reward_extra_infos_dict=reward_extra_infos_dict,
                sample_inputs=sample_inputs,
                sample_outputs=sample_outputs,
                sample_gts=sample_gts,
                sample_scores=sample_scores,
                sample_turns=sample_turns,
                sample_uids=sample_uids,
            )
            print(
                f"[VALIDATE][resume] loaded {len(completed_keys)} completed prompts "
                f"({len(sample_scores)} rollout rows) from {dump_file}"
            )
        elif dump_file and not self._resume_enabled() and os.path.exists(dump_file):
            open(dump_file, "w").close()
            print(f"[VALIDATE][resume] VAL_RESUME=0 -> truncated {dump_file}")

        processed = len(completed_keys)
        print(
            f"[VALIDATE][progress] start: {processed}/{total_prompts} prompts already done; "
            f"chunk_size={chunk_size}, n_rollouts={n_rollouts}, dump={dump_file}"
        )

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # Stable per-prompt resume keys (pre-repeat: one per dataset row).
            n_rows = len(test_batch)
            prompt_keys = [
                self._resume_key_from_nt({k: v[i] for k, v in test_batch.non_tensor_batch.items()})
                for i in range(n_rows)
            ]
            keep_idx = [i for i, key in enumerate(prompt_keys) if key not in completed_keys]
            if not keep_idx:
                print(
                    f"[VALIDATE][progress] {processed}/{total_prompts} prompts done "
                    f"(resume: skipped a fully-completed chunk of {n_rows})"
                )
                continue
            if len(keep_idx) != n_rows:
                test_batch = test_batch[keep_idx]
                prompt_keys = [prompt_keys[i] for i in keep_idx]

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                    dtype=object,
                )

            test_batch = test_batch.repeat(repeat_times=n_rollouts, interleave=True)

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }

            size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            if self.use_rm and "rm_scores" not in test_output_gen_batch_padded.batch.keys():
                self.checkpoint_manager.sleep_replicas()
                batch_reward = self._compute_reward_colocate(test_output_gen_batch_padded)
                test_output_gen_batch_padded = test_output_gen_batch_padded.union(batch_reward)
                self.checkpoint_manager.update_weights(self.global_steps)

            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            input_ids = test_batch.batch["prompts"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]

            reward_tensor, reward_extra_info = extract_reward(test_batch)
            scores = reward_tensor.sum(-1).cpu().tolist()

            # Per-row reward_extra for this chunk. "reward" is the PRE-rewrite
            # sum (== score); the end-of-run `_val_metrics_update` divides it by
            # num_turns exactly once, so it must be persisted un-divided.
            chunk_extra: dict[str, list] = {"reward": list(scores)}
            for key, values in reward_extra_info.items():
                if isinstance(values, np.ndarray):
                    chunk_extra[key] = values.tolist()
                elif isinstance(values, list):
                    chunk_extra[key] = values
                else:
                    chunk_extra[key] = [values]

            data_sources_chunk = test_batch.non_tensor_batch.get(
                "data_source",
                np.array(["unknown"] * reward_tensor.shape[0], dtype=object),
            )
            uids_chunk = [str(u) for u in test_batch.non_tensor_batch["uid"]]
            num_turns_chunk = test_batch.non_tensor_batch.get("__num_turns__", None)

            # Extend the in-memory accumulators.
            sample_inputs.extend(input_texts)
            sample_outputs.extend(output_texts)
            sample_gts.extend(ground_truths)
            sample_scores.extend(scores)
            sample_uids.extend(uids_chunk)
            for key, vals in chunk_extra.items():
                reward_extra_infos_dict[key].extend(vals)
            if num_turns_chunk is not None:
                sample_turns.append(np.asarray(num_turns_chunk))
            data_source_lst.append(
                data_sources_chunk
                if isinstance(data_sources_chunk, np.ndarray)
                else np.array(list(data_sources_chunk), dtype=object)
            )

            # One prompt -> n contiguous rollout rows (repeat interleave=True).
            row_keys = [k for k in prompt_keys for _ in range(n_rollouts)]

            if dump_file:
                self._append_incremental_records(
                    dump_file,
                    row_keys=row_keys,
                    data_sources=list(data_sources_chunk),
                    uids=uids_chunk,
                    inputs=input_texts,
                    outputs=output_texts,
                    gts=ground_truths,
                    scores=scores,
                    chunk_extra=chunk_extra,
                )

            for key in prompt_keys:
                completed_keys.add(key)
            processed += len(prompt_keys)
            print(
                f"[VALIDATE][progress] {processed}/{total_prompts} prompts done "
                f"(+{len(prompt_keys)} this chunk; {total_prompts - processed} remaining)"
            )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # Length-consistency guard (mirrors the parent `_validate`).
        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        if merged:
            return {
                "data_sources": data_source_lst,
                "sample_uids": sample_uids,
                "sample_turns": sample_turns,
                "reward_extra_infos_dict": reward_extra_infos_dict,
            }

        # Feed the per-prompt table builder the same captured lists the parent's
        # `_dump_generations` tee would have produced (the incremental path
        # never calls `_dump_generations`, since it wrote the JSONL itself).
        self._captured_dump_inputs = list(sample_inputs)
        self._captured_dump_outputs = list(sample_outputs)
        self._captured_dump_gts = list(sample_gts)
        self._captured_dump_scores = list(sample_scores)

        if not sample_scores:
            print("[VALIDATE] no prompts processed (all already complete?) — empty metrics")
            self._captured_per_prompt = []
            return {}

        data_sources = np.concatenate(data_source_lst, axis=0)
        return self._val_metrics_update(data_sources, sample_uids, reward_extra_infos_dict, sample_turns)

    def _append_incremental_records(
        self,
        dump_file: str,
        *,
        row_keys: list,
        data_sources: list,
        uids: list,
        inputs: list,
        outputs: list,
        gts: list,
        scores: list,
        chunk_extra: dict[str, list],
    ) -> None:
        """Append one JSONL line per rollout row for a just-finished chunk."""
        n = len(scores)
        lines = []
        for i in range(n):
            rec: dict[str, Any] = {
                "_resume_key": row_keys[i] if i < len(row_keys) else None,
                "data_source": str(data_sources[i]) if i < len(data_sources) else "unknown",
                "uid": str(uids[i]) if i < len(uids) else "",
                "input": inputs[i] if i < len(inputs) else None,
                "output": outputs[i] if i < len(outputs) else None,
                "gts": gts[i] if i < len(gts) else None,
                "score": scores[i],
                "step": self.global_steps,
            }
            for key, vals in chunk_extra.items():
                if i < len(vals):
                    rec[key] = vals[i]
            lines.append(json.dumps(rec, ensure_ascii=False, default=str))
        with open(dump_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _seed_from_incremental(
        self,
        dump_file: str,
        *,
        data_source_lst: list,
        reward_extra_infos_dict: dict[str, list],
        sample_inputs: list,
        sample_outputs: list,
        sample_gts: list,
        sample_scores: list,
        sample_turns: list,
        sample_uids: list,
    ) -> set[str]:
        """Reload an incremental JSONL into the accumulators and return the set
        of completed `_resume_key`s. Mirrors how a chunk extends the
        accumulators, so a resumed run aggregates identically to a single-shot
        run. `reward` is reloaded as the stored PRE-rewrite sum (the end-of-run
        rewrite divides once)."""
        completed: set[str] = set()
        loaded_ds: list = []
        loaded_turns: list = []
        with open(dump_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = rec.get("_resume_key")
                if key is not None:
                    completed.add(key)
                sample_inputs.append(rec.get("input"))
                sample_outputs.append(rec.get("output"))
                sample_gts.append(rec.get("gts"))
                try:
                    score = float(rec.get("score", 0.0) or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                sample_scores.append(score)
                sample_uids.append(str(rec.get("uid", "")))
                loaded_ds.append(str(rec.get("data_source", "unknown")))
                nt = rec.get("num_turns")
                try:
                    loaded_turns.append(int(nt) if nt is not None else 0)
                except (TypeError, ValueError):
                    loaded_turns.append(0)
                # Every non-structural field is a per-rollout reward_extra value
                # (includes "reward", "num_turns", "correctness", "Fast@*", ...).
                for k, v in rec.items():
                    if k in _INCR_STRUCT_KEYS:
                        continue
                    reward_extra_infos_dict[k].append(v)
        if loaded_ds:
            data_source_lst.append(np.array(loaded_ds, dtype=object))
        if loaded_turns:
            sample_turns.append(np.asarray(loaded_turns))
        return completed

    def get_per_prompt_table(self) -> list[dict[str, Any]]:
        """Return the per-prompt rollout table built during the latest
        `do_validate()` call. Empty list when `val_kwargs.n <= 1` or
        when `do_validate()` has not yet been invoked."""
        return getattr(self, "_captured_per_prompt", []) or []


# Re-decorate with the same resource spec as the parent.
DrKernelFullyAsyncRollouter = ray.remote(num_cpus=10, max_concurrency=100)(DrKernelFullyAsyncRollouterImpl)
