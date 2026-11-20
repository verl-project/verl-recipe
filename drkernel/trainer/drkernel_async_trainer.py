# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
DR.Kernel-flavored fully-async PPO trainer.

Subclasses `verl.experimental.fully_async_policy.fully_async_trainer.FullyAsyncTrainer`
and inserts a single `_fit_filter_batch` hook between `_fit_compute_log_prob`
and `_fit_compute_ref_log_prob` in `fit_step`. Verl core stays untouched.

Hook semantics:
- Reads `algorithm.batch_filter` from the config (added by
  `recipe/drkernel/config/drkernel_async_ppo_trainer.yaml`).
- When `batch_filter.enable` is False or the sub-tree is absent, the hook
  is a no-op.
- When enabled, runs the PPOBatchFilter pipeline (see
  `recipe/drkernel/filters/`) and returns the surviving rows.
- On empty result (all samples filtered out), logs a warning and falls back
  to the unfiltered batch so training never stalls.
"""

import logging

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from recipe.drkernel import algos  # noqa: F401  # registers the "trloo" advantage estimator
from recipe.drkernel.rewards.fast_at import (
    FAST_AT_KEY_PREFIX,
    compute_fast_at_indicators,
    resolve_thresholds,
)

from verl import DataProto
from verl.experimental.fully_async_policy.fully_async_trainer import FullyAsyncTrainer as _RemoteFullyAsyncTrainer
from verl.utils.debug import marked_timer

logger = logging.getLogger(__name__)

# `FullyAsyncTrainer` is `@ray.remote(num_cpus=10)`-decorated. Unwrap
# `__ray_actor_class__` to get the underlying class, then re-decorate the
# subclass below.
_BaseFullyAsyncTrainer = _RemoteFullyAsyncTrainer.__ray_actor_class__


class DrKernelFullyAsyncTrainerImpl(_BaseFullyAsyncTrainer):
    """FullyAsyncTrainer + DR.Kernel MRS batch filter."""

    def _fit_collect_metrics(self, batch: DataProto) -> None:
        """Collect parent metrics, then add training-side kernel aggregates.

        Validation Fast@X already comes for free via `process_validation_metrics`,
        which averages each per-trajectory ``Fast@<X>`` indicator across rollouts
        and prompts (logged as ``val-aux/<data_source>/Fast@<X>/mean@N``).
        Training has no equivalent auto-aggregation, so we read per-(traj, turn)
        values out of `batch.non_tensor_batch["turn_results"]` (surfaced by
        `KernelAgentLoop.run` via `extra_fields`) and emit:

        - ``kernel/Fast@<X>``: true cross-(traj, turn) mean of the Fast@X
          indicator — every per-turn 0/1 weighted equally. Matches the
          aggregation semantics of ``kernel/correctness/rate`` etc.
        - ``kernel/Fast@<X>/last``: fraction of trajectories whose LAST
          turn achieved Fast@X. Diagnostic comparison with
          pre-multi-turn runs.
        - ``kernel/Fast@<X>/best``: fraction of trajectories where ANY turn
          achieved Fast@X. Captures backsliding-aware "did the model find
          a Fast@X kernel at some point during revision?"
        - ``kernel/correctness/rate``: fraction of *(trajectory, turn) pairs*
          that passed correctness — i.e. mean across every assistant turn
          in the batch (NOT just the last turn).
        - ``kernel/compilation/rate``: same, but for compilation.
        - ``kernel/is_speedup_positive/rate``: same, but for speedup >= 1+eps.
        - ``kernel/speedup/{mean,max,min}``: speedup distribution across every
          assistant turn in the batch. Failures contribute 0.0.
        - ``kernel/per_turn_reward/{mean,max,min}``: stats over every
          per-turn reward in the batch (sourced from
          `non_tensor_batch["turn_rewards"]`). Bounded by the reward
          function — for ``calculate_reward_speedup`` with default weights
          and ``speedup_reward_upper_bound=3``, ``/max`` ≤ 2.0.
        - ``kernel/reward_term/{correctness,performance}``: derived from the
          per-turn rates above.
        - ``critic/rewards/{mean,max,min}`` and ``critic/score/{mean,max,min}``:
          OVERWRITTEN with per-turn-reward stats (instead of the parent's
          trajectory-sum semantics from
          `verl/trainer/ppo/metric_utils.py:113-145`). This keeps these
          tensorboard tags directly comparable to pre-multi-turn runs,
          where each trajectory had exactly one reward placed at the last
          token.

        Diagnostic ``/last/...`` variants are also emitted whenever the
        per-turn data is available, so the previous last-turn view can be
        compared side-by-side without re-running.

        When per-turn data is absent (legacy paths: validation, single-turn
        ablations, callers other than `KernelAgentLoop`), the metric block
        falls back to last-turn-only computation against the same keys
        (`correctness`, `performance`, `compilation`, `is_speedup_positive`)
        so older callers keep working unchanged.

        The Fast@X threshold list lives in the reward config
        (`reward_model.reward_kwargs.fast_at_thresholds`); this hook discovers
        all `Fast@*` keys by prefix so additions/removals propagate without
        trainer-side changes.
        """
        super()._fit_collect_metrics(batch)

        def _mean(key: str):
            values = batch.non_tensor_batch.get(key)
            if values is None or len(values) == 0:
                return None
            return float(np.mean(np.asarray(values, dtype=np.float64)))

        for key, values in batch.non_tensor_batch.items():
            if not isinstance(key, str) or not key.startswith(FAST_AT_KEY_PREFIX):
                continue
            if values is None or len(values) == 0:
                continue
            self.metrics[f"kernel/{key}"] = float(np.mean(np.asarray(values, dtype=np.float64)))

        # Pull out the reward config once (used for both turn-mean and
        # the reward_term derived metrics).
        reward_kwargs = self.config.reward.get("reward_kwargs", None) if hasattr(self.config, "reward") else None
        try:
            speedup_eps = float(reward_kwargs.get("speedup_eps", 0.01)) if reward_kwargs else 0.01
        except Exception:
            speedup_eps = 0.01

        # Try the per-(traj, turn) aggregation first (preferred path).
        per_turn = self._collect_per_turn_kernel_stats(batch, speedup_eps=speedup_eps)
        if per_turn is not None:
            self.metrics["kernel/correctness/rate"] = per_turn["correctness_rate"]
            self.metrics["kernel/compilation/rate"] = per_turn["compilation_rate"]
            self.metrics["kernel/is_speedup_positive/rate"] = per_turn["is_speedup_positive_rate"]
            self.metrics["kernel/speedup/mean"] = per_turn["speedup_mean"]
            self.metrics["kernel/speedup/max"] = per_turn["speedup_max"]
            self.metrics["kernel/speedup/min"] = per_turn["speedup_min"]
            if "per_turn_reward_mean" in per_turn:
                self.metrics["kernel/per_turn_reward/mean"] = per_turn["per_turn_reward_mean"]
                self.metrics["kernel/per_turn_reward/max"] = per_turn["per_turn_reward_max"]
                self.metrics["kernel/per_turn_reward/min"] = per_turn["per_turn_reward_min"]
                # Overwrite the verl-core `critic/rewards/*` and
                # `critic/score/*` (trajectory-sum-of-token-rewards, parent
                # metric_utils.py:113-145) with the per-turn equivalents so
                # the existing TB tags stay comparable to pre-multi-turn
                # runs. Without this, t11 onwards reads a different
                # quantity than t9 under the same critic/rewards/* tags.
                # Since `token_level_rewards = token_level_scores` (no KL
                # in reward), the two critic/* triples are numerically
                # identical and we overwrite both for consistency.
                self.metrics["critic/rewards/mean"] = per_turn["per_turn_reward_mean"]
                self.metrics["critic/rewards/max"] = per_turn["per_turn_reward_max"]
                self.metrics["critic/rewards/min"] = per_turn["per_turn_reward_min"]
                self.metrics["critic/score/mean"] = per_turn["per_turn_reward_mean"]
                self.metrics["critic/score/max"] = per_turn["per_turn_reward_max"]
                self.metrics["critic/score/min"] = per_turn["per_turn_reward_min"]
            self.metrics["kernel/num_turns/total"] = per_turn["total_turn_count"]
            correctness_rate = per_turn["correctness_rate"]
            speedup_positive_rate = per_turn["is_speedup_positive_rate"]

            # True cross-(traj, turn) Fast@<X> mean, written under the bare
            # `kernel/Fast@<X>` tag so it matches the rest of the per-turn
            # `kernel/...` family (`kernel/correctness/rate`, etc.). The
            # reward manager emits the last-turn variant under
            # `Fast@<X>/last`, which the FAST_AT_KEY_PREFIX auto-aggregation
            # surfaces as `kernel/Fast@<X>/last`.
            fast_at_turn_avg = per_turn.get("fast_at_turn_avg", {}) or {}
            for k, rate in fast_at_turn_avg.items():
                self.metrics[f"kernel/{k}"] = rate

            # Diagnostic last-turn view (the previous behavior) — sourced
            # from the same per-trajectory scalars the legacy path uses.
            last_correctness = _mean("correctness")
            if last_correctness is not None:
                self.metrics["kernel/correctness/last/rate"] = last_correctness
            last_compilation = _mean("compilation")
            if last_compilation is not None:
                self.metrics["kernel/compilation/last/rate"] = last_compilation
            last_speedup_positive = _mean("is_speedup_positive")
            if last_speedup_positive is not None:
                self.metrics["kernel/is_speedup_positive/last/rate"] = last_speedup_positive
            last_speedup_values = batch.non_tensor_batch.get("performance")
            if last_speedup_values is not None and len(last_speedup_values) > 0:
                last_arr = np.asarray(last_speedup_values, dtype=np.float64)
                self.metrics["kernel/speedup/last/mean"] = float(np.mean(last_arr))
                self.metrics["kernel/speedup/last/max"] = float(np.max(last_arr))
                self.metrics["kernel/speedup/last/min"] = float(np.min(last_arr))
        else:
            # Legacy fallback: only last-turn scalars are available. Emit
            # them under the headline keys (same shape as pre-multi-turn).
            correctness_rate = _mean("correctness")
            if correctness_rate is not None:
                self.metrics["kernel/correctness/rate"] = correctness_rate
            compilation_rate = _mean("compilation")
            if compilation_rate is not None:
                self.metrics["kernel/compilation/rate"] = compilation_rate
            speedup_positive_rate = _mean("is_speedup_positive")
            if speedup_positive_rate is not None:
                self.metrics["kernel/is_speedup_positive/rate"] = speedup_positive_rate
            speedup_values = batch.non_tensor_batch.get("performance")
            if speedup_values is not None and len(speedup_values) > 0:
                speedup_arr = np.asarray(speedup_values, dtype=np.float64)
                self.metrics["kernel/speedup/mean"] = float(np.mean(speedup_arr))
                self.metrics["kernel/speedup/max"] = float(np.max(speedup_arr))
                self.metrics["kernel/speedup/min"] = float(np.min(speedup_arr))

        # Weighted contributions to reward = init_correct_weight * correctness
        # + init_performance_weight * is_speedup_positive (the coverage term,
        # if enabled, is logged separately via num_coverage / time_coverage).
        # With the new per-turn rates above, this now reflects the average
        # reward components per (traj, turn) — comparable to
        # ``critic/rewards/mean`` divided by the average number of turns.
        if reward_kwargs is not None:
            w_correct = float(reward_kwargs.get("init_correct_weight", 1.0))
            w_perf = float(reward_kwargs.get("init_performance_weight", 1.0))
            if correctness_rate is not None:
                self.metrics["kernel/reward_term/correctness"] = w_correct * correctness_rate
            if speedup_positive_rate is not None:
                self.metrics["kernel/reward_term/performance"] = w_perf * speedup_positive_rate

    def _collect_per_turn_kernel_stats(
        self,
        batch: DataProto,
        *,
        speedup_eps: float,
    ):
        """Flatten ``non_tensor_batch["turn_results"]`` and
        ``non_tensor_batch["turn_rewards"]`` across every assistant turn in
        the batch, then compute kernel-quality aggregates over the flat
        list. Returns ``None`` if no per-turn data is available (legacy
        callers / validation paths); the caller then falls back to
        last-turn scalars.

        ``fast_at_turn_avg`` in the returned dict is the **true**
        cross-(traj, turn) mean for each Fast@<X> threshold — i.e. every
        per-turn 0/1 indicator contributes one entry to a flat list and
        the mean is over that list. This deliberately differs from the
        per-trajectory ``Fast@<X>/best`` (any-turn-achieved) carried
        through ``reward_extra_info``.
        """
        turn_results_field = batch.non_tensor_batch.get("turn_results")
        if turn_results_field is None or len(turn_results_field) == 0:
            return None

        turn_rewards_field = batch.non_tensor_batch.get("turn_rewards")

        # Threshold list for per-turn Fast@X — resolved from the same
        # config node the reward manager uses, so additions/removals to
        # `reward_kwargs.fast_at_thresholds` flow through automatically.
        reward_kwargs = self.config.reward.get("reward_kwargs", None) if hasattr(self.config, "reward") else None
        fast_at_thresholds = resolve_thresholds(
            reward_kwargs.get("fast_at_thresholds", None) if reward_kwargs else None
        )

        flat_correctness: list = []
        flat_compiled: list = []
        flat_speedup_positive: list = []
        flat_speedup: list = []
        flat_per_turn_reward: list = []
        # Per-threshold flat list of 0/1 indicators across all (traj, turn).
        flat_fast_at: dict[str, list] = {}

        bs = len(turn_results_field)
        for i in range(bs):
            row_results = turn_results_field[i]
            if isinstance(row_results, np.ndarray):
                row_results = row_results.tolist()
            if not isinstance(row_results, list) or len(row_results) == 0:
                continue
            row_rewards = None
            if turn_rewards_field is not None:
                row_rewards = turn_rewards_field[i]
                if isinstance(row_rewards, np.ndarray):
                    row_rewards = row_rewards.tolist()

            for t_idx, result in enumerate(row_results):
                if not isinstance(result, dict):
                    continue
                t_correctness = bool(result.get("correctness", False))
                flat_correctness.append(1.0 if t_correctness else 0.0)
                flat_compiled.append(1.0 if bool(result.get("compiled", False)) else 0.0)
                s = result.get("speedup", 0.0)
                if s is None:
                    s = 0.0
                try:
                    s = float(s)
                except (TypeError, ValueError):
                    s = 0.0
                flat_speedup.append(s)
                flat_speedup_positive.append(1.0 if s >= (1.0 + speedup_eps) else 0.0)
                try:
                    turn_fast_at = compute_fast_at_indicators(t_correctness, s, fast_at_thresholds)
                except Exception:
                    turn_fast_at = {}
                for k, v in turn_fast_at.items():
                    flat_fast_at.setdefault(k, []).append(float(v))
                if row_rewards is not None and t_idx < len(row_rewards):
                    try:
                        flat_per_turn_reward.append(float(row_rewards[t_idx]))
                    except (TypeError, ValueError):
                        pass

        if not flat_correctness:
            return None

        out = {
            "correctness_rate": float(np.mean(flat_correctness)),
            "compilation_rate": float(np.mean(flat_compiled)),
            "is_speedup_positive_rate": float(np.mean(flat_speedup_positive)),
            "speedup_mean": float(np.mean(flat_speedup)),
            "speedup_max": float(np.max(flat_speedup)),
            "speedup_min": float(np.min(flat_speedup)),
            "total_turn_count": float(len(flat_correctness)),
            "fast_at_turn_avg": {k: float(np.mean(vs)) for k, vs in flat_fast_at.items()},
        }
        if flat_per_turn_reward:
            out["per_turn_reward_mean"] = float(np.mean(flat_per_turn_reward))
            out["per_turn_reward_max"] = float(np.max(flat_per_turn_reward))
            out["per_turn_reward_min"] = float(np.min(flat_per_turn_reward))
        return out

    def _fit_compute_reward(self, batch: DataProto) -> DataProto:
        """Compute reward (delegates to parent), then redistribute per-turn
        rewards onto each `response_mask=1` span's last token.

        Why: the parent's `_fit_compute_reward` calls our `KernelAsyncRewardManager`,
        which returns ONE scalar (sum of per-turn rewards). The reward loop
        places that scalar at the trajectory's last valid token. Without
        this hook, `compute_trloo_outcome_advantage` would see a sparse
        terminal reward and TRLOO would degenerate to RLOO-on-final-score.

        The agent loop (`KernelAgentLoop`) has already produced one reward
        per assistant turn (one `response_mask=1` span). We look those up
        from `non_tensor_batch["turn_rewards"]` (surfaced via
        `extra_fields` → `agent_loop.py:981-995`), find each row's spans
        from `response_mask`, and place each turn's reward at the span's
        last token. After this, `compute_trloo_outcome_advantage`'s
        span-sum step reads real per-turn rewards, matching the reference
        DR.Kernel TRLOO semantics.
        """
        batch = super()._fit_compute_reward(batch)
        self._redistribute_per_turn_rewards(batch)
        return batch

    def _redistribute_per_turn_rewards(self, batch: DataProto) -> None:
        """Rewrite `self.reward_tensor` so each turn's scalar reward sits
        at the last token of its `response_mask=1` span.

        No-op if the batch was not produced by `KernelAgentLoop` (e.g.
        validation or ablations where `turn_rewards` is absent) — in that
        case the parent's terminal placement is left alone and TRLOO
        falls back to the old sparse-terminal-reward behavior.
        """
        turn_rewards_field = batch.non_tensor_batch.get("turn_rewards")
        if turn_rewards_field is None:
            return
        response_mask = batch.batch.get("response_mask")
        if response_mask is None or self.reward_tensor is None:
            return

        bs = response_mask.shape[0]
        new_scores = torch.zeros_like(self.reward_tensor)
        rewrote = 0
        for i in range(bs):
            row_turn_rewards = turn_rewards_field[i]
            if row_turn_rewards is None:
                continue
            # `extra_fields` arrives as object dtype np.array entries;
            # each entry is the original Python list from the agent loop.
            if isinstance(row_turn_rewards, np.ndarray):
                row_turn_rewards = row_turn_rewards.tolist()
            if not isinstance(row_turn_rewards, (list, tuple)) or len(row_turn_rewards) == 0:
                continue

            spans = self._mask_runs(response_mask[i].tolist())
            if len(spans) == 0:
                continue
            # Tolerate small length mismatches between per-turn rewards
            # and inferred spans: response_length truncation can chop the
            # final span; the agent loop may have appended a rewardless
            # eval. Align from the front (first N spans ↔ first N
            # rewards) so the earliest turns always get their signal.
            n = min(len(spans), len(row_turn_rewards))
            if n != len(spans) or n != len(row_turn_rewards):
                logger.warning(
                    "[DrKernel] turn_rewards/span count mismatch at row %d: spans=%d turn_rewards=%d (using first %d)",
                    i,
                    len(spans),
                    len(row_turn_rewards),
                    n,
                )
            for (s, e), r in zip(spans[:n], row_turn_rewards[:n], strict=False):
                # Place reward at the LAST token of this span.
                # If the span has any width, e-1 is inside it.
                if e > s:
                    new_scores[i, e - 1] = float(r)
            rewrote += 1

        if rewrote == 0:
            return  # nothing to redistribute, keep parent's placement

        self.reward_tensor = new_scores
        # Keep `batch.batch["rm_scores"]` consistent with what
        # `_fit_compute_advantage` reads — `extract_reward` already wired
        # `self.reward_tensor` from `rm_scores`, so update both.
        if "rm_scores" in batch.batch:
            batch.batch["rm_scores"] = new_scores

    @staticmethod
    def _mask_runs(row_mask: list) -> list:
        """Return contiguous runs of `1` in a 0/1 list as `(start, end)`
        half-open intervals — same conventions as
        `compute_trloo_outcome_advantage` (`core_algos.py:683-697`)."""
        spans: list = []
        in_span = False
        start = 0
        for j, m in enumerate(row_mask):
            if m == 1 and not in_span:
                start = j
                in_span = True
            elif m == 0 and in_span:
                spans.append((start, j))
                in_span = False
        if in_span:
            spans.append((start, len(row_mask)))
        return spans

    def _compute_old_log_prob(self, batch: DataProto):
        """Override of `FullyAsyncTrainer._compute_old_log_prob` that calls
        the grandparent (`RayPPOTrainer._compute_old_log_prob`) via explicit
        class reference instead of `super()`.

        Why: the parent's body
        (`verl/experimental/fully_async_policy/fully_async_trainer.py:484-504`)
        does `super()._compute_old_log_prob(batch)` from within
        `FullyAsyncTrainer`, whose `__class__` cell points at the original
        unwrapped class. Under the `_RemoteFullyAsyncTrainer.__ray_actor_class__`
        / re-`@ray.remote` subclassing pattern used here, that cell does not
        match the actor instance's live MRO and the call raises
        ``TypeError: super(type, obj): obj must be an instance or subtype of
        type`` — only triggered when ``rollout_correction.bypass_mode=False``
        forces this code path on (`separation/ray_trainer.py:494-505`).

        This override mirrors the parent's save/restore logic verbatim but
        bypasses the broken super() by calling
        `RayPPOTrainer._compute_old_log_prob` (which is self-contained and
        uses no super() itself) with `self` explicitly. If upstream changes
        the save/restore semantics, mirror those changes here.
        """
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer

        if self.local_trigger_step == 1:
            self.actor_rollout_wg.save_model_to_cpu(1)
            old_log_prob, old_log_prob_mfu = RayPPOTrainer._compute_old_log_prob(self, batch)
        else:
            self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
            self.actor_rollout_wg.restore_model_from_cpu(1)
            old_log_prob, old_log_prob_mfu = RayPPOTrainer._compute_old_log_prob(self, batch)
            self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
            self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
        return old_log_prob, old_log_prob_mfu

    def _fit_filter_batch(self, batch: DataProto) -> DataProto:
        """Optional MRS batch filter — see module docstring for semantics."""
        cfg = self.config.algorithm.get("batch_filter", None)
        if cfg is None or not bool(cfg.get("enable", False)):
            return batch

        if not getattr(self, "_drkernel_batch_filter_built", False):
            # Lazy import keeps recipe.drkernel imports out of the verl-core
            # FullyAsyncTrainer's import graph.
            from recipe.drkernel.config.batch_filter import build_ppo_filter_config
            from recipe.drkernel.filters import PPOBatchFilter

            ppo_filter_cfg = build_ppo_filter_config(cfg)
            data_cfg = OmegaConf.to_container(self.config.data, resolve=True) if hasattr(self.config, "data") else None
            self._drkernel_batch_filter = PPOBatchFilter(ppo_filter_cfg, data_config=data_cfg)
            self._drkernel_batch_filter_built = True

        from recipe.drkernel.filters import filter_dataproto

        with marked_timer("filter_batch", self.timing_raw, color="cyan"):
            original_size = batch.batch["response_mask"].shape[0]
            filtered, metrics = filter_dataproto(batch, self._drkernel_batch_filter, global_step=self.global_steps)
            filtered_size = filtered.batch["response_mask"].shape[0]

        if metrics.get("batch/critical_empty_batch", 0) == 1:
            logger.warning(
                "[DrKernel] batch filter rejected all %d samples at step %d; falling back to unfiltered batch.",
                original_size,
                self.global_steps,
            )
            # filter_dataproto already returns the original batch in this case.
        elif filtered_size < original_size:
            logger.info(
                "[DrKernel] batch filter kept %d/%d samples (selection_rate=%.2f%%) at step %d",
                filtered_size,
                original_size,
                100.0 * metrics.get("batch/selection_rate", 0),
                self.global_steps,
            )

        for k, v in metrics.items():
            if isinstance(v, (int, float, bool)):
                self.metrics[f"batch_filter/{k}"] = v

        return filtered

    async def fit_step(self, batch_dict: dict = None):
        """fit_step with the MRS batch-filter hook inserted between
        `_fit_compute_log_prob` and `_fit_compute_ref_log_prob`.

        Body kept verbatim from the parent's `fit_step` apart from one new
        line: `batch = self._fit_filter_batch(batch)`. If the parent's body
        ever changes upstream, mirror those changes here too.
        """
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        self.future_reward = None
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch = await self._fit_generate(None)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_filter_batch(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_local_step()
            await self._fit_update_weights()
            self._fit_dump_data(batch)

        await self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_postprocess_step()


# Re-decorate as a Ray remote actor with the same resource spec as the parent.
DrKernelFullyAsyncTrainer = ray.remote(num_cpus=10)(DrKernelFullyAsyncTrainerImpl)
