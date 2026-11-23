# Copyright 2026 Google LLC
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
"""DeepSWE-style TPU recipe hooks: warm GKE sandboxes one training step ahead, and just in time for evaluation.

This module lives in the TPU DeepSWE recipe folder (``verl-recipe/deepswe_tpu/deepswe.py``) and is loaded through
``VERL_USE_EXTERNAL_MODULES`` so core ``verl`` trainer files remain untouched. The decisions about what to warm are
in ``sandbox_lookahead.py``; the sandbox provider only supplies ``prewarm`` and ``unwarm``.
"""

from __future__ import annotations

from collections import defaultdict
import logging
import os
from typing import Any

import numpy as np
from tensordict import TensorDict
import torch
from uni_agent.sandbox.gke import GKEAgentSandbox
from uni_agent.tasks import TaskConfigResolver

from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, register_adv_est
from verl.trainer.ppo.v1.trainer_base import TRAINER_REGISTRY, PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu

from .sandbox_lookahead import LookaheadBatches, PoolKey, WarmPools, sandbox_requests

logger = logging.getLogger(__name__)

_EVAL_BATCH = "eval"


class _DeepSWESandboxLookaheadMixin:
    """Mixin adding sandbox warming to ``PPOTrainer``: one training batch ahead, and just in time for evaluation.

    Evaluation sandboxes are not kept warm for the whole run. They start warming in the background once the rollouts
    of a step that ends in validation are sampled (so they warm during that step's update and weight sync), validation
    waits for them, and they are released as soon as validation ends.
    """

    _sandbox_lookahead_initialized: bool = False
    _warm_pools: WarmPools | None = None
    _lookahead_batches: LookaheadBatches | None = None
    _task_config_resolver: TaskConfigResolver | None = None
    _eval_sandbox_requests: dict[PoolKey, int] | None = None
    _eval_sandboxes_warming: bool = False

    def _resolve_task_config_path(self: PPOTrainer) -> str | None:
        try:
            custom = self.config.actor_rollout_ref.rollout.get("custom", {}) or {}
            af = custom.get("agent_framework", {}) or {}
            runners = af.get("agent_runners", {}) or {}
            task_runner = runners.get("task", {}) or {}
            runner_kwargs = task_runner.get("runner_kwargs", {}) or {}
            path = runner_kwargs.get("task_config_path")
            return str(path) if path else None
        except Exception:
            return None

    def _train_sandbox_requests(self: PPOTrainer, batch: Any) -> dict[PoolKey, int]:
        rollout_n = int(self.config.actor_rollout_ref.rollout.get("n", 8) or 8)
        return sandbox_requests(batch, self._task_config_resolver, per_sample=rollout_n)

    def _init_sandbox_lookahead(self: PPOTrainer) -> None:
        if getattr(self, "_sandbox_lookahead_initialized", False):
            return
        self._sandbox_lookahead_initialized = True

        if os.getenv("VERL_SANDBOX_LOOKAHEAD", "1").strip().lower() in ("0", "false", "no"):
            return

        path = self._resolve_task_config_path()
        self._task_config_resolver = TaskConfigResolver.from_file(path) if path else TaskConfigResolver()
        eval_replicas = int(
            os.getenv(
                "SANDBOX_EVAL_REPLICAS",
                str(self.config.actor_rollout_ref.rollout.val_kwargs.get("n", 1) or 1),
            )
        )
        self._eval_sandbox_requests = (
            sandbox_requests(self.val_dataset, self._task_config_resolver, per_sample=eval_replicas) or None
        )
        self._warm_pools = WarmPools(GKEAgentSandbox.prewarm, GKEAgentSandbox.unwarm)
        self._lookahead_batches = LookaheadBatches(
            self._raw_next_train_batch, self._train_sandbox_requests, self._warm_pools
        )

    def _prime_sandbox_lookahead(self: PPOTrainer) -> None:
        self._init_sandbox_lookahead()
        if self._lookahead_batches is not None:
            self._lookahead_batches.prime()

    def _step_ends_in_validation(self: PPOTrainer) -> bool:
        """Mirror ``PPOTrainer._fit``: validation follows the last step and every ``test_freq``-th step."""
        test_freq = int(self.config.trainer.get("test_freq", 0) or 0)
        if test_freq <= 0:
            return False
        return self.global_steps >= self.total_training_steps or self.global_steps % test_freq == 0

    def _start_eval_sandboxes(self: PPOTrainer) -> None:
        if self._warm_pools is None or not self._eval_sandbox_requests or self._eval_sandboxes_warming:
            return
        self._eval_sandboxes_warming = True
        logger.info(
            "Pre-warming %d evaluation sandbox images for validation at step %d",
            len(self._eval_sandbox_requests),
            self.global_steps,
        )
        self._warm_pools.start(_EVAL_BATCH, self._eval_sandbox_requests)

    def _wait_eval_sandboxes_ready(self: PPOTrainer) -> None:
        """Block until the evaluation sandboxes are Ready, warming them now if no prefetch was started."""
        self._init_sandbox_lookahead()
        self._start_eval_sandboxes()
        if self._eval_sandboxes_warming:
            self._warm_pools.wait(_EVAL_BATCH)

    def _release_eval_sandboxes(self: PPOTrainer) -> None:
        """Let the evaluation warm pools be deleted now that validation has ended."""
        if getattr(self, "_eval_sandboxes_warming", False):
            self._eval_sandboxes_warming = False
            self._warm_pools.finish(_EVAL_BATCH)

    def _finish_sampled_sandbox_batch(self: PPOTrainer) -> None:
        if self._lookahead_batches is not None and self.local_trigger_step == self.parameter_sync_step - 1:
            self._lookahead_batches.finish_oldest()
            if self._step_ends_in_validation():
                self._start_eval_sandboxes()

    def _close_sandbox_lookahead(self: PPOTrainer) -> None:
        self._eval_sandboxes_warming = False
        self._lookahead_batches = None
        pools, self._warm_pools = getattr(self, "_warm_pools", None), None
        if pools is not None:
            pools.close()

    def _raw_next_train_batch(self: PPOTrainer, num_prompts: int | None = None) -> TensorDict:
        return super()._next_train_batch(num_prompts)  # type: ignore[misc]

    def _next_train_batch(self: PPOTrainer, num_prompts: int | None = None) -> TensorDict:
        train_batch_size = self.config.data.train_batch_size
        if num_prompts is None:
            num_prompts = train_batch_size
        lookahead = getattr(self, "_lookahead_batches", None)
        if lookahead is not None and num_prompts == train_batch_size:
            batch = lookahead.next_batch()
        else:
            batch = self._raw_next_train_batch(num_prompts)
            if lookahead is not None:
                # A batch of another size bypasses the lookahead, such as the async trainer's warm-up batches, which
                # arrive as one batch several steps long. Warm it now; it is released like any other batch.
                lookahead.warm_now(batch, num_prompts / train_batch_size)
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch


# If core verl already handles post-sample batch refill (either via an env-var hook
# ``_should_refill_after_sample`` or via the config-driven ``_refill_after_sample`` from
# verl-project/verl#8661 when enabled), defer to it so we never submit two batches per step.
_CORE_HAS_ENV_REFILL_AFTER_SAMPLE = hasattr(PPOTrainerSeparateAsync, "_should_refill_after_sample")


class DeepSWESeparateAsyncPPOTrainer(_DeepSWESandboxLookaheadMixin, PPOTrainerSeparateAsync):
    """Separate-async PPO trainer with one-step-ahead GKE sandbox warming for DeepSWE."""

    def _should_refill_after_sample(self) -> bool:
        num_warmup = self.config.trainer.v1.separate_async.get("num_warmup_batches", 0) or 0
        return (
            num_warmup > 0
            and not self.hybrid_rollout_config.enable_switch
            and os.getenv("VERL_REFILL_AFTER_SAMPLE", "0").strip().lower() in ("1", "true", "yes")
        )

    def _recipe_refills(self) -> bool:
        if _CORE_HAS_ENV_REFILL_AFTER_SAMPLE or bool(getattr(self, "_refill_after_sample", False)):
            return False
        return self._should_refill_after_sample()

    def on_train_begin(self) -> None:
        self._prime_sandbox_lookahead()
        super().on_train_begin()

    def on_train_end(self) -> None:
        try:
            super().on_train_end()
        finally:
            self._close_sandbox_lookahead()

    def on_validate_begin(self) -> None:
        self._wait_eval_sandboxes_ready()
        super().on_validate_begin()

    def on_validate_end(self) -> None:
        try:
            super().on_validate_end()
        finally:
            self._release_eval_sandboxes()

    def on_sample_end(self) -> None:
        self._finish_sampled_sandbox_batch()
        super().on_sample_end()
        if (
            self._recipe_refills()
            and self.local_trigger_step == self.parameter_sync_step - 1
            and self.global_steps < self.total_training_steps
        ):
            self._add_batch_to_generate()

    def prepare_step(self) -> dict:
        if self._recipe_refills():
            return self._wait_for_sampleable_and_switch()
        return super().prepare_step()


class DeepSWESyncPPOTrainer(_DeepSWESandboxLookaheadMixin, PPOTrainerSync):
    """Synchronous PPO trainer with one-step-ahead GKE sandbox warming for DeepSWE."""

    def on_train_begin(self) -> None:
        self._prime_sandbox_lookahead()
        super().on_train_begin()

    def on_train_end(self) -> None:
        try:
            super().on_train_end()
        finally:
            self._close_sandbox_lookahead()

    def on_validate_begin(self) -> None:
        self._wait_eval_sandboxes_ready()
        super().on_validate_begin()

    def on_validate_end(self) -> None:
        try:
            super().on_validate_end()
        finally:
            self._release_eval_sandboxes()

    def on_sample_end(self) -> None:
        self._finish_sampled_sandbox_batch()
        super().on_sample_end()


_LOO_VAR_NOISE_FLOOR = 1e-5
_MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE = 2


@register_adv_est("grpo_loo")
@register_adv_est("grpo-loo")
def compute_grpo_loo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Any = None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group-relative outcome advantage with leave-one-out baseline and LOO std scale."""
    del config, kwargs
    scores = token_level_rewards.sum(dim=-1)
    advantages_1d = torch.zeros_like(scores, dtype=torch.float32)

    id2indices: dict[Any, list[int]] = defaultdict(list)
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2indices[index[i]].append(i)

        for row_indices in id2indices.values():
            group_size = len(row_indices)
            if group_size < _MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE:
                continue
            group_scores = scores[row_indices].float()
            group_mean = group_scores.mean()
            centered = group_scores - group_mean
            loo_centered = centered * (group_size / (group_size - 1))
            sum_sq = (centered * centered).sum()
            if group_size > 2 and sum_sq > 0:
                loo_var = (sum_sq - (centered * centered) * (group_size / (group_size - 1))) / (group_size - 2)
                group_adv = torch.where(
                    loo_var > _LOO_VAR_NOISE_FLOOR * sum_sq,
                    loo_centered / (torch.sqrt(loo_var.clamp_min(0.0)) + epsilon),
                    torch.zeros_like(loo_centered),
                )
            else:
                group_adv = loo_centered
            advantages_1d[row_indices] = group_adv

    advantages = advantages_1d.to(dtype=token_level_rewards.dtype).unsqueeze(-1) * response_mask
    return advantages, advantages


def register_deepswe_trainers() -> None:
    """Register DeepSWE lookahead trainers in ``TRAINER_REGISTRY`` and ``grpo_loo`` in ``ADV_ESTIMATOR_REGISTRY``."""
    TRAINER_REGISTRY["separate_async"] = DeepSWESeparateAsyncPPOTrainer
    TRAINER_REGISTRY["sync"] = DeepSWESyncPPOTrainer
    ADV_ESTIMATOR_REGISTRY["grpo_loo"] = compute_grpo_loo_outcome_advantage
    ADV_ESTIMATOR_REGISTRY["grpo-loo"] = compute_grpo_loo_outcome_advantage


register_deepswe_trainers()
