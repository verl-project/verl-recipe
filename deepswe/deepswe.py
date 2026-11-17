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
"""DeepSWE recipe hooks for 2-slot lookahead GKE / Kubernetes sandbox pre-warming and just-in-time evaluation pools.

This module lives in the DeepSWE recipe folder (``verl-recipe/deepswe/deepswe.py``) and is loaded through
``VERL_USE_EXTERNAL_MODULES`` so core ``verl`` trainer files remain untouched.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import ADV_ESTIMATOR_REGISTRY, register_adv_est
from verl.trainer.ppo.v1.trainer_base import TRAINER_REGISTRY, PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu

try:
    from uni_agent.sandbox.gke_agent_sandbox import (
        PrewarmBatchIterator,
        close_lookahead_warmer,
        ensure_lookahead_batch_warm,
        extract_sandbox_images,
        finish_lookahead_batch_async,
        load_image_map_from_task_config,
        prefetch_lookahead_batch_async,
    )
except ImportError:
    PrewarmBatchIterator = None  # type: ignore[assignment]
    close_lookahead_warmer = None  # type: ignore[assignment]
    ensure_lookahead_batch_warm = None  # type: ignore[assignment]
    extract_sandbox_images = None  # type: ignore[assignment]
    finish_lookahead_batch_async = None  # type: ignore[assignment]
    load_image_map_from_task_config = None  # type: ignore[assignment]
    prefetch_lookahead_batch_async = None  # type: ignore[assignment]

try:
    from uni_agent.framework.entry import AgentFrameworkRolloutAdapter
except ImportError:
    AgentFrameworkRolloutAdapter = object  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

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
    """Group-relative outcome advantage with leave-one-out baseline and LOO std scale (GRPO-LOO).

    Excludes each trajectory from both the baseline mean and the sample standard deviation used to scale its
    advantage:
      - LOO centered score: ``(r_i - mean(r)) * (G / (G - 1)) = r_i - mean(r_{-i})``
      - LOO variance: ``(sum_j (r_j - mean(r))^2 - (r_i - mean(r))^2 * (G / (G - 1))) / (G - 2)``
      - Divided by ``sqrt(loo_var) + epsilon`` when ``loo_var > 1e-5 * sum_sq`` (yielding [-3.615, +3.615] at G=16).
    """
    del config, kwargs
    scores = token_level_rewards.sum(dim=-1)
    advantages_1d = torch.zeros_like(scores, dtype=torch.float32)

    id2indices: dict[Any, list[int]] = defaultdict(list)
    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2indices[index[i]].append(i)

        for row_indices in id2indices.values():
            valid_count = len(row_indices)
            if valid_count < _MIN_VALID_TRAJECTORIES_FOR_ADVANTAGE:
                continue
            idx_tensor = torch.tensor(row_indices, device=scores.device, dtype=torch.long)
            group_scores = scores.index_select(0, idx_tensor).to(dtype=torch.float32)
            n_others = float(valid_count - 1)
            group_mean = group_scores.mean()
            centered = group_scores - group_mean
            scale = float(valid_count) / n_others
            adv = centered * scale

            sum_sq = torch.sum(centered * centered)
            if valid_count - 1 >= 2:
                loo_var = (sum_sq - (centered * centered) * scale) / float(valid_count - 2)
                loo_var = torch.where(
                    loo_var > _LOO_VAR_NOISE_FLOOR * sum_sq,
                    loo_var,
                    torch.zeros_like(loo_var),
                )
            else:
                loo_var = torch.zeros_like(centered)
            loo_std = torch.sqrt(loo_var)
            adv = torch.where(loo_std > 0, adv / (loo_std + epsilon), adv)
            advantages_1d.index_copy_(0, idx_tensor, adv)

        advantages = advantages_1d.to(dtype=scores.dtype).unsqueeze(-1) * response_mask
    return advantages, advantages


class _DeepSWESandboxLookaheadMixin:
    """Mixin adding 2-slot ``PrewarmBatchIterator`` lookahead and just-in-time eval sandbox warming to ``PPOTrainer``.

    Evaluation sandboxes are not kept warm for the whole run. They start warming in the background once the rollouts
    of a step that ends in validation are sampled (so they warm during that step's update and weight sync), validation
    waits for them, and they are released as soon as validation ends.
    """

    _prewarm_batch_iterator: PrewarmBatchIterator | None = None
    _sandbox_lookahead_initialized: bool = False
    _sandbox_image_map: list[Any] | None = None
    _eval_sandbox_images: list[str] | None = None
    _eval_sandbox_replicas: int = 1
    _eval_sandbox_batch_id: str | None = None

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

    def _extract_batch_images(self: PPOTrainer, batch_or_items: Any) -> list[str]:
        if extract_sandbox_images is None:
            return []
        return extract_sandbox_images(batch_or_items, image_map=self._sandbox_image_map)

    def _init_sandbox_lookahead(self: PPOTrainer) -> None:
        if getattr(self, "_sandbox_lookahead_initialized", False):
            return
        self._sandbox_lookahead_initialized = True
        self._prewarm_batch_iterator = None
        self._sandbox_image_map = None
        self._eval_sandbox_images = None
        self._eval_sandbox_batch_id = None

        if (
            PrewarmBatchIterator is None
            or extract_sandbox_images is None
            or ensure_lookahead_batch_warm is None
            or prefetch_lookahead_batch_async is None
            or finish_lookahead_batch_async is None
        ):
            return
        if os.getenv("VERL_SANDBOX_LOOKAHEAD", "1").strip().lower() in ("0", "false", "no"):
            return

        if load_image_map_from_task_config is not None:
            self._sandbox_image_map = load_image_map_from_task_config(self._resolve_task_config_path()) or None

        self._eval_sandbox_images = self._extract_batch_images(self.val_dataset) or None
        self._eval_sandbox_replicas = int(
            os.getenv(
                "SANDBOX_EVAL_REPLICAS",
                str(self.config.actor_rollout_ref.rollout.val_kwargs.get("n", 1) or 1),
            )
        )

        rollout_n = int(self.config.actor_rollout_ref.rollout.get("n", 8) or 8)
        self._prewarm_batch_iterator = PrewarmBatchIterator(
            fetch_batch_fn=self._raw_next_train_batch,
            extract_images_fn=self._extract_batch_images,
            ensure_warm_fn=ensure_lookahead_batch_warm,
            prefetch_async_fn=prefetch_lookahead_batch_async,
            finish_batch_fn=finish_lookahead_batch_async,
            replicas_override=rollout_n,
        )

    def _prime_sandbox_lookahead(self: PPOTrainer) -> None:
        self._init_sandbox_lookahead()
        if self._prewarm_batch_iterator is not None:
            self._prewarm_batch_iterator.prime()

    def _step_ends_in_validation(self: PPOTrainer) -> bool:
        """Mirror ``PPOTrainer._fit``: validation follows the last step and every ``test_freq``-th step."""
        test_freq = int(self.config.trainer.get("test_freq", 0) or 0)
        if test_freq <= 0:
            return False
        return self.global_steps >= self.total_training_steps or self.global_steps % test_freq == 0

    def _prefetch_eval_sandboxes(self: PPOTrainer) -> None:
        """Start warming the evaluation sandboxes in the background ahead of this step's validation."""
        if (
            self._prewarm_batch_iterator is None
            or not self._eval_sandbox_images
            or self._eval_sandbox_batch_id is not None
            or not self._step_ends_in_validation()
        ):
            return
        self._eval_sandbox_batch_id = f"eval_step_{self.global_steps}"
        logger.info(
            "Pre-warming %d evaluation sandbox images for validation at step %d (replicas=%d)",
            len(set(self._eval_sandbox_images)),
            self.global_steps,
            self._eval_sandbox_replicas,
        )
        prefetch_lookahead_batch_async(
            self._eval_sandbox_batch_id,
            self._eval_sandbox_images,
            replicas_override=self._eval_sandbox_replicas,
        )

    def _wait_eval_sandboxes_ready(self: PPOTrainer) -> None:
        """Block until the evaluation sandboxes are Ready, warming them now if no prefetch was started."""
        self._init_sandbox_lookahead()
        if self._prewarm_batch_iterator is None or not self._eval_sandbox_images:
            return
        if self._eval_sandbox_batch_id is None:
            self._eval_sandbox_batch_id = f"eval_step_{self.global_steps}"
        ensure_lookahead_batch_warm(
            self._eval_sandbox_batch_id,
            self._eval_sandbox_images,
            replicas_override=self._eval_sandbox_replicas,
            wait=True,
        )

    def _release_eval_sandboxes(self: PPOTrainer) -> None:
        """Retire the evaluation sandbox batch so its warm pools are torn down after validation."""
        batch_id, self._eval_sandbox_batch_id = getattr(self, "_eval_sandbox_batch_id", None), None
        if batch_id is not None and finish_lookahead_batch_async is not None:
            finish_lookahead_batch_async(batch_id, wait=False)

    def _finish_sampled_sandbox_batch(self: PPOTrainer) -> None:
        if self._prewarm_batch_iterator is not None and self.local_trigger_step == self.parameter_sync_step - 1:
            self._prewarm_batch_iterator.finish_oldest_inflight_batch(wait=False)
            self._prefetch_eval_sandboxes()

    def _close_sandbox_lookahead(self: PPOTrainer) -> None:
        self._eval_sandbox_batch_id = None
        if close_lookahead_warmer is not None:
            close_lookahead_warmer()

    def _raw_next_train_batch(self: PPOTrainer, num_prompts: int | None = None) -> TensorDict:
        return super()._next_train_batch(num_prompts)  # type: ignore[misc]

    def _next_train_batch(self: PPOTrainer, num_prompts: int | None = None) -> TensorDict:
        train_batch_size = self.config.data.train_batch_size
        if num_prompts is None:
            num_prompts = train_batch_size
        if getattr(self, "_prewarm_batch_iterator", None) is not None and num_prompts == train_batch_size:
            batch = self._prewarm_batch_iterator.next_batch()
        else:
            batch = self._raw_next_train_batch(num_prompts)
            if (
                getattr(self, "_prewarm_batch_iterator", None) is not None
                and extract_sandbox_images is not None
                and ensure_lookahead_batch_warm is not None
            ):
                imgs = self._extract_batch_images(batch)
                if imgs:
                    rollout_n = int(self.config.actor_rollout_ref.rollout.get("n", 8) or 8)
                    ensure_lookahead_batch_warm(
                        f"partial_{self.global_steps}_{uuid.uuid4().hex[:6]}",
                        imgs,
                        replicas_override=rollout_n,
                        wait=True,
                    )
        tu.assign_non_tensor_data(batch, "global_steps", self.global_steps)
        return batch


# If core verl already handles post-sample batch refill (either via an env-var hook
# ``_should_refill_after_sample`` or via the config-driven ``_refill_after_sample`` from
# verl-project/verl#8661 when enabled), defer to it so we never submit two batches per step.
_CORE_HAS_ENV_REFILL_AFTER_SAMPLE = hasattr(PPOTrainerSeparateAsync, "_should_refill_after_sample")


class DeepSWESeparateAsyncPPOTrainer(_DeepSWESandboxLookaheadMixin, PPOTrainerSeparateAsync):
    """Separate-async PPO trainer with 2-slot GKE sandbox lookahead pre-warming for DeepSWE."""

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
    """Synchronous PPO trainer with 2-slot GKE sandbox lookahead pre-warming for DeepSWE."""

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


class DeepSWERolloutAdapter(AgentFrameworkRolloutAdapter):
    """Rollout adapter entry point that also ensures DeepSWE trainer hooks are registered."""

    @classmethod
    def create(cls, **kwargs: Any) -> Any:
        register_deepswe_trainers()
        return super().create(**kwargs)


def register_deepswe_trainers() -> None:
    """Register DeepSWE lookahead trainers in ``TRAINER_REGISTRY`` and ``grpo_loo`` in ``ADV_ESTIMATOR_REGISTRY``."""
    TRAINER_REGISTRY["separate_async"] = DeepSWESeparateAsyncPPOTrainer
    TRAINER_REGISTRY["sync"] = DeepSWESyncPPOTrainer
    ADV_ESTIMATOR_REGISTRY["grpo_loo"] = compute_grpo_loo_outcome_advantage
    ADV_ESTIMATOR_REGISTRY["grpo-loo"] = compute_grpo_loo_outcome_advantage


register_deepswe_trainers()
