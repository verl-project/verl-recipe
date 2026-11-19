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
"""DeepSWE-style TPU recipe hooks for 2-slot GKE sandbox lookahead and just-in-time evaluation pools.

This module lives in the TPU DeepSWE recipe folder (``verl-recipe/deepswe_tpu/deepswe.py``) and is loaded through
``VERL_USE_EXTERNAL_MODULES`` so core ``verl`` trainer files remain untouched.
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

from tensordict import TensorDict
from uni_agent.sandbox.gke_agent_sandbox import (
    PrewarmBatchIterator,
    close_lookahead_warmer,
    ensure_lookahead_batch_warm,
    extract_sandbox_images,
    finish_lookahead_batch_async,
    load_image_map_from_task_config,
    prefetch_lookahead_batch_async,
)

from verl.trainer.ppo.v1.trainer_base import TRAINER_REGISTRY, PPOTrainer
from verl.trainer.ppo.v1.trainer_separate_async import PPOTrainerSeparateAsync
from verl.trainer.ppo.v1.trainer_sync import PPOTrainerSync
from verl.utils import tensordict_utils as tu

logger = logging.getLogger(__name__)


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
        return extract_sandbox_images(batch_or_items, image_map=self._sandbox_image_map)

    def _init_sandbox_lookahead(self: PPOTrainer) -> None:
        if getattr(self, "_sandbox_lookahead_initialized", False):
            return
        self._sandbox_lookahead_initialized = True
        self._prewarm_batch_iterator = None
        self._sandbox_image_map = None
        self._eval_sandbox_images = None
        self._eval_sandbox_batch_id = None

        if os.getenv("VERL_SANDBOX_LOOKAHEAD", "1").strip().lower() in ("0", "false", "no"):
            return

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
        if batch_id is not None:
            finish_lookahead_batch_async(batch_id, wait=False)

    def _finish_sampled_sandbox_batch(self: PPOTrainer) -> None:
        if self._prewarm_batch_iterator is not None and self.local_trigger_step == self.parameter_sync_step - 1:
            self._prewarm_batch_iterator.finish_oldest_inflight_batch(wait=False)
            self._prefetch_eval_sandboxes()

    def _close_sandbox_lookahead(self: PPOTrainer) -> None:
        self._eval_sandbox_batch_id = None
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
            if getattr(self, "_prewarm_batch_iterator", None) is not None:
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


def register_deepswe_trainers() -> None:
    """Register DeepSWE lookahead trainers in ``TRAINER_REGISTRY``."""
    TRAINER_REGISTRY["separate_async"] = DeepSWESeparateAsyncPPOTrainer
    TRAINER_REGISTRY["sync"] = DeepSWESyncPPOTrainer


register_deepswe_trainers()
