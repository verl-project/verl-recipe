# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""
DanceGRPO Ray Trainer.

Extends :class:`~verl.trainer.ppo.ray_trainer.RayPPOTrainer` with a
diffusion-specific ``fit()`` loop and a custom ``_create_dataloader`` that
uses :mod:`recipe.dance_grpo.dataset` instead of the default verl dataset.

Changes from the base class:
- ``_create_dataloader``: imports from :mod:`recipe.dance_grpo.dataset`.
- ``fit``: diffusion-aware training loop (no text tokenizer / response mask;
  advantage computation is entirely inside :class:`DiffusionActor.update_policy`).
- ``init_workers``: only supports ``fsdp`` / ``fsdp2`` actor strategy.
"""

import logging
import os
from collections import defaultdict
from typing import Optional

import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, ResourcePoolManager
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.tracking import ValidationGenerationsLogger

__all__ = ["RayDANCETrainer"]

_py_logger = logging.getLogger(__name__)
_py_logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class RayDANCETrainer(RayPPOTrainer):
    """Ray trainer for DanceGRPO diffusion training.

    Inherits the full verl :class:`RayPPOTrainer` infrastructure (resource
    pools, worker groups, checkpoint management, validation) but overrides the
    core training loop methods that are specific to diffusion RL training.

    Key overrides:
    - :meth:`_create_dataloader`: loads :class:`PickAPicDataset` via the
      :mod:`recipe.dance_grpo.dataset` module.
    - :meth:`fit`: diffusion-aware loop — no critic, no KL token-level rewards,
      no advantage estimation on the driver process (all done inside the worker).
    - :meth:`init_workers`: only ``fsdp`` / ``fsdp2`` strategy; no Megatron /
      vLLM / SGLang variants.
    """

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """Initialise the DanceGRPO trainer.

        All arguments mirror :class:`RayPPOTrainer.__init__`.  We avoid
        calling ``super().__init__`` here because the base class constructor
        references modules (e.g. KL controllers, val logger) that do not apply
        to diffusion training.  Instead we replicate only the fields that are
        actually used.

        Args:
            config: Hydra / OmegaConf config tree.
            tokenizer: May be ``None`` for image-only training.
            role_worker_mapping: Maps :class:`Role` to the remote worker class.
            resource_pool_manager: Ray resource pool configuration.
            ray_worker_group_cls: Worker group class (default: RayWorkerGroup).
            processor: Optional HuggingFace processor (unused for text-only).
            reward_fn: Optional callable reward function (unused; reward is
                embedded in the worker).
            val_reward_fn: Optional callable validation reward function.
            train_dataset: Pre-built training dataset (optional).
            val_dataset: Pre-built validation dataset (optional).
            collate_fn: Collate function for the dataloaders (optional).
            train_sampler: Training data sampler (optional).
            device_name: Device string override (falls back to config).
        """
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "DanceGRPO requires hybrid_engine=True"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"Expected ActorRollout or ActorRolloutRef in role_worker_mapping, got {role_worker_mapping.keys()}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device

        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    # ------------------------------------------------------------------
    # Dataloader
    # ------------------------------------------------------------------

    def _create_dataloader(
        self,
        train_dataset: Optional[Dataset],
        val_dataset: Optional[Dataset],
        collate_fn,
        train_sampler: Optional[Sampler],
    ) -> None:
        """Create train / validation dataloaders using :mod:`~recipe.dance_grpo.dataset`.

        If *train_dataset* or *val_dataset* are ``None``, they are constructed
        from the paths in ``config.data.train_files`` / ``config.data.val_files``
        via :func:`~recipe.dance_grpo.dataset.create_rl_dataset`.

        The *collate_fn* defaults to the one returned by
        :func:`~recipe.dance_grpo.dataset.create_rl_dataset`.
        """
        from recipe.dance_grpo.dataset import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset, _train_collate = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                tokenizer=self.tokenizer,
                processor=self.processor,
                is_train=True,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
            if collate_fn is None:
                collate_fn = _train_collate

        if val_dataset is None:
            val_dataset, _val_collate = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                tokenizer=self.tokenizer,
                processor=self.processor,
                is_train=False,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
            if collate_fn is None:
                collate_fn = _val_collate

        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)

        num_workers = int(self.config.data.get("dataloader_num_workers", 0))
        self.collate_fn = collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, "
            f"Size of val dataloader: {len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
        except Exception as exc:
            _py_logger.warning("Could not set total_training_steps in config: %s", exc)

    # ------------------------------------------------------------------
    # Worker initialisation (FSDP/FSDP2 only)
    # ------------------------------------------------------------------

    def init_workers(self) -> None:
        """Initialise Ray worker groups for the DanceGRPO training.

        Only ``fsdp`` / ``fsdp2`` actor strategies are supported.
        """
        from omegaconf import OmegaConf

        from verl.single_controller.ray import RayClassWithInitArgs
        from verl.single_controller.ray.base import create_colocated_worker_cls

        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError(
                f"DanceGRPO only supports fsdp/fsdp2 actor strategy, got "
                f"'{self.config.actor_rollout_ref.actor.strategy}'"
            )

        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        if self.use_rm:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel],
                config=self.config.reward_model,
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # Build worker groups
        all_wg: dict = {}
        wg_kwargs: dict = {"device_name": self.device_name}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            else:
                assert str(Role.ActorRolloutRef) in all_wg
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.actor_rollout_wg = all_wg[str(actor_role)]
        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        self.async_rollout_mode = False

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def _save_checkpoint(self) -> None:
        """Persist DiT weights + optional optimizer state to *local_path*."""
        assert self.global_steps is not None

        local_path = os.path.join(
            self.config.trainer.default_local_dir,
            f"global_step_{self.global_steps}",
        )
        os.makedirs(local_path, exist_ok=True)
        self.actor_rollout_wg.save_checkpoint(
            local_path=local_path,
            hdfs_path=None,
            global_step=self.global_steps,
        )

    def _load_checkpoint(self) -> None:
        """Restore the latest checkpoint from *default_local_dir* if present."""
        resume_from = self.config.trainer.get("resume_from_path", None)
        if resume_from is None:
            resume_from = find_latest_ckpt_path(self.config.trainer.default_local_dir)

        if resume_from is None:
            self.global_steps = 0
        else:
            self.actor_rollout_wg.load_checkpoint(local_path=resume_from)
            # Try to parse the step from the directory name
            try:
                self.global_steps = int(resume_from.rstrip("/").split("_")[-1])
            except ValueError:
                self.global_steps = 0

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def fit(self) -> None:
        """DanceGRPO training loop.

        This is a diffusion-specific loop that:

        1. Loads checkpoint (if available).
        2. Optionally initialises fixed evaluation prompts.
        3. Iterates over training batches, calling:
           - ``actor_rollout_wg.generate_sequences`` – rollout + reward.
           - ``actor_rollout_wg.update_actor`` – GRPO policy update.
        4. Logs metrics, saves checkpoints at the configured frequency, and
           optionally renders fixed evaluation images.

        Advantage computation is *not* performed on the driver process — it is
        fully delegated to :meth:`DiffusionActor.update_policy` inside the worker.
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        # --- Fixed evaluation setup --- #
        _fixed_eval_enabled = bool(self.config.trainer.get("fixed_eval", False))
        if _fixed_eval_enabled:
            import random as _random

            n_eval = min(8, len(self.train_dataset))
            fixed_indices = _random.sample(range(len(self.train_dataset)), n_eval)
            fixed_samples = [self.train_dataset[i] for i in fixed_indices]
            fixed_eval_data = DataProto.from_single_dict(self.collate_fn(fixed_samples))
            fixed_eval_data.meta_info["global_step"] = 0
            fixed_eval_data.meta_info["experiment_name"] = self.config.trainer.get("experiment_name", "dance_grpo")
            self.actor_rollout_wg.init_fixed_eval(fixed_eval_data)

            # Render baseline images before any training (step 0)
            self.actor_rollout_wg.run_fixed_eval(
                DataProto(
                    meta_info={
                        "global_step": 0,
                        "experiment_name": self.config.trainer.get("experiment_name", "dance_grpo"),
                    }
                )
            )

        # --- Progress bar --- #
        progress_bar = tqdm(
            total=self.total_training_steps,
            initial=self.global_steps,
            desc="DanceGRPO Training",
        )

        self.global_steps += 1
        timing_raw: dict = defaultdict(float)
        current_epoch = self.global_steps // max(len(self.train_dataloader), 1)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics: dict = {}
                is_last_step = self.global_steps >= self.total_training_steps

                # ---- Rollout + reward ---- #
                with marked_timer("step", timing_raw):
                    batch: DataProto = DataProto.from_single_dict(batch_dict)
                    batch.meta_info["global_step"] = self.global_steps
                    batch.meta_info["experiment_name"] = self.config.trainer.get("experiment_name", "dance_grpo")

                    with marked_timer("gen", timing_raw, "red"):
                        gen_output = self.actor_rollout_wg.generate_sequences(batch)

                    # ---- Actor update ---- #
                    with marked_timer("update_actor", timing_raw, "red"):
                        actor_output = self.actor_rollout_wg.update_actor(gen_output)

                    # ---- Fixed eval ---- #
                    if _fixed_eval_enabled:
                        with marked_timer("fixed_eval", timing_raw):
                            self.actor_rollout_wg.run_fixed_eval(
                                DataProto(
                                    meta_info={
                                        "global_step": self.global_steps,
                                        "experiment_name": self.config.trainer.get("experiment_name", "dance_grpo"),
                                    }
                                )
                            )

                # ---- Collect actor metrics ---- #
                if actor_output is not None:
                    try:
                        raw = actor_output.meta_info.get("metrics", {})
                        metrics.update({k: (v[0] if isinstance(v, (list, tuple)) else v) for k, v in raw.items()})
                    except Exception as exc:
                        _py_logger.debug("Failed to parse actor metrics: %s", exc)

                # ---- Reward metrics (computed on driver for logging) ---- #
                if gen_output is not None and "rewards" in gen_output.batch.keys():
                    rewards = gen_output.batch["rewards"].float().cpu()
                    grpo_size = int(self.config.actor_rollout_ref.rollout.n)
                    batch_size = rewards.shape[0]
                    advantages = torch.zeros_like(rewards)
                    groups = torch.chunk(torch.arange(batch_size), batch_size // grpo_size)
                    for grp in groups:
                        g_r = rewards[grp]
                        advantages[grp] = (g_r - g_r.mean()) / g_r.std().clamp_min(1e-8)
                    metrics.update(
                        {
                            "train/reward_mean": rewards.mean().item(),
                            "train/reward_max": rewards.max().item(),
                            "train/reward_min": rewards.min().item(),
                            "train/reward_std": rewards.std().item(),
                            "train/advantage_mean": advantages.mean().item(),
                            "train/advantage_max": advantages.max().item(),
                            "train/advantage_min": advantages.min().item(),
                            "train/advantage_std": advantages.std().item(),
                        }
                    )

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                logger.log(data=metrics, step=self.global_steps)

                # ---- Checkpoint ---- #
                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(getattr(self, "max_steps_duration", 0.0), steps_duration)
                esi_close = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close
                ):
                    if esi_close:
                        _py_logger.warning("ESI expiration approaching — forcing checkpoint save.")
                    with marked_timer("save_checkpoint", timing_raw, "green"):
                        self._save_checkpoint()

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    progress_bar.close()
                    return
