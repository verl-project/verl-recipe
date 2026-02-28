# opkd_ray_trainer.py

import uuid
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.trainer.main_ppo import create_rl_sampler
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking


class RayOPKDTrainer(RayPPOTrainer):
    """Distributed OPKD trainer using Ray for scalable offline policy knowledge distillation.

    This trainer uses a student actor and a reference (teacher) policy.
    It repeatedly rolls out each prompt, extracts student top-k statistics,
    queries teacher log probabilities, and updates the actor with a distillation loss.
    """

    def __init__(self, config, tokenizer, **kwargs):
        # Reuse PPO trainer initialization (actors, ref policy worker groups, resource pool, etc.)
        super().__init__(config, tokenizer, **kwargs)
        # Global top-k used by the actor (student) and teacher union subset
        self.topk = self.config.actor_rollout_ref.actor.get("topk", 256)

    def _create_dataloader(
        self,
        train_dataset: Optional[Dataset],
        val_dataset: Optional[Dataset],
        collate_fn,
        train_sampler: Optional[Sampler],
    ):
        """Create train and validation dataloaders for OPKD.

        Assumes datasets are prepared by the caller and only constructs samplers/dataloaders.
        """
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # If no sampler is provided, build a default RL sampler from config
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        # If no collate_fn is provided, use the default RL collate function
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        # Training dataloader using a stateful dataloader to support resuming from checkpoints
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        # Validation dataloader uses full dataset by default if val_batch_size is not specified
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

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty."
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty."

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, "
            f"Size of val dataloader: {len(self.val_dataloader)}"
        )

        # Compute total training steps from dataloader length and epoch count
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        # If total_training_steps is explicitly specified, override automatic calculation
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        # Propagate total_training_steps into optimizer configs so schedulers can use it
        try:
            from omegaconf import open_dict

            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            # Missing optimizer structure is not fatal; training can proceed without this
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def fit(self):
        """Training loop for OPKD with per-prompt repeated rollouts."""
        # Initialize experiment logger/Tracker (e.g. WandB, TensorBoard, etc.)
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        # Restore model, optimizer, and dataloader state if a checkpoint exists
        self._load_checkpoint()

        # Recover current epoch from global step and dataloader length
        current_epoch = self.global_steps // len(self.train_dataloader)

        # Optional rollout skipping/short-circuiting for debugging or ablation
        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            from verl.utils.rollout_skip import RolloutSkip

            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # Progress bar tracks global step across epochs
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # Ensure global_steps starts from at least 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        # Flags controlling which steps should be profiled by the global profiler
        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        # Main training loop over epochs and dataloader batches
        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                # Finalize asynchronous rollout calls if any are pending
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)

                metrics = {}
                timing_raw = {}

                # Start profiling for the current step if configured
                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                # Convert raw batch dict to DataProto for easier manipulation
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # Set generation temperature for actor rollout worker group
                batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

                # Attach unique identifiers per sample to track prompts/rollouts across processing stages
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                    dtype=object,
                )

                # Extract input fields used for generation (prompt-only part)
                gen_batch = self._get_gen_batch(batch)

                # Propagate global step to generation workers for potential logging/scheduling
                gen_batch.meta_info["global_steps"] = self.global_steps

                # Number of rollouts per prompt; n>1 means repeated responses for the same prompt
                rollout_n = getattr(self.config.actor_rollout_ref.rollout, "n", 1)
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    # ------------------------------------------------------------------ #
                    # 1) Rollout: generate responses with the student actor
                    # ------------------------------------------------------------------ #
                    with marked_timer("gen", timing_raw, color="red"):
                        # Repeat the prompt batch n times; by default interleave=True
                        gen_batch_for_gen = gen_batch.repeat(repeat_times=rollout_n, interleave=True)

                        # Synchronous or asynchronous rollout depending on config
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_for_gen)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_for_gen)

                        # Collect generation timing information from remote workers
                        gen_timing = gen_batch_output.meta_info.pop("timing")

                        for k, v in gen_timing.items():
                            # For list values, compute statistics over the list
                            if isinstance(v, list):
                                arr = np.array(v)
                                timing_raw[k + "_mean"] = arr.mean().item()
                                timing_raw[k + "_min"] = arr.min().item()
                                timing_raw[k + "_max"] = arr.max().item()
                                # Keep the max value as the representative latency for this key
                                timing_raw[k] = arr.max().item()
                            else:
                                timing_raw[k] = v

                        # Response lengths are computed based on non-padding tokens
                        response_lens = (
                            (gen_batch_output.batch["responses"] != self.tokenizer.pad_token_id).sum(dim=-1).tolist()
                        )
                        max_len = max(response_lens)
                        min_len = min(response_lens)
                        # Basic sequence length statistics for monitoring
                        metrics.update(
                            {
                                "response_seq_len/average": float(sum(response_lens)) / len(response_lens),
                                "response_seq_len/max": max_len,
                                "response_seq_len/min": min_len,
                                "response_seq_len/max_count": response_lens.count(max_len),
                                "response_seq_len/min_count": response_lens.count(min_len),
                            }
                        )

                    # When rollout_n > 1, repeat the original batch to match generated responses
                    if rollout_n > 1:
                        batch = batch.repeat(repeat_times=rollout_n, interleave=True)

                    # Merge prompt batch and generated data into a single DataProto
                    batch = batch.union(gen_batch_output)

                    # Build a boolean response mask if not already provided in the batch
                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)

                    # Example logging of a generated sentence (first element) for debugging
                    one_attention_mask = batch.batch["attention_mask"][0].to(torch.bool)
                    one_sentence = batch.batch["input_ids"][0]
                    print("INFO:", "generate text done.")
                    print("DEBUG:", self.tokenizer.decode(one_sentence[one_attention_mask].tolist()))

                    # Global token count per sample, used for throughput metrics
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ------------------------------------------------------------------ #
                    # 2) Student statistics: compute top-k indices under the student
                    # ------------------------------------------------------------------ #
                    with marked_timer("student_log_prob", timing_raw, color="blue"):
                        # Compute student top-k tokens for the response span
                        student_topk_index = self.actor_rollout_wg.compute_student_topk_index(batch)
                        batch = batch.union(student_topk_index)

                        # Optional detailed debug metrics (e.g. log prob alignment, heuristics)
                        if "rollout_log_probs" in batch.batch.keys():
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                        print("INFO:", "student log prob done.")

                    # ------------------------------------------------------------------ #
                    # 3) Teacher statistics: compute log prob / logits under reference policy
                    # ------------------------------------------------------------------ #
                    with marked_timer("teacher_log_prob", timing_raw, color="olive"):
                        # Compute teacher information (e.g., union logits, token logprobs)
                        teacher_output = self.ref_policy_wg.compute_teacher_log_prob(batch)
                        print("INFO:", "get teacher knowledge done.")
                        batch = batch.union(teacher_output)

                    # ------------------------------------------------------------------ #
                    # 4) Actor update: distillation loss + gradient step
                    # ------------------------------------------------------------------ #
                    with marked_timer("update_actor", timing_raw, color="red"):
                        # Multi-turn flag is propagated so the actor can adapt its update logic if needed
                        batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                        # Run student policy update with union top-k teacher statistics
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                        print("INFO:", "update actor done.")

                    # Reduce metrics across distributed workers for logging
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # ---------------------------------------------------------------------- #
                # 5) Checkpoint saving logic with ESI-aware forced save
                # ---------------------------------------------------------------------- #
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                # ---------------------------------------------------------------------- #
                # 6) Profiling book-keeping across steps
                # ---------------------------------------------------------------------- #
                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                # Track the maximum step walltime for ESI checkpointing
                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # ---------------------------------------------------------------------- #
                # 7) Aggregate and log metrics for this step
                # ---------------------------------------------------------------------- #
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )

                # Timing metrics (forward, rollout, update, checkpoint, etc.)
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                # Throughput metrics: tokens / second, sequences / second, etc.
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # Curriculum sampler can update its sampling distribution based on batch statistics
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # Log metrics using the configured backend (e.g., WandB/MLflow/TensorBoard)
                logger.log(data=metrics, step=self.global_steps)

                # Advance progress bar and global step counter
                progress_bar.update(1)
                self.global_steps += 1

                # Optional per-step memory snapshot for debugging
                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}",
                        sub_dir=f"step{self.global_steps}",
                    )

                # If this is the last training step, finalize async workers and exit
                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    print(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # Notify dataset that a batch has been consumed (for stateful datasets)
                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
