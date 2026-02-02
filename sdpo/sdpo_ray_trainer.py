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
SDPO (Self-Distillation Policy Optimization) Ray Trainer.

This trainer uses the existing SDPO infrastructure in verl by:
1. Setting policy_loss.loss_mode = "sdpo" to use compute_self_distillation_loss
2. Using self_distillation config for EMA, reprompting, and KL settings
3. Extending OneStepOffRayTrainer for async 2-GPU training

The teacher batch is built by _maybe_build_self_distillation_batch in the parent class.
The self-distillation loss is computed in dp_actor.py when loss_mode="sdpo".

Reference: https://arxiv.org/abs/2601.20802
"""

import asyncio
from pprint import pprint

import numpy as np
import ray
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import apply_kl_penalty, compute_advantage
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics


class SDPORayTrainer(OneStepOffRayTrainer):
    """
    SDPO Ray Trainer that extends OneStepOffRayTrainer.

    This trainer uses the existing SDPO infrastructure:
    - loss_mode="sdpo" triggers compute_self_distillation_loss in dp_actor
    - self_distillation config controls EMA, reprompting, and distillation settings
    - The teacher batch is built using the parent class's _maybe_build_self_distillation_batch

    Key differences from standard PPO/GRPO:
    1. Uses self-distillation loss instead of standard policy loss
    2. Teacher is EMA of reference + training policy conditioned on correct answer
    3. Builds reprompt batch for teacher context
    """

    async def fit(self):
        """
        The training loop of SDPO.

        Uses the existing SDPO infrastructure in verl:
        - _maybe_build_self_distillation_batch builds the teacher batch
        - dp_actor uses compute_self_distillation_loss when loss_mode="sdpo"
        """
        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # Validate SDPO config
        loss_mode = self.config.actor_rollout_ref.actor.policy_loss.get("loss_mode", "vanilla")
        if loss_mode != "sdpo":
            print(f"Warning: SDPO trainer expects loss_mode='sdpo', got '{loss_mode}'")

        self_distillation_cfg = self.config.actor_rollout_ref.actor.get("self_distillation", None)
        if self_distillation_cfg:
            print(f"SDPO Config: ema_rate={self_distillation_cfg.get('ema_update_rate', 0.05)}, "
                  f"alpha={self_distillation_cfg.get('alpha', 0.0)}, "
                  f"is_clip={self_distillation_cfg.get('is_clip', None)}")

        # Load checkpoint
        self._load_checkpoint()

        # Sync rollout weights after loading checkpoint
        self.sync_rollout_weights()
        await self.async_rollout_manager.clear_kv_cache()

        # Validation before training
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="SDPO Training")

        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )

        continuous_iterator = self._create_continuous_iterator()
        batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))

        while batch_data_future is not None:
            do_profile = (
                self.global_steps in self.config.global_profiler.steps
                if self.config.global_profiler.steps is not None
                else False
            )
            if do_profile:
                self.actor_wg.start_profile()
                if not self.hybrid_engine:
                    self.rollout_wg.start_profile()
                if self.use_reference_policy:
                    self.ref_policy_wg.start_profile()
                if self.use_critic:
                    self.critic_wg.start_profile()
                if self.use_rm:
                    self.rm_wg.start_profile()

            metrics = {}
            timing_raw = {}
            is_last_step = self.global_steps >= self.total_training_steps

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )

            with marked_timer("step", timing_raw):
                # Wait for generation
                with marked_timer("gen", timing_raw, color="red"):
                    _metrics, _timing_raw, epoch, batch, future_reward = await batch_data_future
                    timing_raw.update(batch.meta_info["timing"])
                    timing_raw.update(_timing_raw)
                    metrics.update(_metrics)
                    batch.meta_info.pop("timing", None)

                # Sync weights
                with marked_timer("sync_rollout_weights", timing_raw, color="purple"):
                    self.sync_rollout_weights()
                    await self.async_rollout_manager.clear_kv_cache()

                # Start next generation
                if not is_last_step:
                    batch_data_future = asyncio.create_task(self._async_gen_next_batch(continuous_iterator))
                    await asyncio.sleep(0)

                # Compute rewards
                with marked_timer("reward", timing_raw, color="yellow"):
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        reward_tensor = self.rm_wg.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    if self.config.reward_model.launch_reward_fn_async:
                        future_reward = compute_reward_async.remote(
                            data=batch, config=self.config, tokenizer=self.tokenizer
                        )
                    else:
                        reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                await asyncio.sleep(0)

                # Handle rollout correction bypass mode
                rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)

                if bypass_recomputing_logprobs:
                    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode
                    apply_bypass_mode(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
                else:
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'
                await asyncio.sleep(0)

                # Reference policy log probs (for KL and teacher EMA base)
                if self.use_reference_policy:
                    with marked_timer("RefPolicy", timing_raw, color="olive"):
                        if not self.ref_in_actor:
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                        else:
                            ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)
                await asyncio.sleep(0)

                # Critic values (if using)
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self.critic_wg.compute_values(batch)
                        batch = batch.union(values)
                await asyncio.sleep(0)

                with marked_timer("adv", timing_raw, color="brown"):
                    # Get rewards
                    reward_extra_infos_dict: dict[str, list]
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # Apply KL penalty if configured
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    # Rollout correction for decoupled mode
                    if (
                        rollout_corr_config is not None
                        and "rollout_log_probs" in batch.batch
                        and not bypass_recomputing_logprobs
                    ):
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        metrics.update(is_metrics)

                    # Compute GRPO advantages (still needed for advantage estimation)
                    norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )

                    # === SDPO: Build self-distillation batch ===
                    # This is the key SDPO step - builds teacher batch with reprompt context
                    self_distillation_data = self._maybe_build_self_distillation_batch(
                        batch, reward_tensor, reward_extra_infos_dict
                    )
                    if self_distillation_data is not None:
                        self_distillation_batch, self_distillation_metrics = self_distillation_data
                        batch = batch.union(self_distillation_batch)
                        metrics.update(self_distillation_metrics)
                        metrics["sdpo/has_teacher_batch"] = 1.0
                    else:
                        metrics["sdpo/has_teacher_batch"] = 0.0

                await asyncio.sleep(0)

                # Update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self.critic_wg.update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)
                await asyncio.sleep(0)

                # Update actor (uses SDPO loss when loss_mode="sdpo")
                if self.config.trainer.critic_warmup <= self.global_steps:
                    with marked_timer("update_actor", timing_raw, color="red"):
                        rollout_config = self.config.actor_rollout_ref.rollout
                        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
                        batch.meta_info["temperature"] = rollout_config.temperature
                        actor_output = self.actor_rollout_wg.update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)
                await asyncio.sleep(0)

                # Log rollout data
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

            await asyncio.sleep(0)

            # Validation
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics
                metrics.update(val_metrics)
            await asyncio.sleep(0)

            # Checkpoint saving
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

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            # Training metrics
            metrics.update({
                "training/global_step": self.global_steps,
                "training/epoch": epoch,
            })
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if is_last_step:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return
