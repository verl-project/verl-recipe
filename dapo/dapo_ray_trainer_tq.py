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
DAPO Trainer with TransferQueue - asynchronous streaming data management for efficient post-training.
Extends the TransferQueue-based RayPPOTrainer with DAPO-specific logic (dynamic batching, filter_groups, etc.).
"""

import os
import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from tensordict import TensorDict
from tqdm import tqdm
from transfer_queue import BatchMeta

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.reward import compute_reward
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.transferqueue_utils import tqbridge

from verl.experimental.transfer_queue.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer as TQRayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_data_metrics_decorated,
    compute_response_mask,
    compute_reward_decorated,
    compute_timing_metrics_decorated,
    compute_throughout_metrics_decorated,
    compute_val_reward_decorated,
)


@tqbridge(put_data=False)
def _compute_reward_impl(data, reward_fn):
    return compute_reward(data, reward_fn)


class RayDAPOTrainerTQ(TQRayPPOTrainer):
    """
    DAPO Trainer with TransferQueue.
    Uses BatchMeta and tq_client for data flow instead of DataProto, enabling
    distributed data storage and alleviating the single-controller bottleneck.
    """

    def compute_kl_related_metrics(
        self, batch_meta: BatchMeta, metrics: dict, timing_raw: dict
    ) -> BatchMeta:
        """Compute KL-related metrics using BatchMeta and tq_client."""
        batch_meta = compute_response_mask(batch_meta, self.tq_client)

        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob_meta = self.actor_rollout_wg.compute_log_prob(batch_meta)
            data = self.tq_client.get_data(old_log_prob_meta)
            entropys = data["entropys"]
            response_masks = data["response_mask"]
            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
            entropy_agg = agg_loss(
                loss_mat=entropys,
                loss_mask=response_masks,
                loss_agg_mode=loss_agg_mode,
            )
            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
            metrics.update(old_log_prob_metrics)
            batch_meta = batch_meta.union(old_log_prob_meta)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob_meta = self.ref_policy_wg.compute_ref_log_prob(batch_meta)
                else:
                    ref_log_prob_meta = self.actor_rollout_wg.compute_ref_log_prob(batch_meta)
                batch_meta = batch_meta.union(ref_log_prob_meta)

        return batch_meta

    def fit(self):
        """DAPO training loop with TransferQueue (BatchMeta / tq_client)."""
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.gen_steps = 0

        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        self.gen_steps += 1
        last_val_metrics = None

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        timing_raw = defaultdict(float)
        batch_meta_accum: BatchMeta | None = None
        num_prompt_in_batch = 0
        num_gen_batches = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                metrics = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch_dict["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
                )
                num_gen_batches += 1

                repeated_batch_dict = self.repeat_dict(
                    batch_dict,
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                batch_td: TensorDict = self.dict_to_tensordict(repeated_batch_dict)
                partition_id = f"train_{self.global_steps - 1}_gen_{num_gen_batches}"
                gen_meta = self.tq_client.put(data=batch_td, partition_id=partition_id)
                gen_meta.set_extra_info("global_steps", self.global_steps)

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, "red"):
                        gen_output_meta = self.async_rollout_manager.generate_sequences(gen_meta)
                        self.checkpoint_manager.sleep_replicas()
                        timing_raw.update(gen_output_meta.extra_info.get("timing", {}))
                        gen_output_meta.extra_info.pop("timing", None)

                    # TODO (TQ): Support REMAX advantage estimator with TransferQueue
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        raise NotImplementedError(
                            "REMAX advantage estimator is not yet supported with TransferQueue. "
                            "Please use GAE or GRPO."
                        )

                    new_batch_meta = gen_meta.union(gen_output_meta)

                    if self.config.algorithm.use_kl_in_reward:
                        new_batch_meta = self.compute_kl_related_metrics(new_batch_meta, metrics, timing_raw)

                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm and "rm_scores" not in new_batch_meta.field_names:
                            rm_meta = self.rm_wg.compute_rm_score(new_batch_meta)
                            new_batch_meta = new_batch_meta.union(rm_meta)

                        compute_reward_fields = [
                            "responses", "prompts", "attention_mask", "reward_model", "data_source"
                        ]
                        if "rm_scores" in new_batch_meta.field_names:
                            compute_reward_fields.extend(
                                ["rm_scores", *set(new_batch_meta.extra_info.get("reward_extra_keys", []))]
                            )
                        compute_reward_meta = new_batch_meta.select_fields(compute_reward_fields)
                        reward_tensor, reward_extra_infos_dict = compute_reward_decorated(
                            compute_reward_meta, self.reward_fn
                        )
                        reward_td = TensorDict({"token_level_scores": reward_tensor}, batch_size=reward_tensor.size(0))
                        new_batch_meta = self.tq_client.put(data=reward_td, metadata=new_batch_meta)

                        if reward_extra_infos_dict:
                            reward_extra_td = self.dict_to_tensordict(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )
                            new_batch_meta = self.tq_client.put(data=reward_extra_td, metadata=new_batch_meta)

                        if self.config.algorithm.use_kl_in_reward:
                            apply_kl_fields = [
                                "response_mask", "token_level_scores", "old_log_probs", "ref_log_prob"
                            ]
                            apply_kl_meta = new_batch_meta.select_fields(apply_kl_fields)
                            token_level_rewards, kl_metrics = apply_kl_penalty(
                                apply_kl_meta,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            metrics.update(kl_metrics)
                            rewards_td = TensorDict(
                                {"token_level_rewards": token_level_rewards},
                                batch_size=token_level_rewards.size(0),
                            )
                            new_batch_meta = self.tq_client.put(data=rewards_td, metadata=new_batch_meta)
                        else:
                            data = self.tq_client.get_data(new_batch_meta.select_fields(["token_level_scores"]))
                            rewards_td = TensorDict(
                                {"token_level_rewards": data["token_level_scores"]},
                                batch_size=data["token_level_scores"].size(0),
                            )
                            new_batch_meta = self.tq_client.put(data=rewards_td, metadata=new_batch_meta)

                    if not self.config.algorithm.filter_groups.enable:
                        batch_meta = new_batch_meta
                    else:
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            data = self.tq_client.get_data(new_batch_meta.select_fields(["token_level_rewards"]))
                            seq_metric = data["token_level_rewards"].sum(dim=-1).cpu().numpy()
                        elif metric_name == "seq_reward":
                            data = self.tq_client.get_data(new_batch_meta.select_fields(["token_level_scores"]))
                            seq_metric = data["token_level_scores"].sum(dim=-1).cpu().numpy()
                        else:
                            raise ValueError(f"Unsupported filter_groups.metric: {metric_name}")

                        uid_data = self.tq_client.get_data(new_batch_meta.select_fields(["uid"]))
                        uids = uid_data["uid"]
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(uids, seq_metric, strict=True):
                            prompt_uid2metric_vals[str(uid)].append(metric_val)
                        prompt_uid2metric_std = {
                            uid: np.std(vals) for uid, vals in prompt_uid2metric_vals.items()
                        }
                        kept_prompt_uids = [
                            uid for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)
                        kept_traj_idxs = [
                            i for i, uid in enumerate(uids)
                            if str(uid) in kept_prompt_uids
                        ]
                        new_batch_meta = new_batch_meta[kept_traj_idxs]
                        batch_meta_accum = (
                            new_batch_meta
                            if batch_meta_accum is None
                            else BatchMeta.concat([batch_meta_accum, new_batch_meta])
                        )

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            max_num_gen = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen <= 0 or num_gen_batches < max_num_gen:
                                continue
                            else:
                                raise ValueError(
                                    f"Generated {num_gen_batches} batches but only "
                                    f"{num_prompt_in_batch} prompts kept. Check filter_groups."
                                )

                        traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                        batch_meta = batch_meta_accum[:traj_bsz]

                    attention_mask_meta = batch_meta.select_fields(["attention_mask"])
                    balanced_idx = None
                    if self.config.trainer.balance_batch:
                        balanced_idx = self._balance_batch(attention_mask_meta, self.tq_client, metrics=metrics)
                        batch_meta.reorder(balanced_idx)

                    data = self.tq_client.get_data(attention_mask_meta)
                    batch_meta.extra_info["global_token_num"] = torch.sum(data["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch_meta = self.compute_kl_related_metrics(batch_meta, metrics, timing_raw)

                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values_meta = self.critic_wg.compute_values(batch_meta)
                            batch_meta = batch_meta.union(values_meta)

                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch_meta.field_names:
                        corr_data = self.tq_client.get_data(batch_meta)
                        batch_proto = DataProto.from_tensordict(
                            corr_data, meta_info=batch_meta.extra_info.copy()
                        )
                        batch_proto, is_metrics = compute_rollout_correction_and_add_to_batch(
                            batch_proto, rollout_corr_config
                        )
                        metrics.update(is_metrics)
                        corr_td = batch_proto.to_tensordict()
                        batch_meta = self.tq_client.put(data=corr_td, metadata=batch_meta)

                    with marked_timer("adv", timing_raw, "brown"):
                        norm_adv = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)
                        adv_fields = ["response_mask", "token_level_rewards"]
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
                            adv_fields.append("values")
                        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                            adv_fields.append("uid")
                        else:
                            if "uid" in batch_meta.field_names:
                                adv_fields.append("uid")
                            if "reward_baselines" in batch_meta.field_names:
                                adv_fields.append("reward_baselines")

                        adv_meta = batch_meta.select_fields(adv_fields)
                        advantages, returns = compute_advantage(
                            adv_meta,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv,
                            config=self.config.algorithm,
                        )
                        adv_td = TensorDict(
                            {"advantages": advantages, "returns": returns},
                            batch_size=advantages.size(0),
                        )
                        batch_meta = self.tq_client.put(data=adv_td, metadata=batch_meta)

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output_meta = self.critic_wg.update_critic(batch_meta)
                            batch_meta = batch_meta.union(critic_output_meta)
                        critic_output_metrics = reduce_metrics(critic_output_meta.extra_info["metrics"])
                        metrics.update(critic_output_metrics)

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            batch_meta.extra_info["multi_turn"] = (
                                self.config.actor_rollout_ref.rollout.multi_turn.enable
                            )
                            update_actor_fields = [
                                "input_ids", "attention_mask", "position_ids", "prompts", "responses",
                                "response_mask", "old_log_probs", "ref_log_prob", "advantages", "returns",
                                "token_level_rewards", "token_level_scores", "data_source", "reward_model",
                                "extra_info", "uid", "index", "tools_kwargs", "interaction_kwargs", "ability",
                            ]
                            update_actor_meta = batch_meta.select_fields(
                                [f for f in update_actor_fields if f in batch_meta.field_names]
                            )
                            update_actor_meta.set_extra_info("global_token_num", batch_meta.extra_info["global_token_num"])
                            if "temperature" in batch_meta.extra_info:
                                update_actor_meta.set_extra_info("temperature", batch_meta.extra_info["temperature"])
                            actor_output_meta = self.actor_rollout_wg.update_actor(update_actor_meta)
                            batch_meta = batch_meta.union(actor_output_meta)

                        with marked_timer("update_weights", timing_raw, "red"):
                            self.checkpoint_manager.update_weights()

                        actor_output_metrics = reduce_metrics(actor_output_meta.extra_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        log_fields = ["prompts", "responses", "token_level_scores", "reward_model"]
                        if "request_id" in batch_meta.field_names:
                            log_fields.append("request_id")
                        self._log_rollout_data(
                            batch_meta.select_fields(log_fields),
                            reward_extra_infos_dict,
                            timing_raw,
                            rollout_data_dir,
                        )

                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                if self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                ):
                    with marked_timer("save_checkpoint", timing_raw, "green"):
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

                data_metrics_fields = [
                    "token_level_rewards", "token_level_scores", "advantages", "returns",
                    "responses", "attention_mask", "response_mask",
                ]
                if "__num_turns__" in batch_meta.field_names:
                    data_metrics_fields.append("__num_turns__")
                if "tool_call_counts" in batch_meta.field_names:
                    data_metrics_fields.append("tool_call_counts")
                data_metrics_meta = batch_meta.select_fields(data_metrics_fields)
                if balanced_idx is not None:
                    data_metrics_meta.reorder(balanced_idx)
                metrics.update(
                    compute_data_metrics_decorated(batch=data_metrics_meta, use_critic=self.use_critic)
                )
                timing_metrics_meta = batch_meta.select_fields(["responses", "attention_mask"])
                if balanced_idx is not None:
                    timing_metrics_meta.reorder(balanced_idx)
                metrics.update(
                    compute_timing_metrics_decorated(batch=timing_metrics_meta, timing_raw=timing_raw)
                )
                n_gpus = self.resource_pool_manager.get_n_gpus()
                throughput_meta = BatchMeta(
                    samples=[], extra_info={"global_token_num": batch_meta.extra_info["global_token_num"]}
                )
                metrics.update(
                    compute_throughout_metrics_decorated(
                        batch=throughput_meta,
                        timing_raw=timing_raw,
                        n_gpus=n_gpus,
                    )
                )

                self.tq_client.clear_samples(batch_meta)
                timing_raw = defaultdict(float)
                batch_meta_accum = None
                num_prompt_in_batch = 0
                metrics["train/num_gen_batches"] = num_gen_batches
                num_gen_batches = 0

                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                        self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

        checkpoint_dir = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )
        if not os.path.exists(checkpoint_dir):
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)
