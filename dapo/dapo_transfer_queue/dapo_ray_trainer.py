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
DAPO Trainer with TransferQueue: extends recipe.transfer_queue.ray_trainer.RayPPOTrainer,
overrides fit() and compute_kl_related_metrics() to use BatchMeta/tq_client and DAPO logic
(dynamic sampling, filter_groups, etc.).
"""

import os
import uuid
from collections import defaultdict
from pprint import pprint

import numpy as np
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

from transfer_queue import BatchMeta

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import (
    calculate_workload,
    get_seqlen_balanced_partitions,
    log_seqlen_unbalance,
)

from recipe.transfer_queue.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_data_metrics_decorated,
    compute_reward_decorated,
    compute_response_mask,
    compute_throughout_metrics_decorated,
    compute_timing_metrics_decorated,
    compute_val_reward_decorated,
)


class RayDAPOTrainerTQ(RayPPOTrainer):
    """
    DAPO trainer using TransferQueue: data flow uses BatchMeta and tq_client;
    fit() implements DAPO loop (multiple gen batches per step when filter_groups, etc.).
    """

    def compute_kl_related_metrics(self, batch_meta: BatchMeta, metrics: dict, timing_raw: dict) -> BatchMeta:
        """Compute KL-related metrics (response_mask, old_log_prob, ref_log_prob) with BatchMeta/tq_client."""
        if "response_mask" not in batch_meta.field_names:
            response_mask_meta = batch_meta.select_fields(["responses", "attention_mask"])
            response_mask_output_meta = compute_response_mask(response_mask_meta, self.tq_client)
            batch_meta = batch_meta.union(response_mask_output_meta)

        old_log_prob_fields = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "prompts",
            "responses",
            "response_mask",
            "data_source",
            "reward_model",
            "extra_info",
            "uid",
            "index",
            "tools_kwargs",
            "interaction_kwargs",
            "ability",
        ]
        old_log_prob_meta = batch_meta.select_fields(
            [f for f in old_log_prob_fields if f in batch_meta.field_names]
        )
        with marked_timer("old_log_prob", timing_raw, "blue"):
            old_log_prob_output_meta = self.actor_rollout_wg.compute_log_prob(old_log_prob_meta)
            batch_meta = batch_meta.union(old_log_prob_output_meta)

        data = self.tq_client.get_data(old_log_prob_output_meta)
        entropys = data["entropys"]
        response_masks = data["response_mask"]
        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
        entropy_agg = agg_loss(
            loss_mat=entropys,
            loss_mask=response_masks,
            loss_agg_mode=loss_agg_mode,
        )
        metrics["actor/entropy"] = entropy_agg.detach().item()

        if self.use_reference_policy:
            ref_log_prob_fields = [
                "input_ids",
                "attention_mask",
                "position_ids",
                "prompts",
                "responses",
                "response_mask",
                "old_log_probs",
                "data_source",
                "reward_model",
                "extra_info",
                "uid",
                "index",
                "tools_kwargs",
                "interaction_kwargs",
                "ability",
            ]
            ref_log_prob_meta = batch_meta.select_fields(
                [f for f in ref_log_prob_fields if f in batch_meta.field_names]
            )
            with marked_timer("ref", timing_raw, "olive"):
                if not self.ref_in_actor:
                    ref_log_prob_output_meta = self.ref_policy_wg.compute_ref_log_prob(ref_log_prob_meta)
                else:
                    ref_log_prob_output_meta = self.actor_rollout_wg.compute_ref_log_prob(ref_log_prob_meta)
                batch_meta = batch_meta.union(ref_log_prob_output_meta)

        return batch_meta

    def fit(self):
        """DAPO training loop with TransferQueue: BatchMeta/tq_client and filter_groups support."""
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
        batch_list = []
        num_prompt_in_batch = 0
        num_gen_batches = 0
        gen_batch_metas_to_clear = []

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
                    [str(uuid.uuid4()) for _ in range(len(batch_dict["input_ids"]))], dtype=object
                )
                repeated_batch_dict = self.repeat_dict(
                    batch_dict,
                    repeat_times=self.config.actor_rollout_ref.rollout.n,
                    interleave=True,
                )
                batch_td: TensorDict = self.dict_to_tensordict(repeated_batch_dict)
                partition_id = f"train_{self.global_steps - 1}_gen_{num_gen_batches}"
                gen_meta = self.tq_client.put(data=batch_td, partition_id=partition_id)
                gen_meta.set_extra_info("global_steps", self.global_steps)
                gen_batch_metas_to_clear.append(gen_meta)

                num_gen_batches += 1
                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, "red"):
                        gen_output_meta = self.async_rollout_manager.generate_sequences(gen_meta)
                        timing_raw.update(gen_output_meta.extra_info["timing"])
                        gen_output_meta.extra_info.pop("timing", None)
                    # if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    #     with marked_timer("gen_max", timing_raw, "red"):
                    #         gen_baseline_batch = deepcopy(gen_batch)
                    #         gen_baseline_batch.meta_info["do_sample"] = False
                    #         gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                    #         new_batch = new_batch.union(gen_baseline_output)
                    #         # compute reward model score on new_batch
                    #         rm_scores = None
                    #         if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                    #             rm_scores = self.rm_wg.compute_rm_score(new_batch)
                    #             new_batch = new_batch.union(rm_scores)
                    #         reward_baseline_tensor, _ = compute_reward(new_batch, self.reward_fn)
                    #         reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                    #         keys_to_pop = set(gen_baseline_output.batch.keys())
                    #         if rm_scores is not None:
                    #             keys_to_pop.update(rm_scores.batch.keys())
                    #         new_batch.pop(batch_keys=list(keys_to_pop))

                    #         new_batch.batch["reward_baselines"] = reward_baseline_tensor

                    #         del rm_scores, gen_baseline_batch, gen_baseline_output
                    batch_meta: BatchMeta = gen_meta.union(gen_output_meta)

                    # if "response_mask" not in batch_meta.field_names:
                    #     response_mask_output_meta = compute_response_mask(batch_meta, self.tq_client)
                    #     batch_meta = batch_meta.union(response_mask_output_meta)

                    # attention_mask_meta = batch_meta.select_fields(["attention_mask"])
                    # balanced_idx = None
                    # if self.config.trainer.balance_batch:
                    #     balanced_idx = self._balance_batch(attention_mask_meta, self.tq_client, metrics=metrics)
                    #     batch_meta.reorder(balanced_idx)

                    # data = self.tq_client.get_data(attention_mask_meta)
                    # batch_meta.extra_info["global_token_num"] = torch.sum(data["attention_mask"], dim=-1).tolist()

                    # When use_kl_in_reward: must compute old_log_prob & ref_log_prob *before* reward,
                    # because apply_kl_penalty needs them. Same order as original DAPO (dapo_ray_trainer.py).
                    if self.config.algorithm.use_kl_in_reward:
                        batch_meta = self.compute_kl_related_metrics(batch_meta, metrics, timing_raw)

                    with marked_timer("reward", timing_raw, "yellow"):
                        if self.use_rm and "rm_scores" not in batch_meta.field_names:
                            reward_meta = self.rm_wg.compute_rm_score(batch_meta)
                            batch_meta = batch_meta.union(reward_meta)

                        compute_reward_fields = [
                            "responses",
                            "prompts",
                            "attention_mask",
                            "reward_model",
                            "data_source",
                        ]
                        if "extra_info" in batch_meta.field_names:
                            compute_reward_fields.append("extra_info")
                        if "rm_scores" in batch_meta.field_names:
                            compute_reward_fields.append("rm_scores")
                        # DAPORewardManager may access these fields via .get() with defaults
                        for optional_field in ["reward_scores", "__num_turns__"]:
                            if optional_field in batch_meta.field_names:
                                compute_reward_fields.append(optional_field)
                        compute_reward_meta = batch_meta.select_fields(compute_reward_fields)

                        reward_tensor, reward_extra_infos_dict = compute_reward_decorated(
                            compute_reward_meta, self.reward_fn
                        )
                        batch_meta = batch_meta.union(compute_reward_meta)
                        reward_td = TensorDict(
                            {"token_level_scores": reward_tensor},
                            batch_size=reward_tensor.size(0),
                        )
                        batch_meta = self.tq_client.put(data=reward_td, metadata=batch_meta)

                        if reward_extra_infos_dict:
                            reward_extra_infos_dict_new = {
                                k: np.array(v) for k, v in reward_extra_infos_dict.items()
                            }
                            reward_extra_td = self.dict_to_tensordict(reward_extra_infos_dict_new)
                            batch_meta = self.tq_client.put(data=reward_extra_td, metadata=batch_meta)

                        if self.config.algorithm.use_kl_in_reward:
                            apply_kl_penalty_fields = [
                                "response_mask",
                                "token_level_scores",
                                "old_log_probs",
                                "ref_log_prob",
                            ]
                            apply_kl_penalty_meta = batch_meta.select_fields(
                                [f for f in apply_kl_penalty_fields if f in batch_meta.field_names]
                            )
                            token_level_rewards, kl_metrics = apply_kl_penalty(
                                apply_kl_penalty_meta,
                                kl_ctrl=self.kl_ctrl_in_reward,
                                kl_penalty=self.config.algorithm.kl_penalty,
                            )
                            token_level_rewards_td = TensorDict(
                                {"token_level_rewards": token_level_rewards},
                                batch_size=token_level_rewards.size(0),
                            )
                            apply_kl_penalty_meta = self.tq_client.put(
                                data=token_level_rewards_td, metadata=apply_kl_penalty_meta
                            )
                            metrics.update(kl_metrics)
                            batch_meta = batch_meta.union(apply_kl_penalty_meta)
                        else:
                            token_level_scores_meta = batch_meta.select_fields(["token_level_scores"])
                            data = self.tq_client.get_data(token_level_scores_meta)
                            token_level_rewards_td = TensorDict(
                                {"token_level_rewards": data["token_level_scores"]},
                                batch_size=data["token_level_scores"].size(0),
                            )
                            token_level_scores_meta = self.tq_client.put(
                                data=token_level_rewards_td, metadata=token_level_scores_meta
                            )
                            batch_meta = batch_meta.union(token_level_scores_meta)

                    if not self.config.algorithm.filter_groups.enable:
                        data_td = self.tq_client.get_data(batch_meta)
                        new_batch = DataProto.from_tensordict(
                            data_td, meta_info=batch_meta.extra_info.copy()
                        )
                        batch_list = [new_batch]
                        num_prompt_in_batch = self.config.data.train_batch_size
                    else:
                        data_td = self.tq_client.get_data(batch_meta)
                        new_batch = DataProto.from_tensordict(
                            data_td, meta_info=batch_meta.extra_info.copy()
                        )
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"],
                            new_batch.non_tensor_batch[metric_name],
                            strict=True,
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {
                            prompt_uid: np.std(metric_vals)
                            for prompt_uid, metric_vals in prompt_uid2metric_vals.items()
                        }
                        kept_prompt_uids = [
                            uid
                            for uid, std in prompt_uid2metric_std.items()
                            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1
                        ]
                        num_prompt_in_batch += len(kept_prompt_uids)

                        kept_traj_idxs = []
                        for idx, traj_from_prompt_uid in enumerate(new_batch.non_tensor_batch["uid"]):
                            if traj_from_prompt_uid in kept_prompt_uids:
                                kept_traj_idxs.append(idx)
                        new_batch = new_batch[kept_traj_idxs]
                        batch_list.append(new_batch)

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}. "
                                    "Generated too many. Check data difficulty or set max_num_gen_batches=0."
                                )

                    traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                    # Strip all per-batch meta_info before concat so no key can conflict (each gen batch
                    # has different global_token_num, global_steps, etc.). Merged batch gets meta_info set below.
                    if len(batch_list) > 1:
                        for dp in batch_list:
                            dp.meta_info.clear()
                    batch = batch_list[0] if len(batch_list) == 1 else DataProto.concat(batch_list)
                    batch = batch[:traj_bsz]

                    # Set global_token_num on batch before any reorder
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1
                    ).tolist()

                    # Balance on DataProto *before* put, so TQ storage order matches sample order.
                    # Avoids reordering BatchMeta after put (which can misalign scores with rows).
                    balanced_idx = None
                    if self.config.trainer.balance_batch:
                        attention_mask = batch.batch["attention_mask"]
                        batch_size = attention_mask.shape[0]
                        global_seqlen_lst = attention_mask.view(batch_size, -1).sum(-1)
                        global_seqlen_lst = calculate_workload(global_seqlen_lst)
                        world_size = self.actor_rollout_wg.world_size
                        global_partition_lst = get_seqlen_balanced_partitions(
                            global_seqlen_lst, k_partitions=world_size, equal_size=True
                        )
                        for idx, partition in enumerate(global_partition_lst):
                            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
                            ordered_partition = partition[::2] + partition[1::2][::-1]
                            global_partition_lst[idx] = ordered_partition
                        balanced_idx = torch.tensor(
                            [j for partition in global_partition_lst for j in partition]
                        )
                        global_balance_stats = log_seqlen_unbalance(
                            seqlen_list=global_seqlen_lst,
                            partitions=global_partition_lst,
                            prefix="global_seqlen",
                        )
                        metrics.update(global_balance_stats)
                        batch.reorder(balanced_idx)

                    tensordict_update = batch.to_tensordict()
                    for key in list(batch.meta_info.keys()):
                        tensordict_update.pop(key, None)
                    update_batch_meta = self.tq_client.put(
                        data=tensordict_update,
                        partition_id=f"train_{self.global_steps - 1}_update",
                    )
                    for k, v in batch.meta_info.items():
                        update_batch_meta.set_extra_info(k, v)

                    if not self.config.algorithm.use_kl_in_reward:
                        update_batch_meta = self.compute_kl_related_metrics(
                            update_batch_meta, metrics, timing_raw
                        )

                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values_meta = self.critic_wg.compute_values(update_batch_meta)
                            update_batch_meta = update_batch_meta.union(values_meta)

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if (
                        rollout_corr_config is not None
                        and "rollout_log_probs" in update_batch_meta.field_names
                    ):
                        from verl.trainer.ppo.rollout_corr_helper import (
                            compute_rollout_correction_and_add_to_batch,
                        )

                        rollout_data = self.tq_client.get_data(update_batch_meta)
                        rollout_proto = DataProto.from_tensordict(
                            rollout_data, meta_info=update_batch_meta.extra_info.copy()
                        )
                        rollout_proto, is_metrics = compute_rollout_correction_and_add_to_batch(
                            rollout_proto, rollout_corr_config
                        )
                        metrics.update(is_metrics)
                        new_keys = set(rollout_proto.batch.keys()) - set(rollout_data.keys())
                        if new_keys:
                            add_td = TensorDict(
                                {k: rollout_proto.batch[k] for k in new_keys},
                                batch_size=rollout_proto.batch.batch_size[0],
                            )
                            update_batch_meta = self.tq_client.put(
                                data=add_td, metadata=update_batch_meta
                            )

                    with marked_timer("adv", timing_raw, "brown"):
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )
                        compute_advantage_fields = [
                            "response_mask",
                            "token_level_rewards",
                        ]
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
                            compute_advantage_fields.append("values")
                        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GRPO:
                            compute_advantage_fields.append("uid")
                        else:
                            if "uid" in update_batch_meta.field_names:
                                compute_advantage_fields.append("uid")
                            if "reward_baselines" in update_batch_meta.field_names:
                                compute_advantage_fields.append("reward_baselines")

                        compute_advantage_meta = update_batch_meta.select_fields(
                            [f for f in compute_advantage_fields if f in update_batch_meta.field_names]
                        )
                        advantages, returns = compute_advantage(
                            compute_advantage_meta,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )
                        adv_td = TensorDict(
                            {"advantages": advantages, "returns": returns},
                            batch_size=advantages.size(0),
                        )
                        compute_advantage_meta = self.tq_client.put(
                            data=adv_td, metadata=compute_advantage_meta
                        )
                        update_batch_meta = update_batch_meta.union(compute_advantage_meta)

                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output_meta = self.critic_wg.update_critic(update_batch_meta)
                            update_batch_meta = update_batch_meta.union(critic_output_meta)
                        metrics.update(
                            reduce_metrics(critic_output_meta.extra_info["metrics"])
                        )

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, "red"):
                            update_actor_fields = [
                                "input_ids",
                                "attention_mask",
                                "position_ids",
                                "prompts",
                                "responses",
                                "response_mask",
                                "old_log_probs",
                                "ref_log_prob",
                                "advantages",
                                "returns",
                                "token_level_rewards",
                                "token_level_scores",
                                "data_source",
                                "reward_model",
                                "extra_info",
                                "uid",
                                "index",
                                "tools_kwargs",
                                "interaction_kwargs",
                                "ability",
                            ]
                            update_actor_meta = update_batch_meta.select_fields(
                                [f for f in update_actor_fields if f in update_batch_meta.field_names]
                            )
                            update_actor_meta.set_extra_info(
                                "global_token_num",
                                update_batch_meta.get_extra_info("global_token_num"),
                            )
                            update_actor_meta.set_extra_info(
                                "temperature",
                                update_batch_meta.get_extra_info("temperature"),
                            )
                            update_actor_meta.set_extra_info(
                                "multi_turn",
                                self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            )
                            actor_output_meta = self.actor_rollout_wg.update_actor(update_actor_meta)
                            update_batch_meta = update_batch_meta.union(actor_output_meta)
                        metrics.update(
                            reduce_metrics(actor_output_meta.extra_info["metrics"])
                        )

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        log_rollout_fields = [
                            "prompts",
                            "responses",
                            "token_level_scores",
                            "reward_model",
                        ]
                        if "request_id" in update_batch_meta.field_names:
                            log_rollout_fields.append("request_id")
                        log_rollout_meta = update_batch_meta.select_fields(log_rollout_fields)
                        reward_extra_for_log = (
                            reward_extra_infos_dict
                            if num_gen_batches == 1
                            else {}
                        )
                        self._log_rollout_data(
                            log_rollout_meta,
                            reward_extra_for_log,
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

                metrics.update(
                    compute_data_metrics_decorated(batch=update_batch_meta, use_critic=self.use_critic)
                )
                metrics.update(
                    compute_timing_metrics_decorated(
                        batch=update_batch_meta, timing_raw=timing_raw
                    )
                )
                n_gpus = self.resource_pool_manager.get_n_gpus()
                compute_throughout_metrics_meta = BatchMeta(
                    samples=[],
                    extra_info={
                        "global_token_num": update_batch_meta.get_extra_info("global_token_num")
                    },
                )
                metrics.update(
                    compute_throughout_metrics_decorated(
                        batch=compute_throughout_metrics_meta,
                        timing_raw=timing_raw,
                        n_gpus=n_gpus,
                    )
                )
                timing_raw = defaultdict(float)

                for meta in gen_batch_metas_to_clear:
                    self.tq_client.clear_samples(meta)
                self.tq_client.clear_samples(update_batch_meta)
                gen_batch_metas_to_clear = []

                metrics["training/global_step"] = self.global_steps
                metrics["training/epoch"] = epoch
                metrics["train/num_gen_batches"] = num_gen_batches
                batch_list = []
                num_prompt_in_batch = 0
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
