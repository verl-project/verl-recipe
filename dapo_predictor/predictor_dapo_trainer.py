"""Recipe-side DAPO trainer with predictor-driven rollout reordering."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from verl import DataProto
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import AdvantageEstimator, apply_kl_penalty, compute_advantage
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer

from recipe.dapo.dapo_ray_trainer import RayDAPOTrainer

from .predictor_utils import snake_sort_indices


import os
import uuid
from collections import defaultdict
from copy import deepcopy
from pprint import pprint

import numpy as np
import torch
from tqdm import tqdm

from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics
from verl.trainer.ppo.ray_trainer import (
    AdvantageEstimator,
    RayPPOTrainer,
    apply_kl_penalty,
    compute_advantage,
    compute_response_mask,
)
from verl.trainer.ppo.reward import extract_reward
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.metric import reduce_metrics
from verl.utils.profiler import marked_timer
from verl.utils.rollout_skip import RolloutSkip


class PredictorRayDAPOTrainer(RayDAPOTrainer):
    """DAPO trainer that only injects the predictor-specific reorder steps.

    Most heavy lifting is still reused from the current `RayDAPOTrainer` / `RayPPOTrainer` stack:
    reward computation, KL/ref/value computation, actor/critic updates, checkpointing, metrics,
    and rollout manager orchestration all stay on the upstream path.
    """

    def _predictor_cfg(self):
        return self.config.trainer.get("predictor_reorder", {})

    def _predictor_enabled(self) -> bool:
        return self._predictor_cfg().get("enable", False)

    def _build_predictor_order(self, gen_batch: DataProto) -> torch.Tensor:
        """Compute predictor scores and build snake-sort reorder indices."""
        predictor_scores = self.actor_rollout_wg.compute_predictor_score(gen_batch)
        gen_batch = gen_batch.union(predictor_scores)
        dp_world_size = self._get_dp_size(self.actor_rollout_wg, "actor")
        return torch.tensor(
            snake_sort_indices(
                gen_batch.batch["predictor_scores"].tolist(),
                n_samples_per_prompt=self.config.actor_rollout_ref.rollout.n,
                dp_world_size=dp_world_size,
            ),
            dtype=torch.long,
        )

    def _apply_predictor_order(self, batch: DataProto, predictor_order: torch.Tensor | None) -> DataProto:
        """Apply predictor-derived reorder indices to a DataProto batch."""
        if predictor_order is not None:
            if batch.batch is not None:
                batch.reorder(predictor_order)
            else:
                indices_np = predictor_order.detach().cpu().numpy()
                batch.non_tensor_batch = {k: v[indices_np] for k, v in batch.non_tensor_batch.items()}
        return batch

    def _repeat_and_tag_uid(self, batch: DataProto) -> DataProto:
        """Tag each row with a UID and repeat the batch n times per prompt."""
        batch.non_tensor_batch["uid"] = np.array([str(i) for i in range(len(batch.batch))], dtype=object)
        return batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

    @staticmethod
    def _ensure_gen_batch_has_tensors(gen_batch: DataProto, source_batch: DataProto) -> DataProto:
        if gen_batch.batch is None and source_batch.batch is not None:
            gen_batch.batch = source_batch.batch
        return gen_batch

    def _hydrate_gen_batch_model_inputs(self, gen_batch: DataProto) -> DataProto:
        """Ensure gen_batch has input_ids, attention_mask, and position_ids tensors.

        Tokenizes from raw prompts/messages if the model inputs are missing.
        """
        from verl.workers.rollout.schemas import (  
            AsyncRolloutRequest,  
            AsyncRolloutRequestStateEnum,  
            TokenizationSanityCheckModeEnum,  
        )

        import uuid
        from tensordict import TensorDict
        if gen_batch.batch is None:  
            gen_batch.batch = {}  

        batch_keys = set(gen_batch.batch.keys())
        if {"input_ids", "attention_mask", "position_ids"}.issubset(batch_keys):
            return gen_batch

        if "input_ids" not in batch_keys and "prompts" in batch_keys:
            gen_batch.batch["input_ids"] = gen_batch.batch["prompts"]
            batch_keys.add("input_ids")

        if "input_ids" not in batch_keys:
            seqs = None
            messages = None
            if "raw_prompt_ids" in gen_batch.non_tensor_batch:
                raw_prompt_ids = gen_batch.non_tensor_batch["raw_prompt_ids"]
                seqs = raw_prompt_ids.tolist() if isinstance(raw_prompt_ids, np.ndarray) else raw_prompt_ids
            if "messages" in gen_batch.non_tensor_batch:
                messages = gen_batch.non_tensor_batch["messages"]
            elif "raw_prompt" in gen_batch.non_tensor_batch:
                messages = gen_batch.non_tensor_batch["raw_prompt"]
            if messages is not None and len(messages) == 0:
                messages = None
            if messages is not None:
                multi_modal_batch = gen_batch.non_tensor_batch.get("multi_modal_data", None)
                tool_schema_batch = gen_batch.non_tensor_batch.get("tool_schemas", None)
                input_id_list = []
                attn_mask_list = []
                pos_id_list = []
                max_prompt_len = int(self.config.data.get("max_prompt_length", 32768))
                max_response_len = int(self.config.data.get("max_response_length", 8192))
                max_model_len = int(self.config.actor_rollout_ref.rollout.get("max_model_len") or 32768)
                for i, msg in enumerate(messages):
                    multi_modal_data = {"image": [], "video": []}
                    if multi_modal_batch is not None:
                        mm_val = multi_modal_batch[i] if isinstance(multi_modal_batch, np.ndarray) else multi_modal_batch
                        if isinstance(mm_val, dict):
                            multi_modal_data.update(mm_val)
                    tools = None
                    if tool_schema_batch is not None:
                        tool_schema_val = tool_schema_batch[i] if isinstance(tool_schema_batch, np.ndarray) else tool_schema_batch
                        if tool_schema_val:
                            tools = [tool.model_dump() if hasattr(tool, "model_dump") else tool for tool in tool_schema_val]
                    request = AsyncRolloutRequest.model_validate(
                        {
                            "request_id": str(uuid.uuid4()),
                            "state": AsyncRolloutRequestStateEnum.PENDING,
                            "messages": msg,
                            "multi_modal_data": multi_modal_data,
                            "tool_schemas": tools,
                            "reward_scores": {},
                            "max_prompt_len": max_prompt_len,
                            "max_response_len": max_response_len,
                            "max_model_len": max_model_len,
                            "use_inference_chat_template": False,
                            "tokenization_sanity_check_mode": TokenizationSanityCheckModeEnum.DISABLE,
                            "processing_class": self.tokenizer,
                        }
                    )
                    input_ids = request.input_ids.squeeze(0)
                    attention_mask = request.attention_mask.squeeze(0)
                    position_ids = request.position_ids
                    if position_ids.dim() == 2 and position_ids.shape[0] == 1:
                        position_ids = position_ids.squeeze(0)
                    input_id_list.append(input_ids)
                    attn_mask_list.append(attention_mask)
                    pos_id_list.append(position_ids)

                if input_id_list:
                    max_len = max(x.shape[-1] for x in input_id_list)
                    pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                    input_ids = torch.full((len(input_id_list), max_len), fill_value=pad_token_id, dtype=torch.long)
                    attention_mask = torch.zeros((len(attn_mask_list), max_len), dtype=torch.long)
                    is_3d_pos = pos_id_list[0].dim() == 2
                    if is_3d_pos:
                        pos_channels = pos_id_list[0].shape[0]
                        position_ids = torch.zeros((len(pos_id_list), pos_channels, max_len), dtype=torch.long)
                    else:
                        position_ids = torch.zeros((len(pos_id_list), max_len), dtype=torch.long)

                    for i, (iid, am, pid) in enumerate(zip(input_id_list, attn_mask_list, pos_id_list, strict=True)):
                        input_ids[i, : iid.shape[-1]] = iid
                        attention_mask[i, : am.shape[-1]] = am
                        if is_3d_pos:
                            position_ids[i, :, : pid.shape[-1]] = pid
                        else:
                            position_ids[i, : pid.shape[-1]] = pid

                    gen_batch.batch = TensorDict(  
                        source={  
                            "input_ids": input_ids,  
                            "attention_mask": attention_mask,   
                            "position_ids": position_ids,  
                        },  
                        batch_size=(len(input_id_list),),  
                    )  
                    batch_keys.update({"input_ids", "attention_mask", "position_ids"})

            if seqs is not None:
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                max_len = max((len(s) for s in seqs), default=0)
                input_ids = torch.full((len(seqs), max_len), fill_value=pad_token_id, dtype=torch.long)
                for i, seq in enumerate(seqs):
                    if len(seq) > 0:
                        input_ids[i, : len(seq)] = torch.as_tensor(seq, dtype=torch.long)
                gen_batch.batch["input_ids"] = input_ids
                batch_keys.add("input_ids")

        if "attention_mask" not in batch_keys and "input_ids" in batch_keys:
            pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            gen_batch.batch["attention_mask"] = (gen_batch.batch["input_ids"] != pad_token_id).long()
            batch_keys.add("attention_mask")

        if "position_ids" not in batch_keys and "attention_mask" in batch_keys:
            gen_batch.batch["position_ids"] = (torch.cumsum(gen_batch.batch["attention_mask"], dim=-1) - 1).clamp_min(0).long()

        return gen_batch

    @staticmethod
    def _build_reverse_idx_from_uid(before_uid: np.ndarray, after_uid: np.ndarray) -> torch.Tensor:
        """Build a reverse index mapping original positions to post-reorder positions.

        Used to restore data order after DP balancing. Tracks duplicate UIDs
        by counting occurrences so that repeated prompts can be matched correctly.
        """
        before_counts = defaultdict(int)
        before_slots: dict[tuple[str, int], int] = {}
        for idx, uid in enumerate(before_uid.tolist()):
            key = (uid, before_counts[uid])
            before_slots[key] = idx
            before_counts[uid] += 1

        after_counts = defaultdict(int)
        orig_pos_of_after = []
        for uid in after_uid.tolist():
            key = (uid, after_counts[uid])
            if key not in before_slots:
                raise ValueError(f"Cannot restore predictor order: missing uid key {key}")
            orig_pos_of_after.append(before_slots[key])
            after_counts[uid] += 1

        reverse_idx = torch.empty(len(orig_pos_of_after), dtype=torch.long)
        for after_pos, orig_pos in enumerate(orig_pos_of_after):
            reverse_idx[orig_pos] = after_pos
        return reverse_idx

    @staticmethod
    def _prepare_predictor_gen_batch(source_batch: DataProto) -> DataProto:
        """Pop a lightweight gen batch containing only predictor-required keys."""
        batch_keys = []
        if source_batch.batch is not None:
            preferred_batch_keys = ["input_ids", "attention_mask", "position_ids", "prompts"]
            source_batch_keys = set(source_batch.batch.keys())
            batch_keys = [k for k in preferred_batch_keys if k in source_batch_keys]
            if not batch_keys:
                batch_keys = list(source_batch_keys)

        preferred_non_tensor_keys = ["raw_prompt_ids", "multi_modal_data", "multi_modal_inputs"]
        non_tensor_batch_keys = [k for k in preferred_non_tensor_keys if k in source_batch.non_tensor_batch]

        gen_batch = source_batch.pop(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_batch_keys)
        return gen_batch

    @staticmethod
    def _extract_restore_keys(batch: DataProto) -> np.ndarray:
        """Extract unique identifier keys from non_tensor_batch for order restoration."""
        if "uid" in batch.non_tensor_batch:
            return np.asarray(batch.non_tensor_batch["uid"], dtype=object)
        if "extra_info" in batch.non_tensor_batch:
            extra_info = batch.non_tensor_batch["extra_info"]
            keys = []
            for item in extra_info:
                if isinstance(item, dict) and "index" in item:
                    keys.append(item["index"])
                else:
                    keys.append(str(item))
            return np.asarray(keys, dtype=object)
        raise ValueError("Cannot restore order: neither `uid` nor `extra_info.index` found in non_tensor_batch")

    def _maybe_update_predictor(self, gen_batch: DataProto, batch: DataProto, metrics, timing_raw):

        with marked_timer("update_predictor", timing_raw, "orange"):
            # if not self.config.algorithm.filter_groups.enable:
            #         predictor_output = self.actor_rollout_wg.update_predictor(gen_batch, batch)  
            # else:
                # Dynamically extract prompt length  
            prompt_length = batch.batch["prompts"].shape[-1]  
            prompt_input_ids = batch.batch["prompts"]  # [batch_size, prompt_length]  
            prompt_attention_mask = batch.batch["attention_mask"][:, :prompt_length]  # dynamic slice  
            prompt_position_ids = batch.batch["position_ids"][:, :prompt_length]  # dynamic slice  
            
            # Build a standalone prompt batch for predictor training  
            prompt_batch = DataProto.from_dict({  
                "input_ids": prompt_input_ids,  
                "attention_mask": prompt_attention_mask,  
                "position_ids": prompt_position_ids
            }, meta_info=batch.meta_info   )

            predictor_output = self.actor_rollout_wg.update_predictor(prompt_batch, batch)
        metrics.update(reduce_metrics(predictor_output.meta_info.get("metrics", {})))

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        if not self._predictor_enabled():
            return super().fit()
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
        self.max_steps_duration = 0

        # load checkpoint before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.async_rollout_manager)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
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
        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        current_epoch = self.global_steps // len(self.train_dataloader)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
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

                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                num_gen_batches += 1
                # print(f"new_batch{new_batch}")
                gen_batch = self._get_gen_batch(new_batch)
                # print(f"gen_batch{gen_batch}")
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )

                is_last_step = self.global_steps >= self.total_training_steps

                with marked_timer("step", timing_raw):

                    with marked_timer("predictor_score", timing_raw, "purple"):  
                        with marked_timer("predictor_hydrate", timing_raw, "purple"):  
                            predictor_input_batch = gen_batch_output.select(deepcopy=True)  
                            predictor_input_batch = self._hydrate_gen_batch_model_inputs(predictor_input_batch)  
                        predictor_order = self._build_predictor_order(predictor_input_batch)  
                        # predictor_scores = self.actor_rollout_wg.compute_predictor_score(predictor_input_batch)
                        self._apply_predictor_order(gen_batch_output, predictor_order)  
                        # print(f'predictor_scores{predictor_scores}')
                    # generate a batch
                    with marked_timer("gen", timing_raw, "red"):
                        # print(f'gen_batch_output{gen_batch_output}')
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with marked_timer("gen_max", timing_raw, "red"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            # compute reward model score on new_batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                                rm_scores = self._compute_reward_colocate(new_batch)
                                new_batch = new_batch.union(rm_scores)
                            reward_baseline_tensor, _ = extract_reward(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            new_batch.pop(batch_keys=list(keys_to_pop))

                            new_batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(new_batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    if self.config.algorithm.use_kl_in_reward:
                        # We need these metrics for apply_kl_penalty if using kl in reward
                        new_batch = self.compute_kl_related_metrics(new_batch, metrics, timing_raw)
                        # otherwise, we will compute those after dynamic sampling

                    with marked_timer("reward", timing_raw, "yellow"):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm and "rm_scores" not in new_batch.batch.keys():
                            # we first compute reward model score
                            batch_reward = self._compute_reward_colocate(new_batch)
                            new_batch = new_batch.union(batch_reward)

                        # we combine with rule-based rm
                        reward_tensor, reward_extra_infos_dict = extract_reward(new_batch)

                        new_batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            new_batch.non_tensor_batch.update(
                                {k: np.array(v) for k, v in reward_extra_infos_dict.items()}
                            )

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            new_batch, kl_metrics = apply_kl_penalty(
                                new_batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(
                                kl_metrics
                            )  # TODO: This will be cleared if we use multiple genenration batches
                        else:
                            new_batch.batch["token_level_rewards"] = new_batch.batch["token_level_scores"]

                    if not self.config.algorithm.filter_groups.enable:
                        batch = new_batch
                    else:  # NOTE: When prompts after filtering is less than train batch size,
                        # we skip to the next generation batch
                        metric_name = self.config.algorithm.filter_groups.metric
                        if metric_name == "seq_final_reward":
                            # Turn to numpy for easier filtering
                            new_batch.non_tensor_batch["seq_final_reward"] = (
                                new_batch.batch["token_level_rewards"].sum(dim=-1).numpy()
                            )
                        elif metric_name == "seq_reward":
                            new_batch.non_tensor_batch["seq_reward"] = (
                                new_batch.batch["token_level_scores"].sum(dim=-1).numpy()
                            )

                        # Collect the sequence reward for each trajectory
                        prompt_uid2metric_vals = defaultdict(list)
                        for uid, metric_val in zip(
                            new_batch.non_tensor_batch["uid"], new_batch.non_tensor_batch[metric_name], strict=True
                        ):
                            prompt_uid2metric_vals[uid].append(metric_val)

                        prompt_uid2metric_std = {}
                        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
                            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

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
                        batch = new_batch if batch is None else DataProto.concat([batch, new_batch])

                        prompt_bsz = self.config.data.train_batch_size
                        if num_prompt_in_batch < prompt_bsz:
                            print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                            max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                            if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                print(f"{num_gen_batches=}. Keep generating...")
                                self.gen_steps += 1
                                is_last_step = self.global_steps >= self.total_training_steps
                                continue
                            else:
                                raise ValueError(
                                    f"{num_gen_batches=} >= {max_num_gen_batches=}."
                                    + " Generated too many. Please check if your data are too difficult."
                                    + " You could also try set max_num_gen_batches=0 to enable endless trials."
                                )
                        else:
                            # Align the batch
                            traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                            batch = batch[:traj_bsz]

                    self.checkpoint_manager.sleep_replicas()

                    # === Updating ===
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    # print(f'self.config.trainer.balance_batch{self.config.trainer.balance_batch}')
                    # if self.config.trainer.balance_batch:
                    #     self._balance_batch(batch, metrics=metrics)


                    if self.config.trainer.balance_batch:
                        uid_before_balance = batch.non_tensor_batch["uid"].copy()
                        self._balance_batch(batch, metrics=metrics)
                        reverse_idx = self._build_reverse_idx_from_uid(
                            uid_before_balance,
                            batch.non_tensor_batch["uid"],
                        )
                    else:
                        reverse_idx = None
                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, "cyan"):
                            values = self._compute_values(batch)
                            batch = batch.union(values)

                    # Compute rollout correction weights and off-policy metrics (inherited from RayPPOTrainer)
                    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    if rollout_corr_config is not None and "rollout_log_probs" in batch.batch:
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    with marked_timer("adv", timing_raw, "brown"):
                        # compute advantages, executed on the driver process
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

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, "pink"):
                            critic_output = self._update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, "red"):
                            actor_output = self._update_actor(batch)

                        # Update predictor after actor update  
                        if reverse_idx is not None:
                            batch.reorder(reverse_idx)
                        self._maybe_update_predictor(gen_batch, batch, metrics, timing_raw)  

                        # Check if ESI/training plan is close to expiration
                        esi_close_to_expiration = should_save_ckpt_esi(
                            max_steps_duration=self.max_steps_duration,
                            redundant_time=self.config.trainer.esi_redundant_time,
                        )
                        if self.config.trainer.save_freq > 0 and (
                            is_last_step
                            or self.global_steps % self.config.trainer.save_freq == 0
                            or esi_close_to_expiration
                        ):
                            if esi_close_to_expiration:
                                print("Force saving checkpoint: ESI instance expiration approaching.")
                            with marked_timer("save_checkpoint", timing_raw, "green"):
                                self._save_checkpoint()

                        with marked_timer("update_weights", timing_raw, "red"):
                            self.checkpoint_manager.update_weights()
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, "green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

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

                steps_duration = timing_raw.get("step", 0)
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = defaultdict(float)  # clear timing

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0

                # TODO: make a canonical logger that supports various backend
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
        # check if last step checkpint exists
        checkpoint_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        if not os.path.exists(checkpoint_dir):
            # save last step checkpoint
            timing_raw = defaultdict(float)
            with marked_timer("save_checkpoint", timing_raw, "green"):
                self._save_checkpoint()
            metrics = {f"timing/{k}": v for k, v in timing_raw.items()}
            logger.log(data=metrics, step=self.global_steps)

