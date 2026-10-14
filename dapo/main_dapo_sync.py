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
DAPO Sync Trainer with TransferQueue and ReplayBuffer.

Combines:
1. TransferQueue zero-copy data transfer from main_ppo_sync.py
2. DAPO dynamic sampling (filter_groups) from dapo_ray_trainer.py
3. Async Agent Loop with ReplayBuffer from main_ppo_sync.py
4. Conditional KL computation timing from RayDAPOTrainer

Key differences from standard DAPO (dapo_ray_trainer.py):
- Uses KVBatchMeta / tq.kv_batch_get/put instead of DataProto for data flow
- Uses ReplayBuffer for async sampling instead of synchronous generate_sequences
- Uses AgentLoopManagerTQ for fire-and-forget generation
- Supports compute_advantage_for_multi_trajectories for multi-turn agent loops
"""

import logging
import os
import uuid
from collections import defaultdict
from pprint import pprint

import hydra
import numpy as np
import ray

try:
    import transfer_queue as tq
    from transfer_queue import KVBatchMeta
except ImportError:
    print("Please install TQ by calling `pip install TransferQueue==0.1.6` and try again.")
    from verl.utils.transferqueue_utils import KVBatchMeta, tq

from omegaconf import OmegaConf
from tensordict import TensorDict
from tqdm import tqdm

from verl.protocol import DataProto
from verl.single_controller.ray import ResourcePoolManager
from verl.trainer.main_ppo import run_ppo
from verl.trainer.main_ppo_sync import PPOTrainer
from verl.trainer.ppo.ray_trainer import apply_kl_penalty
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.utils import tensordict_utils as tu
from verl.utils.config import validate_config
from verl.utils.debug import marked_timer
from verl.utils.device import auto_set_device
from verl.utils.tracking import Tracking, ValidationGenerationsLogger
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker
from verl.workers.utils.padding import response_to_nested

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class DAPOSyncPPOTrainer(PPOTrainer):
    """DAPO Trainer with TransferQueue, ReplayBuffer, and dynamic sampling.

    Extends PPOTrainer (from main_ppo_sync.py) with:
    1. Dynamic sampling (filter_groups): filters prompts with zero reward variance
       and continues generating until enough valid prompts are collected.
    2. Conditional KL computation: computes KL metrics before or after filtering
       based on whether KL is used in reward.
    3. Multi-batch generation: one training step may include multiple generation batches.
    """

    def __init__(self, config, role_worker_mapping, resource_pool_manager):
        super().__init__(config, role_worker_mapping, resource_pool_manager)
        self.gen_steps = 0

    def compute_kl_related_metrics(self, batch: KVBatchMeta, metrics: dict, timing_raw: dict) -> KVBatchMeta:
        """Compute KL-related metrics (old_log_prob, ref_log_prob) with TQ data flow.

        Timing depends on use_kl_in_reward:
        - True:  called before filter_groups (need ref_log_prob for KL penalty in reward)
        - False: called after filter_groups (avoid computing on filtered-out data)
        """
        with marked_timer("old_log_prob", timing_raw, color="blue"):
            batch = self._compute_old_log_prob(batch, metrics=metrics)

        if self.use_reference_policy:
            with marked_timer("ref", timing_raw, color="olive"):
                batch = self._compute_ref_log_prob(batch, metrics=metrics)

        return batch

    def _filter_groups(self, batch: KVBatchMeta, metrics: dict) -> tuple:
        """Filter prompts with zero reward variance for DAPO.

        Reads reward data from TransferQueue, computes per-prompt metric variance,
        and removes trajectories from prompts whose variance is zero (i.e. all
        trajectories for that prompt received the same reward, providing no
        gradient signal for GRPO).

        Returns:
            filtered_batch: KVBatchMeta with only kept keys
            num_kept_prompts: number of unique prompts kept
        """
        metric_name = self.config.algorithm.filter_groups.metric

        fields = ["uid", "rm_scores", "response_mask"]
        if metric_name == "seq_final_reward":
            fields.append("token_level_rewards")

        data = tq.kv_batch_get(keys=batch.keys, partition_id=batch.partition_id, select_fields=fields)

        if metric_name == "seq_final_reward" and "token_level_rewards" in data:
            metric_values = data["token_level_rewards"].to_padded_tensor().sum(dim=-1).numpy()
        else:
            metric_values = data["rm_scores"].to_padded_tensor().sum(dim=-1).numpy()

        uid_data = data["uid"]
        if hasattr(uid_data, "tolist"):
            uids = uid_data.tolist()
        else:
            uids = list(uid_data)

        prompt_uid2metric_vals = defaultdict(list)
        prompt_uid2key_indices = defaultdict(list)
        for i, (uid, metric_val) in enumerate(zip(uids, metric_values)):
            prompt_uid2metric_vals[uid].append(metric_val)
            prompt_uid2key_indices[uid].append(i)

        prompt_uid2metric_std = {}
        for prompt_uid, metric_vals in prompt_uid2metric_vals.items():
            prompt_uid2metric_std[prompt_uid] = np.std(metric_vals)

        kept_prompt_uids = set()
        for uid, std in prompt_uid2metric_std.items():
            if std > 0 or len(prompt_uid2metric_vals[uid]) == 1:
                kept_prompt_uids.add(uid)

        kept_keys = []
        kept_tags = []
        removed_keys = []
        for i, key in enumerate(batch.keys):
            uid = uids[i]
            if uid in kept_prompt_uids:
                kept_keys.append(key)
                kept_tags.append(batch.tags[i])
            else:
                removed_keys.append(key)

        if removed_keys:
            tq.kv_clear(keys=removed_keys, partition_id=batch.partition_id)

        filtered_batch = KVBatchMeta(
            partition_id=batch.partition_id,
            keys=kept_keys,
            tags=kept_tags,
        )

        total_prompts = len(prompt_uid2metric_std)
        kept_prompts = len(kept_prompt_uids)
        metrics["dapo/filter_ratio"] = 1.0 - kept_prompts / max(total_prompts, 1)
        metrics["dapo/kept_prompts"] = kept_prompts
        metrics["dapo/total_prompts"] = total_prompts

        return filtered_batch, len(kept_prompt_uids)

    def _generate_and_sample(self, batch_dict: dict, timing_raw: dict) -> KVBatchMeta:
        """Generate sequences via AgentLoopManagerTQ and sample from ReplayBuffer.

        Args:
            batch_dict: Raw batch from dataloader
            timing_raw: Timing dictionary

        Returns:
            KVBatchMeta with generated data
        """
        batch_dict["uid"] = np.array(
            [str(uuid.uuid4()) for _ in range(len(batch_dict["raw_prompt"]))], dtype=object
        )
        batch = tu.get_tensordict(batch_dict)
        tu.assign_non_tensor_data(batch, "global_steps", self.gen_steps)
        self.async_rollout_manager.generate_sequences(batch)

        with marked_timer("gen", timing_raw, color="red"):
            gen_batch = self.replay_buffer.sample(partition_id="train", global_steps=self.gen_steps)
        gen_batch.extra_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
        self.checkpoint_manager.sleep_replicas()

        return gen_batch

    def _merge_kv_batch_metas(self, batch_list: list) -> KVBatchMeta:
        """Merge multiple KVBatchMeta objects into one.

        All batches must share the same partition_id.
        """
        if len(batch_list) == 1:
            return batch_list[0]

        all_keys = []
        all_tags = []
        for b in batch_list:
            all_keys.extend(b.keys)
            all_tags.extend(b.tags)

        return KVBatchMeta(
            partition_id=batch_list[0].partition_id,
            keys=all_keys,
            tags=all_tags,
        )

    def fit(self):
        """DAPO training loop with dynamic sampling and TransferQueue."""
        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        self._load_checkpoint()
        self.checkpoint_manager.update_weights()

        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        current_epoch = self.global_steps // len(self.train_dataloader)
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        self.global_steps += 1
        self.gen_steps += 1
        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        last_val_metrics = None
        timing_raw = defaultdict(float)

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                is_last_step = self.global_steps >= self.total_training_steps
                metrics, num_gen_batches = {}, 0

                self._start_profiling()
                with marked_timer("step", timing_raw):
                    # === Dynamic sampling loop ===
                    batch_list = []
                    all_gen_batch_metas = []
                    num_prompt_in_batch = 0

                    while True:
                        num_gen_batches += 1
                        self.gen_steps += 1

                        # 1. Generate sequences
                        gen_batch = self._generate_and_sample(batch_dict, timing_raw)
                        all_gen_batch_metas.append(gen_batch)

                        # 2. [OPTIONAL] Compute reward with colocated reward model
                        if self.reward_loop_manager.reward_loop_worker_handles is None:
                            with marked_timer("reward", timing_raw, color="yellow"):
                                gen_batch = self._compute_reward_colocate(gen_batch, metrics=metrics)

                        # 3. Conditional KL computation (before filtering if use_kl_in_reward)
                        if self.config.algorithm.use_kl_in_reward:
                            gen_batch = self.compute_kl_related_metrics(gen_batch, metrics, timing_raw)

                            # Apply KL penalty to get token_level_rewards
                            with marked_timer("kl_penalty", timing_raw, color="olive"):
                                kl_fields = ["response_mask", "rm_scores", "old_log_probs", "ref_log_prob"]
                                kl_data = tq.kv_batch_get(
                                    keys=gen_batch.keys,
                                    partition_id=gen_batch.partition_id,
                                    select_fields=kl_fields,
                                )
                                kl_data = DataProto(batch=kl_data.to_padded_tensor())
                                kl_data.batch["token_level_scores"] = kl_data.batch["rm_scores"]
                                kl_data, kl_metrics = apply_kl_penalty(
                                    kl_data,
                                    kl_ctrl=self.kl_ctrl_in_reward,
                                    kl_penalty=self.config.algorithm.kl_penalty,
                                )
                                metrics.update(kl_metrics)
                                token_level_rewards = response_to_nested(
                                    kl_data.batch["token_level_rewards"],
                                    kl_data.batch["response_mask"],
                                )
                                tq.kv_batch_put(
                                    keys=gen_batch.keys,
                                    partition_id=gen_batch.partition_id,
                                    fields=TensorDict(
                                        {"token_level_rewards": token_level_rewards},
                                        batch_size=len(gen_batch),
                                    ),
                                )

                        # 4. Filter groups or keep all
                        if not self.config.algorithm.filter_groups.enable:
                            batch_list.append(gen_batch)
                            num_prompt_in_batch = self.config.data.train_batch_size
                            break
                        else:
                            filtered_batch, num_kept = self._filter_groups(gen_batch, metrics)
                            batch_list.append(filtered_batch)
                            num_prompt_in_batch += num_kept

                            prompt_bsz = self.config.data.train_batch_size
                            if num_prompt_in_batch < prompt_bsz:
                                print(f"{num_prompt_in_batch=} < {prompt_bsz=}")
                                max_num_gen_batches = self.config.algorithm.filter_groups.max_num_gen_batches
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f"{num_gen_batches=}. Keep generating...")
                                    is_last_step = self.global_steps >= self.total_training_steps
                                    continue
                                else:
                                    raise ValueError(
                                        f"{num_gen_batches=} >= {max_num_gen_batches=}. "
                                        "Generated too many. Please check if your data are too difficult. "
                                        "You could also try set max_num_gen_batches=0 to enable endless trials."
                                    )
                            else:
                                break

                    # === Merge batches ===
                    batch = self._merge_kv_batch_metas(batch_list)

                    # Align batch size
                    traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                    if len(batch) > traj_bsz:
                        batch = KVBatchMeta(
                            partition_id=batch.partition_id,
                            keys=batch.keys[:traj_bsz],
                            tags=batch.tags[:traj_bsz],
                        )

                    # 5. Balance batch across data parallel groups
                    self._balance_batch(batch, metrics=metrics)

                    # 6. Conditional KL computation (after filtering if not use_kl_in_reward)
                    if not self.config.algorithm.use_kl_in_reward:
                        batch = self.compute_kl_related_metrics(batch, metrics, timing_raw)

                    # 7. [OPTIONAL] Compute critic values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            batch = self._compute_values(batch, metrics=metrics)

                    # 8. Compute advantage and return
                    with marked_timer("adv", timing_raw, color="brown"):
                        batch = self._compute_advantage(batch, metrics=metrics)

                    # 9. [OPTIONAL] Update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            batch = self._update_critic(batch, metrics=metrics)

                    # 10. Update actor
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch = self._update_actor(batch, metrics=metrics)

                    # 11. Save checkpoint
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step or self.global_steps % self.config.trainer.save_freq == 0
                    ):
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                    # 12. Update weights from trainer to rollout
                    with marked_timer("update_weights", timing_raw, color="red"):
                        self.checkpoint_manager.update_weights()

                self._stop_profiling()

                # 13. Validate
                if self.config.trainer.test_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.test_freq == 0
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # 14. Record metrics
                self._compute_metrics(batch, metrics, timing_raw, global_steps=self.global_steps, epoch=epoch)
                metrics["train/num_gen_batches"] = num_gen_batches

                # 15. Cleanup transfer queue and replay buffer
                for gen_batch in all_gen_batch_metas:
                    tq.kv_clear(keys=gen_batch.keys, partition_id=gen_batch.partition_id)
                    self.replay_buffer.remove(gen_batch.partition_id, gen_batch.keys)
                tq.kv_clear(keys=batch.keys, partition_id=batch.partition_id)
                self.replay_buffer.remove(batch.partition_id, batch.keys)

                self.logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1
                self.gen_steps += 1

                timing_raw = defaultdict(float)

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


class DAPOSyncTaskRunner:
    def __init__(self) -> None:
        self.role_worker_mapping = {}
        self.mapping = {}

    def add_actor_rollout_worker(self, config):
        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None

        role = Role.ActorRolloutRef if need_reference_policy(config) and not ref_in_actor else Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(ActorRolloutRefWorker)
        self.mapping[role] = "global_pool"

    def add_critic_worker(self, config):
        if need_critic(config):
            self.role_worker_mapping[Role.Critic] = ray.remote(TrainingWorker)
            self.mapping[Role.Critic] = "global_pool"

    def init_resource_pool_mgr(self, config):
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        if config.reward.reward_model.enable_resource_pool:
            if config.reward.reward_model.n_gpus_per_node <= 0:
                raise ValueError("config.reward.reward_model.n_gpus_per_node must be greater than 0")
            if config.reward.reward_model.nnodes <= 0:
                raise ValueError("config.reward.reward_model.nnodes must be greater than 0")

            reward_pool = [config.reward.reward_model.n_gpus_per_node] * config.reward.reward_model.nnodes
            resource_pool_spec["reward_pool"] = reward_pool
            self.mapping[Role.RewardModel] = "reward_pool"
        else:
            config.reward.reward_model.nnodes = config.trainer.nnodes
            config.reward.reward_model.n_gpus_per_node = config.trainer.n_gpus_per_node
            self.mapping[Role.RewardModel] = "global_pool"

        self.resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=self.mapping)

    def run(self, config):
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        tq.init(config.transfer_queue)

        self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.init_resource_pool_mgr(config)

        trainer = DAPOSyncPPOTrainer(
            config=config,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=self.resource_pool_manager,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="dapo_sync_trainer", version_base=None)
def main(config):
    auto_set_device(config)

    config.transfer_queue.enable = True

    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DAPOSyncTaskRunner))


if __name__ == "__main__":
    main()
