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

import uuid
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import open_dict
from recipe.partial_rollout.agent_loop.agent_loop import PartialRolloutAgentLoopManager
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
from tqdm import tqdm

from verl.experimental.separation.ray_trainer import SeparateRayPPOTrainer
from verl.protocol import DataProto
from verl.single_controller.ray import RayWorkerGroup, ResourcePoolManager
from verl.trainer.ppo.core_algos import AdvantageEstimator
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.trainer.ppo.utils import Role, WorkerType
from verl.utils.debug import marked_timer


class PartialRolloutRayPPOTrainer(SeparateRayPPOTrainer):
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
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
            processor,
            reward_fn,
            val_reward_fn,
            train_dataset,
            val_dataset,
            collate_fn,
            train_sampler,
            device_name,
        )
        # PartialRollout-specific override: route agent_loop_manager_class to PartialRolloutAgentLoopManager.
        # Done here (not in init_workers) so config setup is colocated with the rest
        # of __init__, and init_workers stays focused on worker creation.
        # open_dict: rollout.yaml's `agent` block omits agent_loop_manager_class even
        # though the AgentLoopConfig dataclass defines it, so the loaded node is a
        # struct-locked plain dict that rejects new keys without escaping struct mode.
        with open_dict(self.config.actor_rollout_ref.rollout["agent"]):
            self.config.actor_rollout_ref.rollout["agent"]["agent_loop_manager_class"] = (
                f"{PartialRolloutAgentLoopManager.__module__}.{PartialRolloutAgentLoopManager.__qualname__}"
            )
        # Fail fast on unsupported advantage estimators rather than after a wasted rollout.
        assert self.config.algorithm.adv_estimator != AdvantageEstimator.REMAX, (
            "PartialRolloutRayPPOTrainer does not support the ReMax advantage estimator yet"
        )
        self.stale_trajectory_processed = 0

    def init_workers(self):
        from recipe.partial_rollout.llm_server import PartialRolloutLLMServerManager
        from recipe.partial_rollout.prompt_manager import RolloutPromptManager

        import verl.trainer.ppo.ray_trainer as upstream_ray_trainer

        self.rollout_prompt_manager = RolloutPromptManager.remote(self.config, self.tokenizer)

        # Skip SeparateRayPPOTrainer.init_workers (its _create_actor_rollout_classes
        # is NotImplementedError and PartialRollout doesn't override it) and go straight to
        # RayPPOTrainer's setup, which already wires up async_rollout_manager and
        # checkpoint_manager.
        #
        # RayPPOTrainer.init_workers hardcodes `LLMServerManager.create(...)` (no
        # FQN config knob like AgentLoopManager has), so to swap in
        # PartialRolloutLLMServerManager we replace the symbol on the upstream module for
        # the duration of the call. Restored in finally so any other importer
        # of `verl.trainer.ppo.ray_trainer.LLMServerManager` keeps the upstream
        # class.
        original_lsm = upstream_ray_trainer.LLMServerManager
        upstream_ray_trainer.LLMServerManager = PartialRolloutLLMServerManager
        try:
            RayPPOTrainer.init_workers(self)
        finally:
            upstream_ray_trainer.LLMServerManager = original_lsm

        # PartialRolloutAgentLoopManager.create defers worker spawn until both the prompt
        # manager and the LLMServerManager (for cancel/resume around
        # update_weights) are available; provide them here.
        self.async_rollout_manager.init_agent_loop_workers(self.rollout_prompt_manager, self.llm_server_manager)

    def _fit_generate(self, data_loader_iter) -> DataProto:
        metrics = self.metrics
        timing_raw = self.timing_raw

        need_more = True
        while need_more:
            try:
                batch_dict = next(data_loader_iter)
            except StopIteration:
                # Loader exhausted: build a dummy gen_batch so the rest of this
                # step still runs and any in-flight prompts in rollout_prompt_manager
                # can be drained via generate_sequences/pull_batch.
                # global_steps is required by PartialRollout generate_sequences (asserted
                # in agent_loop.py); the rest of meta_info is unused here since
                # generate_sequences pulls real batches from rollout_prompt_manager.
                # Source the batch_size from config (matches RolloutPromptManager)
                # rather than self.train_dataloader.batch_size, which is None when
                # the DataLoader is built with a batch_sampler.
                dummy_batch_size = self.config.data.get("gen_batch_size", self.config.data.train_batch_size)
                # Match the gen_batch contract: every row carries a uid in
                # non_tensor_batch (PartialRolloutAgentLoopManager.generate_sequences
                # derives num_rollout_prompts from len(uid)). Use unique uuids
                # so any leak into prompt_manager's UID-collision guard can't
                # accidentally match a real in-flight prompt.
                gen_batch = DataProto(
                    batch=TensorDict({}, batch_size=(dummy_batch_size,)),
                    non_tensor_batch={
                        "uid": np.array([str(uuid.uuid4()) for _ in range(dummy_batch_size)], dtype=object),
                    },
                    meta_info={"global_steps": self.global_steps},
                )
                break
            batch = self._fit_get_batch(batch_dict)
            gen_batch = self._get_gen_batch(batch)
            # pass global_steps to trace
            gen_batch.meta_info["global_steps"] = self.global_steps
            need_more = ray.get(self.rollout_prompt_manager.push_batch.remote(batch, gen_batch))

        gen_batch_output = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

        with marked_timer("gen", timing_raw, color="red"):
            if self.curr_step_profile:
                self.async_rollout_manager.start_profile(global_step=self.global_steps)
            # Prime each rollout server's global_steps; see PartialRolloutLLMServerManager.set_global_steps
            # for why this can't be left to checkpoint_manager.update_weights alone.
            self.llm_server_manager.set_global_steps(self.global_steps)
            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)
            self.checkpoint_manager.sleep_replicas()
            if self.curr_step_profile:
                self.async_rollout_manager.stop_profile()

            # PartialRolloutAgentLoopManager.generate_sequences doesn't populate
            # `meta_info["timing"]` (it pulls a pre-assembled batch via
            # `pull_batch.remote()` instead of fanning out per-call). Outer
            # `marked_timer("gen", ...)` above still captures the wall-clock
            # for this stage; only the per-stage sub-timings
            # (generate_sequences / tool_calls / compute_score) are missing.
            timing = gen_batch_output.meta_info.pop("timing", None)
            if timing:
                timing_raw.update(timing)
            self._collect_metrics_from_samples(gen_batch_output, metrics)

        # gen_batch_output already carries the repeated/unioned batch built by
        # the agent loop, so use it directly instead of repeating + unioning here.
        batch = gen_batch_output

        if "response_mask" not in batch.batch.keys():
            batch.batch["response_mask"] = compute_response_mask(batch)
        # Balance the number of valid tokens across DP ranks.
        # NOTE: This usually changes the order of data in the `batch`,
        # which won't affect the advantage calculation (since it's based on uid),
        # but might affect the loss calculation (due to the change of mini-batching).
        if self.config.trainer.balance_batch:
            self._balance_batch(batch, metrics=metrics)

        # compute global_valid tokens
        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
        # get images_seqlens. `multi_modal_inputs` is only populated by the
        # multi-modal dataloader path; `.get(..., [])` keeps text-only runs
        # working without a KeyError.
        images_seqlens_all = []
        for multi_modal_input in batch.non_tensor_batch.get("multi_modal_inputs", []):
            if "image_grid_thw" not in multi_modal_input.keys():
                continue
            images_seqlens_all.extend(multi_modal_input["images_seqlens"].tolist())
        batch.meta_info["images_seqlens"] = images_seqlens_all
        return batch

    def _collect_metrics_from_samples(self, batch, metrics):
        """
        Collect metrics from samples
        """
        if hasattr(batch, "meta_info") and batch.meta_info:
            trajectory_param_versions = batch.meta_info["trajectory_param_versions"]
            stale_traj_count = sum(1 for v in trajectory_param_versions if self.global_steps - v >= 1)
            self.stale_trajectory_processed += stale_traj_count
            metrics.update(
                {
                    "fully_async/count/stale_trajectory_processed": self.stale_trajectory_processed,
                    "fully_async/count/current_param_version": self.global_steps,
                }
            )
            for key, value in batch.meta_info.items():
                if key.startswith("fully_async") or key.startswith("timing_s"):
                    metrics[key] = value

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.

        !!!
        The logic of fit is consistent with that of fit_refactor;
        if any modifications are made, apply them to both methods simultaneously.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        self.logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint and update weights before doing anything
        self._load_checkpoint()
        self.checkpoint_manager.update_weights(self.global_steps)

        current_epoch = self.global_steps // len(self.train_dataloader)
        # Number of batches already consumed in current_epoch before the
        # checkpoint was taken. On resume the stateful dataloader yields only
        # `len - start_in_epoch` batches for the current epoch, so the inner
        # loop below caps at that count for the resumed epoch.
        start_in_epoch = self.global_steps % len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            self.logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        self.progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        if self.global_steps > self.total_training_steps:
            # Resumed from a checkpoint that already met or exceeded the
            # configured limit; nothing left to do. Without this guard the loop
            # below would unconditionally run one more step before checking
            # is_last_step.
            return
        self.last_val_metrics = None
        self.max_steps_duration = 0

        self.prev_step_profile = False
        self.curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        self.next_step_profile = False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            # Pass the iterator (not a batch_dict) so _fit_generate can consume
            # multiple items per step under backpressure. Assumes a stateful
            # dataloader: on resume, iter(self.train_dataloader) yields only
            # the remaining batches in the current epoch, so no manual skip
            # is needed — but the inner loop must also cap at the matching
            # remaining count, otherwise the leftover iterations fall into
            # the dummy gen_batch path and burn step counts. Subsequent
            # epochs (epoch != current_epoch) run a full pass.
            self.epoch = epoch
            data_loader_iter = iter(self.train_dataloader)
            remaining = len(self.train_dataloader) - (start_in_epoch if epoch == current_epoch else 0)
            for _ in range(remaining):
                self.fit_step(data_loader_iter)
                if self.is_last_step:
                    return

    def fit_step(self, data_loader_iter):
        self.metrics = {"training/global_step": self.global_steps, "training/epoch": self.epoch}
        self.timing_raw = {}
        # reward message
        self.reward_tensor = None
        self.reward_extra_infos_dict = {}

        self._fit_prepare_step()
        self._fit_start_profile()

        with marked_timer("step", self.timing_raw):
            batch = self._fit_generate(data_loader_iter)
            batch = self._fit_compute_reward(batch)
            batch = self._fit_compute_log_prob(batch)
            batch = self._fit_compute_ref_log_prob(batch)
            batch = self._fit_compute_critic(batch)
            batch = self._fit_compute_advantage(batch)
            batch = self._fit_update_critic(batch)
            batch = self._fit_update_actor(batch)
            self._fit_update_weights()
            self._fit_dump_data(batch)

        self._fit_validate()
        self._fit_save_checkpoint()
        self._fit_stop_profile()
        self._fit_collect_metrics(batch)
        self._fit_experimental(batch)
        self._fit_postprocess_step()
