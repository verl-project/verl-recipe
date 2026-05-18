# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Individual Contributor: Brilliant Hanabi, furunding
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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import asyncio
import os
import socket

import hydra
import ray

from verl.trainer.ppo.ray_trainer import ResourcePoolManager
from verl.utils.device import auto_set_ascend_device_name
from verl.trainer.ppo.utils import Role


def create_resource_pool_manager(config, roles: list) -> ResourcePoolManager:
    """
    Create resource pool manager

    Args:
        config: Configuration object
        roles: List of roles that need to create resource pools

    Returns:
        ResourcePoolManager: Resource pool manager
    """
    resource_pool_spec = {}
    mapping = {}

    # Actor resource pool

    assert config.trainer.n_gpus_per_node > 0, "config.trainer.n_gpus_per_node must be greater than 0"
    assert config.trainer.nnodes > 0, "config.trainer.nnodes must be greater than 0"

    trainer_pool = [config.trainer.n_gpus_per_node] * config.trainer.nnodes
    resource_pool_spec["trainer_pool"] = trainer_pool

    # Map training-related roles to the same resource pool
    mapping[Role.Actor] = "trainer_pool"

    # Rollout resource pool
    assert config.rollout.n_gpus_per_node > 0, "config.rollout.n_gpus_per_node must be greater than 0"
    assert config.rollout.nnodes > 0, "config.rollout.nnodes must be greater than 0"

    rollout_pool = [config.rollout.n_gpus_per_node] * config.rollout.nnodes
    resource_pool_spec["rollout_pool"] = rollout_pool
    mapping[Role.Rollout] = "rollout_pool"

    return ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)


def create_role_worker_mapping(config):
    """
    Create mapping from roles to worker classes

    Args:
        config: Configuration object

    Returns:
        dict: Mapping from roles to worker classes
    """
    # GKD now supports two backends:
    # - "megatron": existing Megatron-LM split actor / rollout workers.
    # - "fsdp" / "fsdp2": new FSDP / FSDP2 split actor / rollout workers,
    #   reusing the disaggregated weight-sync path from
    #   ``recipe.one_step_off_policy.fsdp_workers``.
    actor_strategy = config.actor_rollout_ref.actor.strategy
    if actor_strategy == "megatron":
        from verl.single_controller.ray import RayWorkerGroup

        from megatron_workers import (
            MegatronOnPolicyDistillActorWorker,
            MegatronOnPolicyDistillRolloutWorker,
        )

        rollout_cls = MegatronOnPolicyDistillRolloutWorker
        actor_cls = MegatronOnPolicyDistillActorWorker
        ray_worker_group_cls = RayWorkerGroup

    elif actor_strategy in ("fsdp", "fsdp2"):
        from verl.single_controller.ray import RayWorkerGroup

        from fsdp_workers import (
            FSDPOnPolicyDistillActorWorker,
            FSDPOnPolicyDistillRolloutWorker,
        )

        rollout_cls = FSDPOnPolicyDistillRolloutWorker
        actor_cls = FSDPOnPolicyDistillActorWorker
        ray_worker_group_cls = RayWorkerGroup

    else:
        raise NotImplementedError(
            f"Unsupported actor_rollout_ref.actor.strategy: {actor_strategy}. "
            "Expected one of: 'megatron', 'fsdp', 'fsdp2'."
        )

    # Map roles to their corresponding remote worker classes.
    role_worker_mapping = {
        Role.Rollout: ray.remote(rollout_cls),
        Role.Actor: ray.remote(actor_cls),
    }

    return role_worker_mapping, ray_worker_group_cls


@ray.remote(num_cpus=10, max_concurrency=100)  # please make sure main_task is not scheduled on head
class TaskRunner:
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.
    """

    def run(self, config):
        """Execute the main PPO training workflow.

        This method sets up the distributed training environment, initializes
        workers, datasets, and reward functions, then starts the training process.

        Args:
            config: Training configuration object containing all parameters needed
                   for setting up and running the PPO training process.
        """
        # Print the initial configuration. `resolve=True` will evaluate symbolic values.
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        # Lazy import to avoid heavy deps (e.g. megatron) when only the entry
        # point is loaded for hydra config resolution.
        from ray_trainer import OnPolicyDistillTrainer

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        # Download the checkpoint from HDFS to the local machine.
        # `use_shm` determines whether to use shared memory, which could lead to faster model loading if turned on
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor.
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # Used for multimodal LLM, could be None
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Version validation for vllm.
        if config.actor_rollout_ref.rollout.name in ["vllm"]:
            from verl.utils.vllm import is_version_ge

            if config.actor_rollout_ref.model.get("lora_rank", 0) > 0:
                if not is_version_ge(pkg="vllm", minver="0.7.3"):
                    raise NotImplementedError("PPO LoRA is not supported before vllm 0.7.3")

        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)

        resource_pool_manager = create_resource_pool_manager(config, role_worker_mapping.keys())

        from verl.trainer.main_ppo import create_rl_sampler
        from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn

        # Create training and validation datasets.
        train_dataset = RLHFDataset(config.data.train_files, tokenizer, config.data, None)

        if config.data.val_files:
            val_dataset = RLHFDataset(config.data.val_files, tokenizer, config.data, None)
        else:
            val_dataset = None

        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the PPO trainer.
        trainer = OnPolicyDistillTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers of the trainer.
        trainer.init_workers()
        # Start the training process.
        asyncio.run(trainer.fit())


@hydra.main(config_path="config", config_name="on_policy_distill_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config_dict: Hydra configuration dictionary containing training parameters.
    """
    from time import time
    from verl.trainer.main_ppo import run_ppo

    start_time = time()

    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_ascend_device_name(config)

    run_ppo(config, task_runner_class=TaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
