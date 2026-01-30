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
SDPO (Self-Distillation Policy Optimization) main entry point.

This module provides the entry point for SDPO training, which extends the
one-step-off-policy trainer with self-distillation advantages.
"""

import asyncio
import os
import socket

import hydra
import ray

from verl.experimental.one_step_off_policy.main_ppo import create_resource_pool_manager, create_role_worker_mapping
from verl.experimental.one_step_off_policy.utils import need_critic
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
from verl.trainer.ppo.reward import load_reward_manager
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

from .sdpo_ray_trainer import SDPORayTrainer


@ray.remote(num_cpus=10, max_concurrency=100)
class SDPOTaskRunner:
    """Task runner for SDPO training."""

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"SDPOTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))

        OmegaConf.resolve(config)

        role_worker_mapping, ray_worker_group_cls = create_role_worker_mapping(config)

        # validate config
        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        # Download the checkpoint from HDFS to the local machine
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )

        # Instantiate the tokenizer and processor
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Load the reward manager for training and validation
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0, **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1, **config.reward_model.get("reward_kwargs", {})
        )

        resource_pool_manager = create_resource_pool_manager(config, role_worker_mapping.keys())

        from verl.utils.dataset.rl_dataset import collate_fn

        # Create training and validation datasets
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor, max_samples=config.data.get("val_max_samples", -1)
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Initialize the SDPO trainer
        trainer = SDPORayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
            device_name=config.trainer.device,
        )
        # Initialize the workers
        trainer.init_workers()
        # Start the training process
        asyncio.run(trainer.fit())


@hydra.main(config_path="config", config_name="sdpo_trainer", version_base=None)
def main(config):
    from time import time

    from verl.trainer.main_ppo import run_ppo

    start_time = time()

    # Automatically set device for NPU if applicable
    auto_set_device(config)

    run_ppo(config, task_runner_class=SDPOTaskRunner)
    print(f"SDPO training completed. Total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
