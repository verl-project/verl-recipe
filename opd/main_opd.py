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
Online Policy Distillation (OPD) training entry point.

Uses verl's standard GRPO/PPO pipeline with an external teacher model
providing per-token KL distillation loss. The teacher is queried via
ZMQ after each rollout, and teacher log probs are injected as ref_log_prob
for the existing use_kl_loss mechanism.

Example:
    # Start teacher server first (see recipe/gkd/teacher/)
    # Then run:
    python3 -m recipe.opd.main_opd \\
        actor_rollout_ref.model.path=/path/to/student_model \\
        +teacher.server_ip=127.0.0.1 \\
        +teacher.server_port=15555 \\
        +teacher.kl_loss_coef=0.01 \\
        ...
"""

import os
import socket
import sys

import hydra
import ray
from omegaconf import OmegaConf


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Extract teacher config from hydra overrides
    teacher_ip = OmegaConf.select(config, "teacher.server_ip", default="127.0.0.1")
    teacher_port = OmegaConf.select(config, "teacher.server_port", default=15555)
    kl_loss_coef = OmegaConf.select(config, "teacher.kl_loss_coef", default=0.01)

    @ray.remote(num_cpus=1)
    class OPDTaskRunner:
        """Custom TaskRunner that uses OPDTrainer for online distillation."""

        def __init__(self):
            self.role_worker_mapping = {}
            self.mapping = {}

        def run(self, config):
            from omegaconf import OmegaConf, open_dict

            from verl.single_controller.ray import RayWorkerGroup
            from verl.trainer.main_ppo import TaskRunner, create_rl_dataset, create_rl_sampler
            from verl.trainer.ppo.ray_trainer import ResourcePoolManager
            from verl.utils import hf_processor, hf_tokenizer
            from verl.utils.dataset.rl_dataset import collate_fn
            from verl.utils.fs import copy_to_local

            print(f"OPDTaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
            OmegaConf.resolve(config)

            # Disable use_kl_loss at config level;
            # OPDTrainer will re-enable it with correct teacher settings
            with open_dict(config):
                config.actor_rollout_ref.actor.use_kl_loss = False

            # Standard verl setup
            local_path = copy_to_local(config.actor_rollout_ref.model.path)
            trust = config.data.get("trust_remote_code", False)
            tokenizer = hf_tokenizer(local_path, trust_remote_code=trust)
            processor = hf_processor(local_path, trust_remote_code=trust)

            # Setup workers (ActorRollout only, no separate Ref model needed)
            base = TaskRunner()
            base.add_actor_rollout_worker(config)
            if hasattr(base, "add_reward_worker"):
                base.add_reward_worker(config)

            resource_pool_spec = {
                "global_pool": [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            }
            resource_pool_manager = ResourcePoolManager(
                resource_pool_spec=resource_pool_spec,
                mapping=base.mapping,
            )

            train_dataset = create_rl_dataset(
                config.data.train_files, config.data, tokenizer, processor, is_train=True
            )
            val_dataset = create_rl_dataset(
                config.data.val_files, config.data, tokenizer, processor, is_train=False
            )
            train_sampler = create_rl_sampler(config.data, train_dataset)

            # Use OPDTrainer instead of RayPPOTrainer
            from recipe.opd.opd_trainer import OPDTrainer

            trainer = OPDTrainer(
                config=config,
                tokenizer=tokenizer,
                processor=processor,
                role_worker_mapping=base.role_worker_mapping,
                resource_pool_manager=resource_pool_manager,
                ray_worker_group_cls=RayWorkerGroup,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                collate_fn=collate_fn,
                train_sampler=train_sampler,
                teacher_ip=teacher_ip,
                teacher_port=teacher_port,
                kl_loss_coef=kl_loss_coef,
            )

            trainer.init_workers()
            trainer.fit()

    run_ppo(config, task_runner_class=OPDTaskRunner)


if __name__ == "__main__":
    main()
