# Copyright 2026 Huawei Technologies Co., Ltd. and/or its affiliates
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

import traceback

import hydra
import ray

from verl.experimental.fully_async_policy.fully_async_main import FullyAsyncTaskRunner
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role
from verl.utils.device import auto_set_device

from .reward_queue import RewardQueueClient, create_reward_queue
from .rollouter import Rollouter
from .trainer import Trainer

ray_metadata = FullyAsyncTaskRunner.__ray_metadata__
OriginalTaskRunner = ray_metadata.modified_class


@ray.remote(num_cpus=1)
class TaskRunner(OriginalTaskRunner):
    """
    Ray remote class for executing distributed PPO training tasks.
    """

    def run(self, config):
        print("[ASYNC MAIN] Starting fully async PPO training...")
        self._initialize_components(config)
        enable_reward_queue = config.async_training.get("enable_reward_queue", False)
        if enable_reward_queue:
            self._init_reward_queue_components(config)
        self._run_training_loop()

    def _init_reward_queue_components(self, config):
        max_reward_queue_size = ray.get(self.components["rollouter"].get_max_reward_queue_size.remote())
        print(f"[ASYNC MAIN] Creating RewardQueue... max_reward_queue_size {max_reward_queue_size}")
        reward_queue = create_reward_queue(config, max_reward_queue_size)
        reward_queue_client = RewardQueueClient(reward_queue)
        self.components["reward_queue"] = reward_queue
        self.components["reward_queue_client"] = reward_queue_client

        ray.get(self.components["rollouter"].set_reward_queue_client.remote(self.components["reward_queue_client"]))

        reward_loop_worker_handles = ray.get(self.components["rollouter"].get_reward_loop_worker_handles.remote())
        if reward_loop_worker_handles:
            ray.get(self.components["rollouter"].set_reward_loop_worker_handles.remote(reward_loop_worker_handles))
            print(f"[ASYNC MAIN] Set reward_loop_worker_handles: {len(reward_loop_worker_handles)} workers")
        else:
            print("[ASYNC MAIN] WARNING: No reward_loop_worker_handles available, scoring will not work")

    def _create_rollouter(self, config) -> None:
        print("[ASYNC MAIN] Starting create rollouter...")
        rollouter = Rollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        # set_hybrid_worker_group must be called BEFORE init_workers() so that
        # _init_async_rollout_manager can pass the hybrid WG to ALM.create().
        if "hybrid_worker_group" in self.components:
            ray.get(rollouter.set_hybrid_worker_group.remote(self.components["hybrid_worker_group"]))
            print("[ASYNC MAIN] Hybrid worker group injected into rollouter")

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[ASYNC MAIN] Rollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[ASYNC MAIN] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = Trainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[ASYNC MAIN] FullyAsyncTrainer created and initialized successfully")


@hydra.main(config_path="config", config_name="fully_async", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    # Ensure async training config exists
    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    from time import time

    start_time = time()
    auto_set_device(config)
    # TODO: unify rollout config with actor_rollout_ref
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node
    config = migrate_legacy_reward_impl(config)

    is_train_over = False
    auto_resume_on_error = config.trainer.get("auto_resume_on_error", False)
    while not is_train_over:
        try:
            run_ppo(config, task_runner_class=TaskRunner)
            is_train_over = True
        except (ray.exceptions.RayTaskError, Exception) as e:
            print(e, str(traceback.format_exc()))
            # raise e
        finally:
            ray.shutdown()

        if not auto_resume_on_error:
            break

    print("training process successfully!")
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
