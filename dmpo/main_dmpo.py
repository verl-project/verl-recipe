# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.device import auto_set_device

from .dmpo_patch import apply_dmpo_patch


class DMPOTaskRunner(TaskRunner):
    """Task runner that uses the DMPO actor rollout worker."""

    def add_actor_rollout_worker(self, config):
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        from .dmpo_worker import DMPOActorRolloutRefWorker

        actor_rollout_cls = DMPOActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
        if lora_rank <= 0:
            lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
        ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        if need_reference_policy(config) and not ref_in_actor:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout

        self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
        self.mapping[role] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls


@hydra.main(config_path="config", config_name="dmpo_trainer", version_base=None)
def main(config):
    apply_dmpo_patch()
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DMPOTaskRunner))


if __name__ == "__main__":
    main()
