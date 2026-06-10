# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
"""Dynamo training entry point."""

import hydra
import ray

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunner, run_ppo
from verl.utils.device import auto_set_device


def _register_dynamo_rollout():
    """Register recipe-side Dynamo rollout support in the current process."""
    from verl.workers.rollout.replica import RolloutReplicaRegistry

    def _load_dynamo():
        from recipe.dynamo.dynamo_async_server import DynamoReplica

        return DynamoReplica

    RolloutReplicaRegistry.register("dynamo", _load_dynamo)


class DynamoTaskRunner(TaskRunner):
    def run(self, config):
        _register_dynamo_rollout()
        return super().run(config)


@hydra.main(config_path="config", config_name="dynamo_trainer", version_base=None)
def main(config):
    _register_dynamo_rollout()
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DynamoTaskRunner))


if __name__ == "__main__":
    main()
