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
"""Dynamo training entry point.

Thin wrapper around ``verl.trainer.main_ppo`` that only swaps the hydra config
name. Unlike verl's own ``main()`` it also runs ``migrate_legacy_reward_impl``
before validation, preserving compatibility with older recipe configs.

The V0/V1 dispatch follows ``trainer.use_v1`` exactly like upstream:
V1 (default) runs TaskRunnerV1 → ``trainer.v1.trainer_mode``
(sync | colocate_async | separate_async); V0 remains reachable with
``trainer.use_v1=false`` until upstream removes it (deprecated, v0.9.0).
"""

import hydra

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


@hydra.main(config_path="config", config_name="dynamo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    manager_fqn = (config.actor_rollout_ref.rollout.get("agent", {}) or {}).get("agent_loop_manager_class") or ""
    if config.trainer.use_v1 and "DynamoAgentLoopManager" in str(manager_fqn):
        raise ValueError(
            "trainer.use_v1=true with agent_loop_manager_class=DynamoAgentLoopManager: the legacy "
            "manager does not write TransferQueue and violates the V1 contract. Use "
            "--config-name=dynamo_trainer_v1_colocate (agent_loop_manager_class=null) for V1, or "
            "set trainer.use_v1=false for the legacy path."
        )
    if config.trainer.use_v1:
        run_ppo(config, task_runner_class=TaskRunnerV1)
    else:
        from verl.trainer.main_ppo_v0 import TaskRunner

        run_ppo(config, task_runner_class=TaskRunner)


if __name__ == "__main__":
    main()
