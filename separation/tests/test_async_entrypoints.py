# Copyright 2025 Meituan Ltd. and/or its affiliates
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

import subprocess
import sys

import pytest
from omegaconf import OmegaConf


@pytest.mark.parametrize(
    "module, config_name, strategy",
    [
        ("recipe.fully_async_policy.fully_async_main", "fully_async_ppo_trainer", "fsdp"),
        ("recipe.fully_async_policy.fully_async_main", "fully_async_ppo_megatron_trainer", "megatron"),
        ("recipe.one_step_off_policy.main_ppo", "one_step_off_ppo_trainer", "fsdp"),
        ("recipe.one_step_off_policy.main_ppo", "one_step_off_ppo_megatron_trainer", "megatron"),
    ],
)
def test_entrypoint_loads_pinned_verl_config(module, config_name, strategy):
    result = subprocess.run(
        [sys.executable, "-m", module, "--config-name", config_name, "--cfg", "job"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stderr
    config = OmegaConf.create(result.stdout)
    assert config.actor_rollout_ref.actor.strategy == strategy
    assert config.rollout.n_gpus_per_node > 0


def test_async_trainers_inherit_verl_v0():
    from recipe.fully_async_policy.fully_async_rollouter import FullyAsyncRollouter
    from recipe.fully_async_policy.fully_async_trainer import FullyAsyncTrainer
    from recipe.one_step_off_policy.ray_trainer import OneStepOffRayTrainer
    from recipe.separation.ray_trainer import SeparateRayPPOTrainer

    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    assert issubclass(SeparateRayPPOTrainer, RayPPOTrainer)
    assert issubclass(OneStepOffRayTrainer, SeparateRayPPOTrainer)
    assert issubclass(FullyAsyncTrainer.__ray_metadata__.modified_class, SeparateRayPPOTrainer)
    assert issubclass(FullyAsyncRollouter.__ray_metadata__.modified_class, SeparateRayPPOTrainer)
