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
"""SPPO entry point for the current verl core (v1 trainer + engine workers).

``main_sppo.py`` next to this file is the legacy entry point: it builds
``RaySPPOTrainer`` / ``SPPOActorRolloutRefWorker`` subclasses pinned to verl
bcb6386 (see ``REQUIRED_VERL.txt``) and no longer imports against this core.
This module is the port described in ``sppo_core_algos.py`` -- SPPO as a
registered advantage estimator plus a registered policy loss, on top of the stock
trainer, so it only has to do two things:

1. import ``sppo_core_algos`` to populate both registries, and
2. hand ``run_ppo`` a TaskRunner subclass *defined here*, so that Ray imports
   this module inside the trainer actor too -- that actor is where
   ``compute_advantage`` looks ``sppo`` up, and it is a separate process from the
   one running this ``main()``.

Same pattern as ``recipe/dapo/main_dapo.py``.

The actor workers compute the policy loss in yet another process; they pick the
registration up through ``actor_rollout_ref.model.external_lib`` (imported by
``HFModelConfig.__post_init__``), which the launch script sets to
``[recipe.sppo.sppo_core_algos]``.

Usage::

    python3 -m recipe.sppo.main_sppo_v1 \
        algorithm.adv_estimator=sppo \
        +algorithm.sppo_eta=1.0 \
        actor_rollout_ref.actor.policy_loss.loss_mode=sppo \
        actor_rollout_ref.model.external_lib='[recipe.sppo.sppo_core_algos]'

Hydra reads the stock ``verl/trainer/config/ppo_trainer.yaml``; SPPO needs no
config group of its own because every knob it adds is a registry entry plus
``algorithm.sppo_eta``.
"""

import hydra
import ray

from verl.trainer.main_ppo import TaskRunnerV1, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device

# Importing this module registers @register_adv_est("sppo") and
# @register_policy_loss("sppo"); the import is the point, the names are not used.
from . import sppo_core_algos  # noqa: F401

# `TaskRunnerV1` is already `@ray.remote`-decorated in main_ppo, i.e. an
# ActorClass; `__ray_actor_class__` is the plain class underneath it, which is
# what can be subclassed. Fall back to the object itself for Ray versions that
# hand out the plain class.
_BaseTaskRunner = getattr(TaskRunnerV1, "__ray_actor_class__", TaskRunnerV1)


class SPPOTaskRunner(_BaseTaskRunner):
    """Stock v1 TaskRunner.

    Deliberately empty: SPPO changes no trainer behaviour. Its only job is to
    make the trainer actor import this module (and therefore register SPPO) when
    Ray instantiates it.
    """


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    """SPPO training entry point; mirrors ``verl.trainer.main_ppo.main``."""
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)

    if not config.trainer.get("use_v1", True):
        raise ValueError(
            "recipe.sppo.main_sppo_v1 requires trainer.use_v1=True: SPPO is registered as an advantage "
            "estimator plus a policy loss for the v1 trainer / engine workers. For the legacy subclass "
            "implementation use recipe.sppo.main_sppo with the verl commit pinned in REQUIRED_VERL.txt."
        )

    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )

    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(SPPOTaskRunner))


if __name__ == "__main__":
    main()
