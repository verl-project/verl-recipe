# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""
Entry point for the DR.Kernel recipe — fully-async PPO with MRS batch filter.

Subclasses `verl.experimental.fully_async_policy.fully_async_main.FullyAsyncTaskRunner`
to swap the trainer instantiation for `DrKernelFullyAsyncTrainer`. Verl core
stays untouched.

Run:
    python -m recipe.drkernel.main \\
        --config-path=recipe/drkernel/config \\
        --config-name=drkernel_async_ppo_trainer \\
        ...

(Or any equivalent Hydra invocation that resolves to
`recipe/drkernel/config/drkernel_async_ppo_trainer.yaml`.)
"""

from time import time

import hydra
import ray
from recipe.drkernel import (
    algos,  # noqa: F401  # registers the "trloo" advantage estimator
    rollout_veto_patch,  # noqa: F401  # overrides "bypass_mode" loss with MRS veto
)
from recipe.drkernel.trainer.drkernel_async_rollouter import DrKernelFullyAsyncRollouter
from recipe.drkernel.trainer.drkernel_async_trainer import DrKernelFullyAsyncTrainer

from verl.experimental.fully_async_policy.fully_async_main import (
    FullyAsyncTaskRunner as _RemoteFullyAsyncTaskRunner,
)
from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.experimental.separation.utils import create_resource_pool_manager
from verl.trainer.ppo.utils import Role, need_reward_model
from verl.utils.device import auto_set_device

# Unwrap to the underlying class so we can subclass it.
_BaseFullyAsyncTaskRunner = _RemoteFullyAsyncTaskRunner.__ray_actor_class__


class DrKernelFullyAsyncTaskRunnerImpl(_BaseFullyAsyncTaskRunner):
    """FullyAsyncTaskRunner that instantiates DR.Kernel's
    trainer/rollouter subclasses instead of the upstream defaults.
    Everything else is inherited."""

    def _create_rollouter(self, config) -> None:
        """Swap in `DrKernelFullyAsyncRollouter` so validation reports
        `val-core/<src>/reward/mean@N` as the per-(traj, turn) mean
        instead of the per-trajectory sum-of-turn-rewards."""
        print("[DrKernel] Starting create rollouter...")
        rollouter = DrKernelFullyAsyncRollouter.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=None,
            resource_pool_manager=create_resource_pool_manager(config, roles=[Role.Rollout]),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(rollouter.init_workers.remote())
        ray.get(rollouter.set_max_required_samples.remote())

        self.components["rollouter"] = rollouter
        print("[DrKernel] DrKernelFullyAsyncRollouter created and initialized successfully")

    def _create_trainer(self, config) -> None:
        print("[DrKernel] Starting create trainer...")
        trainer_role_mapping = {
            role: worker_cls
            for role, worker_cls in self.components["role_worker_mapping"].items()
            if role != Role.Rollout
        }

        trainer = DrKernelFullyAsyncTrainer.remote(
            config=config,
            tokenizer=self.components["tokenizer"],
            role_worker_mapping=trainer_role_mapping,
            resource_pool_manager=create_resource_pool_manager(config, roles=list(trainer_role_mapping.keys())),
            ray_worker_group_cls=self.components["ray_worker_group_cls"],
            processor=self.components["processor"],
            device_name=config.trainer.device,
        )

        ray.get(trainer.init_workers.remote())
        self.components["trainer"] = trainer
        print("[DrKernel] DrKernelFullyAsyncTrainer created and initialized successfully")


# Same resource spec as upstream FullyAsyncTaskRunner.
DrKernelFullyAsyncTaskRunner = ray.remote(num_cpus=1)(DrKernelFullyAsyncTaskRunnerImpl)


@hydra.main(config_path="config", config_name="drkernel_async_ppo_trainer", version_base=None)
def main(config):
    from verl.trainer.main_ppo import run_ppo

    if not hasattr(config, "async_training"):
        raise RuntimeError("must set async_training config")

    assert config.async_training.use_trainer_do_validate is False, "use_trainer_do_validate is not ready to use."

    if need_reward_model(config) and config.async_training.use_trainer_do_validate:
        raise NotImplementedError("use_trainer_do_validate with GenRM/DisRM is not yet supported.")

    start_time = time()
    auto_set_device(config)
    config.actor_rollout_ref.rollout.nnodes = config.rollout.nnodes
    config.actor_rollout_ref.rollout.n_gpus_per_node = config.rollout.n_gpus_per_node

    # Resolve all OmegaConf interpolations BEFORE legacy-reward migration.
    # The yaml uses `${reward_model}` to forward the reward_model block into
    # `custom_reward_function.reward_kwargs.reward_config`. `migrate_legacy_
    # reward_impl` *deletes* `config.reward_model` after copying its known
    # fields elsewhere, so a lazy interpolation would raise
    # `InterpolationKeyError` on first access in the Ray reward worker.
    # Resolving here turns the interpolation into a concrete dict snapshot
    # that survives the migration intact.
    from omegaconf import OmegaConf

    OmegaConf.resolve(config)

    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=DrKernelFullyAsyncTaskRunner)
    print(f"total time: {time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
