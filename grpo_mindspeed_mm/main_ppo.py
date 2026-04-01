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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.utils import need_reference_policy
from verl.utils.device import auto_set_device, is_cuda_available


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config)


# Define a function to run the PPO-like training process
def run_ppo(config, task_runner_class=None) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
        task_runner_class: For recipe to change TaskRunner.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(MMTaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()
    ray.get(runner.run.remote(config))

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_kwargs.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


class MMTaskRunner(TaskRunner):
    """Ray remote class for executing distributed PPO training tasks.

    This class encapsulates the main training logic and runs as a Ray remote actor
    to enable distributed execution across multiple nodes and GPUs.

    Attributes:
        role_worker_mapping: Dictionary mapping Role enums to Ray remote worker classes
        mapping: Dictionary mapping Role enums to resource pool IDs for GPU allocation
    """

    def add_actor_rollout_worker(self, config):
        """Add actor rollout worker based on the actor strategy."""
        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import Role

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        # use new model engine implementation
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup

            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            # NOTE: In new model engine, ref policy and actor rollout are in same ActorRolloutRefWorker,
            # while in legacy model engine, ref policy is in a separate ActorRolloutRefWorker.
            if need_reference_policy(config) and not ref_in_actor:
                role = Role.ActorRolloutRef
            else:
                role = Role.ActorRollout
            self.role_worker_mapping[role] = ray.remote(actor_rollout_cls)
            self.mapping[role] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls

        # Note: sync mode validation is now handled in RolloutConfig.__post_init__
        # Always use async worker since sync mode is deprecated and rejected
        from .mm_workers import AsyncMMActorRolloutRefWorker

        actor_rollout_cls = AsyncMMActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup

        self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
        self.mapping[Role.ActorRollout] = "global_pool"
        return actor_rollout_cls, ray_worker_group_cls


if __name__ == "__main__":
    main()
