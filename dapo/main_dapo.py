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

import os
import socket

import hydra
import ray
from omegaconf import OmegaConf

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env
from verl.utils.device import auto_set_device, is_cuda_available

from .dapo_ray_trainer import RayDAPOTrainer


@hydra.main(config_path="config", config_name="dapo_trainer", version_base=None)
def main(config):
    # Automatically set `config.trainer.device = npu` when running on Ascend NPU.
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)

    run_ppo(config)


def run_ppo(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    try:
        if (
            is_cuda_available
            and config.global_profiler.tool == "nsys"
            and OmegaConf.select(config.global_profiler, "steps") is not None
            and len(OmegaConf.select(config.global_profiler, "steps")) > 0
        ):
            nsight_options = OmegaConf.to_container(
                config.global_profiler.global_tool_config.nsys.controller_nsight_options
            )
            runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
        else:
            runner = TaskRunner.remote()
        ray.get(runner.run.remote(config))
    finally:
        if ray.is_initialized():
            ray.shutdown()


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
class TaskRunner:
    def run(self, config):
        # print initial config
        from pprint import pprint

        from omegaconf import OmegaConf

        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")

        pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
        OmegaConf.resolve(config)

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.actor_rollout_ref.model.path)

        # instantiate tokenizer
        from verl.utils import hf_processor, hf_tokenizer

        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        # used for multimodal LLM, could be none
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        from verl.single_controller.ray import RayWorkerGroup
        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
        from verl.trainer.ppo.utils import need_reference_policy

        use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")
        ray_worker_group_cls = RayWorkerGroup

        role_worker_mapping = {}
        mapping = {}
        global_pool_id = "global_pool"

        # define actor/rollout worker class
        if use_legacy_worker_impl == "disable":
            from verl.workers.engine_workers import ActorRolloutRefWorker

            actor_rollout_cls = ActorRolloutRefWorker

            lora_rank = config.actor_rollout_ref.model.get("lora", {}).get("rank", 0)
            if lora_rank <= 0:
                lora_rank = config.actor_rollout_ref.model.get("lora_rank", 0)
            ref_in_actor = lora_rank > 0 or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
            if need_reference_policy(config) and not ref_in_actor:
                actor_role = Role.ActorRolloutRef
            else:
                actor_role = Role.ActorRollout
            role_worker_mapping[actor_role] = ray.remote(actor_rollout_cls)
            mapping[actor_role] = global_pool_id
        else:
            if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
            elif config.actor_rollout_ref.actor.strategy == "megatron":
                from verl.workers.megatron_workers import AsyncActorRolloutRefWorker
            else:
                raise NotImplementedError

            actor_rollout_cls = AsyncActorRolloutRefWorker
            role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
            mapping[Role.ActorRollout] = global_pool_id

        # define critic worker class
        if config.critic.strategy in {"fsdp", "fsdp2"}:
            from verl.workers.fsdp_workers import CriticWorker
        elif config.critic.strategy == "megatron":
            from verl.workers.megatron_workers import CriticWorker
        else:
            raise NotImplementedError
        role_worker_mapping[Role.Critic] = ray.remote(CriticWorker)
        mapping[Role.Critic] = global_pool_id

        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }

        # reward model
        if config.reward.reward_model.enable:
            if config.reward.reward_model.get("strategy") in {"fsdp", "fsdp2"}:
                from verl.workers.fsdp_workers import RewardModelWorker
            elif config.reward.reward_model.get("strategy") == "megatron":
                from verl.workers.megatron_workers import RewardModelWorker
            else:
                raise NotImplementedError
            role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
            mapping[Role.RewardModel] = global_pool_id

        # reference model (fused into ActorRolloutRefWorker when disable)
        if use_legacy_worker_impl != "disable":
            if need_reference_policy(config):
                role_worker_mapping[Role.RefPolicy] = ray.remote(actor_rollout_cls)
                mapping[Role.RefPolicy] = global_pool_id

        resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

        trainer = RayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
        )
        trainer.init_workers()
        trainer.fit()


if __name__ == "__main__":
    main()
