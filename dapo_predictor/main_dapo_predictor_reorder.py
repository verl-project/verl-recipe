"""DAPO entrypoint with legacy predictor-driven reorder support."""

import os
import socket

import hydra
import ray
from recipe.dapo.main_dapo import DAPOTaskRunner
from recipe.dapo_predictor.predictor_dapo_trainer import PredictorRayDAPOTrainer
from recipe.dapo_predictor.predictor_worker import PredictorAsyncActorRolloutRefWorker

from verl.experimental.reward_loop import migrate_legacy_reward_impl
from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler, run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device


class PredictorDAPOTaskRunner(DAPOTaskRunner):
    def add_actor_rollout_worker(self, config):
        if config.actor_rollout_ref.actor.strategy in {"fsdp", "fsdp2"}:
            from verl.single_controller.ray import RayWorkerGroup
            from verl.trainer.ppo.ray_trainer import Role

            actor_rollout_cls = PredictorAsyncActorRolloutRefWorker
            ray_worker_group_cls = RayWorkerGroup
            self.role_worker_mapping[Role.ActorRollout] = ray.remote(actor_rollout_cls)
            self.mapping[Role.ActorRollout] = "global_pool"
            return actor_rollout_cls, ray_worker_group_cls
        return super().add_actor_rollout_worker(config)

    def run(self, config):
        from pprint import pprint

        from omegaconf import OmegaConf, open_dict

        from verl.utils import hf_processor, hf_tokenizer
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.utils.fs import copy_to_local

        print(f"TaskRunner hostname: {socket.gethostname()}, PID: {os.getpid()}")
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        trainer_predictor_cfg = OmegaConf.select(config, "trainer.predictor_reorder", default=None)
        if trainer_predictor_cfg is not None:
            with open_dict(config.actor_rollout_ref):
                config.actor_rollout_ref.predictor_reorder = OmegaConf.create(
                    OmegaConf.to_container(trainer_predictor_cfg, resolve=True)
                )
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path, use_shm=config.actor_rollout_ref.model.get("use_shm", False)
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        resource_pool_manager = self.init_resource_pool_mgr(config)
        train_dataset = create_rl_dataset(
            config.data.train_files,
            config.data,
            tokenizer,
            processor,
            is_train=True,
            max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files,
            config.data,
            tokenizer,
            processor,
            is_train=False,
            max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)
        trainer = PredictorRayDAPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="../dapo/config", config_name="dapo_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    config = migrate_legacy_reward_impl(config)
    run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(PredictorDAPOTaskRunner))


if __name__ == "__main__":
    main()
