from __future__ import annotations

from functools import partial
from pprint import pprint

import hydra
import ray
import transfer_queue as tq
from omegaconf import DictConfig, OmegaConf
from recipe.echo.loss import echo_ppo_loss

from verl.single_controller.base.decorator import Dispatch, register
from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import Role, need_critic, need_reference_policy
from verl.trainer.ppo.v1 import AgentLoopManagerTQ, PPOTrainerSync
from verl.utils.config import omega_conf_to_dataclass, validate_config
from verl.utils.device import auto_set_device
from verl.utils.logging_utils import configure_verl_logging
from verl.workers.engine_workers import ActorRolloutRefWorker


class EchoActorRolloutRefWorker(ActorRolloutRefWorker):
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        super().init_model()
        if "actor" not in self.role:
            return

        actor_config = omega_conf_to_dataclass(self.config.actor)
        aux_token_loss_coeff = float(self.config.echo_aux_token_loss_coeff)
        self.loss_fn = partial(
            echo_ppo_loss,
            config=actor_config,
            aux_token_loss_coeff=aux_token_loss_coeff,
        )
        self.actor.set_loss_fn(self.loss_fn)


class EchoPPOTrainerSync(PPOTrainerSync):
    def _init_resource_pool_mgr(self):
        super()._init_resource_pool_mgr()
        if Role.ActorRolloutRef in self.role_worker_mapping:
            role = Role.ActorRolloutRef
        else:
            role = Role.ActorRollout
        self.role_worker_mapping[role] = ray.remote(EchoActorRolloutRefWorker)


@ray.remote
class EchoTaskRunner:
    def run(self, config: DictConfig):
        configure_verl_logging()
        if config.trainer.v1.trainer_mode != "sync":
            raise ValueError("The ECHO recipe currently supports trainer.v1.trainer_mode=sync only")

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)

        trainer = None
        succeeded = False
        try:
            trainer = EchoPPOTrainerSync(config=config)
            trainer.init()
            agent_loop_manager = AgentLoopManagerTQ.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(agent_loop_manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path="../../verl/trainer/config", config_name="ppo_trainer", version_base=None)
def main(config: DictConfig):
    auto_set_device(config)
    validate_config(
        config=config,
        use_reference_policy=need_reference_policy(config),
        use_critic=need_critic(config),
    )
    run_ppo(config, task_runner_class=EchoTaskRunner)


if __name__ == "__main__":
    main()
