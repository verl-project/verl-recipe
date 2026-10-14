"""Hydra/Ray entry point for shared HACPO training."""

from pprint import pprint

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.utils import need_critic, need_reference_policy
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.logging_utils import configure_verl_logging

from .hacpo_config import build_agent_runtime_specs, validate_hacpo_config


@ray.remote
class HacpoTaskRunner:
    """Initialize one AgentLoopManager per policy around the shared trainer."""

    def __init__(self) -> None:
        self.trainer = None
        self.agent_loop_managers = {}

    def _init_agent_loop_managers(self) -> None:
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ

        clients = self.trainer.get_llm_clients()
        for spec in self.trainer.runtime_agent_specs:
            manager_fqn = spec.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
            manager_cls = load_class_from_fqn(manager_fqn, "AgentLoopManager") if manager_fqn else AgentLoopManagerTQ
            self.agent_loop_managers[spec.agent_id] = manager_cls.create(
                config=spec.config,
                llm_client=clients[spec.agent_id],
                teacher_client=None,
                reward_loop_worker_handles=self.trainer.get_reward_handles(spec.agent_id),
            )

    def run(self, config: DictConfig) -> None:
        configure_verl_logging()
        import transfer_queue as tq

        from .hacpo_ray_trainer import HacpoTrainer

        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config, resolve=True))
        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = HacpoTrainer(config=config)
            self.trainer.init()
            self._init_agent_loop_managers()
            self.trainer.fit(self.agent_loop_managers)
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path="config", config_name="hacpo_trainer", version_base=None)
def main(config: DictConfig) -> None:
    auto_set_device(config)
    specs = build_agent_runtime_specs(config)
    validate_hacpo_config(config, specs)
    for spec in specs:
        validate_config(
            config=spec.config,
            use_reference_policy=need_reference_policy(spec.config),
            use_critic=need_critic(spec.config),
        )
    run_ppo(config, task_runner_class=HacpoTaskRunner)


if __name__ == "__main__":
    main()
