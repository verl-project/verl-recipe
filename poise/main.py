# Copyright 2026 POISE authors
# SPDX-License-Identifier: Apache-2.0
"""Launch POISE through veRL's Ray and TransferQueue runtime."""

import hydra
import ray
import transfer_queue as tq
from omegaconf import OmegaConf

from verl.trainer.main_ppo import run_ppo
from verl.trainer.ppo.v1 import AgentLoopManagerTQ
from verl.utils.config import validate_config
from verl.utils.device import auto_set_device
from verl.utils.logging_utils import configure_verl_logging

from .trainer import PoiseTrainer, validate_poise_config


@ray.remote
class PoiseTaskRunner:
    def run(self, config):
        configure_verl_logging()
        config.transfer_queue.enable = True
        OmegaConf.resolve(config)
        tq.init(config.transfer_queue)
        trainer, succeeded = None, False
        try:
            trainer = PoiseTrainer(config)
            trainer.init()
            manager = AgentLoopManagerTQ.create(
                config=config,
                llm_client=trainer.get_llm_client(),
                teacher_client=trainer.get_teacher_client(),
                reward_loop_worker_handles=trainer.get_reward_handles(),
            )
            trainer.fit(manager)
            succeeded = True
        finally:
            try:
                tracking = getattr(trainer, "logger", None)
                if tracking:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


@hydra.main(config_path="config", config_name="qwen3_4b", version_base=None)
def main(config):
    auto_set_device(config)
    validate_poise_config(config)
    validate_config(config, use_reference_policy=False, use_critic=False)
    run_ppo(config, task_runner_class=PoiseTaskRunner)


if __name__ == "__main__":
    main()
