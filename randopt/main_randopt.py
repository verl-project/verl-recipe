import os
import socket
import tempfile
import time
from pprint import pprint

import hydra
import ray
from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.utils.device import auto_set_device

try:
    from recipe.randopt.randopt_ray_trainer import RandOptRayTrainer
    from recipe.randopt.task_utils import create_prompt_processor, create_reward_fn, create_vote_fns, load_data
except ModuleNotFoundError:
    from randopt.randopt_ray_trainer import RandOptRayTrainer
    from randopt.task_utils import create_prompt_processor, create_reward_fn, create_vote_fns, load_data


@hydra.main(config_path="config", config_name="randopt_trainer", version_base=None)
def main(config):
    auto_set_device(config)
    run_randopt(config)


def run_randopt(config) -> None:
    print(f"RandOpt hostname: {socket.gethostname()}, PID: {os.getpid()}")
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    if not ray.is_initialized():
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        if not ray_init_kwargs:
            ray_init_kwargs = {
                "address": "local",
                "include_dashboard": False,
                "ignore_reinit_error": True,
                "_temp_dir": tempfile.mkdtemp(prefix=f"ray_randopt_{int(time.time())}_"),
            }
        ray.init(**OmegaConf.to_container(ray_init_kwargs, resolve=True))

    data_config = OmegaConf.to_container(config.data, resolve=True)
    train_data = load_data(config.data.train_files)
    eval_data = load_data(config.data.val_files) if config.data.get("val_files") else []
    train_max_samples = int(config.data.get("train_max_samples", -1))
    val_max_samples = int(config.data.get("val_max_samples", -1))
    if train_max_samples > 0:
        train_data = train_data[:train_max_samples]
    if val_max_samples > 0:
        eval_data = eval_data[:val_max_samples]

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.path,
        trust_remote_code=config.model.get("trust_remote_code", False),
    )
    prompt_processor = create_prompt_processor(data_config)
    reward_fn = create_reward_fn(data_config)
    vote_answer_fn, vote_correct_fn = create_vote_fns(data_config)

    trainer = RandOptRayTrainer(
        config=config,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        train_data=train_data,
        eval_data=eval_data,
        prompt_processor=prompt_processor,
        vote_answer_fn=vote_answer_fn,
        vote_correct_fn=vote_correct_fn,
    )
    trainer.init_workers(config.model.path)
    trainer.fit()

    if ray.is_initialized():
        ray.shutdown()


if __name__ == "__main__":
    main()
