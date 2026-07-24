import os
import time
from pathlib import Path
from typing import cast

import datasets
import tinker
from tinker_cookbook import cli_utils, model_info, renderers
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.common import compute_mean_nll
from tinker_cookbook.supervised.data import conversation_to_datum
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig
from tinker_cookbook.tokenizer_utils import get_tokenizer

from ..utils import model_name_slug


async def run_no_robot_test(base_url, model_name, tokenizer_name_or_path=None):
    tokenizer_name_or_path = tokenizer_name_or_path or model_name
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=tokenizer_name_or_path,
        renderer_name=renderer_name,
        max_length=2048,
        batch_size=16,
        train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
    )
    config = train.Config(
        log_path="/tmp/tinker-sft-norobot-demo",
        model_name=model_name,
        renderer_name=renderer_name,
        dataset_builder=chat_datasets.NoRobotsBuilder(common_config=common),
        learning_rate=1e-5,
        num_epochs=1,
        lora_rank=0,
        max_steps=100,
        eval_every=25,
        save_every=0,
        base_url=base_url,
        # WandB
        wandb_project="verl-tinker-ci",
        wandb_name=f"sft-norobot-{model_name_slug(model_name)}",
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    await train.main(config)


# the task above is from tinker cookbook, which will call asample at the end requiring
# rollout deployment. the task below is written by use, which does not require
# rollout deployment


async def run_no_robot_direct_sft_test(base_url, model_name, tokenizer_name_or_path=None):
    tokenizer_name_or_path = tokenizer_name_or_path or model_name
    log_path = os.environ.get("TINKER_NOROBOT_DIRECT_LOG_PATH", "/tmp/direct-sft-norobot")
    batch_size = int(os.environ.get("TINKER_NOROBOT_DIRECT_BATCH_SIZE", "16"))
    max_steps = int(os.environ.get("TINKER_NOROBOT_DIRECT_MAX_STEPS", "100"))
    max_length = int(os.environ.get("TINKER_NOROBOT_DIRECT_MAX_LENGTH", "2048"))
    learning_rate = float(os.environ.get("TINKER_NOROBOT_DIRECT_LEARNING_RATE", "1e-5"))
    lora_rank = int(os.environ.get("TINKER_NOROBOT_DIRECT_LORA_RANK", "0"))

    cli_utils.check_log_dir(log_path, behavior_if_exists="delete")
    Path(log_path).mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(tokenizer_name_or_path)
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    renderer = renderers.get_renderer(renderer_name, tokenizer=tokenizer)

    dataset = cast(datasets.DatasetDict, datasets.load_dataset("HuggingFaceH4/no_robots"))
    train_dataset = dataset["train"].shuffle(seed=0)
    num_batches = len(train_dataset) // batch_size
    num_steps = min(max_steps, num_batches)
    if num_steps <= 0:
        raise ValueError(f"No full training batches available for batch_size={batch_size}")

    service_client = tinker.ServiceClient(base_url=base_url)
    training_client = await service_client.create_lora_training_client_async(
        base_model=model_name,
        rank=lora_rank,
    )

    print(
        f"Starting direct No Robots SFT: steps={num_steps}, batch_size={batch_size}, "
        f"max_length={max_length}, renderer={renderer_name}"
    )
    for step in range(num_steps):
        start_time = time.time()
        batch_start = step * batch_size
        batch_end = batch_start + batch_size
        batch_rows = train_dataset.select(range(batch_start, batch_end))
        batch = [
            conversation_to_datum(
                row["messages"],
                renderer,
                max_length,
                TrainOnWhat.LAST_ASSISTANT_MESSAGE,
            )
            for row in batch_rows
        ]

        fwd_bwd_future = await training_client.forward_backward_async(batch, loss_fn="cross_entropy")
        optim_step_future = await training_client.optim_step_async(
            tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
        )

        fwd_bwd_result = await fwd_bwd_future.result_async()
        optim_result = await optim_step_future.result_async()

        train_logprobs = [output["logprobs"] for output in fwd_bwd_result.loss_fn_outputs]
        train_weights = [datum.loss_fn_inputs["weights"] for datum in batch]
        metrics = {
            "step": step,
            "num_sequences": len(batch),
            "num_tokens": sum(datum.model_input.length for datum in batch),
            "train_mean_nll": compute_mean_nll(train_logprobs, train_weights),
            "time_total": time.time() - start_time,
        }
        if optim_result.metrics:
            metrics.update(optim_result.metrics)
        print(f"direct No Robots SFT metrics: {metrics}")
