from tinker_cookbook import checkpoint_utils, cli_utils, model_info
from tinker_cookbook.recipes.math_rl import math_env
from tinker_cookbook.rl.train import Config, KLReferenceConfig, main

from ..utils import model_name_slug


async def run_math_rl_gsm8k_test(base_url, model_name, tokenizer_name_or_path=None):
    tokenizer_name_or_path = tokenizer_name_or_path or model_name
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=model_name,
        explicit_renderer_name=model_info.get_recommended_renderer_name(model_name),
        load_checkpoint_path=None,
        base_url=base_url,
    )

    group_size = 16
    groups_per_batch = 16

    dataset_builder = math_env.get_math_dataset_builder(
        dataset_name="gsm8k",
        batch_size=groups_per_batch,
        model_name_for_tokenizer=tokenizer_name_or_path,
        renderer_name=renderer_name,
        group_size=group_size,
        seed=0,
    )

    config = Config(
        model_name=model_name,
        renderer_name=renderer_name,
        dataset_builder=dataset_builder,
        max_steps=30,
        # Training
        learning_rate=1e-5,
        lora_rank=0,
        max_tokens=4096,
        temperature=0.7,
        kl_penalty_coef=0.01,
        kl_reference_config=KLReferenceConfig(base_model=model_name),
        # Logging / saving
        wandb_project="verl-tinker-ci",
        wandb_name=f"math-rl-gsm8k-{model_name_slug(model_name)}",
        log_path="/tmp/tinker-gsm8k-rl-smoke",
        eval_every=0,
        save_every=0,
        compute_post_kl=False,
        # Service
        base_url=base_url,
        load_checkpoint_path=None,
        # Loss
        loss_fn="importance_sampling",
        loss_fn_config=None,
        # No async/off-policy mode for simplest version
        async_config=None,
        stream_minibatch_config=None,
    )

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")

    await main(config)
