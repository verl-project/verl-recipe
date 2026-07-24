from tinker_cookbook import checkpoint_utils, cli_utils, model_info
from tinker_cookbook.distillation import sdft
from tinker_cookbook.recipes.sdft.datasets import (
    SciKnowEvalSDFTBuilder,
)

from ..utils import model_name_slug


async def run_sdft_single_task_test(base_url: str, model_name: str, tokenizer_name_or_path: str | None = None):
    tokenizer_name_or_path = tokenizer_name_or_path or model_name
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=model_name,
        explicit_renderer_name=model_info.get_recommended_renderer_name(model_name),
        load_checkpoint_path=None,
        base_url=base_url,
    )

    group_size = 1
    groups_per_batch = 4

    dataset_builder = SciKnowEvalSDFTBuilder(
        groups_per_batch=groups_per_batch,
        group_size=group_size,
        model_name_for_tokenizer=tokenizer_name_or_path,
        renderer_name=renderer_name,
        domain="Chemistry",
    )

    train_dataset, test_dataset = await dataset_builder()

    config = sdft.Config(
        model_name=model_name,
        renderer_name=renderer_name,
        lora_rank=128,
        base_url=base_url,
        # Training
        learning_rate=2e-5,
        max_tokens=256,
        temperature=1.0,
        # SDFT-specific
        topk=20,
        reverse=False,
        teacher_sync_every=None,
        max_context_length=32768,
        # Optimizer
        num_substeps=1,
        # Logging / saving
        wandb_project="verl-tinker-ci",
        wandb_name=f"sdft-sciknoweval-{model_name_slug(model_name)}",
        log_path="/tmp/tinker-sciknoweval-sdft-smoke",
        eval_every=0,
        save_every=0,
        # Service / checkpoint
        load_checkpoint_path=None,
        max_steps=50,
    )

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")

    await sdft.main(config, train_dataset, test_dataset)
