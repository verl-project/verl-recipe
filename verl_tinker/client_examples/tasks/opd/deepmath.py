import os

from tinker_cookbook import checkpoint_utils, cli_utils, model_info
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDatasetBuilder,
    TeacherConfig,
)

from ..utils import model_name_slug

DEFAULT_TEACHER_MODEL = "Qwen/Qwen3-30B-A3B"


async def run_opd_deepmath_test(base_url: str, model_name: str, tokenizer_name_or_path: str | None = None):
    """One-step OPD smoke run with actor rollouts and a frozen teacher sampler."""

    tokenizer_name_or_path = tokenizer_name_or_path or model_name
    teacher_model = os.environ.get("TINKER_TEACHER_MODEL", DEFAULT_TEACHER_MODEL)
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=model_name,
        explicit_renderer_name=model_info.get_recommended_renderer_name(model_name),
        load_checkpoint_path=None,
        base_url=base_url,
    )

    groups_per_batch = 2
    dataset_builder = PromptOnlyDatasetBuilder(
        dataset_name="deepmath",
        groups_per_batch=groups_per_batch,
        group_size=2,
        model_name_for_tokenizer=tokenizer_name_or_path,
        renderer_name=renderer_name,
        max_prompt_tokens=512,
    )
    dataset_config = DistillationDatasetConfig(
        dataset_builder=dataset_builder,
        teacher_config=TeacherConfig(base_model=teacher_model),
        groups_per_batch=groups_per_batch,
    )

    config = train_on_policy.Config(
        learning_rate=1e-5,
        dataset_configs=[dataset_config],
        model_name=model_name,
        renderer_name=renderer_name,
        lora_rank=0,
        max_tokens=512,
        temperature=1.0,
        kl_penalty_coef=1.0,
        kl_discount_factor=0.0,
        num_substeps=1,
        loss_fn="importance_sampling",
        loss_fn_config=None,
        wandb_project="verl-tinker-ci",
        wandb_name=f"opd-deepmath-{model_name_slug(model_name)}-teacher-{model_name_slug(teacher_model)}",
        log_path="/tmp/tinker-deepmath-opd-smoke",
        base_url=base_url,
        load_checkpoint_path=None,
        compute_post_kl=False,
        eval_every=0,
        save_every=0,
        max_steps=50,
    )

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    await train_on_policy.main(config)
