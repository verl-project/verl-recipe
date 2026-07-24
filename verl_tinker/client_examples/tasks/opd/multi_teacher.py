from tinker_cookbook import checkpoint_utils, cli_utils, model_info
from tinker_cookbook.distillation import train_on_policy
from tinker_cookbook.distillation.datasets import (
    DistillationDatasetConfig,
    PromptOnlyDatasetBuilder,
    TeacherConfig,
)

from ..utils import model_name_slug

DEEPMATH_TEACHER_MODEL = "Qwen/Qwen3-32B"
TULU3_TEACHER_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"


async def run_opd_multi_teacher_test(
    base_url: str,
    model_name: str,
    tokenizer_name_or_path: str | None = None,
):
    """Nightly-sized DeepMath + Tulu3 multi-teacher OPD workload."""

    tokenizer_name_or_path = tokenizer_name_or_path or model_name
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=model_name,
        explicit_renderer_name=model_info.get_recommended_renderer_name(model_name),
        load_checkpoint_path=None,
        base_url=base_url,
    )

    groups_per_teacher = 16
    dataset_configs = [
        DistillationDatasetConfig(
            dataset_builder=PromptOnlyDatasetBuilder(
                dataset_name="deepmath",
                groups_per_batch=groups_per_teacher,
                group_size=2,
                model_name_for_tokenizer=tokenizer_name_or_path,
                renderer_name=renderer_name,
                max_prompt_tokens=1024,
            ),
            teacher_config=TeacherConfig(base_model=DEEPMATH_TEACHER_MODEL),
            groups_per_batch=groups_per_teacher,
        ),
        DistillationDatasetConfig(
            dataset_builder=PromptOnlyDatasetBuilder(
                dataset_name="tulu3",
                groups_per_batch=groups_per_teacher,
                group_size=2,
                model_name_for_tokenizer=tokenizer_name_or_path,
                renderer_name=renderer_name,
                max_prompt_tokens=1024,
            ),
            teacher_config=TeacherConfig(base_model=TULU3_TEACHER_MODEL),
            groups_per_batch=groups_per_teacher,
        ),
    ]

    config = train_on_policy.Config(
        learning_rate=2e-6,
        dataset_configs=dataset_configs,
        model_name=model_name,
        renderer_name=renderer_name,
        lora_rank=0,
        max_tokens=2048,
        temperature=1.0,
        kl_penalty_coef=1.0,
        kl_discount_factor=0.0,
        num_substeps=1,
        loss_fn="importance_sampling",
        loss_fn_config=None,
        wandb_project="verl-tinker-ci",
        wandb_name=(
            f"opd-multi-{model_name_slug(model_name)}"
            f"-deepmath-{model_name_slug(DEEPMATH_TEACHER_MODEL)}"
            f"-tulu3-{model_name_slug(TULU3_TEACHER_MODEL)}"
        ),
        log_path="/tmp/tinker-multi-teacher-opd-demo",
        base_url=base_url,
        load_checkpoint_path=None,
        compute_post_kl=True,
        eval_every=0,
        save_every=0,
        max_steps=100,
    )

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    await train_on_policy.main(config)
