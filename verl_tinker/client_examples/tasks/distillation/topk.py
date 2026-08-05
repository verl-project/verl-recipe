from tinker_cookbook import checkpoint_utils, cli_utils
from tinker_cookbook.distillation import train_off_policy
from tinker_cookbook.distillation.datasets import TeacherConfig
from tinker_cookbook.recipes.distillation.off_policy_reasoning import OpenThoughts3Builder
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from ..utils import model_name_slug, recommended_renderer_name

TEACHER_MODEL = "Qwen/Qwen3-30B-A3B"


async def run_topk_distillation_test(
    base_url: str,
    model_name: str,
    tokenizer_name_or_path: str | None = None,
    lite: bool = False,
):
    """Run Cookbook off-policy top-K distillation against a dedicated teacher."""

    tokenizer_name_or_path = tokenizer_name_or_path or model_name
    renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=model_name,
        explicit_renderer_name=recommended_renderer_name(model_name),
        load_checkpoint_path=None,
        base_url=base_url,
    )

    batch_size = 2 if lite else 4
    max_steps = 20 if lite else 200
    common_config = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=tokenizer_name_or_path,
        renderer_name=renderer_name,
        max_length=512 if lite else 1024,
        batch_size=batch_size,
        train_on_what=None,
    )
    dataset_builder = OpenThoughts3Builder(
        common_config=common_config,
        buffer_size=batch_size * max_steps,
        max_prompts=batch_size * max_steps,
    )
    dataset_config = train_off_policy.DatasetWithTeacher(
        dataset_builder=dataset_builder,
        teacher_config=TeacherConfig(base_model=TEACHER_MODEL),
    )

    config = train_off_policy.Config(
        learning_rate=2e-5,
        dataset_configs=[dataset_config],
        model_name=model_name,
        recipe_name="verl_tinker_topk_distillation",
        renderer_name=renderer_name,
        lora_rank=0,
        n_teacher_targets=20,
        teacher_concurrency=batch_size,
        batch_size=batch_size,
        save_every=0,
        eval_every=0,
        max_steps=max_steps,
        load_checkpoint_path=None,
        log_path="/tmp/tinker-topk-distillation-demo",
        wandb_project="verl-tinker-ci",
        wandb_name=(f"topk-distillation-{model_name_slug(model_name)}-teacher-{model_name_slug(TEACHER_MODEL)}"),
        base_url=base_url,
    )

    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    await train_off_policy.main(config)
