from tinker_cookbook import cli_utils, model_info
from tinker_cookbook.recipes.chat_sl import chat_datasets
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import ChatDatasetBuilderCommonConfig

from ..utils import model_name_slug


async def run_tulu3_test(base_url, model_name, tokenizer_name_or_path=None):
    tokenizer_name_or_path = tokenizer_name_or_path or model_name
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=tokenizer_name_or_path,
        renderer_name=renderer_name,
        max_length=4096,
        batch_size=32,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    config = train.Config(
        log_path="/tmp/cookbook-sl-basic",
        model_name=model_name,
        renderer_name=renderer_name,
        dataset_builder=chat_datasets.Tulu3Builder(common_config=common),
        learning_rate=3e-5,
        num_epochs=1,
        max_steps=10,
        eval_every=0,
        save_every=0,
        base_url=base_url,
        # WandB
        wandb_project="verl-tinker-ci",
        wandb_name=f"sft-tulu3-{model_name_slug(model_name)}",
    )
    cli_utils.check_log_dir(config.log_path, behavior_if_exists="delete")
    await train.main(config)
