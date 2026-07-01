import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import chz
import datasets
import tinker
from tinker_cookbook import checkpoint_utils, cli_utils, model_info, renderers
from tinker_cookbook.eval.benchmarks import BenchmarkConfig, BenchmarkResult, run_benchmark
from tinker_cookbook.eval.benchmarks import gsm8k as _gsm8k_benchmark  # noqa: F401
from tinker_cookbook.recipes.math_rl import math_env
from tinker_cookbook.renderers import TrainOnWhat
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.supervised import train as sft_train
from tinker_cookbook.supervised.data import SupervisedDatasetFromHFDataset, conversation_to_datum
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer


DATASET_NAME = "gsm8k"

WANDB_PROJECT = "verl-remote-actor-ci"

EXPERIMENT_ROOT = Path(os.environ.get("TINKER_GSM8K_EXPERIMENT_ROOT", "/tmp/tinker-gsm8k-sft-then-rl-demo"))
SFT_LOG_PATH = str(EXPERIMENT_ROOT / "sft")
RL_LOG_PATH = str(EXPERIMENT_ROOT / "rl")
EVAL_LOG_PATH = EXPERIMENT_ROOT / "eval"

SFT_BATCH_SIZE = 64
SFT_MAX_LENGTH = 2048
SFT_MAX_STEPS = int(os.environ.get("TINKER_GSM8K_SFT_MAX_STEPS", "50"))
SFT_LEARNING_RATE = 2e-5

RL_GROUP_SIZE = 16
RL_GROUPS_PER_BATCH = 16
RL_MAX_STEPS = int(os.environ.get("TINKER_GSM8K_RL_MAX_STEPS", "100"))
RL_MAX_TOKENS = int(os.environ.get("TINKER_GSM8K_RL_MAX_TOKENS", "1024"))
RL_LEARNING_RATE = 1e-6

EVAL_NUM_EXAMPLES = int(os.environ.get("TINKER_GSM8K_EVAL_NUM_EXAMPLES", "100"))
EVAL_MAX_TOKENS = int(os.environ.get("TINKER_GSM8K_EVAL_MAX_TOKENS", "1024"))
EVAL_TEMPERATURE = float(os.environ.get("TINKER_GSM8K_EVAL_TEMPERATURE", "0.0"))
EVAL_CONCURRENCY = int(os.environ.get("TINKER_GSM8K_EVAL_CONCURRENCY", "32"))


@dataclass(frozen=True)
class RequiredCheckpoint:
    state_path: str
    sampler_path: str


def _format_gsm8k_solution(answer: str) -> str:
    final_answer = math_env.extract_gsm8k_final_answer(answer)
    reasoning = answer.split("####", 1)[0].strip()
    if reasoning:
        return f"{reasoning}\n\n\\boxed{{{final_answer}}}"
    return f"\\boxed{{{final_answer}}}"


@chz.chz
class Gsm8kSFTBuilder(ChatDatasetBuilder):
    seed: int = 0

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        dataset = cast(datasets.DatasetDict, datasets.load_dataset("openai/gsm8k", name="main"))
        train_dataset = dataset["train"].shuffle(seed=self.seed)
        train_on_what = (
            TrainOnWhat(self.common_config.train_on_what)
            if self.common_config.train_on_what
            else TrainOnWhat.LAST_ASSISTANT_MESSAGE
        )

        def map_fn(row: dict) -> tinker.Datum:
            messages = math_env.MathEnv.standard_fewshot_prefix() + [
                {
                    "role": "user",
                    "content": row["question"] + math_env.MathEnv.question_suffix(),
                },
                {
                    "role": "assistant",
                    "content": _format_gsm8k_solution(row["answer"]),
                },
            ]
            return conversation_to_datum(
                messages,
                self.renderer,
                self.common_config.max_length,
                train_on_what,
            )

        return (
            SupervisedDatasetFromHFDataset(
                train_dataset,
                batch_size=self.common_config.batch_size,
                map_fn=map_fn,
            ),
            None,
        )


def _checkpoint_field(checkpoint: Any, field_name: str) -> str | None:
    return checkpoint.get(field_name) if isinstance(checkpoint, dict) else getattr(checkpoint, field_name)


def _get_required_checkpoint(log_path: str) -> RequiredCheckpoint:
    checkpoint = checkpoint_utils.get_last_checkpoint(log_path)
    if checkpoint is None:
        raise RuntimeError(f"No checkpoint found in {log_path}")
    state_path = _checkpoint_field(checkpoint, "state_path")
    sampler_path = _checkpoint_field(checkpoint, "sampler_path")
    if state_path is None or sampler_path is None:
        raise RuntimeError(f"Checkpoint in {log_path} must include both state_path and sampler_path: {checkpoint}")
    return RequiredCheckpoint(state_path=state_path, sampler_path=sampler_path)


def _build_eval_renderer(renderer_name: str, tokenizer_name_or_path: str) -> renderers.Renderer:
    tokenizer = get_tokenizer(tokenizer_name_or_path)
    return renderers.get_renderer(renderer_name, tokenizer=tokenizer)


async def _create_sampling_client(
    base_url: str,
    *,
    model_name: str,
    model_path: str | None = None,
) -> tinker.SamplingClient:
    service_client = tinker.ServiceClient(base_url=base_url)
    if model_path is not None:
        return await service_client.create_sampling_client_async(model_path=model_path)
    return await service_client.create_sampling_client_async(base_model=model_name)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _metrics_from_benchmark_result(
    stage: str,
    result: BenchmarkResult,
    *,
    model_path: str | None,
    save_dir: Path,
) -> dict[str, Any]:
    num_examples = result.num_examples
    error_rate = result.num_errors / num_examples if num_examples else 0.0
    truncated_rate = result.num_truncated / num_examples if num_examples else 0.0
    return {
        "stage": stage,
        "dataset": "openai/gsm8k",
        "config": "main",
        "split": "test",
        "num_examples": num_examples,
        "num_correct": result.num_correct,
        "pass_rate": result.score,
        "accuracy": result.score,
        "score_completed": result.score_completed,
        "num_errors": result.num_errors,
        "error_rate": error_rate,
        "num_truncated": result.num_truncated,
        "truncated_rate": truncated_rate,
        "time_seconds": result.time_seconds,
        "model_path": model_path,
        "benchmark_save_dir": str(save_dir),
        "test_metrics_path": str(save_dir / "gsm8k" / "result.json"),
        "test_trajectories_path": str(save_dir / "gsm8k" / "trajectories.jsonl"),
        "benchmark_metrics": result.metrics,
    }


def _print_eval_metrics(metrics: dict[str, Any]) -> None:
    print(
        f"[{metrics['stage']}] GSM8K pass_rate={metrics['pass_rate']:.3f} "
        f"({metrics['num_correct']}/{metrics['num_examples']}), "
        f"completed_pass_rate={metrics['score_completed']:.3f}, "
        f"errors={metrics['num_errors']}, "
        f"truncated={metrics['num_truncated']}"
    )


async def _evaluate_stage(
    *,
    stage: str,
    base_url: str,
    model_name: str,
    renderer: renderers.Renderer,
    model_path: str | None = None,
) -> dict[str, Any]:
    sampling_client = await _create_sampling_client(base_url, model_name=model_name, model_path=model_path)
    save_dir = EVAL_LOG_PATH / stage
    result = await run_benchmark(
        "gsm8k",
        sampling_client,
        renderer,
        BenchmarkConfig(
            max_examples=EVAL_NUM_EXAMPLES,
            concurrency=EVAL_CONCURRENCY,
            max_tokens=EVAL_MAX_TOKENS,
            temperature=EVAL_TEMPERATURE,
            save_dir=str(save_dir),
        ),
    )

    metrics = _metrics_from_benchmark_result(stage, result, model_path=model_path, save_dir=save_dir)
    _print_eval_metrics(metrics)
    return {"metrics": metrics}


def _write_experiment_summary(stage_results: Sequence[dict[str, Any]], model_name: str) -> None:
    summary = {
        "dataset": "openai/gsm8k",
        "train_split": "train",
        "test_split": "test",
        "model_name": model_name,
        "sft_max_steps": SFT_MAX_STEPS,
        "rl_max_steps": RL_MAX_STEPS,
        "eval_num_examples": EVAL_NUM_EXAMPLES,
        "eval_temperature": EVAL_TEMPERATURE,
        "stages": [result["metrics"] for result in stage_results],
    }
    _write_json(EVAL_LOG_PATH / "eval_summary.json", summary)


def _print_final_eval_summary(stage_results: Sequence[dict[str, Any]]) -> None:
    print("\nFinal GSM8K eval summary:")
    for result in stage_results:
        _print_eval_metrics(result["metrics"])


async def run_math_sft_rl_gsm8k_test(base_url: str, model_name: str, tokenizer_name_or_path: str | None = None):
    tokenizer_name_or_path = tokenizer_name_or_path or model_name
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    eval_renderer = _build_eval_renderer(renderer_name, tokenizer_name_or_path)

    cli_utils.check_log_dir(str(EVAL_LOG_PATH), behavior_if_exists="delete")
    base_eval = await _evaluate_stage(
        stage="base",
        base_url=base_url,
        model_name=model_name,
        renderer=eval_renderer,
    )

    sft_common = ChatDatasetBuilderCommonConfig(
        model_name_for_tokenizer=tokenizer_name_or_path,
        renderer_name=renderer_name,
        max_length=SFT_MAX_LENGTH,
        batch_size=SFT_BATCH_SIZE,
        train_on_what=TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )
    sft_config = sft_train.Config(
        log_path=SFT_LOG_PATH,
        model_name=model_name,
        renderer_name=renderer_name,
        dataset_builder=Gsm8kSFTBuilder(common_config=sft_common),
        learning_rate=SFT_LEARNING_RATE,
        num_epochs=1,
        max_steps=SFT_MAX_STEPS,
        eval_every=0,
        save_every=0,
        base_url=base_url,
        wandb_project=WANDB_PROJECT,
        wandb_name="sft-stage",
    )

    cli_utils.check_log_dir(sft_config.log_path, behavior_if_exists="delete")
    await sft_train.main(sft_config)
    sft_checkpoint = _get_required_checkpoint(sft_config.log_path)
    after_sft_eval = await _evaluate_stage(
        stage="after_sft",
        base_url=base_url,
        model_name=model_name,
        renderer=eval_renderer,
        model_path=sft_checkpoint.sampler_path,
    )

    rl_renderer_name = await checkpoint_utils.resolve_renderer_name_from_checkpoint_or_default_async(
        model_name=model_name,
        explicit_renderer_name=renderer_name,
        load_checkpoint_path=sft_checkpoint.state_path,
        base_url=base_url,
    )
    rl_warmup_dataset_builder = math_env.get_math_dataset_builder(
        dataset_name=DATASET_NAME,
        batch_size=1,
        model_name_for_tokenizer=tokenizer_name_or_path,
        renderer_name=rl_renderer_name,
        group_size=RL_GROUP_SIZE,
        seed=0,
    )
    rl_warmup_config = rl_train.Config(
        model_name=model_name,
        renderer_name=rl_renderer_name,
        dataset_builder=rl_warmup_dataset_builder,
        max_steps=1,
        learning_rate=RL_LEARNING_RATE,
        max_tokens=RL_MAX_TOKENS,
        temperature=0.7,
        kl_penalty_coef=0.0,
        wandb_project=WANDB_PROJECT,
        wandb_name="rl-1-step",
        log_path=RL_LOG_PATH,
        eval_every=0,
        save_every=0,
        compute_post_kl=False,
        base_url=base_url,
        load_checkpoint_path=sft_checkpoint.state_path,
        loss_fn="importance_sampling",
        loss_fn_config=None,
        async_config=None,
        stream_minibatch_config=None,
    )

    cli_utils.check_log_dir(rl_warmup_config.log_path, behavior_if_exists="delete")
    await rl_train.main(rl_warmup_config)
    rl_warmup_checkpoint = _get_required_checkpoint(rl_warmup_config.log_path)
    after_rl_1_eval = await _evaluate_stage(
        stage="after_rl_1",
        base_url=base_url,
        model_name=model_name,
        renderer=eval_renderer,
        model_path=rl_warmup_checkpoint.sampler_path,
    )

    if RL_MAX_STEPS > 1:
        rl_dataset_builder = math_env.get_math_dataset_builder(
            dataset_name=DATASET_NAME,
            batch_size=RL_GROUPS_PER_BATCH,
            model_name_for_tokenizer=tokenizer_name_or_path,
            renderer_name=rl_renderer_name,
            group_size=RL_GROUP_SIZE,
            seed=0,
        )
        rl_config = rl_train.Config(
            model_name=model_name,
            renderer_name=rl_renderer_name,
            dataset_builder=rl_dataset_builder,
            max_steps=RL_MAX_STEPS,
            learning_rate=RL_LEARNING_RATE,
            max_tokens=RL_MAX_TOKENS,
            temperature=0.7,
            kl_penalty_coef=0.01,
            kl_reference_config=rl_train.KLReferenceConfig(
                base_model=model_name,
                load_checkpoint_path=sft_checkpoint.state_path,
            ),
            wandb_project=WANDB_PROJECT,
            wandb_name="rl-stage",
            log_path=RL_LOG_PATH,
            eval_every=0,
            save_every=0,
            compute_post_kl=False,
            base_url=base_url,
            load_checkpoint_path=sft_checkpoint.state_path,
            loss_fn="importance_sampling",
            loss_fn_config=None,
            async_config=None,
            stream_minibatch_config=None,
        )
        await rl_train.main(rl_config)
        rl_checkpoint = _get_required_checkpoint(rl_config.log_path)

        after_rl_eval = await _evaluate_stage(
            stage="after_rl",
            base_url=base_url,
            model_name=model_name,
            renderer=eval_renderer,
            model_path=rl_checkpoint.sampler_path,
        )
        stage_results = [base_eval, after_sft_eval, after_rl_1_eval, after_rl_eval]
    else:
        stage_results = [base_eval, after_sft_eval, after_rl_1_eval]

    _write_experiment_summary(stage_results, model_name)
    _print_final_eval_summary(stage_results)
