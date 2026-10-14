"""Ray/vLLM trainer for the RandOpt recipe."""

import gc
import json
import os
import time
from collections import Counter
from datetime import datetime
from typing import Any

import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf
from ray.util.placement_group import placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from tqdm import tqdm
from vllm import LLM, SamplingParams
from vllm.utils import get_ip, get_open_port

from verl.utils.tracking import Tracking


class RandOptLLM(LLM):
    """vLLM wrapper that keeps CUDA visibility under Ray control."""

    def __init__(self, *args, **kwargs):
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        super().__init__(*args, **kwargs)


class RandOptRayTrainer:
    """Zeroth-order RandOpt trainer using parallel vLLM engines."""

    def __init__(
        self,
        config: DictConfig,
        tokenizer,
        reward_fn,
        train_data: list[dict[str, Any]],
        eval_data: list[dict[str, Any]],
        prompt_processor=None,
        vote_answer_fn=None,
        vote_correct_fn=None,
    ):
        self.config = config
        self.randopt_config = config.randopt
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.train_data = train_data
        self.eval_data = eval_data
        self.prompt_processor = prompt_processor
        self.vote_answer_fn = vote_answer_fn
        self.vote_correct_fn = vote_correct_fn
        self.engines = []
        self.placement_groups = []
        self._debug_samples_printed = 0

        seed = self.randopt_config.get("global_seed")
        if seed is not None:
            self._set_global_seed(int(seed))

    def _set_global_seed(self, seed: int) -> None:
        import random

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    def init_workers(self, model_path: str) -> None:
        self._launch_engines(model_path)
        self._init_inter_engine_group()

    def _launch_engines(self, model_path: str) -> None:
        num_engines = int(self.randopt_config.num_engines)
        tensor_parallel_size = int(self.randopt_config.get("tensor_parallel_size", 1))
        precision = self.randopt_config.get("precision", "bfloat16")
        worker_extension_cls = self.randopt_config.get(
            "worker_extension_cls", "recipe.randopt.worker_extension.WorkerExtension"
        )

        bundles = [{"GPU": 1, "CPU": 0} for _ in range(tensor_parallel_size)]
        placement_groups = [placement_group(bundles, lifetime="detached") for _ in range(num_engines)]
        ray.get([group.ready() for group in placement_groups])

        strategies = [
            PlacementGroupSchedulingStrategy(
                placement_group=group,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=0,
            )
            for group in placement_groups
        ]

        self.engines = [
            ray.remote(num_cpus=0, num_gpus=0, scheduling_strategy=strategy)(RandOptLLM).remote(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                distributed_executor_backend="ray",
                worker_extension_cls=worker_extension_cls,
                dtype=precision,
                enable_prefix_caching=self.randopt_config.get("enable_prefix_caching", False),
                enforce_eager=self.randopt_config.get("enforce_eager", True),
                gpu_memory_utilization=self.randopt_config.get("gpu_memory_utilization", 0.85),
                disable_log_stats=True,
            )
            for strategy in strategies
        ]
        self.placement_groups = placement_groups

    def _init_inter_engine_group(self) -> None:
        master_address = get_ip()
        master_port = get_open_port()
        ray.get(
            [
                engine.collective_rpc.remote(
                    "init_inter_engine_group",
                    args=(master_address, master_port, rank, len(self.engines)),
                )
                for rank, engine in enumerate(self.engines)
            ]
        )

    def fit(self) -> None:
        logging_dir = self._create_logging_dir()
        logger = Tracking(
            project_name=self.config.trainer.get("project_name", "randopt"),
            experiment_name=self.config.trainer.get("experiment_name", "randopt-run"),
            default_backend=self.config.trainer.get("logger", ["console"]),
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        self._save_config(logging_dir)

        train_prompts = self._build_prompts(self.train_data)
        sigma_values = self._get_sigma_values()
        population_size = int(self.randopt_config.population_size)
        global_seed = int(self.randopt_config.get("global_seed", 42))
        top_k_values = self._get_top_k_values(population_size)

        progress = tqdm(range(1), desc="RandOpt")
        try:
            for iteration in progress:
                iteration_start = time.time()
                rng = np.random.default_rng(seed=global_seed + iteration)
                seeds = rng.choice(2**31, size=population_size, replace=False).tolist()
                sigmas = rng.choice(sigma_values, size=population_size).tolist()
                perturbations = [(int(seed), float(sigma)) for seed, sigma in zip(seeds, sigmas, strict=True)]
                self._log_progress(
                    f"iteration {iteration + 1}: evaluating {population_size} perturbations "
                    f"across {len(self.engines)} engines"
                )
                perturbation_metrics = self._evaluate_population(perturbations, train_prompts, iteration)
                metrics = self._summarize_train_metrics(perturbation_metrics, iteration, time.time() - iteration_start)
                logger.log(data=metrics, step=iteration)
                progress.set_postfix({"reward": f"{metrics['train/reward_mean']:.4f}"}, refresh=False)
                self._log_progress(
                    f"iteration {iteration + 1}: train reward mean={metrics['train/reward_mean']:.4f}, "
                    f"max={metrics['train/reward_max']:.4f}, elapsed={metrics['train/iteration_time']:.1f}s"
                )

                top_perturbations = self._select_top_perturbations(perturbation_metrics, top_k_values)
                if self.eval_data:
                    self._log_progress(f"iteration {iteration + 1}: evaluating base model")
                    logger.log(data=self._evaluate_model(iteration), step=iteration)
                    self._log_progress(f"iteration {iteration + 1}: evaluating ensembles")
                    ensemble_metrics = self._evaluate_ensemble(top_perturbations, top_k_values, iteration)
                    logger.log(data=ensemble_metrics, step=iteration)

                perturbation_metrics.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            progress.close()
            if hasattr(logger, "finish"):
                logger.finish()
            self._cleanup()

    def _evaluate_population(
        self,
        perturbations: list[tuple[int, float]],
        train_prompts,
        iteration: int,
    ) -> dict[tuple[int, float], dict]:
        perturbation_metrics = {}
        num_engines = len(self.engines)
        total_batches = (len(perturbations) + num_engines - 1) // num_engines
        for start in range(0, len(perturbations), num_engines):
            batch_perturbations = perturbations[start : start + num_engines]
            batch_idx = start // num_engines + 1
            processed = min(start + len(batch_perturbations), len(perturbations))
            self._log_progress(
                f"iteration {iteration + 1}: population batch {batch_idx}/{total_batches} "
                f"({processed}/{len(perturbations)} perturbations) perturbing weights"
            )
            ray.get(
                [
                    self.engines[index].collective_rpc.remote(
                        "perturb_self_weights", args=(int(seed), float(sigma), False)
                    )
                    for index, (seed, sigma) in enumerate(batch_perturbations)
                ]
            )
            sampling_params = SamplingParams(
                temperature=self.randopt_config.get("temperature", 0.0),
                seed=int(self.randopt_config.get("global_seed", 42)) + iteration,
                max_tokens=int(self.randopt_config.get("max_tokens", 1024)),
            )
            self._log_progress(
                f"iteration {iteration + 1}: population batch {batch_idx}/{total_batches} generating "
                f"{len(train_prompts)} prompts on {len(batch_perturbations)} engines"
            )
            outputs_per_engine = ray.get(
                [
                    self.engines[index].generate.remote(train_prompts, sampling_params, use_tqdm=False)
                    for index, _ in enumerate(batch_perturbations)
                ]
            )
            self._log_progress(
                f"iteration {iteration + 1}: population batch {batch_idx}/{total_batches} restoring weights"
            )
            ray.get(
                [
                    self.engines[index].collective_rpc.remote(
                        "restore_self_weights", args=(int(seed), float(sigma), False)
                    )
                    for index, (seed, sigma) in enumerate(batch_perturbations)
                ]
            )
            self._log_progress(
                f"iteration {iteration + 1}: population batch {batch_idx}/{total_batches} scoring outputs"
            )
            for index, perturbation in enumerate(batch_perturbations):
                perturbation_metrics[perturbation] = self._compute_metrics(outputs_per_engine[index], self.train_data)
            del outputs_per_engine
        return perturbation_metrics

    def _get_sigma_values(self) -> list[float]:
        sigma_list = self.randopt_config.get("sigma_list", None)
        if sigma_list is None:
            return [float(self.randopt_config.sigma)]
        if isinstance(sigma_list, str):
            return [float(value.strip()) for value in sigma_list.split(",")]
        return [float(value) for value in sigma_list]

    def _get_top_k_values(self, population_size: int) -> list[int]:
        top_k_values = self.randopt_config.get("top_k_values", None)
        if top_k_values is not None and len(top_k_values) > 0:
            values = [int(value) for value in top_k_values]
        else:
            ratios = self.randopt_config.get("top_k_ratios", [0.02, 0.1])
            if isinstance(ratios, str):
                ratios = [float(value.strip()) for value in ratios.split(",")]
            values = [max(1, int(float(ratio) * population_size)) for ratio in ratios]
        return sorted({min(population_size, value) for value in values})

    def _select_top_perturbations(
        self,
        perturbation_metrics: dict[tuple[int, float], dict],
        top_k_values: list[int],
    ) -> list[tuple[int, float]]:
        max_k = max(top_k_values)
        ranked = sorted(perturbation_metrics.items(), key=lambda item: item[1]["avg_reward"], reverse=True)
        print("\n[RandOpt] Top perturbations by train reward:")
        for rank, ((seed, sigma), metrics) in enumerate(ranked[: min(max_k, 10)], start=1):
            print(f"  {rank}. seed={seed}, sigma={sigma}, reward={metrics['avg_reward']:.4f}")
        return [(int(seed), float(sigma)) for (seed, sigma), _ in ranked[:max_k]]

    def _evaluate_ensemble(
        self,
        top_perturbations: list[tuple[int, float]],
        top_k_values: list[int],
        step: int,
    ) -> dict[str, float]:
        if self.vote_answer_fn is None or self.vote_correct_fn is None:
            return {"ensemble/skipped": 1.0, "training/global_step": step}

        eval_prompts = self._build_prompts(self.eval_data)
        sampling_params = SamplingParams(
            temperature=0.0,
            seed=int(self.randopt_config.get("global_seed", 42)),
            max_tokens=int(self.randopt_config.get("max_tokens", 1024)),
        )

        max_k = min(max(top_k_values), len(top_perturbations))
        all_answers: list[list[str] | None] = [None] * max_k
        num_engines = len(self.engines)
        total_batches = (max_k + num_engines - 1) // num_engines

        for start in range(0, max_k, num_engines):
            batch_perturbations = top_perturbations[start : start + num_engines]
            batch_idx = start // num_engines + 1
            self._log_progress(
                f"ensemble batch {batch_idx}/{total_batches}: perturbing top perturbations "
                f"({start + len(batch_perturbations)}/{max_k})"
            )
            ray.get(
                [
                    self.engines[index].collective_rpc.remote(
                        "perturb_self_weights", args=(int(seed), float(sigma), False)
                    )
                    for index, (seed, sigma) in enumerate(batch_perturbations)
                ]
            )
            self._log_progress(
                f"ensemble batch {batch_idx}/{total_batches}: generating {len(eval_prompts)} eval prompts"
            )
            batch_outputs = ray.get(
                [
                    self.engines[index].generate.remote(eval_prompts, sampling_params, use_tqdm=False)
                    for index, _ in enumerate(batch_perturbations)
                ]
            )
            self._log_progress(f"ensemble batch {batch_idx}/{total_batches}: restoring weights")
            ray.get(
                [
                    self.engines[index].collective_rpc.remote(
                        "restore_self_weights", args=(int(seed), float(sigma), False)
                    )
                    for index, (seed, sigma) in enumerate(batch_perturbations)
                ]
            )

            for local_idx, global_idx in enumerate(range(start, start + len(batch_perturbations))):
                answers_for_model = []
                for output, task_data in zip(batch_outputs[local_idx], self.eval_data, strict=False):
                    answer, is_valid, _ = self.vote_answer_fn(output.outputs[0].text, task_data)
                    answers_for_model.append(answer if is_valid else "")
                all_answers[global_idx] = answers_for_model
            del batch_outputs

        metrics = {"training/global_step": step}
        previous_k = None
        previous_final_answers = None
        previous_correct_flags = None
        for k_value in top_k_values:
            if k_value > max_k:
                continue
            correct = 0
            final_answers = []
            correct_flags = []
            valid_vote_counts = []
            for sample_idx, task_data in enumerate(self.eval_data):
                answers = [
                    model_answers[sample_idx]
                    for model_answers in all_answers[:k_value]
                    if model_answers is not None and model_answers[sample_idx]
                ]
                valid_vote_counts.append(len(answers))
                final_answer = ""
                if answers:
                    final_answer = Counter(answers).most_common(1)[0][0]
                    is_correct = self.vote_correct_fn(final_answer, task_data)
                    if is_correct:
                        correct += 1
                else:
                    is_correct = False
                final_answers.append(final_answer)
                correct_flags.append(is_correct)
            accuracy = correct / len(self.eval_data) * 100 if self.eval_data else 0.0
            metrics[f"ensemble/top_{k_value}_accuracy"] = accuracy
            metrics[f"ensemble/top_{k_value}_correct"] = float(correct)
            metrics[f"ensemble/top_{k_value}_valid_vote_mean"] = float(np.mean(valid_vote_counts))
            metrics[f"ensemble/top_{k_value}_valid_vote_min"] = float(np.min(valid_vote_counts))
            metrics[f"ensemble/top_{k_value}_coverage"] = float(
                np.mean([count > 0 for count in valid_vote_counts]) * 100.0
            )
            if previous_final_answers is not None and previous_correct_flags is not None:
                changed = sum(
                    before != after for before, after in zip(previous_final_answers, final_answers, strict=True)
                )
                correctness_changed = sum(
                    before != after for before, after in zip(previous_correct_flags, correct_flags, strict=True)
                )
                metrics[f"ensemble/top_{previous_k}_to_top_{k_value}_prediction_changed"] = float(changed)
                metrics[f"ensemble/top_{previous_k}_to_top_{k_value}_correctness_changed"] = float(correctness_changed)
                print(
                    f"[RandOpt] ensemble top-{previous_k}->top-{k_value}: "
                    f"prediction changed on {changed}/{len(self.eval_data)}, "
                    f"correctness changed on {correctness_changed}/{len(self.eval_data)}"
                )
            print(f"[RandOpt] ensemble top-{k_value}: {accuracy:.2f}% ({correct}/{len(self.eval_data)})")
            previous_k = k_value
            previous_final_answers = final_answers
            previous_correct_flags = correct_flags
        return metrics

    def _log_progress(self, message: str) -> None:
        print(f"[RandOpt progress] {message}", flush=True)

    def _compute_metrics(
        self,
        outputs,
        task_datas: list[dict[str, Any]],
        debug_prefix: str | None = None,
    ) -> dict[str, Any]:
        rewards = []
        format_rewards = []
        answer_rewards = []
        for sample_idx, (output, task_data) in enumerate(zip(outputs, task_datas, strict=False)):
            response = output.outputs[0].text
            result = self.reward_fn(response, task_data)
            if isinstance(result, dict):
                rewards.append(float(result.get("reward", 0.0)))
                reward_info = result.get("reward_info", {})
                format_rewards.append(float(reward_info.get("format_reward", 0.0)))
                answer_rewards.append(float(reward_info.get("answer_reward", 0.0)))
            else:
                rewards.append(float(result))
            debug_max_samples = int(self.randopt_config.get("debug_max_samples", 4))
            if debug_prefix and self._debug_samples_printed < debug_max_samples:
                print(f"\n[RandOpt debug] {debug_prefix} sample")
                print(f"sample_idx={sample_idx}")
                print(f"task_data={task_data}")
                print(f"response={response[:1000]}")
                print(f"reward={result}\n")
                self._debug_samples_printed += 1
        return {
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "avg_format": float(np.mean(format_rewards)) if format_rewards else 0.0,
            "avg_answer": float(np.mean(answer_rewards)) if answer_rewards else 0.0,
            "accuracy": float(np.mean([reward > 0 for reward in answer_rewards]) * 100.0) if answer_rewards else 0.0,
        }

    def _evaluate_model(self, step: int) -> dict[str, float]:
        if not self.eval_data:
            return {}
        eval_prompts = self._build_prompts(self.eval_data)
        sampling_params = SamplingParams(
            temperature=0.0,
            seed=int(self.randopt_config.get("global_seed", 42)),
            max_tokens=int(self.randopt_config.get("max_tokens", 1024)),
        )
        outputs = ray.get(self.engines[0].generate.remote(eval_prompts, sampling_params, use_tqdm=False))
        metrics = self._compute_metrics(
            outputs,
            self.eval_data,
            debug_prefix=f"eval step {step}" if self.randopt_config.get("debug_print_samples", False) else None,
        )
        return {
            "eval/avg_reward": metrics["avg_reward"],
            "eval/format_reward": metrics["avg_format"],
            "eval/answer_reward": metrics["avg_answer"],
            "eval/accuracy": metrics["accuracy"],
            "training/global_step": step,
        }

    def _build_prompts(self, data: list[dict[str, Any]]):
        if self.prompt_processor:
            return [self.prompt_processor(row, self.tokenizer) for row in data]
        return [row.get("prompt", row.get("context")) for row in data]

    def _summarize_train_metrics(
        self, perturbation_metrics: dict[tuple[int, float], dict], iteration: int, elapsed: float
    ) -> dict[str, float]:
        rewards = [metrics["avg_reward"] for metrics in perturbation_metrics.values()]
        return {
            "train/reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "train/reward_std": float(np.std(rewards)) if rewards else 0.0,
            "train/reward_min": float(np.min(rewards)) if rewards else 0.0,
            "train/reward_max": float(np.max(rewards)) if rewards else 0.0,
            "train/iteration_time": elapsed,
            "training/global_step": iteration,
        }

    def _create_logging_dir(self) -> str:
        base_dir = self.config.trainer.get("default_local_dir", "/tmp/verl/randopt_checkpoints")
        logging_dir = os.path.join(base_dir, f"randopt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(logging_dir, exist_ok=True)
        return logging_dir

    def _save_config(self, logging_dir: str) -> None:
        with open(os.path.join(logging_dir, "config.json"), "w") as f:
            json.dump(OmegaConf.to_container(self.config, resolve=True), f, indent=2)

    def _cleanup(self) -> None:
        for engine in self.engines:
            try:
                ray.kill(engine)
            except Exception:
                pass
        for group in self.placement_groups:
            try:
                remove_placement_group(group)
            except Exception:
                pass
