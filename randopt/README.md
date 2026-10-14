# RandOpt

## Required `verl` Version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repository, install mode, and copy-pastable `pip` instruction.

## Overview

RandOpt is a LLM post-training algorithm introduced in [**Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights**](https://arxiv.org/abs/2603.12228) (ICML 2026 Spotlight). It samples Gaussian perturbations around a pretrained model, evaluates the perturbed models in parallel with vLLM, selects the top-performing perturbations, and evaluates them with majority-vote ensembling.

Project page: <https://thickets.mit.edu>

## Installation

Install the pinned `verl` version and runtime dependencies:

```bash
pip install verl==0.7.1
pip install vllm ray pandas pyarrow tqdm
```

If running from a `verl` checkout, initialize the recipe submodule first:

```bash
git submodule update --init --recursive recipe
```

## Run

Run the Countdown example from the `verl` repository root. The following command expects prepared Countdown parquet files:

```bash
python3 -m recipe.randopt.main_randopt \
    model.path=Qwen/Qwen2.5-3B-Instruct \
    data.task_type=countdown \
    data.train_files=data/countdown/train.parquet \
    data.val_files=data/countdown/test.parquet \
    randopt.worker_extension_cls=recipe.randopt.worker_extension.WorkerExtension
```

For a quick test with generated toy Countdown data:

```bash
python3 -m recipe.randopt.run_countdown_example
```

Common overrides:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m recipe.randopt.main_randopt \
    model.path=Qwen/Qwen2.5-7B-Instruct \
    data.task_type=countdown \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/test.parquet \
    randopt.num_engines=4 \
    randopt.tensor_parallel_size=1 \
    "randopt.sigma_list=[0.0005,0.001,0.002]" \
    "randopt.top_k_ratios=[0.02,0.1]" \
    randopt.worker_extension_cls=recipe.randopt.worker_extension.WorkerExtension \
    trainer.n_gpus_per_node=4
```


If you are running from the standalone `verl-recipe` repository root instead of from `verl`, drop the `recipe.` prefix:

```bash
python3 -m randopt.run_countdown_example
python3 -m randopt.main_randopt ...
```

## Test Result

The following test was run on six H200 GPUs with `Qwen/Qwen2.5-1.5B-Instruct`, `population_size=500`, 20 toy Countdown training examples, 200 validation examples, and one RandOpt iteration:

```text
train/reward_mean: 0.1045
train/reward_std: 0.0664
train/reward_min: 0.0130
train/reward_max: 0.3820
ensemble/top_10_accuracy: 42.0%
ensemble/top_50_accuracy: 64.0%
```

With `top_k_ratios=[0.02,0.1]`, `population_size=500` evaluates top-10 and top-50 majority-vote ensembles. The base model may score poorly on the toy Countdown examples, but the ensemble metrics should still be emitted at the end of a successful smoke test run.

## Custom Tasks

For a custom dataset, set `data.task_type=custom` and provide a reward function and optional prompt processor:

```bash
python3 -m recipe.randopt.main_randopt \
    data.task_type=custom \
    data.train_files=/path/to/train.parquet \
    data.val_files=/path/to/test.parquet \
    data.reward_fn_path=/path/to/reward.py \
    data.reward_fn_name=my_reward_fn \
    data.prompt_processor_path=/path/to/prompts.py \
    data.prompt_processor_name=my_prompt_processor
```

The reward function should accept `(response: str, task_data: dict)` and return either a float or a dict with a `reward` field.

## Citation

```bibtex
@misc{gan2026neuralthickets,
      title={Neural Thickets: Diverse Task Experts Are Dense Around Pretrained Weights},
      author={Yulu Gan and Phillip Isola},
      year={2026},
      eprint={2603.12228},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2603.12228},
}
```
