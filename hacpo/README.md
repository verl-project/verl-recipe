# HACPO

HACPO is the policy optimization algorithm introduced in *Heterogeneous Agent Collaborative Reinforcement Learning*. It enables heterogeneous language models to learn from a joint set of verified trajectories while remaining independent policies at inference time.

[![arXiv](https://img.shields.io/badge/arXiv-2603.02604-b31b1b.svg?style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/pdf/2603.02604)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Paper-FFD21E.svg?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/papers/2603.02604)
[![GitHub](https://img.shields.io/badge/GitHub-Original%20Code-181717.svg?style=flat-square&logo=github&logoColor=white)](https://github.com/zzx-peter/HACRL-code)

![Overview of HACPO](./figures/overview.png)

*Given the same prompts, heterogeneous policies contribute trajectories to a shared pool. HACPO uses capability- and distribution-aware corrections when updating each policy. Figure from the paper.*

## Overview

For every training step, the recipe:

1. sends one prompt batch to both policies and collects independently generated, verified responses;
2. stores response text, reward, source policy, and rollout-time sequence log-probability in a tokenizer-neutral trajectory pool;
3. builds a training batch for each learner from the complete trajectory set, retokenizing only responses produced by the other policy; and
4. updates both policies with the HACPO objective before proceeding to the next prompt batch.

The objective implements the four mechanisms introduced in the paper: Agent-Capability-Aware Advantage Estimation, the Model Capabilities Discrepancy Coefficient, Exponential Importance Sampling, and Stepwise Clipping. Self-generated trajectories use GSPO-style clipping, while cross-policy trajectories use the source sequence probability and the learner's token-level probabilities.

Each policy has its own tokenizer-aware reward workers, actor/reference worker group, async vLLM rollout servers, and checkpoint state. The two runtimes share one eight-GPU resource pool and are activated sequentially; inactive parameters, optimizer state, and rollout replicas are offloaded or put to sleep. This makes Qwen-to-Qwen and cross-tokenizer Qwen-to-Llama training use the same path.

The current release supports synchronous, text-only, single-turn, full-parameter mutual learning between two policies. The runtime interfaces are indexed by agent, while training with three or more heterogeneous agents remains future work.

## Components

| File | Responsibility |
| --- | --- |
| [`main_hacpo.py`](main_hacpo.py) | Hydra/Ray entry point that initializes the HACPO training runtime. |
| [`hacpo_config.py`](hacpo_config.py) and [`config/hacpo_trainer.yaml`](config/hacpo_trainer.yaml) | Agent configuration, HACPO hyperparameters, and runtime validation. |
| [`hacpo_ray_trainer.py`](hacpo_ray_trainer.py) | Coordinates the two policy runtimes, shared trajectories, updates, validation, and joint checkpoints. |
| [`hacpo_trajectory.py`](hacpo_trajectory.py) | Stores tokenizer-neutral trajectories and retokenizes cross-policy responses for each learner. |
| [`hacpo_workers.py`](hacpo_workers.py) | Connects HACPO batches and losses to verl's V1 actor and rollout workers. |
| [`hacpo_core_algos.py`](hacpo_core_algos.py) | Implements capability-aware advantages, importance ratios, clipping, and the HACPO policy loss. |
| [`math_reward.py`](math_reward.py) | Adapts the reference implementation's boxed exact-match reward to rollout and validation. |
| [`run_hacpo_math_8gpu.sh`](run_hacpo_math_8gpu.sh) and the model-pair launchers | Define the shared eight-GPU setup and the two paper-aligned model presets. |

## Requirements

This recipe is tested with `verl` commit `bf48903d93e4618531d3bbae96551a889007dd8b`. See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the pinned version and installation command.

From the parent `verl` checkout, install verl and the additional runtime dependency:

```bash
pip install -e .
pip install TransferQueue==0.1.8
```

Install the vLLM/FSDP stack supported by the pinned verl revision. The provided presets require eight NVIDIA GPUs and two local Hugging Face model directories.

The paper uses the 7.5k MATH training split and evaluates on seven datasets: MATH-500, MATH, GSM8K, AIME 2025 (`test16`), AMC 2023, Minerva Math, and OlympiadBench. Convert the required splits to verl's `RLHFDataset` parquet format before use.

## Training

Run from the parent `verl` checkout. `VAL_PATHS` accepts colon-separated parquet paths and normally contains MATH-500 during training.

```bash
export QWEN3_4B_MODEL=/models/Qwen3-4B-Base
export QWEN3_1P7B_MODEL=/models/Qwen3-1.7B-Base
export TRAIN_FILE=/data/math/train.parquet
export VAL_PATHS=/data/math500.parquet
export CHECKPOINT_DIR=/outputs/hacpo/checkpoints
export TENSORBOARD_DIR=/outputs/hacpo/tensorboard

bash recipe/hacpo/run_hacpo_qwen3_4b_qwen3_1p7b_8gpu.sh
```

For Qwen3-4B-Base and Llama-3.2-3B-Instruct, set `LLAMA3P2_3B_MODEL` and run:

```bash
bash recipe/hacpo/run_hacpo_qwen3_4b_llama3p2_3b_8gpu.sh
```

Both presets follow the paper training setup: batch size 128, eight rollouts per prompt, 4096 response tokens, no Qwen thinking, greedy MATH-500 validation every three steps, TensorBoard logging, and a final checkpoint.

Each `global_step_*` checkpoint stores both policies under `agents/<agent-id>/actor`, together with the dataloader state, capability history, and agent manifest required for a joint resume.
