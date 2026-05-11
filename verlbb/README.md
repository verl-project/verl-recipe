# VerlBB

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repository, install mode, and copy-pastable `pip` / `git` instructions.

## Overview

VerlBB is a recipe for running verl SFT and GRPO with Bumblebee as an external training engine. The recipe keeps the integration thin:

- verl still owns trainers, datasets, rollout orchestration, and algorithm logic.
- Bumblebee owns model construction, parallelism, optimizer/offload, checkpoint, and weight export.
- The adapter package under [`verlbb/`](verlbb/) only registers `strategy=bumblebee` and translates verl batches into Bumblebee's THD/no-padding runtime contract.

The included scripts also keep `BACKEND=megatron` as a reference path so the same data/model settings can be compared against verl's Megatron backend.

## Prerequisites

Install or expose these packages before running the scripts:

- `verl` at the version described in [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt).
- Bumblebee runtime, either installed in the active environment or exposed with `BUMBLEBEE_ROOT=/path/to/bumblebee`.
- Megatron-LM and mbridge when using `BACKEND=megatron`, exposed with `MEGATRON_ROOT` and `MBRIDGE_ROOT` if they are not installed.

Optional source-tree overrides:

```bash
export VERL_ROOT=/path/to/verl
export BUMBLEBEE_ROOT=/path/to/bumblebee
export MEGATRON_ROOT=/path/to/Megatron-LM
export MBRIDGE_ROOT=/path/to/mbridge
```

The scripts do not launch through a cluster scheduler. For multi-node runs, start the same script per node with the standard `torchrun` or Ray environment variables for your environment.

## SFT

The SFT script expects a messages parquet input and uses verl's native SFT trainer.

```bash
export MODEL_PATH=/path/to/qwen3-moe-hf
export TRAIN_FILES=/path/to/train.parquet
export VAL_FILES=/path/to/val.parquet

BACKEND=bumblebee bash verlbb/scripts/run_qwen3moe_sft.sh
```

Common knobs:

- `BACKEND=bumblebee|megatron`
- `TP_SIZE`, `PP_SIZE`, `VPP_SIZE`, `CP_SIZE`, `EP_SIZE`, `ETP_SIZE`
- `TOTAL_STEPS`, `TRAIN_BATCH_SIZE`, `MICRO_BATCH_SIZE`, `MAX_TOKENS_PER_GPU`
- `PARAM_OFFLOAD`, `OPTIMIZER_OFFLOAD`, `GRAD_OFFLOAD`
- `ATTENTION_BACKEND=flash`
- `DRY_RUN=1` to print the resolved command without running it

## GRPO on GSM8K

The GRPO script expects GSM8K-style train/validation parquet files with a `prompt` field.

```bash
export MODEL_PATH=/path/to/qwen3-moe-hf
export TRAIN_FILE=/path/to/gsm8k/train.parquet
export VAL_FILE=/path/to/gsm8k/test.parquet

BACKEND=bumblebee bash verlbb/scripts/run_qwen3moe_gsm8k_grpo.sh
```

Common knobs:

- `BACKEND=bumblebee|megatron`
- `TP_SIZE`, `PP_SIZE`, `VPP_SIZE`, `CP_SIZE`, `EP_SIZE`, `ETP_SIZE`
- `TOTAL_STEPS`, `TRAIN_BATCH_SIZE`, `ROLLOUT_N`
- `ROLLOUT_GPU_MEMORY_UTILIZATION`, default `0.7`
- `ALL_OFFLOAD`, or the individual `PARAM_OFFLOAD`, `OPTIMIZER_OFFLOAD`, `GRAD_OFFLOAD`
- `USE_FUSED_KERNELS=True`
- `ATTENTION_BACKEND=flash`
- `DRY_RUN=1` to print the resolved command without running it

## Outputs

By default the scripts write logs, file-logger JSONL, command snapshots, and checkpoints under `verlbb/outputs/...`. Override `OUTPUT_ROOT`, `LOG_FILE`, `JSONL_FILE`, `CMD_FILE`, or `CKPT_DIR` to redirect artifacts.
