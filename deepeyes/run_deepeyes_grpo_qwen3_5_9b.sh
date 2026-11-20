#!/usr/bin/env bash
# DeepEyes GRPO for Qwen3.5-9B on Ascend NPU.
# Uses a smaller batch than 4B to fit 64 GB HBM.
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-9B}
export GEN_TP=${GEN_TP:-2}
export ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.55}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}

exec bash "${SCRIPT_DIR}/run_deepeyes_grpo_qwen3_5.sh" "$@"
