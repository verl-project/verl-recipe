#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${QWEN3_4B_MODEL:?Set QWEN3_4B_MODEL to Qwen3-4B-Base}"
: "${QWEN3_1P7B_MODEL:?Set QWEN3_1P7B_MODEL to Qwen3-1.7B-Base}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the HACPO math training parquet}"
: "${VAL_PATHS:?Set VAL_PATHS to colon-separated validation parquets}"

export AGENT_A_ID=${AGENT_A_ID:-qwen3_4b_base}
export AGENT_A_MODEL=${AGENT_A_MODEL:-${QWEN3_4B_MODEL}}
export AGENT_B_ID=${AGENT_B_ID:-qwen3_1p7b_base}
export AGENT_B_MODEL=${AGENT_B_MODEL:-${QWEN3_1P7B_MODEL}}
export PROJECT_NAME=${PROJECT_NAME:-hacpo_qwen3_4b_qwen3_1p7b}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-hacpo_qwen3_4b_qwen3_1p7b_seed42}

exec bash "${SCRIPT_DIR}/run_hacpo_math_8gpu.sh" "$@"
