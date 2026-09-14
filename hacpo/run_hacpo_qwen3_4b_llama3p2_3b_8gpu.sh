#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

: "${QWEN3_4B_MODEL:?Set QWEN3_4B_MODEL to Qwen3-4B-Base}"
: "${LLAMA3P2_3B_MODEL:?Set LLAMA3P2_3B_MODEL to Llama-3.2-3B-Instruct}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the HACPO math training parquet}"
: "${VAL_PATHS:?Set VAL_PATHS to colon-separated validation parquets}"

export AGENT_A_ID=${AGENT_A_ID:-qwen3_4b_base}
export AGENT_A_MODEL=${AGENT_A_MODEL:-${QWEN3_4B_MODEL}}
export AGENT_B_ID=${AGENT_B_ID:-llama3p2_3b_instruct}
export AGENT_B_MODEL=${AGENT_B_MODEL:-${LLAMA3P2_3B_MODEL}}
export PROJECT_NAME=${PROJECT_NAME:-hacpo_qwen3_4b_llama3p2_3b}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-hacpo_qwen3_4b_llama3p2_3b_seed42}

exec bash "${SCRIPT_DIR}/run_hacpo_math_8gpu.sh" "$@"
