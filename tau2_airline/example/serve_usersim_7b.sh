#!/usr/bin/env bash
# ==============================================================================
# tau2 user-simulator server — Qwen2.5-7B-Instruct, OpenAI-compatible, port 18001
# Runs on GPU0; the policy trainer runs on GPU1 (dual-5090 split, mirrors the
# hand-written pipeline's usersim/policy separation).
#
# The tau2 UserSimulator talks to this endpoint via litellm as "openai/usersim"
# with api_base=http://127.0.0.1:18001/v1. Airline users have no tools, so a
# plain chat server suffices (tool-call parser left off).
#
# Usage:  bash serve_usersim_7b.sh    (Ctrl-C to stop)
# ==============================================================================
set -xeuo pipefail

# ROOT = workspace holding models/ and caches; VENV optional (empty = current env)
ROOT=${ROOT:-$PWD}
VENV=${VENV:-}
MODEL=${USERSIM_MODEL:-$ROOT/models/qwen25-7b-instruct}
PORT=${USERSIM_PORT:-18001}
GPU=${USERSIM_GPU:-0}

export CUDA_VISIBLE_DEVICES=$GPU
export VLLM_CACHE_ROOT=$ROOT/.vllmcache
export TRITON_CACHE_DIR=$ROOT/.triton
export HF_HOME=$ROOT/.hfhome
export VLLM_USE_V1=1

[[ -n "$VENV" ]] && source "$VENV/bin/activate"

# Inference-only 7B fits comfortably on one 5090 (~16GB weights + kv-cache).
python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name usersim \
    --port "$PORT" \
    --host 127.0.0.1 \
    --gpu-memory-utilization ${USERSIM_UTIL:-0.80} \
    --max-model-len 8192 \
    --enforce-eager \
    --disable-log-requests \
    2>&1 | tee "$ROOT/usersim_server.log"
