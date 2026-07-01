#!/bin/bash
# VeRL-backed training-only Tinker server entrypoint for Qwen3-1.7B SFT.
#
# This script can be run directly in an interactive job, or used as the
# entrypoint for a formal Merlin launch config.
#
# Optional env vars:
#   TINKER_SERVER_MODEL                  Model path override for the config.
#   TINKER_SERVER_NNODES                 Number of Ray worker nodes in the server config.
#   TINKER_SERVER_N_GPUS_PER_NODE        Number of GPUs per Ray worker node in the server config.
#   RAY_ADDRESS                          Ray address. Defaults to local.
# Platform-injected:
#   BYTED_RAY_SERVE_PROXY_TARGET_PORT    Server bind port
#   BYTED_RAY_SERVE_PROXY_PSM            PSM service name for client discovery

set -euo pipefail

SERVE_PORT="${BYTED_RAY_SERVE_PROXY_TARGET_PORT:-8000}"
MODEL_PATH="${TINKER_SERVER_MODEL:-/mnt/hdfs/mlsys/models/Qwen3-1.7B}"
RAY_ADDRESS="${RAY_ADDRESS:-local}"
SERVER_CONFIG="verl-recipes/src/verl_recipes/verl_tinker_server/config/quick_start/actor.yaml"

echo "=================================================="
echo "verl-recipes Tinker server SFT (Qwen3-1.7B)"
echo "  Serve port: ${SERVE_PORT}"
echo "  Config:     ${SERVER_CONFIG}"
echo "  Model:      ${MODEL_PATH}"
echo "  Ray:        ${RAY_ADDRESS}"
echo "  PSM:        ${BYTED_RAY_SERVE_PROXY_PSM:-<unset>}"
echo "=================================================="

export PYTHONPATH="verl-recipes/src${PYTHONPATH:+:${PYTHONPATH}}"
PY_EXEC=""
if [ -n "${BYTED_RAY_JOB_RUNTIME_PATH:-}" ] && [ -f "${BYTED_RAY_JOB_RUNTIME_PATH}" ]; then
    PY_EXEC=$(awk -F'"' '/^py_executable:/{print $2}' "${BYTED_RAY_JOB_RUNTIME_PATH}")
fi
echo "  Python:     ${PY_EXEC:-python}"

SERVER_ENV=(
    TINKER_SERVER_PORT="${SERVE_PORT}"
    RAY_ADDRESS="${RAY_ADDRESS}"
    PYTHONUNBUFFERED=1
    RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-1}"
)
if [ -n "${MODEL_PATH}" ]; then
    SERVER_ENV+=(TINKER_SERVER_MODEL="${MODEL_PATH}")
fi

if [ -n "${PY_EXEC}" ]; then
    read -r -a PY_EXEC_ARGS <<< "${PY_EXEC}"
    env "${SERVER_ENV[@]}" "${PY_EXEC_ARGS[@]}" -- python -m verl_recipes.verl_tinker_server.start \
        --config "${SERVER_CONFIG}" &
else
    env "${SERVER_ENV[@]}" python -m verl_recipes.verl_tinker_server.start \
        --config "${SERVER_CONFIG}" &
fi

SERVER_PID=$!
trap 'kill "${SERVER_PID}" 2>/dev/null || true' EXIT

wait "${SERVER_PID}"
