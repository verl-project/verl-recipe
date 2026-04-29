#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run manual_dynamo_smoke.sh inside the dynamo vllm-runtime container,
# on a Slurm + enroot/pyxis cluster (EOS-style).
#
# Defaults pulled from recipe/dapo/slurm_qwen30b.sh: account=general_sa,
# partition=batch. Override via env if needed.

set -euo pipefail

ACCOUNT="${ACCOUNT:-general_sa}"
PARTITION="${PARTITION:-batch}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
# EOS batch partition allocates whole nodes; --gres is not supported.
# We constrain to GPU 0 inside the container via CUDA_VISIBLE_DEVICES.
CONTAINER="${CONTAINER:-/lustre/fsw/general_sa/sopyang/images/dynamo_vllm_1.0.0.sqsh}"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SMOKE_SH="${SCRIPT_DIR}/manual_dynamo_smoke.sh"
[[ -f "$SMOKE_SH" ]] || { echo "missing $SMOKE_SH" >&2; exit 1; }
[[ -f "$CONTAINER" ]] || { echo "missing container $CONTAINER (pull it with: enroot import -o $CONTAINER docker://nvcr.io#nvidia/ai-dynamo/vllm-runtime:1.0.0)" >&2; exit 1; }

LOG_DIR="${LOG_DIR:-/lustre/fsw/general_sa/sopyang/dynamo_smoke_logs/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "[wrapper] logs -> $LOG_DIR"

HOME_DIR="/lustre/fsw/general_sa/sopyang"
MOUNTS="${HOME_DIR}:${HOME_DIR}"

# Pass log dir into the inner script via env so we capture frontend/worker logs
# on the shared lustre path even after the container exits.
exec srun \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIME_LIMIT" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --job-name="dynamo-smoke" \
    --output="$LOG_DIR/srun.out" \
    --error="$LOG_DIR/srun.err" \
    --container-image="$CONTAINER" \
    --container-mounts="$MOUNTS" \
    --container-workdir="$HOME_DIR" \
    bash -c "LOG_DIR=$LOG_DIR HF_HOME=$HOME_DIR/.hf_cache bash $SMOKE_SH"
