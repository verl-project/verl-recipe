#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run m1_registration_smoke.py inside a verl-equipped container on a
# Slurm compute node. CPU-only; no GPU is allocated.
#
# Defaults assume the dynamo container; override CONTAINER if you want to
# verify against a different verl env (e.g. vllm017_latest.sqsh).

set -euo pipefail

ACCOUNT="${ACCOUNT:-general_sa}"
PARTITION="${PARTITION:-batch}"
TIME_LIMIT="${TIME_LIMIT:-00:15:00}"
CONTAINER="${CONTAINER:-/lustre/fsw/general_sa/sopyang/images/dynamo_vllm_1.0.0.sqsh}"
VERL_REPO="${VERL_REPO:-/lustre/fsw/general_sa/sopyang/rl/verl_0211/verl}"
HOME_DIR="/lustre/fsw/general_sa/sopyang"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SMOKE_PY="${SCRIPT_DIR}/m1_registration_smoke.py"
[[ -f "$SMOKE_PY" ]] || { echo "missing $SMOKE_PY" >&2; exit 1; }
[[ -f "$CONTAINER" ]] || { echo "missing container $CONTAINER" >&2; exit 1; }

LOG_DIR="${LOG_DIR:-${HOME_DIR}/dynamo_smoke_logs/m1_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "[wrapper] logs -> $LOG_DIR"

# pip install verl in --no-deps mode (deps already satisfied by the container's
# existing torch/ray/vllm stack); then run the smoke. Any pip churn is captured
# in the log so first-run install latency is visible but not fatal.
INNER_CMD="set -euo pipefail; \
    pip install --no-deps --quiet -e ${VERL_REPO} 2>&1 | tail -n 5; \
    export PYTHONPATH=${VERL_REPO}:\${PYTHONPATH:-}; \
    python ${SMOKE_PY}"

exec srun \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIME_LIMIT" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --job-name="general_sa-dynamo.m1_smoke" \
    --output="$LOG_DIR/srun.out" \
    --error="$LOG_DIR/srun.err" \
    --container-image="$CONTAINER" \
    --container-mounts="${HOME_DIR}:${HOME_DIR}" \
    --container-workdir="$VERL_REPO" \
    bash -c "$INNER_CMD"
