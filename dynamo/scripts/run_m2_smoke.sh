#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run m2_contract_smoke.py inside a verl-equipped container.
# CPU-only; no GPU is needed (no Ray actor instantiation).

set -euo pipefail

ACCOUNT="${ACCOUNT:-general_sa}"
PARTITION="${PARTITION:-batch}"
TIME_LIMIT="${TIME_LIMIT:-00:15:00}"
CONTAINER="${CONTAINER:-/lustre/fsw/general_sa/sopyang/images/vllm017_latest.sqsh}"
VERL_REPO="${VERL_REPO:-/lustre/fsw/general_sa/sopyang/rl/verl_0211/verl}"
HOME_DIR="/lustre/fsw/general_sa/sopyang"

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
SMOKE_PY="${SCRIPT_DIR}/m2_contract_smoke.py"
[[ -f "$SMOKE_PY" ]] || { echo "missing $SMOKE_PY" >&2; exit 1; }
[[ -f "$CONTAINER" ]] || { echo "missing container $CONTAINER" >&2; exit 1; }

LOG_DIR="${LOG_DIR:-${HOME_DIR}/dynamo_smoke_logs/m2_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "[wrapper] logs -> $LOG_DIR"

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
    --job-name="general_sa-dynamo.m2_smoke" \
    --output="$LOG_DIR/srun.out" \
    --error="$LOG_DIR/srun.err" \
    --container-image="$CONTAINER" \
    --container-mounts="${HOME_DIR}:${HOME_DIR}" \
    --container-workdir="$VERL_REPO" \
    bash -c "$INNER_CMD"
