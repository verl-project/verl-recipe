#!/usr/bin/env bash
# Experiment 1 — the three-arm "correct test" under the shaped reward:
#   none    — plain GRPO            (phantom gradient active)
#   std     — shaped-score gate    (official DAPO recipes instead filter on binary acc)
#   outcome — binary-outcome gate   (drops all-same-outcome groups -> kills the phantom)
# Advantage-zeroing (not mask-zeroing): the token-mean denominator is unchanged,
# so the comparison is free of the 1/live_frac learning-rate confound.
# Repeat this chain to get independent replicates (README reports n=3).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${GATE_ROOT:-$(pwd)/work}
RES=$ROOT/gate_results
mkdir -p "$RES"
STEPS=${STEPS:-40}
TAG=${TAG:-r1}          # replicate tag: r1, r2, ... (README's runs used per-run tags)

run() {
  local mode="$1"
  local exp="gate_${mode}_${TAG}" log="$ROOT/gate_${mode}_${TAG}.log"
  echo "======== $(date) START $exp ========"
  env GPU="${GPU:-0}" ADV=grpo LR="${LR:-1e-4}" MAX_STEPS="$STEPS" GUTIL="${GUTIL:-0.3}" \
      GATE_MODE="$mode" GATE_LAMBDA="${GATE_LAMBDA:-0.30}" EXP="$exp" \
      GATE_ROOT="$ROOT" VERL_VENV="${VERL_VENV:-$ROOT/venv}" \
      bash "$HERE/run_gate_arm.sh"
  if ! grep -aqE "acc[^ ]*mean@1:" "$log"; then
    echo "[ABORT] $exp produced no validation accuracy; inspect $log" >&2
    return 1
  fi
  {
    echo "# $exp done $(date)"
    echo -n "em: "; grep -aoE "acc[^ ]*mean@1:np.float64\([0-9.]+\)" "$log" | sed -E "s/.*\(([0-9.]+)\)/\1/" | cut -c1-6 | paste -sd" " -
    echo -n "gate_last: "; grep -a "\[gate\]" "$log" | tail -1 | grep -aoE "live_frac=[0-9.]+ dropped_rows=[0-9/]+" || true; echo
    echo -n "len_last: "; grep -aoE "response_length/mean:[0-9]+" "$log" | tail -1 || true
  } > "$RES/${exp}.md"
  echo "ARM_DONE $exp"; cat "$RES/${exp}.md"
}

run none
run std
run outcome
echo "CHAIN_COMPLETE $(date)"
