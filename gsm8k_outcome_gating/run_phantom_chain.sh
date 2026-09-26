#!/usr/bin/env bash
# Experiment 3 — reverse phantom (GATE_PHANTOM=long): the length term flips so
# the LONGEST wrong answer scores best. Probes whether phantom direction
# matters (it does — see README: the Goodhart asymmetry).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${GATE_ROOT:-$(pwd)/work}
RES=$ROOT/gate_results
mkdir -p "$RES"
STEPS=${STEPS:-40}

run() {
  local mode="$1"
  local exp="gate_${mode}_ph030" log="$ROOT/gate_${mode}_ph030.log"
  echo "======== $(date) START $exp (phantom=long lambda=0.30) ========"
  env GPU="${GPU:-0}" ADV=grpo LR="${LR:-1e-4}" MAX_STEPS="$STEPS" GUTIL="${GUTIL:-0.3}" \
      GATE_MODE="$mode" GATE_LAMBDA=0.30 GATE_PHANTOM=long EXP="$exp" \
      GATE_ROOT="$ROOT" VERL_VENV="${VERL_VENV:-$ROOT/venv}" \
      bash "$HERE/run_gate_arm.sh"
  if ! grep -aqE "acc[^ ]*mean@1:" "$log"; then
    echo "[ABORT] $exp produced no validation accuracy; inspect $log" >&2
    return 1
  fi
  {
    echo "# $exp (phantom=long) done $(date)"
    echo -n "em: "; grep -aoE "acc[^ ]*mean@1:np.float64\([0-9.]+\)" "$log" | sed -E "s/.*\(([0-9.]+)\)/\1/" | cut -c1-6 | paste -sd" " -
    echo -n "gate_last: "; grep -a "\[gate\]" "$log" | tail -1 | grep -aoE "live_frac=[0-9.]+ dropped_rows=[0-9/]+" || true; echo
    echo -n "len_curve: "; grep -aoE "response_length/mean:[0-9]+" "$log" | sed -E "s/.*:([0-9]+)/\1/" | paste -sd" " - || true
  } > "$RES/${exp}.md"
  echo "ARM_DONE $exp"; cat "$RES/${exp}.md"
}

run none
run std
run outcome
echo "CHAIN_COMPLETE $(date)"
