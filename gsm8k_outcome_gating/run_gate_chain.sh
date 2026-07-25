#!/usr/bin/env bash
# Experiment 1 — the three-arm "correct test" under the shaped reward:
#   none    — plain GRPO            (phantom gradient active)
#   std     — DAPO-style std gate   (keeps all-fail groups with std>0 -> misses the phantom)
#   outcome — binary-outcome gate   (drops all-same-outcome groups -> kills the phantom)
# Advantage-zeroing (not mask-zeroing): the token-mean denominator is unchanged,
# so the comparison is free of the 1/live_frac learning-rate confound.
# Repeat this chain to get independent replicates (README reports n=3).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${GATE_ROOT:-$(pwd)/work}
PY=${VERL_VENV:-$ROOT/venv}/bin/python
RES=$ROOT/gate_results
mkdir -p "$RES"
STEPS=${STEPS:-40}
TAG=${TAG:-r1}          # replicate tag: r1, r2, ... (README's runs used per-run tags)

run() {
  local mode="$1" exp="gate_${mode}_${TAG}" log="$ROOT/gate_${mode}_${TAG}.log"
  echo "======== $(date) START $exp ========"
  "$PY" -m ray stop --force >/dev/null 2>&1
  for p in $(pgrep -u "$(id -un)" -f "verl.trainer.main_ppo"); do kill -9 "$p" 2>/dev/null; done
  sleep 10
  env GPU="${GPU:-0}" ADV=grpo LR="${LR:-1e-4}" MAX_STEPS="$STEPS" GUTIL="${GUTIL:-0.3}" \
      GATE_SHAPED=1 GATE_MODE="$mode" GATE_LAMBDA="${GATE_LAMBDA:-0.30}" EXP="$exp" \
      GATE_ROOT="$ROOT" VERL_VENV="${VERL_VENV:-$ROOT/venv}" \
      bash "$HERE/run_gate_arm.sh" >> "$log" 2>&1
  {
    echo "# $exp done $(date)"
    echo -n "em: "; grep -aoE "acc[^ ]*mean@1:np.float64\([0-9.]+\)" "$log" | sed -E "s/.*\(([0-9.]+)\)/\1/" | cut -c1-6 | paste -sd" "
    echo -n "gate_last: "; grep -a "\[gate\]" "$log" | tail -1 | grep -aoE "live_frac=[0-9.]+ dropped_rows=[0-9/]+"; echo
    echo -n "len_last: "; grep -aoE "response_length/mean:[0-9]+" "$log" | tail -1
  } > "$RES/${exp}.md"
  echo "ARM_DONE $exp"; cat "$RES/${exp}.md"
}

run none
run std
run outcome
echo "CHAIN_COMPLETE $(date)"
