#!/usr/bin/env bash
# Experiment 2 — dose-response over phantom strength lambda.
# The lambda=0.30 anchor comes from run_gate_chain.sh (n=3 in README); this
# chain adds lambda in {0.50, 0.10} x {none, std, outcome}. The 0.50/none arm
# runs first as a canary: a stronger phantom must collapse faster than 0.30 —
# if it doesn't, the GATE_LAMBDA plumbing is broken and you find out in run 1
# (also check step-1 metrics: critic/rewards/min must equal exactly -lambda).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=${GATE_ROOT:-$(pwd)/work}
PY=${VERL_VENV:-$ROOT/venv}/bin/python
RES=$ROOT/gate_results
mkdir -p "$RES"
STEPS=${STEPS:-40}

run() {
  local mode="$1" lam="$2" tag="$3"
  local exp="gate_${mode}_lam${tag}" log="$ROOT/${exp}.log"
  echo "======== $(date) START $exp (lambda=$lam) ========"
  "$PY" -m ray stop --force >/dev/null 2>&1
  for p in $(pgrep -u "$(id -un)" -f "verl.trainer.main_ppo"); do kill -9 "$p" 2>/dev/null; done
  sleep 10
  env GPU="${GPU:-0}" ADV=grpo LR="${LR:-1e-4}" MAX_STEPS="$STEPS" GUTIL="${GUTIL:-0.3}" \
      GATE_SHAPED=1 GATE_MODE="$mode" GATE_LAMBDA="$lam" EXP="$exp" \
      GATE_ROOT="$ROOT" VERL_VENV="${VERL_VENV:-$ROOT/venv}" \
      bash "$HERE/run_gate_arm.sh" >> "$log" 2>&1
  {
    echo "# $exp (lambda=$lam) done $(date)"
    echo -n "em: "; grep -aoE "acc[^ ]*mean@1:np.float64\([0-9.]+\)" "$log" | sed -E "s/.*\(([0-9.]+)\)/\1/" | cut -c1-6 | paste -sd" "
    echo -n "gate_last: "; grep -a "\[gate\]" "$log" | tail -1 | grep -aoE "live_frac=[0-9.]+ dropped_rows=[0-9/]+"; echo
    echo -n "len_last: "; grep -aoE "response_length/mean:[0-9]+" "$log" | tail -1
  } > "$RES/${exp}.md"
  echo "ARM_DONE $exp"; cat "$RES/${exp}.md"
}

run none    0.50 050
run std     0.50 050
run outcome 0.50 050
run none    0.10 010
run std     0.10 010
run outcome 0.10 010
echo "CHAIN_COMPLETE $(date)"
