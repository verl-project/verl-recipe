#!/usr/bin/env bash
# Terminal sync evaluation: 256 batches x 16 task slots x 16 rollouts = 65,536 rollouts.
# Repeat the terminal evaluation manifest 16 times.
set -euo pipefail

RECIPE_DIR=recipe/obsspec
DATA_DIR="${DATA_DIR:-${HOME}/data/terminal_obsspec}"

DATASET=endless_terminals \
EVAL_MANIFEST_PATH="${DATA_DIR}/endless_terminals_eval_manifest.json" \
EVAL_REPORT_LEVEL="${EVAL_REPORT_LEVEL:-basic}" \
bash "${RECIPE_DIR}/reproduce/eval_4k.sh" +data.eval_manifest_repeat=16 "$@"
