#!/usr/bin/env bash
set -xeuo pipefail

# Recipe dir = this script's directory (rename-safe; no hardcoded recipe path).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repo root = two levels up from recipe/<this-folder>/
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

python3 -m verl.trainer.main_ppo \
    --config-path="$SCRIPT_DIR/config" \
    --config-name=grpo \
    "hydra.searchpath=[file://${REPO_ROOT}/verl/trainer/config]" \
    "custom_reward_function.path=${SCRIPT_DIR}/reward_function.py"
