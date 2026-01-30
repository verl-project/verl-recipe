#!/usr/bin/env bash
# GRPO Baseline Smoke Test
#
# Minimal smoke test version of grpo_baseline.sh for quick verification.
# Uses minimal batch sizes and only 3 training steps.
#
# Usage:
#   bash grpo_baseline_smoke.sh
#
# This script should complete in a few minutes and verify that:
# - Ray initializes correctly
# - Model loads on both GPUs (vLLM rollout + Megatron training)
# - Weight sync between actor and rollout works
# - Training loop completes without errors

set -x

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Run the baseline script in smoke test mode
SMOKE_TEST=1 bash "${SCRIPT_DIR}/grpo_baseline.sh" "$@"
