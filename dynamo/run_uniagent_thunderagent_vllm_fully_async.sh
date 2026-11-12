#!/usr/bin/env bash
# The same launcher with separate training/rollout pools.
set -euo pipefail
: "${RAY_CLUSTER_ADDRESS:?Set RAY_CLUSTER_ADDRESS to the prepared 16-GPU Ray cluster}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_uniagent_thunderagent_vllm.sh" uniagent_mode=separate "$@"
