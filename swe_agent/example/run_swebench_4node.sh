#!/usr/bin/env bash
# =============================================================================
# SWE-bench VERL Training - Multi-Node
#
# Thin wrapper: sets multi-node defaults, then delegates to run_swebench.sh.
# Default topology: head + 1 worker, 8 GPUs each = 16 GPUs.
#
# Prerequisites:
#   1. Start Ray cluster (see below — this script does it automatically)
#   2. Prepare data: see run_swebench.sh header
#
# Usage:
#   HEAD_IP=<head> WORKER_IPS_STR=<w1> bash run_swebench_4node.sh
#   bash run_swebench_4node.sh trainer.total_epochs=5  # Hydra overrides pass through
#   NNODES=4 WORKER_IPS_STR="<w1> <w2> <w3>" bash run_swebench_4node.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ================= Cluster topology =================
HEAD_IP=${MASTER_ADDR:?"MASTER_ADDR must be set (e.g. export MASTER_ADDR=192.168.1.100)"}
WORKER_IPS_STR=${WORKER_IPS_STR:?"WORKER_IPS_STR must be set (space-separated IPs, e.g. '192.168.1.101 192.168.1.102')"}
IFS=' ' read -ra WORKER_IP_LIST <<< "$WORKER_IPS_STR"
SSH_PORT=${SSH_PORT:-22}

NUM_WORKERS=${#WORKER_IP_LIST[@]}
TOTAL_NODES=$(( NUM_WORKERS + 1 ))

if [ -n "${NNODES:-}" ] && [ "$NNODES" -ne "$TOTAL_NODES" ]; then
    echo "[ERROR] NNODES ($NNODES) does not match worker topology derived from WORKER_IPS ($TOTAL_NODES nodes total)."
    echo "        Fix WORKER_IPS or unset NNODES."
    exit 1
fi
export NNODES=$TOTAL_NODES
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3-4b-swebench-${TOTAL_NODES}node-v1}

# Multi-node networking
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp96s0f0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp96s0f0}
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-enp96s0f0}
export MASTER_ADDR=${MASTER_ADDR:-$HEAD_IP}
export RAY_ADDRESS=${RAY_ADDRESS:-auto}

export DATA_DIR=${DATA_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)/data/swe_bench_small}

WORK_BASE=${WORK_BASE:-$HOME/workspace}
RAY_TMPDIR=$WORK_BASE/ray_tmp
RAY_PORT=6379

log() { echo "[$(date '+%H:%M:%S')] $1"; }

ssh_worker() {
    local ip="$1"; shift
    ssh -n -p "$SSH_PORT" -o StrictHostKeyChecking=no -o BatchMode=yes root@"$ip" "$@"
}

# ================= Step 1: Setup Ray cluster =================
log "Setting up ${TOTAL_NODES}-node Ray cluster: HEAD=$HEAD_IP + workers=${WORKER_IPS_STR}"

for ip in "${WORKER_IP_LIST[@]}"; do
    if ! ssh_worker "$ip" "echo connected" >/dev/null 2>&1; then
        log "ERROR: cannot connect to worker $ip via SSH port $SSH_PORT"
        exit 1
    fi
done

log "Stopping existing Ray processes..."
ray stop --force 2>/dev/null || true
for ip in "${WORKER_IP_LIST[@]}"; do
    ssh_worker "$ip" "ray stop --force 2>/dev/null || true" 2>&1 || true
done
sleep 2

log "Starting Ray head on $HEAD_IP..."
mkdir -p "$RAY_TMPDIR"
RAY_TMPDIR="$RAY_TMPDIR" ray start \
    --head \
    --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --num-gpus="$GPUS_PER_NODE" \
    --temp-dir="$RAY_TMPDIR"
sleep 3

for ip in "${WORKER_IP_LIST[@]}"; do
    log "Starting Ray worker on $ip..."
    ssh_worker "$ip" "
        mkdir -p $RAY_TMPDIR
        export RAY_TMPDIR=$RAY_TMPDIR
        ray start --address=${HEAD_IP}:${RAY_PORT} --num-gpus=$GPUS_PER_NODE --temp-dir=$RAY_TMPDIR
    " 2>&1
    log "  Worker $ip joined"
done
sleep 3

log "Verifying cluster..."
ray status
NODES=$(python3 -c "import ray; ray.init(address='auto'); print(len(ray.nodes()))" 2>/dev/null || echo "?")
log "Ray cluster ready: $NODES nodes"

if [ "$NODES" != "$TOTAL_NODES" ] && [ "$NODES" != "?" ]; then
    log "ERROR: Expected $TOTAL_NODES nodes but got $NODES"
    exit 1
fi

# ================= Step 2: Launch training =================
echo "=========================================="
echo "SWE-bench ${TOTAL_NODES}-Node Launch"
echo "  HEAD:       $HEAD_IP"
for ip in "${WORKER_IP_LIST[@]}"; do
    echo "  WORKER:     $ip"
done
echo "  GPUs:       $NNODES × $GPUS_PER_NODE = $(( NNODES * GPUS_PER_NODE ))"
echo "  Batch size: $TRAIN_BATCH_SIZE"
echo "  Data:       $DATA_DIR"
echo "  Dashboard:  http://${HEAD_IP}:8265"
echo "=========================================="

exec bash "$SCRIPT_DIR/run_swebench.sh" "$@"
