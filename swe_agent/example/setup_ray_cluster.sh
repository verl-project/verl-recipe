#!/usr/bin/env bash
# =============================================================================
# Setup 4-node Ray cluster for SWE-Agent VERL Training
# =============================================================================
# Topology:
#   HEAD:    <HEAD_IP>    (Ray head + Dashboard :8265)
#   WORKER1: <WORKER_IP_1> (via SSH)
#   WORKER2: <WORKER_IP_2> (via SSH)
#   WORKER3: <WORKER_IP_3> (via SSH)
# =============================================================================
set -euo pipefail

HEAD_IP=${HEAD_IP:?"HEAD_IP must be set (e.g. export HEAD_IP=192.168.1.100)"}
# shellcheck disable=SC2206
WORKER_IPS=(${WORKER_IPS_STR:?"WORKER_IPS_STR must be set (space-separated IPs, e.g. '192.168.1.101 192.168.1.102')"})
SSH_PORT=${SSH_PORT:-22}
RAY_PORT=6379
DASHBOARD_PORT=8265

WORK_BASE=${WORK_BASE:-$HOME/workspace}
RAY_TMPDIR=$WORK_BASE/ray_tmp

log() { echo "[$(date '+%H:%M:%S')] $1"; }

ssh_worker() {
    local ip="$1"; shift
    ssh -n -p "$SSH_PORT" -o StrictHostKeyChecking=no -o BatchMode=yes root@"$ip" "$@"
}

# =================== Step 1: Stop any existing Ray ===================
log "Stopping any existing Ray processes..."
ray stop --force 2>/dev/null || true

for ip in "${WORKER_IPS[@]}"; do
    log "  Stopping Ray on $ip..."
    ssh_worker "$ip" "ray stop --force 2>/dev/null || true" 2>&1 || true
done
sleep 2

# =================== Step 2: Start Ray head ===================
log "Starting Ray head on $HEAD_IP..."
mkdir -p "$RAY_TMPDIR"

RAY_TMPDIR="$RAY_TMPDIR" ray start \
    --head \
    --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$DASHBOARD_PORT \
    --num-gpus=8 \
    --temp-dir="$RAY_TMPDIR"

sleep 3
log "Ray head started. Dashboard: http://${HEAD_IP}:${DASHBOARD_PORT}"

# =================== Step 3: Start Ray workers ===================
for ip in "${WORKER_IPS[@]}"; do
    log "Starting Ray worker on $ip..."
    ssh_worker "$ip" "
        mkdir -p $RAY_TMPDIR
        export RAY_TMPDIR=$RAY_TMPDIR
        ray start \
            --address=${HEAD_IP}:${RAY_PORT} \
            --num-gpus=8 \
            --temp-dir=$RAY_TMPDIR
    " 2>&1
    log "  Worker $ip started"
done

sleep 5

# =================== Step 4: Verify cluster ===================
log "Verifying cluster status..."
ray status

NODES=$(python3 -c "import ray; ray.init(address='auto'); print(len(ray.nodes()))" 2>/dev/null || echo "?")
log ""
log "=========================================="
log "Ray cluster ready: $NODES nodes"
log "  HEAD:    $HEAD_IP"
for ip in "${WORKER_IPS[@]}"; do
    log "  WORKER:  $ip"
done
log "  Dashboard: http://${HEAD_IP}:${DASHBOARD_PORT}"
log "=========================================="
