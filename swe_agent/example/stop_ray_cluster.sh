#!/usr/bin/env bash
# =============================================================================
# Stop 4-node Ray cluster and clean up Docker containers
# =============================================================================
set -euo pipefail

HEAD_IP=${HEAD_IP:?"HEAD_IP must be set (e.g. export HEAD_IP=192.168.1.100)"}
# shellcheck disable=SC2206
WORKER_IPS=(${WORKER_IPS_STR:?"WORKER_IPS_STR must be set (space-separated IPs)"})
SSH_PORT=${SSH_PORT:-22}

log() { echo "[$(date '+%H:%M:%S')] $1"; }

ssh_worker() {
    local ip="$1"; shift
    ssh -n -p "$SSH_PORT" -o StrictHostKeyChecking=no -o BatchMode=yes root@"$ip" "$@"
}

# =================== Stop Ray on all nodes ===================
log "Stopping Ray on all nodes..."

for ip in "${WORKER_IPS[@]}"; do
    log "  Stopping $ip..."
    ssh_worker "$ip" "
        ray stop --force 2>/dev/null || true
        docker ps -q --filter ancestor=swerex-python:3.11 2>/dev/null | xargs -r docker stop 2>/dev/null || true
        docker ps -q --filter 'name=sweb' 2>/dev/null | xargs -r docker stop 2>/dev/null || true
    " 2>&1 || true
done

log "  Stopping head ($HEAD_IP)..."
ray stop --force 2>/dev/null || true
docker ps -q --filter "ancestor=swerex-python:3.11" 2>/dev/null | xargs -r docker stop 2>/dev/null || true
docker ps -q --filter "name=sweb" 2>/dev/null | xargs -r docker stop 2>/dev/null || true

sleep 2

# =================== Verify ===================
log "Checking for remaining processes..."
REMAINING=$(docker ps -q --filter "ancestor=swerex-python:3.11" 2>/dev/null | wc -l)
log "  Head swerex containers: $REMAINING"

for ip in "${WORKER_IPS[@]}"; do
    R=$(ssh_worker "$ip" "docker ps -q --filter ancestor=swerex-python:3.11 2>/dev/null | wc -l" 2>/dev/null || echo "?")
    log "  $ip swerex containers: $R"
done

log ""
log "=== Cluster stopped ==="
