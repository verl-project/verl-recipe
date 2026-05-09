#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# v1 smoke test for the recipe/dynamo backend (single node, generation only).
# Submits a 1-node SLURM job that:
#   1. enroots into vllm017_latest.sqsh
#   2. installs ai-dynamo / ai-dynamo-runtime / pyzmq / aiohttp if missing
#   3. starts a single-node Ray cluster
#   4. runs scripts/smoke_dynamo_v1.py against a small model
#
# Pass: smoke_dynamo_v1.py exits 0 with "PASS:" in its log.

set -euo pipefail

ACCOUNT="${ACCOUNT:-general_sa}"
PARTITION="${PARTITION:-batch}"
TIME_LIMIT="${TIME_LIMIT:-00:30:00}"
CONTAINER="${CONTAINER:-/lustre/fsw/general_sa/sopyang/images/vllm017_latest.sqsh}"
WORKSPACE="${WORKSPACE:-/lustre/fsw/general_sa/sopyang}"
MODEL="${MODEL:-Qwen/Qwen2.5-0.5B-Instruct}"
TP="${TP:-1}"
GPUS="${GPUS:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"

LOG_DIR="${LOG_DIR:-${WORKSPACE}/rl/verl_0211/slurm/logs/dynamo_v1_smoke_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$LOG_DIR"
echo "[wrapper] logs -> $LOG_DIR"

[[ -f "$CONTAINER" ]] || { echo "missing container $CONTAINER" >&2; exit 1; }

# Inner script written to a tempfile to avoid shell-quoting hell in --wrap.
INNER_SH="${LOG_DIR}/inner.sh"
cat > "$INNER_SH" <<'BASH'
#!/usr/bin/env bash
set -euo pipefail
set -x

export PIP_CACHE_DIR=/workspace/hf_models/.cache/pip
export HF_HOME=/workspace/hf_models/.cache/huggingface
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME"

VERL_REPO=/workspace/rl/verl_0211/verl
export PYTHONPATH="${VERL_REPO}:${PYTHONPATH:-}"
cd "$VERL_REPO"

# 1. ensure deps
# Pin to ai-dynamo 1.0.2: 1.1.0 introduced multimodal_utils.protocol that
# requires MultiModalUUIDDict from vllm.inputs (vLLM 0.18+), incompatible
# with the vllm017_latest container.
DYNAMO_VER="${DYNAMO_VER:-1.0.2}"
# pip is idempotent if the version already matches, so we always run install.
# (Using `pip show | awk` for skip-if-installed fails under `set -o pipefail`
# when the package isn't there yet.)
pip install --pre --no-deps "ai-dynamo==${DYNAMO_VER}" "ai-dynamo-runtime==${DYNAMO_VER}" 2>&1 | tail -5
python3 -c "import dynamo, dynamo.runtime, dynamo.llm; print('dynamo SDK loaded')"
python3 -c "import zmq" 2>/dev/null || pip install pyzmq 2>&1 | tail -1
python3 -c "import aiohttp" 2>/dev/null || pip install aiohttp 2>&1 | tail -1
# nixl is a transitive dep of dynamo.common.multimodal (audio_loader); we
# don't actually use it but the dynamo.vllm import chain requires the module
# to be importable.
python3 -c "import nixl" 2>/dev/null || pip install nixl 2>&1 | tail -3
python3 -c "import verl; print('verl', verl.__version__)"

# Static etcd + nats-server binaries cached in /workspace/dynamo_bin so this
# pulls only once per cluster lifetime.
DYN_BIN_DIR=/workspace/dynamo_bin
mkdir -p "$DYN_BIN_DIR"
ETCD_VER=v3.5.21
NATS_VER=v2.10.22
if [[ ! -x "$DYN_BIN_DIR/etcd" ]]; then
    cd /tmp
    curl -sSL "https://github.com/etcd-io/etcd/releases/download/${ETCD_VER}/etcd-${ETCD_VER}-linux-amd64.tar.gz" \
        | tar xz
    install -m 755 "etcd-${ETCD_VER}-linux-amd64/etcd"     "$DYN_BIN_DIR/etcd"
    install -m 755 "etcd-${ETCD_VER}-linux-amd64/etcdctl"  "$DYN_BIN_DIR/etcdctl"
    rm -rf "etcd-${ETCD_VER}-linux-amd64"*
    cd -
fi
if [[ ! -x "$DYN_BIN_DIR/nats-server" ]]; then
    cd /tmp
    curl -sSL "https://github.com/nats-io/nats-server/releases/download/${NATS_VER}/nats-server-${NATS_VER}-linux-amd64.tar.gz" \
        | tar xz
    install -m 755 "nats-server-${NATS_VER}-linux-amd64/nats-server" "$DYN_BIN_DIR/nats-server"
    rm -rf "nats-server-${NATS_VER}-linux-amd64"*
    cd -
fi
export PATH="$DYN_BIN_DIR:$PATH"
etcd --version | head -1
nats-server --version

# Surface dynamo subprocess logs out of the container's /tmp so we can read
# them on the host post-mortem.
export VERL_DYNAMO_LOG_DIR="${VERL_DYNAMO_LOG_DIR:-/workspace/rl/verl_0211/slurm/dynamo_logs/${SLURM_JOB_ID:-manual}}"
mkdir -p "$VERL_DYNAMO_LOG_DIR"
echo "[inner] dynamo subprocess logs -> $VERL_DYNAMO_LOG_DIR"

# 2. start single-node Ray head
HEAD_IP=$(hostname -I | awk '{print $1}')
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address="$HEAD_IP" --port=6379
sleep 5
ray status

# 3. run smoke
set +e
python3 recipe/dynamo/scripts/smoke_dynamo_v1.py \
    --model "${SMOKE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" \
    --tp "${SMOKE_TP:-1}" \
    --gpus-per-node "${SMOKE_GPUS:-1}" \
    --gpu-memory-utilization "${SMOKE_GPU_MEM_UTIL:-0.5}" \
    --max-model-len "${SMOKE_MAX_MODEL_LEN:-4096}" \
    --enforce-eager
RC=$?
set -e

ray stop --force >/dev/null 2>&1 || true
echo "[wrapper] smoke exit code: $RC"
exit $RC
BASH
chmod +x "$INNER_SH"

# Translate host path → container path for the inner script (container only
# sees /workspace, which mounts $WORKSPACE on the host).
INNER_SH_IN_CONTAINER="${INNER_SH/${WORKSPACE}//workspace}"

JOB_ID=$(sbatch \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIME_LIMIT" \
    --nodes=1 \
    --ntasks-per-node=1 \
    --job-name="dynamo-v1-smoke" \
    --output="$LOG_DIR/srun.out" \
    --error="$LOG_DIR/srun.err" \
    --export="ALL,SMOKE_MODEL=${MODEL},SMOKE_TP=${TP},SMOKE_GPUS=${GPUS},SMOKE_MAX_MODEL_LEN=${MAX_MODEL_LEN},SMOKE_GPU_MEM_UTIL=${GPU_MEM_UTIL}" \
    --wrap="srun --container-image=${CONTAINER} --container-mounts=${WORKSPACE}:/workspace bash ${INNER_SH_IN_CONTAINER}" \
    --parsable)

echo "[wrapper] submitted job $JOB_ID"
echo "[wrapper] watch: tail -F ${LOG_DIR}/srun.out"
echo "$JOB_ID" > "$LOG_DIR/job_id"
echo "$LOG_DIR" > "${WORKSPACE}/rl/verl_0211/slurm/logs/.last_dynamo_smoke_dir"
