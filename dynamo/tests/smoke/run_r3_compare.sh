#!/usr/bin/env bash
# R3 parity smoke: stock SGLang (what verl's native server runs) vs the recipe's
# Dynamo SGLang sidecar stack behind the frontend's native /generate.
#
# The candidate commands are the ones DynamoHttpServer builds for
#   engine=sglang, sglang.launch_mode=sidecar, generate_api=generate,
#   rollout.enable_rollout_routing_replay=True
# (regenerate them from _build_sglang_cmd/_build_sglang_sidecar_cmd/_start_frontend
# if those change). The reference gets the same engine flags minus the
# sidecar-only ones, so both sides run identical kernels.
#
# Needs: sglang>=0.5.16 with native gRPC (0.5.21 tested), ai-dynamo with the
# SGLang sidecar and /generate (1.7.0.dev20261110 or newer), etcd and nats-server
# on PATH (or ETCD_BIN/NATS_BIN), 2*TP GPUs.
#
#   MODEL=/models/Qwen3-Coder-30B-A3B-Instruct TP=2 bash run_r3_compare.sh
set -euo pipefail

MODEL=${MODEL:?set MODEL to the model directory}
TP=${TP:-2}
MEM=${MEM:-0.7}
CTX=${CTX:-8192}
OUT=${OUT:-/tmp/r3_compare_$(date +%Y%m%d_%H%M%S)}
PROMPT_LENS=${PROMPT_LENS:-"128 1024 4096 7000"}
MAX_NEW=${MAX_NEW:-256}
READY_TIMEOUT=${READY_TIMEOUT:-1800}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ETCD_BIN=${ETCD_BIN:-etcd}
NATS_BIN=${NATS_BIN:-nats-server}
NS=r3test

mkdir -p "$OUT"
echo "logs: $OUT"
REF_GPUS=$(seq -s, 0 $((TP - 1)))
CAND_GPUS=$(seq -s, "$TP" $((2 * TP - 1)))

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    sleep 2
    for pid in "${PIDS[@]}"; do kill -9 "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT

wait_http() {  # url name
    local deadline=$((SECONDS + READY_TIMEOUT))
    until curl -sf -o /dev/null "$1"; do
        if ((SECONDS > deadline)); then echo "timeout waiting for $2 ($1)"; exit 1; fi
        for pid in "${PIDS[@]}"; do
            kill -0 "$pid" 2>/dev/null || { echo "a process died while waiting for $2; see $OUT"; tail -n 40 "$OUT"/*.log; exit 1; }
        done
        sleep 5
    done
    echo "ready: $2"
}

python - <<'EOF'
import importlib.metadata as md
for p in ("sglang", "ai-dynamo", "ai-dynamo-runtime"):
    try:
        print(f"{p}=={md.version(p)}")
    except md.PackageNotFoundError:
        raise SystemExit(f"{p} is not installed")
EOF

# --- control plane the recipe would start (etcd + NATS) ---------------------
"$ETCD_BIN" --data-dir "$OUT/etcd" --listen-client-urls http://127.0.0.1:2379 \
    --advertise-client-urls http://127.0.0.1:2379 >"$OUT/etcd.log" 2>&1 &
PIDS+=($!)
"$NATS_BIN" -js -p 4222 -sd "$OUT/nats" >"$OUT/nats.log" 2>&1 &
PIDS+=($!)
export ETCD_ENDPOINTS=http://127.0.0.1:2379 NATS_SERVER=nats://127.0.0.1:4222
export DYN_NAMESPACE=$NS DYN_DISCOVERY_BACKEND=etcd DYN_SDK_DISABLE_ANSI_LOGGING=1

ENGINE_FLAGS=(--model-path "$MODEL" --served-model-name r3model --tp-size "$TP"
    --mem-fraction-static "$MEM" --context-length "$CTX"
    --enable-return-routed-experts --attention-backend flashinfer)

# --- reference: stock SGLang HTTP server (verl native path) -----------------
CUDA_VISIBLE_DEVICES=$REF_GPUS python -m sglang.launch_server --host 127.0.0.1 --port 30000 \
    --nccl-port 29610 "${ENGINE_FLAGS[@]}" >"$OUT/reference.log" 2>&1 &
PIDS+=($!)

# --- candidate: recipe sidecar shard + frontend ------------------------------
CUDA_VISIBLE_DEVICES="" python -m dynamo.sglang.sidecar --grpc-endpoint 127.0.0.1:2049 \
    --grpc-startup-deadline-secs "$READY_TIMEOUT" >"$OUT/sidecar.log" 2>&1 &
PIDS+=($!)
CUDA_VISIBLE_DEVICES=$CAND_GPUS python -m sglang.launch_server --host 127.0.0.1 --port 2048 \
    --grpc-port 2049 --incremental-streaming-output --nccl-port 29611 \
    "${ENGINE_FLAGS[@]}" >"$OUT/engine.log" 2>&1 &
PIDS+=($!)
DYN_SGLANG_ENABLE_GENERATE=1 python -m dynamo.frontend --http-port 8000 --http-host 127.0.0.1 \
    --router-mode round-robin --discovery-backend etcd --namespace-prefix "$NS" \
    >"$OUT/frontend.log" 2>&1 &
PIDS+=($!)

wait_http http://127.0.0.1:30000/health_generate reference
wait_http http://127.0.0.1:2048/health_generate candidate-engine
deadline=$((SECONDS + READY_TIMEOUT))
until curl -sf http://127.0.0.1:8000/v1/models | grep -q r3model; do
    if ((SECONDS > deadline)); then echo "frontend never listed the sidecar worker"; tail -n 40 "$OUT"/sidecar.log "$OUT"/frontend.log; exit 1; fi
    sleep 5
done
echo "ready: frontend lists r3model"

# shellcheck disable=SC2086
python "$HERE/r3_compare.py" --model "$MODEL" \
    --reference http://127.0.0.1:30000 --candidate http://127.0.0.1:8000 \
    --prompt-lens $PROMPT_LENS --max-new-tokens "$MAX_NEW" --dump "$OUT/responses.jsonl" \
    2>&1 | tee "$OUT/compare.txt"
exit "${PIPESTATUS[0]}"
