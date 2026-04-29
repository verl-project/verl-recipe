#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Manual smoke test for Dynamo + vLLM on this machine.
# Goal: confirm dynamo & vllm are installed correctly and a Qwen3-0.6B
# request round-trips end-to-end, BEFORE any verl integration. If this
# fails, recipe/dynamo cannot work either — fix install first.
#
# What it does:
#   1. Starts `python -m dynamo.frontend` on $HTTP_PORT (file discovery).
#   2. Starts `python -m dynamo.vllm` on GPU 0 (file discovery, kv events off).
#   3. Polls /v1/models until the worker registers (or timeout).
#   4. Issues one /v1/chat/completions request and prints the response.
#   5. Tears both processes down on exit.
#
# Override via env: MODEL, HTTP_PORT, DYN_SYSTEM_PORT, GPU, TIMEOUT_SEC, LOG_DIR.

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
HTTP_PORT="${HTTP_PORT:-8000}"
DYN_SYSTEM_PORT="${DYN_SYSTEM_PORT:-8081}"
GPU="${GPU:-0}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
LOG_DIR="${LOG_DIR:-$(mktemp -d -t dynamo-smoke-XXXXXX)}"

mkdir -p "$LOG_DIR"
echo "[smoke] logs -> $LOG_DIR"
echo "[smoke] model=$MODEL http_port=$HTTP_PORT system_port=$DYN_SYSTEM_PORT gpu=$GPU"

cleanup() {
    rc=$?
    echo "[smoke] cleaning up (rc=$rc)"
    [[ -n "${FRONTEND_PID:-}" ]] && kill "$FRONTEND_PID" 2>/dev/null || true
    [[ -n "${WORKER_PID:-}" ]] && kill "$WORKER_PID" 2>/dev/null || true
    wait 2>/dev/null || true
    if [[ $rc -ne 0 ]]; then
        echo "[smoke] FAILED. Tail of logs:"
        for f in "$LOG_DIR"/frontend.log "$LOG_DIR"/worker.log; do
            [[ -f "$f" ]] && { echo "===== $f ====="; tail -n 50 "$f"; }
        done
    fi
    exit $rc
}
trap cleanup EXIT

# 1. frontend
python -m dynamo.frontend \
    --http-port "$HTTP_PORT" \
    --discovery-backend file \
    > "$LOG_DIR/frontend.log" 2>&1 &
FRONTEND_PID=$!
echo "[smoke] frontend pid=$FRONTEND_PID"

# 2. vllm worker
DYN_SYSTEM_PORT="$DYN_SYSTEM_PORT" CUDA_VISIBLE_DEVICES="$GPU" \
    python -m dynamo.vllm \
        --model "$MODEL" \
        --enforce-eager \
        --discovery-backend file \
        --kv-events-config '{"enable_kv_cache_events": false}' \
        > "$LOG_DIR/worker.log" 2>&1 &
WORKER_PID=$!
echo "[smoke] worker pid=$WORKER_PID"

# 3. wait for /v1/models to list the model
echo "[smoke] waiting up to ${TIMEOUT_SEC}s for worker to register..."
deadline=$(( $(date +%s) + TIMEOUT_SEC ))
while :; do
    # Bail early if either process died.
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
        echo "[smoke] frontend died before serving"; exit 1
    fi
    if ! kill -0 "$WORKER_PID" 2>/dev/null; then
        echo "[smoke] worker died before registering"; exit 1
    fi

    body="$(curl -fsS "http://localhost:${HTTP_PORT}/v1/models" 2>/dev/null || true)"
    if echo "$body" | grep -q "\"$MODEL\""; then
        echo "[smoke] worker registered."
        break
    fi

    if [[ $(date +%s) -ge $deadline ]]; then
        echo "[smoke] timeout waiting for worker registration"
        exit 1
    fi
    sleep 2
done

# 4. one chat completion
echo "[smoke] issuing chat completion..."
resp="$(curl -fsS "http://localhost:${HTTP_PORT}/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\": \"$MODEL\",
         \"messages\": [{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],
         \"max_tokens\": 8,
         \"temperature\": 0}")"
echo "[smoke] response:"
echo "$resp"

if echo "$resp" | grep -q '"choices"'; then
    echo "[smoke] PASS"
else
    echo "[smoke] response missing choices"; exit 1
fi
