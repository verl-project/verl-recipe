#!/bin/bash
# atropos + verl GRPO training with atropos scoring environments
#
# works with any atropos environment
#
# flow:
# 1. start atropos trajectory API (buffers scored rollouts)
# 2. start verl trainer (FSDP + internal vLLM in HYBRID mode, sleep/wake)
# 3. wait for trainer init, read internal vLLM address from ready file
# 4. start generate proxy on :9004 -> internal vLLM (round-robin for multi-GPU)
# 5. start atropos environment (generates via proxy, scores deterministically)
#
# weight sync: verl's checkpoint_manager.update_weights() handles sleep/wake and
# weight transfer to internal vLLM every step. zero-copy on naive backend.
#
# prerequisites:
#   pip install verl[vllm,atropos]
#   git clone https://github.com/NousResearch/atropos.git && cd atropos && pip install -e .
#
# usage:
#   bash recipe/atropos/run_atropos.sh
#   ATROPOS_ENV=gsm8k_server bash recipe/atropos/run_atropos.sh
#

set -euo pipefail

export HYDRA_FULL_ERROR=1
export VLLM_ATTENTION_BACKEND=FLASHINFER

ATROPOS_ENV="${ATROPOS_ENV:-gsm8k_server}"
MODEL="${MODEL:-Qwen/Qwen3-1.7B}"
LR="${LR:-5e-6}"
ATROPOS_API_PORT="${ATROPOS_API_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-9004}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GROUP_SIZE="${GROUP_SIZE:-4}"
TOTAL_STEPS="${TOTAL_STEPS:-50}"
ATROPOS_DIR="${ATROPOS_DIR:-../atropos}"
READY_FILE=$(mktemp /tmp/verl_atropos_ready.XXXXXX)
TOTAL_SEQS=$((BATCH_SIZE * GROUP_SIZE))
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-${TOTAL_SEQS}}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1536}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.45}"
GRAD_CLIP="${GRAD_CLIP:-1.0}"
ENTROPY_COEFF="${ENTROPY_COEFF:-0}"
SAVE_FREQ="${SAVE_FREQ:-10}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-true}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-true}"
N_GPUS="${N_GPUS:-1}"
STEPS_PER_EVAL="${STEPS_PER_EVAL:-}"
PIDS=()
cleanup() {
    echo "Cleaning up..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    pkill -f "generate_proxy" 2>/dev/null || true
    ray stop 2>/dev/null || true
    rm -f "$READY_FILE"
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# kill leftovers
ray stop 2>/dev/null || true
pkill -u "$USER" -f "vllm.entrypoints|generate_proxy|run-api|main_atropos|environments/.*\.py" 2>/dev/null || true
rm -f "$READY_FILE"
sleep 2

echo "=== GPU status ==="
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader

echo "=== Step 1: Starting Atropos Trajectory API ==="
run-api --port "${ATROPOS_API_PORT}" > /tmp/api.log 2>&1 &
PIDS+=($!)
sleep 3

echo "=== Step 2: Starting verl trainer (internal vLLM, HYBRID mode) ==="
python3 -m recipe.atropos.main_atropos \
    actor_rollout_ref.model.path="${MODEL}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.val_before_train=false \
    data.train_batch_size="${BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    actor_rollout_ref.rollout.n="${GROUP_SIZE}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.enforce_eager=true \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TOTAL_SEQS}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.actor.grad_clip="${GRAD_CLIP}" \
    actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF}" \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    'actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]' \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.model.use_remove_padding=false \
    '+actor_rollout_ref.model.override_config={attn_implementation: sdpa}' \
    atropos.api_url="http://localhost:${ATROPOS_API_PORT}" \
    atropos.poll_timeout=900 \
    atropos.ready_file="${READY_FILE}" \
    atropos.proxy_url="http://localhost:${PROXY_PORT}" \
    atropos.proxy_drain_timeout=300 \
    'trainer.logger=["console","wandb"]' \
    trainer.project_name=verl-atropos \
    trainer.experiment_name="${MODEL##*/}_${ATROPOS_ENV}_grpo" > /tmp/trainer.log 2>&1 &
TRAINER_PID=$!
PIDS+=($TRAINER_PID)

echo "Waiting for trainer to finish initialization..."
for i in $(seq 1 600); do
    if [ -f "$READY_FILE" ]; then
        echo "Trainer ready! (took ${i}s)"
        break
    fi
    if ! kill -0 "$TRAINER_PID" 2>/dev/null; then
        echo "ERROR: Trainer died during init. Log:"
        tail -60 /tmp/trainer.log
        exit 1
    fi
    if [ "$i" -eq 600 ]; then
        echo "ERROR: Trainer init timeout (600s)"
        tail -30 /tmp/trainer.log
        exit 1
    fi
    sleep 1
done

VLLM_ADDRESSES=$(cat "$READY_FILE")
echo "Internal vLLM addresses: ${VLLM_ADDRESSES}"

# convert comma-separated addr:port pairs to http:// URLs for the proxy
BACKEND_URLS=$(echo "$VLLM_ADDRESSES" | tr ',' '\n' | while read -r addr; do printf "http://%s," "$addr"; done | sed 's/,$//')

echo "=== Step 3: Starting generate proxy on port ${PROXY_PORT} ==="
python3 -m recipe.atropos.generate_proxy \
    --backend-url "${BACKEND_URLS}" \
    --model "${MODEL}" \
    --port "${PROXY_PORT}" \
    --drain-timeout 300 \
    --generation-timeout 300 > /tmp/proxy.log 2>&1 &
PROXY_PID=$!
PIDS+=($PROXY_PID)

for i in $(seq 1 60); do
    if curl -s "http://localhost:${PROXY_PORT}/health" > /dev/null 2>&1; then
        echo "Proxy ready (took ${i}s)"
        break
    fi
    if ! kill -0 "$PROXY_PID" 2>/dev/null; then
        echo "ERROR: Proxy died. Log:"
        tail -30 /tmp/proxy.log
        exit 1
    fi
    if [ "$i" -eq 60 ]; then
        echo "ERROR: Proxy timeout (60s)"
        tail -30 /tmp/proxy.log
        exit 1
    fi
    sleep 1
done

echo "=== Step 4: Starting ${ATROPOS_ENV} environment ==="
cd "${ATROPOS_DIR}"

python3 "environments/${ATROPOS_ENV}.py" serve \
    --env.rollout_server_url "http://localhost:${ATROPOS_API_PORT}" \
    --env.group_size "${GROUP_SIZE}" \
    --env.max_token_length "${MAX_RESPONSE_LENGTH}" \
    --env.tokenizer_name "${MODEL}" \
    ${STEPS_PER_EVAL:+--env.steps_per_eval "${STEPS_PER_EVAL}"} \
    --env.use_wandb true \
    --openai.server_type vllm \
    --openai.base_url "http://localhost:${PROXY_PORT}/v1" \
    --openai.model_name "${MODEL}" \
    --slurm false > /tmp/env.log 2>&1 &
PIDS+=($!)

echo "=== All services started. Waiting for trainer to complete... ==="
RESULT=0
wait $TRAINER_PID || RESULT=$?
echo "=== Training complete (exit ${RESULT}) ==="
tail -50 /tmp/trainer.log
