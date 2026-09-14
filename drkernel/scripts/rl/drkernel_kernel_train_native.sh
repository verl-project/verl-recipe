#!/bin/bash
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# DR.Kernel kernel-RL training launcher — DR.Kernel-NATIVE reward path.
#
# Wiring (matches upstream DR.Kernel layout):
#   --config-name=drkernel_kernel_trainer_native
#   reward_manager: kernel_async (DR.Kernel-style)
#   compute_kernel_reward_batch via AsyncKernelRewardManager
#
# Required env (must be set):
#   KERNELGYM_SERVER_URL   e.g. http://npu-4:8002
#   MODEL_PATH             path to the SFT-warmed checkpoint
#   TRAIN_FILES            list-form, e.g. [/data/train.parquet]
#   VAL_FILES              list-form, e.g. [/data/val.parquet]

set -x
set -eo pipefail

ray stop --force 2>/dev/null || true
pkill -9 -f 'recipe.drkernel.main' 2>/dev/null || true
sleep 2

TRAIN_FILES=/data/nfs/ahmad/dataset/thinking/training_data_thinking_npu.parquet
VAL_FILES=/data/nfs/ahmad/dataset/thinking/validation_data_thinking_npu.parquet
# MODEL_PATH=/data/nfs/model/Qwen3-4B
MODEL_PATH=/data/nfs/model/Qwen3-8B
export KERNELGYM_SERVER_URL=http://7.150.10.206:8002

# ---------- Ascend / vLLM env ----------
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1
export VLLM_ASCEND_ENABLE_NZ=0
export VLLM_USE_V1=1

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# ---------- Defaults ----------
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-16}"
N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-10}"
N_GPUS_TRAINING=$((NGPUS_PER_NODE - N_GPUS_ROLLOUT))

STALENESS="${STALENESS:-0.2}"
SYNC_STEP="${SYNC_STEP:-4}"
REQUIRE_BATCHES="${REQUIRE_BATCHES:-4}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-True}"
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-$((64 * 160))}"
TEST_FREQ="${TEST_FREQ:-10000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-200}"
SAVE_FREQ="${SAVE_FREQ:--1}"

ALGORITHM="${ALGORITHM:-trloo}"
ROLLOUT_N="${ROLLOUT_N:-6}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-4}"
MAX_TURN="${MAX_TURN:-3}"
ENABLE_MRS="${ENABLE_MRS:-False}"
ENABLE_BYPASS_MODE="${ENABLE_BYPASS_MODE:-True}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-24576}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
GEN_TP="${GEN_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.75}"

USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

REWARD_MAX_CONCURRENT="${REWARD_MAX_CONCURRENT:-32}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-100}"
DETECT_DECOY_KERNEL="${DETECT_DECOY_KERNEL:-True}"

# Fast@X metrics: comma-separated speedup thresholds X (over Torch
# reference). Each X yields `kernel/Fast@<X>` (training) and
# `val-aux/<data_source>/Fast@<X>/mean@N` (validation).
FAST_AT_THRESHOLDS="${FAST_AT_THRESHOLDS:-0.4,0.6,0.8,1.0,1.2}"

PROJECT_NAME="${PROJECT_NAME:-drkernel_async}"
# EXP_NAME="${EXP_NAME:-drkernel_kernel_4b_async_native}"
EXP_NAME="8b-test1"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-checkpoints/${PROJECT_NAME}/${EXP_NAME}}"

# ---------- Resolve repo root + PYTHONPATH ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

mkdir -p "logs/${PROJECT_NAME}"

# ---------- Launch ----------
python -m recipe.drkernel.main \
    --config-name=drkernel_kernel_trainer_native \
    \
    `# data` \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.truncation='right' \
    \
    `# model + actor (static-bsz)` \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    \
    `# rollout` \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURN}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURN}" \
    \
    `# algorithm` \
    algorithm.adv_estimator="${ALGORITHM}" \
    algorithm.batch_filter.enable="${ENABLE_MRS}" \
    algorithm.rollout_correction.bypass_mode="${ENABLE_BYPASS_MODE}" \
    \
    `# reward (DR.Kernel-native: kernel_async manager + reward_kwargs blob)` \
    reward_model.server_url="${KERNELGYM_SERVER_URL}" \
    reward_model.reward_kwargs.max_concurrent="${REWARD_MAX_CONCURRENT}" \
    reward_model.reward_kwargs.num_perf_trials="${NUM_PERF_TRIALS}" \
    reward_model.reward_kwargs.detect_decoy_kernel="${DETECT_DECOY_KERNEL}" \
    reward_model.reward_kwargs.fast_at_thresholds="[${FAST_AT_THRESHOLDS}]" \
    \
    `# resource pools` \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAINING}" \
    rollout.nnodes="${NNODES}" \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    rollout.total_rollout_steps="${TOTAL_ROLLOUT_STEPS}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.device=npu \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.logger='["console","tensorboard"]' \
    trainer.val_before_train=False \
    trainer.rollout_data_dir="./rollout_dump/${PROJECT_NAME}/run_${EXP_NAME}" \
    \
    `# async tuning` \
    async_training.staleness_threshold="${STALENESS}" \
    async_training.trigger_parameter_sync_step="${SYNC_STEP}" \
    async_training.require_batches="${REQUIRE_BATCHES}" \
    async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
    \
    "$@" 2>&1 | tee "logs/${PROJECT_NAME}/run_${EXP_NAME}.log"
