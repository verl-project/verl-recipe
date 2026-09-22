#!/bin/bash
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

set -x
set -eo pipefail

# ---------- Resolve repo root + PYTHONPATH ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------- Start KernelGYM reward server ----------
# Patched KernelGYM checkout created by recipe/drkernel/kernelgym_npu/setup_kernelgym.sh.
# Override KERNELGYM_DIR to use a checkout elsewhere.
KERNELGYM_DIR="${KERNELGYM_DIR:-${REPO_ROOT}/recipe/drkernel/kernelgym_npu/KernelGYM}"
if [ ! -d "${KERNELGYM_DIR}" ]; then
    echo "KernelGYM checkout not found at ${KERNELGYM_DIR}." >&2
    echo "Run: bash ${REPO_ROOT}/recipe/drkernel/kernelgym_npu/setup_kernelgym.sh" >&2
    exit 1
fi
cd "${KERNELGYM_DIR}"
mkdir -p logs && cd logs && rm -rf * && cd ..
bash start_all_with_monitor.sh
cd "${REPO_ROOT}"

pkill -9 -f 'recipe.drkernel.main' 2>/dev/null || true
sleep 2

# ---------- Run identity ----------
PROJECT_NAME="${PROJECT_NAME:-drkernel_async}"
EXP_NAME="${EXP_NAME:-8b}"

# ---------- Data + model ----------
TRAIN_FILES="${TRAIN_FILES:-/home/dataset/training_drkernel_2k_difficulty-filtered_v3.parquet}"
VAL_FILES="${VAL_FILES:-/home/dataset/validation_drkernel_level2_v3.parquet}"
MODEL_PATH="${MODEL_PATH:-/home/model/Qwen3-8B}"

export KERNELGYM_SERVER_URL="${KERNELGYM_SERVER_URL:-http://127.0.0.1:8002}"
export KERNELGYM_FEEDBACK_MODE=drop_keys
export KERNELGYM_FEEDBACK_DROP_KEYS="metadata,num_custom_kernel,num_total_kernels,custom_kernel_cuda_time_in_profiling_us,total_kernel_run_time_in_profiling_us,task_id,submitted_at,completed_at,processing_time,profiling"

# ---------- NPU / vLLM env ----------
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

# ---------- Resource pools ----------
NNODES_TRAIN="${NNODES_TRAIN:-1}"
NNODES_ROLLOUT="${NNODES_ROLLOUT:-1}"
N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-10}"
N_GPUS_TRAINING="${N_GPUS_TRAINING:-16}"

# ---------- Async tuning ----------
STALENESS="${STALENESS:-0.1}"
SYNC_STEP="${SYNC_STEP:-1}"
REQUIRE_BATCHES="${REQUIRE_BATCHES:-16}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-True}"
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-$((64 * 160))}"
TEST_FREQ="${TEST_FREQ:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-200}"
SAVE_FREQ="${SAVE_FREQ:-10}"

# ---------- Algorithm ----------
ALGORITHM="${ALGORITHM:-trloo}"
ROLLOUT_N="${ROLLOUT_N:-16}"
PPO_MINI_BATCH="${PPO_MINI_BATCH:-1}"
MAX_TURN="${MAX_TURN:-3}"

ENABLE_BYPASS_MODE="${ENABLE_BYPASS_MODE:-False}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-24576}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
GEN_TP="${GEN_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.75}"

USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

ENTROPY_CHECKPOINTING="${ENTROPY_CHECKPOINTING:-True}"
ENTROPY_FROM_LOGITS_WITH_CHUNKING="${ENTROPY_FROM_LOGITS_WITH_CHUNKING:-True}"

# ---------- Reward ----------
REWARD_MAX_CONCURRENT="${REWARD_MAX_CONCURRENT:-32}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-100}"
DETECT_DECOY_KERNEL="${DETECT_DECOY_KERNEL:-True}"

REWARD_FUNC_NAME="${REWARD_FUNC_NAME:-calculate_reward_weighted}"
INIT_CORRECT_WEIGHT="${INIT_CORRECT_WEIGHT:-1.0}"
INIT_PERFORMANCE_WEIGHT="${INIT_PERFORMANCE_WEIGHT:-1.0}"
SPEEDUP_EPS="${SPEEDUP_EPS:-0.01}"
SPEEDUP_REWARD_UPPER_BOUND="${SPEEDUP_REWARD_UPPER_BOUND:-3.0}"
SPEEDUP_REWARD_LOWER_BOUND="${SPEEDUP_REWARD_LOWER_BOUND:-0.0}"

COVERAGE_REWARD_ENABLE="${COVERAGE_REWARD_ENABLE:-False}"
COVERAGE_REWARD_WEIGHT="${COVERAGE_REWARD_WEIGHT:-0.5}"
COVERAGE_REWARD_TYPE="${COVERAGE_REWARD_TYPE:-time_coverage}"

FAST_AT_THRESHOLDS="${FAST_AT_THRESHOLDS:-0.0,0.1,0.3,0.5,0.6,0.8,1.0,1.2}"

# ---------- Kernelgym setup ----------
KERNELGYM_TOOLKIT="${KERNELGYM_TOOLKIT:-sandbox_v3}"

# ---------- MRS (paper-faithful rollout correction) ----------
ROLLOUT_RS="${ROLLOUT_RS:-seq_mean_k1}"
ROLLOUT_RS_THRESHOLD="${ROLLOUT_RS_THRESHOLD:-0.999001_1.001001}"
ROLLOUT_TOKEN_VETO_THRESHOLD="${ROLLOUT_TOKEN_VETO_THRESHOLD:-1e-4}"
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"

DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXP_NAME}}"

cd "${REPO_ROOT}"
mkdir -p "logs/${PROJECT_NAME}"

# ---------- Launch ----------
python -m recipe.drkernel.main \
    --config-name=drkernel_kernel_trainer_native \
    \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.truncation='right' \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr="${LEARNING_RATE}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.entropy_checkpointing="${ENTROPY_CHECKPOINTING}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}" \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking="${ENTROPY_FROM_LOGITS_WITH_CHUNKING}" \
    actor_rollout_ref.ref.entropy_checkpointing="${ENTROPY_CHECKPOINTING}" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    \
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
    algorithm.adv_estimator="${ALGORITHM}" \
    algorithm.rollout_correction.bypass_mode="${ENABLE_BYPASS_MODE}" \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    algorithm.rollout_correction.rollout_rs="${ROLLOUT_RS}" \
    algorithm.rollout_correction.rollout_rs_threshold="${ROLLOUT_RS_THRESHOLD}" \
    algorithm.rollout_correction.rollout_token_veto_threshold="${ROLLOUT_TOKEN_VETO_THRESHOLD}" \
    \
    reward_model.server_url="${KERNELGYM_SERVER_URL}" \
    reward_model.reward_func_name="${REWARD_FUNC_NAME}" \
    reward_model.reward_kwargs.max_concurrent="${REWARD_MAX_CONCURRENT}" \
    reward_model.reward_kwargs.num_perf_trials="${NUM_PERF_TRIALS}" \
    reward_model.reward_kwargs.detect_decoy_kernel="${DETECT_DECOY_KERNEL}" \
    reward_model.reward_kwargs.toolkit="${KERNELGYM_TOOLKIT}" \
    reward_model.reward_kwargs.fast_at_thresholds="[${FAST_AT_THRESHOLDS}]" \
    reward_model.reward_kwargs.init_correct_weight="${INIT_CORRECT_WEIGHT}" \
    reward_model.reward_kwargs.init_performance_weight="${INIT_PERFORMANCE_WEIGHT}" \
    reward_model.reward_kwargs.speedup_eps="${SPEEDUP_EPS}" \
    reward_model.reward_kwargs.speedup_reward_upper_bound="${SPEEDUP_REWARD_UPPER_BOUND}" \
    reward_model.reward_kwargs.speedup_reward_lower_bound="${SPEEDUP_REWARD_LOWER_BOUND}" \
    reward_model.reward_kwargs.coverage_reward.enable="${COVERAGE_REWARD_ENABLE}" \
    reward_model.reward_kwargs.coverage_reward.weight="${COVERAGE_REWARD_WEIGHT}" \
    reward_model.reward_kwargs.coverage_reward.reward_type="${COVERAGE_REWARD_TYPE}" \
    \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAINING}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
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
    trainer.validation_data_dir="./rollout_validation_dump/${PROJECT_NAME}/run_${EXP_NAME}" \
    \
    async_training.staleness_threshold="${STALENESS}" \
    async_training.trigger_parameter_sync_step="${SYNC_STEP}" \
    async_training.require_batches="${REQUIRE_BATCHES}" \
    async_training.partial_rollout="${PARTIAL_ROLLOUT}" \
    \
    +ray_kwargs.ray_init.runtime_env.env_vars.KERNELGYM_FEEDBACK_MODE="${KERNELGYM_FEEDBACK_MODE}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.KERNELGYM_TOOLKIT="${KERNELGYM_TOOLKIT}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.KERNELGYM_FEEDBACK_DROP_KEYS="'${KERNELGYM_FEEDBACK_DROP_KEYS}'" \
    \
    "$@" 2>&1 | tee "logs/${PROJECT_NAME}/run_${EXP_NAME}.log"
