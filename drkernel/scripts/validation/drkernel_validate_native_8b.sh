#!/bin/bash
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# DR.Kernel kernel-RL standalone validation launcher.
#
# Runs DrKernelFullyAsyncRollouter.do_validate() once against a merged HF
# checkpoint and writes metrics (JSON + TensorBoard). Rollout-only: no
# trainer pool, no parameter sync. Reuses the same multi-turn /
# kernel_async-reward path as training, so val-core / val-aux metrics are
# directly comparable to the curves logged at test_freq during training.
#
# Set VAL_N>1 (with VAL_DO_SAMPLE=True) to evaluate naive Pass@N.

set -x
set -eo pipefail

# ---------- Resolve repo root + PYTHONPATH ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------- Start KernelGYM reward server ----------
# Patched KernelGYM checkout created by recipe/drkernel/kernelgym_npu/setup_kernelgym.sh.
# Override KERNELGYM_DIR to use a checkout elsewhere, or comment this block out to
# validate against an already-running server (set KERNELGYM_SERVER_URL accordingly).
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

pkill -9 -f 'recipe.drkernel.main_validate' 2>/dev/null || true
sleep 2

# ---------- Run identity ----------
PROJECT_NAME="${PROJECT_NAME:-drkernel_async}"
EXP_NAME="${EXP_NAME:-8b-val}"

# ---------- Data + model ----------
# MODEL_PATH must be a *merged HF* checkpoint dir (e.g. produced from a
# training checkpoint with scripts/merge_to_hf.sh).
MODEL_PATH="${MODEL_PATH:-/home/model/Qwen3-8B}"
VAL_FILES="${VAL_FILES:-/home/dataset/validation_drkernel_level2_v3.parquet}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-${REPO_ROOT}/outputs/${PROJECT_NAME}/${EXP_NAME}}"

export KERNELGYM_SERVER_URL="${KERNELGYM_SERVER_URL:-http://127.0.0.1:8002}"
export KERNELGYM_FEEDBACK_MODE="${KERNELGYM_FEEDBACK_MODE:-drop_keys}"
export KERNELGYM_FEEDBACK_DROP_KEYS="${KERNELGYM_FEEDBACK_DROP_KEYS:-metadata,num_custom_kernel,num_total_kernels,custom_kernel_cuda_time_in_profiling_us,total_kernel_run_time_in_profiling_us,task_id,submitted_at,completed_at,processing_time,profiling}"

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

# ---------- Resource pools (rollout-only; trainer pool gets 0 NPUs) ----------
NNODES_ROLLOUT="${NNODES_ROLLOUT:-${NNODES:-1}}"
N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-10}"

# ---------- Validation sampling ----------
# VAL_N=1 = single rollout per prompt. Set VAL_N>1 + VAL_DO_SAMPLE=True for Pass@N.
VAL_N="${VAL_N:-1}"
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-True}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"

MAX_TURN="${MAX_TURN:-3}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-24576}"

GEN_TP="${GEN_TP:-1}"
ROLLOUT_GPU_MEM_UTIL="${ROLLOUT_GPU_MEM_UTIL:-0.75}"

# ---------- Reward (kernel_async manager + reward_kwargs blob) ----------
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

# Fast@X metrics: comma-separated speedup thresholds X (over Torch reference).
FAST_AT_THRESHOLDS="${FAST_AT_THRESHOLDS:-0.0,0.1,0.3,0.5,0.6,0.8,1.0,1.2}"

mkdir -p "logs/${PROJECT_NAME}"

# ---------- Launch ----------
# data.train_files is set to VAL_FILES: the rollouter builds a train dataset
# at init even when only validating; it's never iterated but must be valid.
python -m recipe.drkernel.main_validate \
    --config-name=drkernel_kernel_trainer_native \
    \
    data.train_files="${VAL_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.return_raw_chat=True \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.truncation='right' \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.hybrid_engine=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURN}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURN}" \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE}" \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${TOP_K}" \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    \
    reward_model.server_url="${KERNELGYM_SERVER_URL}" \
    reward_model.reward_func_name="${REWARD_FUNC_NAME}" \
    reward_model.reward_kwargs.max_concurrent="${REWARD_MAX_CONCURRENT}" \
    reward_model.reward_kwargs.num_perf_trials="${NUM_PERF_TRIALS}" \
    reward_model.reward_kwargs.detect_decoy_kernel="${DETECT_DECOY_KERNEL}" \
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
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    rollout.total_rollout_steps=1 \
    trainer.nnodes="${NNODES_ROLLOUT}" \
    trainer.n_gpus_per_node=0 \
    trainer.total_epochs=1 \
    trainer.test_freq=1 \
    trainer.save_freq=-1 \
    trainer.val_before_train=False \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.device=npu \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.logger='["console","tensorboard"]' \
    trainer.validation_data_dir="./rollout_validation_dump/${PROJECT_NAME}/run_${EXP_NAME}" \
    \
    async_training.use_trainer_do_validate=False \
    \
    +ray_kwargs.ray_init.runtime_env.env_vars.KERNELGYM_FEEDBACK_MODE="${KERNELGYM_FEEDBACK_MODE}" \
    +ray_kwargs.ray_init.runtime_env.env_vars.KERNELGYM_FEEDBACK_DROP_KEYS="'${KERNELGYM_FEEDBACK_DROP_KEYS}'" \
    \
    "$@" 2>&1 | tee "logs/${PROJECT_NAME}/val_${EXP_NAME}.log"
