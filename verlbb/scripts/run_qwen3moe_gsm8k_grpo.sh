#!/usr/bin/env bash
set -euo pipefail

if [[ "${VERBOSE:-0}" == "1" ]]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -L)"
RECIPE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd -L)"

add_pythonpath() {
  local path="${1:-}"
  if [[ -n "${path}" ]]; then
    export PYTHONPATH="${path}:${PYTHONPATH:-}"
  fi
}

add_pythonpath "${RECIPE_ROOT}"
add_pythonpath "${VERL_ROOT:-}"
add_pythonpath "${BUMBLEBEE_ROOT:-}"
add_pythonpath "${MEGATRON_ROOT:-}"
add_pythonpath "${MBRIDGE_ROOT:-}"

export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export RAY_memory_monitor_refresh_ms="${RAY_memory_monitor_refresh_ms:-0}"

if [[ "${DISABLE_VLLM_EXPANDABLE_SEGMENTS:-True}" == "True" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF//expandable_segments:True/}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF//,,/,}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF#,}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF%,}"
  if [[ -z "${PYTORCH_CUDA_ALLOC_CONF}" ]]; then
    unset PYTORCH_CUDA_ALLOC_CONF
  fi

  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF//expandable_segments:True/}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF//,,/,}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF#,}"
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF%,}"
  if [[ -z "${PYTORCH_ALLOC_CONF}" ]]; then
    unset PYTORCH_ALLOC_CONF
  fi
fi

: "${MODEL_PATH:?set MODEL_PATH to a Hugging Face checkpoint directory or model id}"
TRAIN_FILE="${TRAIN_FILE:-${TRAIN_FILES:-}}"
VAL_FILE="${VAL_FILE:-${VAL_FILES:-}}"
: "${TRAIN_FILE:?set TRAIN_FILE or TRAIN_FILES to a GSM8K train parquet path}"
: "${VAL_FILE:?set VAL_FILE or VAL_FILES to a GSM8K validation parquet path}"

BACKEND="${BACKEND:-bumblebee}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${RECIPE_ROOT}/outputs/qwen3moe_gsm8k_grpo}"
PROJECT_NAME="${PROJECT_NAME:-verlbb-qwen3moe-gsm8k-grpo}"

NUM_GPUS="${NUM_GPUS:-8}"
NNODES="${NNODES:-1}"

TOTAL_STEPS="${TOTAL_STEPS:-100}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
SAVE_FREQ="${SAVE_FREQ:--1}"
TEST_FREQ="${TEST_FREQ:--1}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-null}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-null}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-null}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-512}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-128}"
ROLLOUT_N="${ROLLOUT_N:-2}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-2048}"
INFER_MAX_TOKEN_LEN_PER_GPU="${INFER_MAX_TOKEN_LEN_PER_GPU:-2048}"
LOG_PROB_MICRO_BATCH_SIZE_PER_GPU="${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU:-1}"

TP_SIZE="${TP_SIZE:-2}"
PP_SIZE="${PP_SIZE:-1}"
VPP_SIZE="${VPP_SIZE:-null}"
CP_SIZE="${CP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-8}"
ETP_SIZE="${ETP_SIZE:-1}"
DTYPE="${DTYPE:-bfloat16}"
BB_IMPL="${BB_IMPL:-lite}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-flash}"
USE_FUSED_KERNELS="${USE_FUSED_KERNELS:-True}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
USE_REMOVE_PADDING="${USE_REMOVE_PADDING:-True}"
ALL_OFFLOAD="${ALL_OFFLOAD:-True}"
PARAM_OFFLOAD="${PARAM_OFFLOAD:-${ALL_OFFLOAD}}"
OPTIMIZER_OFFLOAD="${OPTIMIZER_OFFLOAD:-${ALL_OFFLOAD}}"
GRAD_OFFLOAD="${GRAD_OFFLOAD:-${ALL_OFFLOAD}}"
OPTIMIZER_STATE_OFFLOAD_FRACTION="${OPTIMIZER_STATE_OFFLOAD_FRACTION:-1.0}"
OPTIMIZER_CPU_OFFLOAD="${OPTIMIZER_CPU_OFFLOAD:-True}"
USE_PRECISION_AWARE_OPTIMIZER="${USE_PRECISION_AWARE_OPTIMIZER:-True}"
DECOUPLED_WEIGHT_DECAY="${DECOUPLED_WEIGHT_DECAY:-True}"

ROLLOUT_TP_SIZE="${ROLLOUT_TP_SIZE:-2}"
ROLLOUT_DP_SIZE="${ROLLOUT_DP_SIZE:-1}"
ROLLOUT_EP_SIZE="${ROLLOUT_EP_SIZE:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
ROLLOUT_MAX_MODEL_LEN="${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-16}"

LR="${LR:-1e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
SEED="${SEED:-1}"
DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=("$@")

case "${BACKEND}" in
  megatron|bumblebee)
    ;;
  *)
    echo "Unsupported BACKEND=${BACKEND}. Expected megatron or bumblebee." >&2
    exit 1
    ;;
esac

BB_VPP_SIZE="${VPP_SIZE}"
if [[ "${BB_VPP_SIZE}" == "null" ]]; then
  BB_VPP_SIZE=1
fi

RUN_NAME="${RUN_NAME:-qwen3moe_gsm8k_grpo_${BACKEND}_tp${TP_SIZE}_pp${PP_SIZE}_cp${CP_SIZE}_ep${EP_SIZE}_etp${ETP_SIZE}_n${ROLLOUT_N}}"
CKPT_DIR="${CKPT_DIR:-${OUTPUT_ROOT}/checkpoints/${RUN_NAME}}"
LOG_FILE="${LOG_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.log}"
JSONL_FILE="${JSONL_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.jsonl}"
CMD_FILE="${CMD_FILE:-${OUTPUT_ROOT}/${RUN_NAME}.cmd.sh}"

mkdir -p "${OUTPUT_ROOT}" "${CKPT_DIR}" "$(dirname "${LOG_FILE}")" "$(dirname "${JSONL_FILE}")" "$(dirname "${CMD_FILE}")"
export VERL_FILE_LOGGER_PATH="${JSONL_FILE}"
export VERL_FILE_LOGGER_ROOT="${OUTPUT_ROOT}"

CACHE_ROOT="${VERLBB_CACHE_ROOT:-${TMPDIR:-/tmp}/verlbb}"
mkdir -p "${CACHE_ROOT}/pycache_${USER:-user}" "${CACHE_ROOT}/torchinductor_${USER:-user}" "${CACHE_ROOT}/triton_${USER:-user}"
export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-${CACHE_ROOT}/pycache_${USER:-user}}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${CACHE_ROOT}/torchinductor_${USER:-user}}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton_${USER:-user}}"

COMMON_ARGS=(
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${VAL_FILE}"
  "data.prompt_key=prompt"
  "data.train_batch_size=${TRAIN_BATCH_SIZE}"
  "data.train_max_samples=${TRAIN_MAX_SAMPLES}"
  "data.val_max_samples=${VAL_MAX_SAMPLES}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.truncation=left"
  "data.return_raw_chat=True"
  "data.trust_remote_code=True"
  "data.dataloader_num_workers=0"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.trust_remote_code=True"
  "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}"
  "actor_rollout_ref.model.use_fused_kernels=${USE_FUSED_KERNELS}"
  "actor_rollout_ref.rollout.name=vllm"
  "actor_rollout_ref.rollout.mode=async"
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}"
  "actor_rollout_ref.rollout.temperature=1.0"
  "actor_rollout_ref.rollout.top_p=1.0"
  "actor_rollout_ref.rollout.top_k=-1"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOG_PROB_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${INFER_MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE}"
  "actor_rollout_ref.rollout.data_parallel_size=${ROLLOUT_DP_SIZE}"
  "actor_rollout_ref.rollout.expert_parallel_size=${ROLLOUT_EP_SIZE}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${ROLLOUT_MAX_NUM_BATCHED_TOKENS}"
  "actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}"
  "actor_rollout_ref.rollout.enable_chunked_prefill=True"
  "actor_rollout_ref.rollout.enable_prefix_caching=True"
  "actor_rollout_ref.rollout.enforce_eager=True"
  "actor_rollout_ref.rollout.free_cache_engine=True"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.val_kwargs.temperature=1.0"
  "actor_rollout_ref.actor.use_dynamic_bsz=${USE_DYNAMIC_BSZ}"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.ppo_epochs=1"
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU}"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}"
  "actor_rollout_ref.actor.optim.lr=${LR}"
  "actor_rollout_ref.actor.optim.weight_decay=${WEIGHT_DECAY}"
  "algorithm.adv_estimator=grpo"
  "algorithm.use_kl_in_reward=False"
  "algorithm.kl_ctrl.kl_coef=0.0"
  "reward.reward_manager.name=naive"
  "trainer.logger=[console,file]"
  "trainer.project_name=${PROJECT_NAME}"
  "trainer.experiment_name=${RUN_NAME}"
  "trainer.default_local_dir=${CKPT_DIR}"
  "trainer.total_epochs=${TOTAL_EPOCHS}"
  "trainer.total_training_steps=${TOTAL_STEPS}"
  "trainer.val_before_train=False"
  "trainer.test_freq=${TEST_FREQ}"
  "trainer.save_freq=${SAVE_FREQ}"
  "trainer.resume_mode=${RESUME_MODE}"
  "trainer.resume_from_path=${RESUME_FROM_PATH}"
  "trainer.nnodes=${NNODES}"
  "trainer.n_gpus_per_node=${NUM_GPUS}"
  "trainer.use_legacy_worker_impl=disable"
)

if [[ "${BACKEND}" == "megatron" ]]; then
  BACKEND_ARGS=(
    "model_engine=megatron"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.megatron.dtype=${DTYPE}"
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${TP_SIZE}"
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${PP_SIZE}"
    "actor_rollout_ref.actor.megatron.virtual_pipeline_model_parallel_size=${VPP_SIZE}"
    "actor_rollout_ref.actor.megatron.context_parallel_size=${CP_SIZE}"
    "actor_rollout_ref.actor.megatron.expert_model_parallel_size=${EP_SIZE}"
    "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${ETP_SIZE}"
    "actor_rollout_ref.actor.megatron.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.megatron.optimizer_offload=${OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.actor.megatron.grad_offload=${GRAD_OFFLOAD}"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${OPTIMIZER_STATE_OFFLOAD_FRACTION}"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=${USE_PRECISION_AWARE_OPTIMIZER}"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${OPTIMIZER_CPU_OFFLOAD}"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.decoupled_weight_decay=${DECOUPLED_WEIGHT_DECAY}"
    "actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend=${ATTENTION_BACKEND}"
  )
else
  BACKEND_ARGS=(
    "hydra.searchpath=[pkg://verlbb.config]"
    "actor@actor_rollout_ref.actor=bumblebee_actor"
    "actor_rollout_ref.actor.use_torch_compile=False"
    "actor_rollout_ref.actor.engine.dtype=${DTYPE}"
    "actor_rollout_ref.actor.engine.impl=${BB_IMPL}"
    "actor_rollout_ref.actor.engine.tp=${TP_SIZE}"
    "actor_rollout_ref.actor.engine.pp=${PP_SIZE}"
    "actor_rollout_ref.actor.engine.vpp=${BB_VPP_SIZE}"
    "actor_rollout_ref.actor.engine.cp=${CP_SIZE}"
    "actor_rollout_ref.actor.engine.ep=${EP_SIZE}"
    "actor_rollout_ref.actor.engine.etp=${ETP_SIZE}"
    "actor_rollout_ref.actor.engine.param_offload=${PARAM_OFFLOAD}"
    "actor_rollout_ref.actor.engine.optimizer_offload=${OPTIMIZER_OFFLOAD}"
    "actor_rollout_ref.actor.engine.grad_offload=${GRAD_OFFLOAD}"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.offload_fraction=${OPTIMIZER_STATE_OFFLOAD_FRACTION}"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=${USE_PRECISION_AWARE_OPTIMIZER}"
    "+actor_rollout_ref.actor.optim.override_optimizer_config.decoupled_weight_decay=${DECOUPLED_WEIGHT_DECAY}"
    "actor_rollout_ref.actor.engine.attention_backend_override=${ATTENTION_BACKEND}"
    "actor_rollout_ref.actor.engine.impl_cfg.use_thd=True"
  )
fi

COMMAND=(
  python3
  -m
  verl.trainer.main_ppo
  "${COMMON_ARGS[@]}"
  "${BACKEND_ARGS[@]}"
  "${EXTRA_ARGS[@]}"
)

printf '%q ' "${COMMAND[@]}" > "${CMD_FILE}"
printf '\n' >> "${CMD_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi

echo "[${BACKEND}] output_root=${OUTPUT_ROOT}"
echo "[${BACKEND}] log=${LOG_FILE}"
echo "[${BACKEND}] jsonl=${JSONL_FILE}"
echo "[${BACKEND}] cmd=${CMD_FILE}"

set +e
"${COMMAND[@]}" 2>&1 | tee "${LOG_FILE}"
cmd_rc="${PIPESTATUS[0]}"
set -e
exit "${cmd_rc}"
