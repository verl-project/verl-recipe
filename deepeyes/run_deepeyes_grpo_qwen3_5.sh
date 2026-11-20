#!/usr/bin/env bash
# DeepEyes GRPO on Ascend NPU with vLLM-Ascend.
#   4B: run_deepeyes_grpo_qwen3_5_4b.sh
#   9B: run_deepeyes_grpo_qwen3_5_9b.sh
# Requires DeepEyes parquet files (scripts/prepare_data.py) and an
# OpenAI-compatible judge server. All parameters are environment overridable.
set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"

########################### user-adjustable ###########################
MODEL_ID=${MODEL_ID:-Qwen/Qwen3.5-4B}
MODEL_PATH=${MODEL_PATH:-"${HOME}/models/${MODEL_ID##*/}"}
DATASET_DIR=${DATASET_DIR:-"${HOME}/data/deepeyes"}
TRAIN_FILE=${TRAIN_FILE:-"${DATASET_DIR}/train.parquet"}
VAL_FILE=${VAL_FILE:-"${DATASET_DIR}/val.parquet"}

# OpenAI-compatible judge endpoint (must expose /v1/models).
export LLM_AS_A_JUDGE_BASE=${LLM_AS_A_JUDGE_BASE:-""}

PROJECT_NAME=${PROJECT_NAME:-DeepEyes-Qwen3_5}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-DeepEyes-${MODEL_ID##*/}}
CKPTS_DIR=${CKPTS_DIR:-"${HOME}/ckpts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/logs/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-2}
# -1 shards over all training devices.
FSDP_SIZE=${FSDP_SIZE:--1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
ROLLOUT_N=${ROLLOUT_N:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-8192}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-16384}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-32768}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
SAVE_FREQ=${SAVE_FREQ:-8}
TEST_FREQ=${TEST_FREQ:-80}
LOGGER=${LOGGER:-"['console','tensorboard']"}
# Prompt filtering renders every sample; one process is too slow for 47k rows.
FILTER_WORKERS=${FILTER_WORKERS:-32}
# Interpreter with verl and vllm-ascend installed.
PYTHON=${PYTHON:-python3}
########################### end user-adjustable ###########################

########################### Ascend environment ###########################
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1500}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_HOST_SOCKET_PORT_RANGE=${HCCL_HOST_SOCKET_PORT_RANGE:-60000-60050}
export HCCL_NPU_SOCKET_PORT_RANGE=${HCCL_NPU_SOCKET_PORT_RANGE:-61000-61050}
# With NOSET=1 verl calls set_device(physical_id), so the visible list must start
# at 0. For a device subset set RAY_NOSET_VISIBLE_DEVICES=0 and GEN_TP=1.
RAY_NOSET_VISIBLE_DEVICES=${RAY_NOSET_VISIBLE_DEVICES:-1}
if [[ "${RAY_NOSET_VISIBLE_DEVICES}" == "1" ]]; then
    export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
else
    unset RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES || true
fi
# Dynamo tracing and NZ layout are not needed on vLLM-Ascend.
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export VLLM_ASCEND_ENABLE_NZ=${VLLM_ASCEND_ENABLE_NZ:-0}
# More HCCL streams in ACL-graph mode.
export HCCL_OP_EXPANSION_MODE=${HCCL_OP_EXPANSION_MODE:-AIV}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-"${LOG_DIR}/tensorboard"}

if [[ ! -d "${MODEL_PATH}" ]]; then
    echo "Model not found: ${MODEL_PATH}. Download ${MODEL_ID} or set MODEL_PATH." >&2
    exit 1
fi
if [[ ! -f "${TRAIN_FILE}" || ! -f "${VAL_FILE}" ]]; then
    echo "Dataset not found: ${TRAIN_FILE} / ${VAL_FILE}" >&2
    echo "Run: python3 ${SCRIPT_DIR}/scripts/prepare_data.py --output-dir ${DATASET_DIR}" >&2
    exit 1
fi
if [[ "${RAY_NOSET_VISIBLE_DEVICES}" == "1" && -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
    first_device=${ASCEND_RT_VISIBLE_DEVICES%%,*}
    if [[ "${first_device}" != "0" ]]; then
        echo "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES} is not 0-based, so verl's" >&2
        echo "set_device(physical_id) will fail with 107001. Use a 0-based list, or set" >&2
        echo "RAY_NOSET_VISIBLE_DEVICES=0 and GEN_TP=1 for a device subset." >&2
        exit 1
    fi
fi
if [[ -z "${LLM_AS_A_JUDGE_BASE}" ]]; then
    echo "LLM_AS_A_JUDGE_BASE is empty: accuracy reward will stay at 0." >&2
fi
# Reward scoring runs in Ray actors, which do not inherit the shell environment
# of an already started Ray cluster.
JUDGE_ENV=()
for var in LLM_AS_A_JUDGE_BASE LLM_AS_A_JUDGE_MODEL; do
    if [[ -n "${!var:-}" ]]; then
        JUDGE_ENV+=("+ray_kwargs.ray_init.runtime_env.env_vars.${var}=${!var}")
    fi
done
if ! "${PYTHON}" -c "import verl" >/dev/null 2>&1; then
    echo "Interpreter '${PYTHON}' cannot import verl. Activate the training environment or set PYTHON=<path>." >&2
    exit 1
fi

mkdir -p "${LOG_DIR}" "${CKPTS_DIR}"

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
LOG_FILE="${LOG_DIR}/${start_time}.log"

########################### parameter arrays ###########################
DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=${FILTER_WORKERS}
    data.truncation='error'
    data.image_key=images
)

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    # Qwen3.5 GDN kernels have no NPU varlen path.
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_fused_kernels=False
)

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.entropy_coeff=0.0
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=1
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.offload_policy=True
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1
    actor_rollout_ref.ref.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.mode=async
    # Ray actors do not inherit the launch directory; keep the tool config path absolute.
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${SCRIPT_DIR}/configs/image_zoom_in_tool_config.yaml"
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
)

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger="${LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
)

########################### launch ###########################
if ((${#JUDGE_ENV[@]})); then
    set -- "${JUDGE_ENV[@]}" "$@"
fi
PYTHONUNBUFFERED=1 "${PYTHON}" -m verl.trainer.main_ppo \
    --config-path="${SCRIPT_DIR}/configs" \
    --config-name=deepeyes_multiturn_grpo_qwen3_5 \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"
