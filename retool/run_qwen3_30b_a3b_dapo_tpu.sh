#!/usr/bin/env bash
# ReTool Stage 2 (Multi-Turn DAPO RL with Code Interpreter) | Qwen3-30B-A3B (MoE)
# TorchTitan Trainer (SimpleFSDP=16, CP=1, EP=1) + 2x Async vLLM Rollout Replicas (2 x TP=16, tpu-sync Weight Sync)
# TPU v5p (1x Trainer + 2x Rollout 2x2x4 Slices = 48 Chips)
#
# Usage (from a clean verl checkout root):
#   bash recipe/retool/run_qwen3_30b_a3b_dapo_tpu.sh

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# If invoked on a workstation/login host where the active python3 does not have Hydra/Ray Worker installed,
# automatically port-forward to the target GKE Ray cluster and submit this script via `ray job submit`.
if [[ "${RAY_JOB_SUBMITTED:-0}" != "1" ]] && ! python3 -m hydra --help >/dev/null 2>&1; then
    set +x
    if [[ -f "${VERL_ROOT}/.env" ]]; then
        # shellcheck disable=SC1091
        source "${VERL_ROOT}/.env"
    fi
    RAY_BIN="$(command -v ray || echo "${HOME}/tunix_env/bin/ray")"
    if [[ ! -x "${RAY_BIN}" ]]; then
        echo "ERROR: 'ray' CLI not found on PATH or at ${HOME}/tunix_env/bin/ray" >&2
        exit 1
    fi
    PF_PID=""
    cleanup_pf() {
        if [[ -n "${PF_PID}" ]] && kill -0 "${PF_PID}" 2>/dev/null; then
            kill "${PF_PID}" 2>/dev/null || true
            wait "${PF_PID}" 2>/dev/null || true
        fi
    }
    trap cleanup_pf EXIT INT TERM

    if [[ -z "${RAY_ADDRESS:-}" ]]; then
        export KUBECONFIG="${KUBECONFIG:-${HOME}/.kube/kubeconfig_bodaborg_v5p_nap}"
        RAY_NAMESPACE="${RAY_NAMESPACE:-trellis}"
        RAY_CLUSTER_NAME="${RAY_CLUSTER_NAME:-verl-nap-aaskhat-retool}"
        RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-$((28300 + RANDOM % 500))}"
        kubectl port-forward -n "${RAY_NAMESPACE}" "svc/${RAY_CLUSTER_NAME}-head-svc" "${RAY_DASHBOARD_PORT}:8265" >/dev/null 2>&1 &
        PF_PID=$!
        for _ in {1..20}; do
            if curl -fsS --max-time 2 "http://127.0.0.1:${RAY_DASHBOARD_PORT}/api/version" >/dev/null 2>&1; then
                break
            fi
            sleep 1
        done
        RAY_ADDRESS="http://127.0.0.1:${RAY_DASHBOARD_PORT}"
    fi

    export RAY_JOB_SUBMITTED=1
    export VERL_ROOT
    RUNTIME_ENV_JSON="$(python3 - <<'PY'
import json
import os

forward_keys = [
    "RAY_JOB_SUBMITTED",
    "TPU_ACCELERATOR_TYPE",
    "NNODES_TRAINER",
    "N_CHIPS_TRAINER",
    "NNODES_ROLLOUT",
    "N_CHIPS_ROLLOUT",
    "ROLLOUT_TP",
    "ROLLOUT_EP",
    "ROLLOUT_MAX_NUM_SEQS",
    "ROLLOUT_GPU_MEMORY_UTILIZATION",
    "ROLLOUT_LOG_STATS",
    "CONTEXT_PARALLEL_SIZE",
    "ACTOR_MAX_TOKEN_LEN_PER_GPU",
    "LOG_PROB_MAX_TOKEN_LEN_PER_GPU",
    "ENABLE_THINKING",
    "MAX_PROMPT_LENGTH",
    "MAX_RESPONSE_LENGTH",
    "MAX_TURNS",
    "N_RESP_PER_PROMPT",
    "N_RESP_PER_PROMPT_VAL",
    "ACTOR_LR",
    "CLIP_GRAD",
    "TRAIN_BATCH_SIZE",
    "PPO_MINI_BATCH_SIZE",
    "TOTAL_TRAINING_STEPS",
    "TOTAL_EPOCHS",
    "TEST_FREQ",
    "SAVE_FREQ",
    "SAVE_CONTENTS",
    "RESUME_MODE",
    "VAL_BEFORE_TRAIN",
    "VAL_ONLY",
    "RAY_DATA_HOME",
    "DATA_ROOT",
    "SFT_CKPT_ROOT",
    "SFT_EXPERIMENT_NAME",
    "MODEL_PATH",
    "CKPTS_DIR",
    "ROLLOUT_DATA_DIR",
    "VALIDATION_DATA_DIR",
    "PROJECT_NAME",
    "EXPERIMENT_NAME",
    "LOGGER_LIST",
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "WANDB_PROJECT",
    "WANDB_MODE",
]
env_vars = {
    "PYTHONPATH": ".",
    "PYTHONUNBUFFERED": "1",
    "VERL_PLATFORM": "tpu",
    "VERL_USE_EXTERNAL_MODULES": "verl_hardware_plugin",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
}
for k in forward_keys:
    val = os.environ.get(k)
    if val is not None and val != "":
        env_vars[k] = val
for k, v in os.environ.items():
    if k.startswith("VERL_TPU_") and v != "":
        env_vars[k] = v

py_modules = []
for candidate in (
    os.path.join(os.environ.get("VERL_ROOT", "."), "..", "verl-hardware-plugin", "verl_hardware_plugin"),
    os.path.expanduser("~/verl-hardware-plugin/verl_hardware_plugin"),
):
    if os.path.isdir(candidate):
        py_modules.append(os.path.abspath(candidate))
        break

runtime_env = {
    "env_vars": env_vars,
    "excludes": [".git", "recipe/.git", "tests", "docs", ".github", "tasks"],
}
if py_modules:
    runtime_env["py_modules"] = py_modules
print(json.dumps(runtime_env))
PY
)"
    echo "Submitting recipe/retool/$(basename "${BASH_SOURCE[0]}") to Ray cluster at ${RAY_ADDRESS}..."
    "${RAY_BIN}" job submit \
        --address "${RAY_ADDRESS}" \
        --working-dir "${VERL_ROOT}" \
        --runtime-env-json "${RUNTIME_ENV_JSON}" \
        -- bash "recipe/retool/$(basename "${BASH_SOURCE[0]}")" "$@"
    exit $?
fi

export VERL_PLATFORM=tpu
export VERL_USE_EXTERNAL_MODULES=verl_hardware_plugin
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v5p}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

RECIPE_LIBTPU_FLAGS=("--xla_tpu_use_enhanced_launch_barrier=false")
if [[ "${TPU_ACCELERATOR_TYPE}" == "v5p" ]]; then
    RECIPE_LIBTPU_FLAGS+=("--xla_tpu_scoped_vmem_limit_kib=32768")
fi
for flag in "${RECIPE_LIBTPU_FLAGS[@]}"; do
    if [[ " ${LIBTPU_INIT_ARGS:-} " != *" ${flag%%=*}="* ]]; then
        LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+${LIBTPU_INIT_ARGS} }${flag}"
    fi
done
export LIBTPU_INIT_ARGS

set +x
if [[ -f "${VERL_ROOT}/.env" ]]; then
    # shellcheck disable=SC1091
    source "${VERL_ROOT}/.env"
fi
set -x
if [[ "${WANDB_BASE_URL:-}" == *"forge.coreweave.com"* ]]; then
    unset WANDB_BASE_URL
fi
export WANDB_BASE_URL="${WANDB_BASE_URL:-https://api.wandb.ai}"
export WANDB_ENTITY="${WANDB_ENTITY:-google-trellis}"
export WANDB_PROJECT="${WANDB_PROJECT:-verl-torchtpu-grpo-test}"
export WANDB_MODE="${WANDB_MODE:-online}"

# ================= data/model/tool =================
RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
DATA_ROOT="${DATA_ROOT:-${RAY_DATA_HOME}}"
SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-/data/aaskhat}"

dapo_math_17k="${DATA_ROOT}/dataset/BytedTsinghua-SIA/DAPO-Math-17k"
aime_2024="${DATA_ROOT}/dataset/Maxwell-Jia/AIME_2024"
aime_2025="${DATA_ROOT}/dataset/yentinglin/aime_2025"

sft_experiment_name="${SFT_EXPERIMENT_NAME:-multiturn-sft-qwen3-30b-a3b-v5p}"
DEFAULT_SFT_SRC="${SFT_CKPT_ROOT}/checkpoint/${sft_experiment_name}/global_step_372/huggingface"
model_path="${MODEL_PATH:-${DEFAULT_SFT_SRC}}"

train_files="['${dapo_math_17k}']"
test_files="['${aime_2025}', '${aime_2024}']"

retool_py_path="pkg://recipe.retool.retool"

project_name="${PROJECT_NAME:-${WANDB_PROJECT:-verl-torchtpu-grpo-test}}"
experiment_name="${EXPERIMENT_NAME:-qwen3-30b-a3b_retool_dapo_tpu}"
default_local_dir="${CKPTS_DIR:-/tmp/checkpoint/${experiment_name}}"

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss="${USE_KL_LOSS:-False}"
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns="${MAX_TURNS:-16}"
max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-16384}"
max_model_len=$((max_prompt_length + max_response_length))
actor_lr="${ACTOR_LR:-2e-6}"
min_lr_factor="${MIN_LR_FACTOR:-1.0}"
clip_grad="${CLIP_GRAD:-0.0625}"

train_batch_size="${TRAIN_BATCH_SIZE:-64}"
ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-64}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-16}"
n_resp_per_prompt_val="${N_RESP_PER_PROMPT_VAL:-8}"

# ================= TPU v5p slice topology & MoE parallelism =================
export NNODES_TRAINER="${NNODES_TRAINER:-4}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-4}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-8}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-4}"

TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))

# 2 x 2x2x4 TPU v5p rollout slices -> 2 vLLM replicas with TP=16, EP=16.
ROLLOUT_TP="${ROLLOUT_TP:-16}"
ROLLOUT_EP="${ROLLOUT_EP:-${ROLLOUT_TP}}"
ROLLOUT_BLOCK_SIZE="${ROLLOUT_BLOCK_SIZE:-256}"
ROLLOUT_ENABLE_PREFIX_CACHING="${ROLLOUT_ENABLE_PREFIX_CACHING:-True}"
PARAMETER_SYNC_STEP="${PARAMETER_SYNC_STEP:-$((train_batch_size / ppo_mini_batch_size))}"
NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-2}"
MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-4}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-16}"

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
TRAINER_EP="${TRAINER_EP:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
DEFAULT_DP_SHARD=$((TOTAL_TRAINER_CHIPS / (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-${DEFAULT_DP_SHARD}}"

USE_SIMPLE_FSDP="${USE_SIMPLE_FSDP:-True}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-True}"
PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-2048}"
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-20480}"
LOG_PROB_MAX_TOKEN_LEN_PER_GPU="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-20480}"
actor_max_token_len_per_gpu="${ACTOR_MAX_TOKEN_LEN_PER_GPU}"
log_prob_max_token_len_per_gpu="${LOG_PROB_MAX_TOKEN_LEN_PER_GPU}"
torchtitan_max_seq_len=$(( (actor_max_token_len_per_gpu > log_prob_max_token_len_per_gpu ? actor_max_token_len_per_gpu : log_prob_max_token_len_per_gpu) * CONTEXT_PARALLEL_SIZE ))
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_NEVER}"
CHECKPOINT_BACKEND="${CHECKPOINT_BACKEND:-tpu-sync}"
VERIFY_PARITY="${VERIFY_PARITY:-off}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.85}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-512}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-4096}"
ROLLOUT_LOG_STATS="${ROLLOUT_LOG_STATS:-True}"
if [[ "${ROLLOUT_LOG_STATS}" == "True" || "${ROLLOUT_LOG_STATS}" == "true" ]]; then
    ROLLOUT_DISABLE_LOG_STATS=False
else
    ROLLOUT_DISABLE_LOG_STATS=True
fi

# Off-policy correction & bypass_mode for separate_async overlap
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-2.0}"
BYPASS_MODE="${BYPASS_MODE:-True}"
LOSS_ARGS=("algorithm.rollout_correction.bypass_mode=${BYPASS_MODE}")
if [[ "${BYPASS_MODE}" == "True" || "${BYPASS_MODE}" == "true" ]]; then
    LOSS_ARGS+=("actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode")
fi

VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
VAL_ONLY="${VAL_ONLY:-False}"
if [[ "${VAL_ONLY}" == "True" || "${VAL_ONLY}" == "true" ]]; then
    VAL_BEFORE_TRAIN=True
fi
SAVE_FREQ="${SAVE_FREQ:-10}"
MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-2}"
TEST_FREQ="${TEST_FREQ:-25}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
TRUNCATION="${TRUNCATION:-left}"
LOGGER_LIST="${LOGGER_LIST:-['console','wandb']}"
ENABLE_THINKING="${ENABLE_THINKING:-True}"
RESUME_MODE="${RESUME_MODE:-auto}"
SAVE_CONTENTS="${SAVE_CONTENTS:-model,optimizer,extra}"

EXTRA_ARGS=()
if [[ -n "${ROLLOUT_DATA_DIR:-}" ]]; then
    EXTRA_ARGS+=("trainer.rollout_data_dir=${ROLLOUT_DATA_DIR}")
fi
if [[ -n "${VALIDATION_DATA_DIR:-}" ]]; then
    EXTRA_ARGS+=("trainer.validation_data_dir=${VALIDATION_DATA_DIR}")
fi
if [[ "${TPU_ACCELERATOR_TYPE}" == "v5p" ]]; then
    export USE_MOE_SPARSE_CORE="${USE_MOE_SPARSE_CORE:-0}"
    export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:+${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY},}USE_MOE_SPARSE_CORE"
    EXTRA_ARGS+=(
        "+ray_kwargs.ray_init.runtime_env.env_vars.USE_MOE_SPARSE_CORE='${USE_MOE_SPARSE_CORE}'"
        "+ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RAY_EXTRA_ENV_VARS_TO_COPY='${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY}'"
    )
fi

python3 -m verl.trainer.main_ppo \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=separate_async \
    trainer.v1.separate_async.num_warmup_batches="${NUM_WARMUP_BATCHES}" \
    trainer.v1.separate_async.parameter_sync_step="${PARAMETER_SYNC_STEP}" \
    trainer.v1.sampler.max_off_policy_threshold="${MAX_OFF_POLICY_THRESHOLD}" \
    transfer_queue.enable=True \
    model_engine=torchtitan \
    algorithm.adv_estimator="${adv_estimator}" \
    algorithm.use_kl_in_reward="${use_kl_in_reward}" \
    algorithm.kl_ctrl.kl_coef="${kl_coef}" \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    "${LOSS_ARGS[@]}" \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.return_raw_chat=True \
    data.train_batch_size="${train_batch_size}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    +data.max_length="${max_model_len}" \
    +data.max_token_len_per_gpu="${actor_max_token_len_per_gpu}" \
    data.filter_overlong_prompts=False \
    data.truncation="${TRUNCATION}" \
    +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}" \
    +data.pad_mode=no_padding \
    data.custom_cls.path="${retool_py_path}" \
    data.custom_cls.name=CustomRLHFDataset \
    reward.custom_reward_function.path="${retool_py_path}" \
    reward.custom_reward_function.name=compute_score \
    actor_rollout_ref.actor.strategy=torchtitan \
    "actor_rollout_ref.actor.checkpoint.save_contents=[${SAVE_CONTENTS}]" \
    actor_rollout_ref.model.path="${model_path}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss="${use_kl_loss}" \
    actor_rollout_ref.actor.kl_loss_coef="${kl_loss_coef}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low="${clip_ratio_low}" \
    actor_rollout_ref.actor.clip_ratio_high="${clip_ratio_high}" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr="${actor_lr}" \
    actor_rollout_ref.actor.optim.min_lr_factor="${min_lr_factor}" \
    actor_rollout_ref.actor.optim.clip_grad="${clip_grad}" \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${ppo_mini_batch_size}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${actor_max_token_len_per_gpu}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${log_prob_max_token_len_per_gpu}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${log_prob_max_token_len_per_gpu}" \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.actor.torchtitan.use_torch_compile=True \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_splash_attention=True \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.tpu_eager_mode="${TPU_EAGER_MODE}" \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_simple_fsdp="${USE_SIMPLE_FSDP}" \
    +actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.use_fused_linear_cross_entropy=True \
    actor_rollout_ref.actor.torchtitan.pad_to_length=True \
    actor_rollout_ref.actor.torchtitan.pad_to_length_bucket="${PAD_TO_LENGTH_BUCKET}" \
    actor_rollout_ref.actor.torchtitan.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.expert_parallel_size="${TRAINER_EP}" \
    actor_rollout_ref.actor.torchtitan.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
    actor_rollout_ref.actor.torchtitan.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    actor_rollout_ref.actor.torchtitan.pipeline_parallel_size=1 \
    actor_rollout_ref.actor.torchtitan.attn_type=varlen \
    actor_rollout_ref.actor.torchtitan.activation_checkpoint=full \
    actor_rollout_ref.actor.torchtitan.max_seq_len="${torchtitan_max_seq_len}" \
    actor_rollout_ref.ref.torchtitan.use_torch_compile=True \
    actor_rollout_ref.ref.torchtitan.max_seq_len="${torchtitan_max_seq_len}" \
    actor_rollout_ref.ref.torchtitan.pad_to_length=True \
    actor_rollout_ref.ref.torchtitan.pad_to_length_bucket="${PAD_TO_LENGTH_BUCKET}" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
    actor_rollout_ref.rollout.enable_prefix_caching="${ROLLOUT_ENABLE_PREFIX_CACHING}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.expert_parallel_size="${ROLLOUT_EP}" \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.block_size="${ROLLOUT_BLOCK_SIZE}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.disable_log_stats="${ROLLOUT_DISABLE_LOG_STATS}" \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.checkpoint_engine.backend="${CHECKPOINT_BACKEND}" \
    "+actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.${CHECKPOINT_BACKEND}.verify_parity=${VERIFY_PARITY}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns="${max_turns}" \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${max_turns}" \
    actor_rollout_ref.rollout.multi_turn.function_tool_path="${retool_py_path}" \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.n="${n_resp_per_prompt}" \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n="${n_resp_per_prompt_val}" \
    trainer.logger="${LOGGER_LIST}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    trainer.nnodes="${NNODES_TRAINER}" \
    actor_rollout_ref.rollout.nnodes="${NNODES_ROLLOUT}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    +rollout.nnodes="${NNODES_ROLLOUT}" \
    +rollout.n_gpus_per_node="${N_CHIPS_ROLLOUT}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.val_only="${VAL_ONLY}" \
    trainer.log_val_generations=20 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.max_actor_ckpt_to_keep="${MAX_CKPT_TO_KEEP}" \
    trainer.resume_mode="${RESUME_MODE}" \
    trainer.default_local_dir="${default_local_dir}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
