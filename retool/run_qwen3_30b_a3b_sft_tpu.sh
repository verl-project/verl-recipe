#!/usr/bin/env bash
# ReTool Stage 1 (Cold-Start Multi-Turn SFT) | Qwen3-30B-A3B (MoE) | ReTool-SFT
# TorchTitan Engine (SimpleFSDP=32, CP=1, EP=1) | TPU v6e-8 (4x v6e-8 = 32 Chips)
#
# Usage (from a clean verl checkout root):
#   bash recipe/retool/run_qwen3_30b_a3b_sft_tpu.sh

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
        if [[ -z "${KUBECONFIG:-}" ]]; then
            if [[ -f "/tmp/kc_v7x" ]] && KUBECONFIG=/tmp/kc_v7x kubectl get svc -n "${RAY_NAMESPACE:-default}" "${RAY_CLUSTER_NAME:-verl-nap-aaskhat-v7x}-head-svc" >/dev/null 2>&1; then
                export KUBECONFIG="/tmp/kc_v7x"
                RAY_CLUSTER_NAME="${RAY_CLUSTER_NAME:-verl-nap-aaskhat-v7x}"
                export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-tpu7x}"
            elif [[ -f "/tmp/kc_v6e" ]]; then
                export KUBECONFIG="/tmp/kc_v6e"
                RAY_CLUSTER_NAME="${RAY_CLUSTER_NAME:-ray-tpu-v6e-cluster}"
                export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-8}"
            fi
        fi
        RAY_NAMESPACE="${RAY_NAMESPACE:-default}"
        RAY_CLUSTER_NAME="${RAY_CLUSTER_NAME:-ray-tpu-v6e-cluster}"
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
    "CONTEXT_PARALLEL_SIZE",
    "MAX_LENGTH",
    "TRAIN_BATCH_SIZE",
    "TOTAL_EPOCHS",
    "TOTAL_TRAINING_STEPS",
    "TEST_FREQ",
    "SAVE_FREQ",
    "SAVE_CONTENTS",
    "RAY_DATA_HOME",
    "SFT_CKPT_ROOT",
    "TRAIN_DATA",
    "EVAL_DATA",
    "MODEL_PATH",
    "SAVE_PATH",
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
if os.environ.get("VERL_UPLOAD_HARDWARE_PLUGIN", "0") == "1":
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
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-v6e-8}"
export RAY_OVERRIDE_JOB_RUNTIME_ENV=1
export PYTHONUNBUFFERED=1
export RAY_memory_monitor_refresh_ms=0
export RAY_memory_usage_threshold=0.99

RECIPE_LIBTPU_FLAGS=(
    "--xla_tpu_use_enhanced_launch_barrier=false"
    "--xla_tpu_enable_bitcast_reduce_fusion_graduation=false"
    "--xla_tpu_enable_sublane_major_scaling_bitcast_fusion=false"
    "--xla_tpu_bf16_emission_mode=PROMOTE_F32"
)
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

project_name="${PROJECT_NAME:-${WANDB_PROJECT:-verl-torchtpu-grpo-test}}"
experiment_name="${EXPERIMENT_NAME:-multiturn-sft-qwen3-30b-a3b-tpu}"

RAY_DATA_HOME="${RAY_DATA_HOME:-/data/jialei}"
SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-/tmp}"
TRAIN_DATA="${TRAIN_DATA:-${RAY_DATA_HOME}/dataset/ReTool-SFT/data/train-00000-of-00001.parquet}"
EVAL_DATA="${EVAL_DATA:-${RAY_DATA_HOME}/dataset/ReTool-SFT/data/train-00000-of-00001.parquet}"
MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/Qwen3-30B-A3B}"
SAVE_PATH="${SAVE_PATH:-${SFT_CKPT_ROOT}/checkpoint/${experiment_name}}"

# Default TPU topology: 4 hosts x 8 chips = 32 trainer chips on v6e-8, or 1 host x 8 TensorCores on tpu7x 2x2x1.
if [[ -z "${NNODES_TRAINER:-}" ]]; then
    if [[ "${TPU_ACCELERATOR_TYPE}" == v6e* ]]; then
        export NNODES_TRAINER=4
    else
        export NNODES_TRAINER=1
    fi
fi
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-8}"
TOTAL_TRAINER_CHIPS=$((NNODES_TRAINER * N_CHIPS_TRAINER))

TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
EXPERT_PARALLEL_SIZE="${EXPERT_PARALLEL_SIZE:-1}"
CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-$((TOTAL_TRAINER_CHIPS / (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))}"
TPU_EAGER_MODE="${TPU_EAGER_MODE:-DEFER_AND_FUSE}"
OPTIM_IMPL="${OPTIM_IMPL:-foreach}"
USE_SIMPLE_FSDP="${USE_SIMPLE_FSDP:-True}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-2048}"

# Train at 16384 max_length with SimpleFSDP=32, CP=1 on 32 TPU v6e chips
# (6 epochs x 62 steps/epoch at train_batch_size=32 = 372 total steps)
MAX_LENGTH="${MAX_LENGTH:-16384}"
TRUNCATION="${TRUNCATION:-right}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-$((MICRO_BATCH_SIZE_PER_GPU * MAX_LENGTH / CONTEXT_PARALLEL_SIZE))}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-6}"
SAVE_FREQ="${SAVE_FREQ:-186}"
TEST_FREQ="${TEST_FREQ:-62}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-32}"
LOGGER_LIST="${LOGGER_LIST:-['console','wandb']}"
SAVE_CONTENTS="${SAVE_CONTENTS:-hf_model}"
if [[ "${SAVE_CONTENTS}" == "none" ]]; then
    SAVE_CONTENTS=""
fi

EXTRA_ARGS=()
if [[ -n "${TOTAL_TRAINING_STEPS:-}" ]]; then
    EXTRA_ARGS+=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

python3 -m verl.trainer.sft_trainer_ray \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${EVAL_DATA}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    data.max_length="${MAX_LENGTH}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.messages_key=messages \
    data.ignore_input_ids_mismatch=True \
    data.pad_mode=no_padding \
    data.truncation="${TRUNCATION}" \
    data.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    model=hf_model \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=True \
    engine=torchtitan \
    engine.tensor_parallel_size="${TENSOR_PARALLEL_SIZE}" \
    engine.pipeline_parallel_size=1 \
    engine.context_parallel_size="${CONTEXT_PARALLEL_SIZE}" \
    engine.data_parallel_shard_size="${DATA_PARALLEL_SHARD_SIZE}" \
    engine.expert_parallel_size="${EXPERT_PARALLEL_SIZE}" \
    engine.use_torch_compile=True \
    +engine.engine_kwargs.tpu.use_splash_attention=True \
    +engine.engine_kwargs.tpu.use_simple_fsdp="${USE_SIMPLE_FSDP}" \
    +engine.engine_kwargs.tpu.tpu_eager_mode="${TPU_EAGER_MODE}" \
    +engine.engine_kwargs.tpu.optim_impl="${OPTIM_IMPL}" \
    +engine.engine_kwargs.tpu.use_fused_linear_cross_entropy=True \
    engine.pad_to_length=True \
    engine.pad_to_length_bucket="${PAD_TO_LENGTH_BUCKET}" \
    engine.activation_checkpoint=full \
    engine.reshard_after_forward=always \
    engine.attn_type=varlen \
    engine.max_seq_len="${MAX_LENGTH}" \
    optim=torchtitan \
    optim.lr=1e-5 \
    optim.lr_warmup_steps_ratio=0.1 \
    optim.weight_decay=0.01 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_factor=0.1 \
    optim.decay_type=cosine \
    "checkpoint.save_contents=[${SAVE_CONTENTS}]" \
    trainer.default_local_dir="${SAVE_PATH}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${experiment_name}" \
    trainer.logger="${LOGGER_LIST}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.resume_mode=disable \
    trainer.device=tpu \
    trainer.nnodes="${NNODES_TRAINER}" \
    trainer.n_gpus_per_node="${N_CHIPS_TRAINER}" \
    "${EXTRA_ARGS[@]}" \
    "$@"
