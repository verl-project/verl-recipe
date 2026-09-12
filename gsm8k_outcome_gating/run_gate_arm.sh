#!/usr/bin/env bash
# ==============================================================================
# Single-arm launcher — GSM8K x Qwen2.5-1.5B-Instruct, GRPO + LoRA(r=32), 1 GPU.
# This is the launcher that produced every number in README.md (machine-specific
# absolute paths replaced by env vars; hydra flags unchanged). Ran on a single
# RTX 5090: hybrid engine, rollout TP=1, no flash-attn (sdpa fallback).
#
# Env knobs:
#   GATE_ROOT     work dir holding data/ models/ logs   (default: ./work)
#   VERL_VENV     venv with the pinned verl installed   (default: $GATE_ROOT/venv)
#   MODEL_PATH    policy model                          (default: $GATE_ROOT/models/qwen25-1.5b-instruct)
#   GPU           CUDA device index                     (default: 0)
#   ADV           advantage estimator                   (default: grpo)
#   LR            actor lr                              (default: 1e-4)
#   MAX_STEPS     total training steps                  (default: 40)
#   GUTIL         vllm gpu_memory_utilization           (default: 0.3)
#   EXP           experiment name / log file stem       (default: gate_grpo)
#   PRECHECK_MAX  refuse to start if GPU already uses more MiB (default: 15000)
#   HF_MIRROR=1   route HF downloads through hf-mirror.com
#   The gate itself (read by the hook / reward inside Ray workers):
#   GATE_SHAPED=1 route the gsm8k reward through gate_shaped_reward.py
#   GATE_MODE     none | std | outcome                  (which arm)
#   GATE_LAMBDA   phantom strength                      (default: 0.30)
#   GATE_PHANTOM  short | long                          (default: short)
#   REALSEED      inject rollout.seed + data.seed       (default: off)
# ==============================================================================
set -xeuo pipefail

ROOT=${GATE_ROOT:-$(pwd)/work}
VENV=${VERL_VENV:-$ROOT/venv}

# ---- keep every cache / tmp off the system disk ----
export TMPDIR=$ROOT/tmp
export HF_HOME=$ROOT/.hfhome
[ -n "${HF_MIRROR:-}" ] && export HF_ENDPOINT=https://hf-mirror.com
export VLLM_CACHE_ROOT=$ROOT/.vllmcache
export TRITON_CACHE_DIR=$ROOT/.triton
export TORCHINDUCTOR_CACHE_DIR=$ROOT/.inductor
export XDG_CACHE_HOME=$ROOT/.xdgcache
export RAY_TMPDIR=$ROOT/tmp
export VLLM_USE_V1=1
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

export CUDA_VISIBLE_DEVICES=${GPU:-0}

# ---- shared-machine precheck: refuse to start on a busy GPU ----
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES}" | tr -d " ")
echo "[precheck] GPU ${CUDA_VISIBLE_DEVICES} used ${used} MiB"
if [ "${used}" -gt "${PRECHECK_MAX:-15000}" ]; then
  echo "[ABORT] GPU ${CUDA_VISIBLE_DEVICES} is busy (${used} MiB). Pick another: GPU=1 bash $0"
  exit 1
fi

source "$VENV/bin/activate"
cd "$ROOT"

MODEL_PATH=${MODEL_PATH:-$ROOT/models/qwen25-1.5b-instruct}
TRAIN_FILE=$ROOT/data/gsm8k/train.parquet
TEST_FILE=$ROOT/data/gsm8k/test.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=${ADV:-grpo} \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-4} \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GUTIL:-0.3} \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.use_v1=False \
    trainer.critic_warmup=0 \
    trainer.logger=[console,tensorboard] \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.project_name=gate_study \
    trainer.experiment_name=${EXP:-gate_grpo} \
    trainer.default_local_dir=$ROOT/ckpts/gate_study \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=${MAX_STEPS:-40} \
    ${REALSEED:+actor_rollout_ref.rollout.seed=$REALSEED} \
    ${REALSEED:+data.seed=$REALSEED} \
    2>&1 | tee "$ROOT/${EXP:-gate_grpo}.log"
