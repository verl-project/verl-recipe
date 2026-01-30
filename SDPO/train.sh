#!/usr/bin/env bash
# SDPO (Self-Distillation Policy Optimization) Training Script
#
# This script trains Qwen models on geo3k using SDPO with 2 GPUs:
# - 1 GPU for training (actor)
# - 1 GPU for rollout (vLLM)
#
# Usage:
#   bash train.sh [ENGINE]
#
# Arguments:
#   ENGINE: Rollout engine to use (default: vllm)
#
# Environment Variables:
#   HF_MODEL_PATH: Path to the model (default: Qwen/Qwen2.5-VL-3B-Instruct)
#   TRAIN_FILE: Path to training data (default: $HOME/data/geo3k/train.parquet)
#   TEST_FILE: Path to test data (default: $HOME/data/geo3k/test.parquet)
#   SMOKE_TEST: Set to 1 for smoke test mode with minimal iterations

set -x
ENGINE=${1:-vllm}
export CUDA_DEVICE_MAX_CONNECTIONS=1  # For megatron communication/computation overlapping

# ==============================================================================
# Model and Data Configuration
# ==============================================================================

HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen/Qwen2.5-VL-3B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/geo3k/train.parquet"}
TEST_FILE=${TEST_FILE:-"$HOME/data/geo3k/test.parquet"}

# ==============================================================================
# Training Configuration
# ==============================================================================

# Smoke test mode - minimal iterations for verification
if [ "${SMOKE_TEST:-0}" = "1" ]; then
    echo "=== SMOKE TEST MODE ==="
    TOTAL_TRAIN_STEPS=3
    TRAIN_BATCH_SIZE=4
    PPO_MINI_BATCH_SIZE=2
    VAL_BATCH_SIZE=4
    N_RESP_PER_PROMPT=1
    TEST_FREQ=2
    SAVE_FREQ=0  # Don't save checkpoints in smoke test
else
    TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-100}
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
    PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
    VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-16}
    N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-1}
    TEST_FREQ=${TEST_FREQ:-10}
    SAVE_FREQ=${SAVE_FREQ:-50}
fi

# Rollout settings
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096
TEMPERATURE=0.7
TOP_P=0.9

# SDPO settings
EMA_UPDATE_RATE=${EMA_UPDATE_RATE:-0.05}
IS_CLIP=${IS_CLIP:-2.0}
ALPHA=${ALPHA:-0.0}  # 0.0 = forward KL

# ==============================================================================
# Resource Configuration (2 GPUs: 1 training, 1 rollout)
# ==============================================================================

ROLLOUT_N_GPUS=1
TRAINER_N_GPUS=1
ROLLOUT_TP=1
ACTOR_TP=1

# ==============================================================================
# Run Training
# ==============================================================================

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VERL_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

python3 -m verl.experimental.one_step_off_policy.main_ppo \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name='sdpo_trainer.yaml' \
    trainer.n_gpus_per_node=$TRAINER_N_GPUS \
    rollout.n_gpus_per_node=$ROLLOUT_N_GPUS \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.temperature=$TEMPERATURE \
    actor_rollout_ref.rollout.top_p=$TOP_P \
    actor_rollout_ref.rollout.n=$N_RESP_PER_PROMPT \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.rollout_is=sequence \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    trainer.total_training_steps=$TOTAL_TRAIN_STEPS \
    trainer.test_freq=$TEST_FREQ \
    trainer.save_freq=$SAVE_FREQ \
    trainer.project_name="sdpo_geo3k" \
    trainer.experiment_name="sdpo_${ENGINE}" \
    trainer.logger=console \
    ++trainer.val_before_train=True
