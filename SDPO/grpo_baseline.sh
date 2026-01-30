#!/usr/bin/env bash
# GRPO Baseline Training Script (for comparison with SDPO)
#
# This script trains Qwen3-VL-4B on geo3k using GRPO with async 2-GPU setup:
# - 1 GPU for rollout (vLLM)
# - 1 GPU for training (FSDP2)
#
# Usage:
#   bash grpo_baseline.sh [ENGINE]
#
# Arguments:
#   ENGINE: Rollout engine to use (default: vllm)
#
# Environment Variables:
#   HF_MODEL_PATH: Path to the model (default: Qwen/Qwen3-VL-4B-Instruct)
#   TRAIN_FILE: Path to training data (default: $HOME/data/geo3k/train.parquet)
#   TEST_FILE: Path to test data (default: $HOME/data/geo3k/test.parquet)
#   SMOKE_TEST: Set to 1 for smoke test mode with minimal iterations

set -x
ENGINE=${1:-vllm}

# ==============================================================================
# Model and Data Configuration
# ==============================================================================

HF_MODEL_PATH=${HF_MODEL_PATH:-"Qwen/Qwen3-VL-4B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"$HOME/data/geo3k/train.parquet"}
TEST_FILE=${TEST_FILE:-"$HOME/data/geo3k/test.parquet"}

# ==============================================================================
# Training Configuration
# ==============================================================================

# Smoke test mode - minimal iterations for verification
if [ "${SMOKE_TEST:-0}" = "1" ]; then
    echo "=== SMOKE TEST MODE ==="
    TOTAL_EPOCHS=1
    TRAIN_BATCH_SIZE=4
    PPO_MINI_BATCH_SIZE=2
    PPO_MICRO_BATCH_SIZE=1
    N_RESP_PER_PROMPT=2
    TEST_FREQ=1
    SAVE_FREQ=-1  # Don't save checkpoints in smoke test
    MAX_PROMPT_LENGTH=512
    MAX_RESPONSE_LENGTH=512
else
    TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
    TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
    PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
    PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-4}
    N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-5}
    TEST_FREQ=${TEST_FREQ:-5}
    SAVE_FREQ=${SAVE_FREQ:-20}
    MAX_PROMPT_LENGTH=1024
    MAX_RESPONSE_LENGTH=2048
fi

# ==============================================================================
# Resource Configuration (2 GPUs: 1 training, 1 rollout)
# ==============================================================================

NNODES=1
ROLLOUT_N_GPUS=1
TRAINER_N_GPUS=1
ROLLOUT_TP=1

# ==============================================================================
# Run Training
# ==============================================================================

# Get the directory of this script for config path
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

python3 -m verl.experimental.one_step_off_policy.main_ppo \
    --config-path="${SCRIPT_DIR}/config" \
    --config-name='grpo_baseline.yaml' \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=$MAX_RESPONSE_LENGTH \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$HF_MODEL_PATH" \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$N_RESP_PER_PROMPT \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.strategy=fsdp2 \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=True \
    algorithm.rollout_correction.rollout_is=sequence \
    algorithm.rollout_correction.rollout_is_threshold=2.0 \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='grpo_baseline_geo3k' \
    trainer.experiment_name='qwen3_vl_4b_grpo_async' \
    trainer.nnodes=$NNODES \
    trainer.n_gpus_per_node=$TRAINER_N_GPUS \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$TEST_FREQ \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=True \
    rollout.nnodes=$NNODES \
    rollout.n_gpus_per_node=$ROLLOUT_N_GPUS \
    "$@"
