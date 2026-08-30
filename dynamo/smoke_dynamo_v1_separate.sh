#!/usr/bin/env bash
set -xeuo pipefail

# Two-GPU V1 separate_async smoke for the Dynamo rollout path.
#
# Split placement on one node: trainer = 1 GPU (hybrid dynamo pool, slept
# outside validation), standalone rollout = 1 GPU (CheckpointEngineWorker +
# dynamo stack). Exercises per step:
#   on_sample_end: switch_to_trainer (hybrid abort + sleep)
#   train step (Decoupled PPO across parameter_sync_step mini steps)
#   on_step_end:   standalone update_weights — abort -> release_kv ->
#                  nccl first hop -> CUDA-IPC second hop -> resume_kv -> resume
#
# Requires: verl >= REQUIRED_VERL.txt pin, TransferQueue, cupy-cuda12x (the
# nccl checkpoint-engine backend registers only when cupy imports), recipe
# mounted as recipe/dynamo under the verl repo root.

project_name=${PROJECT_NAME:-verl-dynamo}
exp_name=${EXP_NAME:-dynamo-v1-separate-smoke}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
ROLLOUT_NNODES=${ROLLOUT_NNODES:-1}
ROLLOUT_NGPUS_PER_NODE=${ROLLOUT_NGPUS_PER_NODE:-1}
TOTAL_STEPS=${TOTAL_STEPS:-2}
PARAMETER_SYNC_STEP=${PARAMETER_SYNC_STEP:-2}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-1}
TRAIN_BATCH_SIZE=$((PARAMETER_SYNC_STEP * PPO_MINI_BATCH_SIZE))
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register

# bypass_mode: use rollout logprobs as old_log_probs directly (matches the
# uni-agent claude_code recipe). The Decoupled save/restore alternative is
# DTensor-based and a 1-GPU trainer's FSDP2 wrap yields no DTensor params —
# multi-GPU runs may drop this and exercise Decoupled PPO instead.
python3 -m recipe.dynamo.main_dynamo \
    --config-name=dynamo_trainer_v1_separate \
    algorithm.rollout_correction.bypass_mode=true \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.val_batch_size=1 \
    data.train_max_samples="${TRAIN_BATCH_SIZE}" \
    data.val_max_samples=1 \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.name=dynamo \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES}" \
    actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_NGPUS_PER_NODE}" \
    trainer.v1.separate_async.parameter_sync_step="${PARAMETER_SYNC_STEP}" \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=round-robin \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.total_epochs=100 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    "$@"

echo "PASS: Dynamo V1 separate_async smoke completed"
