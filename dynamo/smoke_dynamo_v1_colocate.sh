#!/usr/bin/env bash
set -xeuo pipefail

# One-GPU V1 colocate_async smoke for the Dynamo rollout path.
#
# Unlike smoke_dynamo_v1.sh (val-only), this runs REAL training steps to
# exercise the full colocate_async choreography per step:
#   on_sample_end: abort_all_requests (vLLM pause) -> sleep (free weights+KV)
#   train step
#   on_step_end:   update_weights (CUDA-IPC) -> resume_generation
#
# Requires: verl >= REQUIRED_VERL.txt pin (V1 trainer), TransferQueue
# (pip install TransferQueue), and the recipe mounted as recipe/dynamo under
# the verl repo root (hydra searchpath is CWD-relative).

project_name=${PROJECT_NAME:-verl-dynamo}
exp_name=${EXP_NAME:-dynamo-v1-colocate-smoke}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
TOTAL_STEPS=${TOTAL_STEPS:-2}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register

python3 -m recipe.dynamo.main_dynamo \
    --config-name=dynamo_trainer_v1_colocate \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=2 \
    data.val_batch_size=1 \
    data.train_max_samples=2 \
    data.val_max_samples=1 \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
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

echo "PASS: Dynamo V1 colocate_async smoke completed"
