#!/usr/bin/env bash
set -xeuo pipefail

# One-GPU, validation-only smoke for the Dynamo rollout path.
project_name=${PROJECT_NAME:-verl-dynamo}
exp_name=${EXP_NAME:-dynamo-v1-smoke}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register

# FLEXKV=1 turns this same colocated smoke into the FlexKV L2 acceptance run.
FLEXKV=${FLEXKV:-0}
flexkv_args=()
if [ "$FLEXKV" = "1" ]; then
  export FLEXKV_CPU_CACHE_GB=${FLEXKV_CPU_CACHE_GB:-16} FLEXKV_ENABLE_MPS=${FLEXKV_ENABLE_MPS:-0}
  flexkv_args=(++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_flexkv=True)
fi
# cold first-load + compile of the model can exceed the 600 s default window
export VERL_DYNAMO_FE_READY_TIMEOUT=${VERL_DYNAMO_FE_READY_TIMEOUT:-2400}

python3 -m recipe.dynamo.main_dynamo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=1 \
    data.val_batch_size=1 \
    data.train_max_samples=1 \
    data.val_max_samples=1 \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.rollout.name=dynamo \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=round-robin \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=false \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    trainer.total_training_steps=1 \
    trainer.save_freq=-1 \
    "${flexkv_args[@]}" \
    "$@"

echo "PASS: Dynamo validation smoke completed"
