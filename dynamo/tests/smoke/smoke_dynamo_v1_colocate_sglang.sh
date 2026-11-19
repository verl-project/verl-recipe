#!/usr/bin/env bash
set -xeuo pipefail

# V1 colocate_async smoke on the SGLANG engine: gate/abort/sleep-wake/weight
# injection + real training steps on a tiny model.
#
# Environment prerequisites (see README "Required versions"):
#   pip install "ai-dynamo[sglang]"  + the #11640 incremental-logprobs backport
#   PYTORCH_CUDA_ALLOC_CONF must be unset (torch_memory_saver).
python3 -c "import dynamo.sglang" || {
  echo "dynamo.sglang not importable — install ai-dynamo[sglang] first" >&2
  exit 2
}
unset PYTORCH_CUDA_ALLOC_CONF || true

project_name=${PROJECT_NAME:-verl-dynamo}
exp_name=${EXP_NAME:-dynamo-v1-sglang-smoke}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
TOTAL_STEPS=${TOTAL_STEPS:-2}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe.dynamo.register}"
DYNAMO_CONFIG_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../config" && pwd)

python3 -m verl.trainer.main_ppo \
    --config-path "${DYNAMO_CONFIG_DIR}" \
    --config-name=dynamo_trainer_v1_colocate_sglang \
    "ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_USE_EXTERNAL_MODULES}'" \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    trainer.logger=console \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    "$@"
