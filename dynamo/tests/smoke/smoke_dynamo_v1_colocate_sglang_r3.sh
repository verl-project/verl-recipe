#!/usr/bin/env bash
set -xeuo pipefail

# End-to-end R3 smoke: Megatron colocate_async training with rollouts through
# the Dynamo SGLang sidecar and the frontend's native /generate, replaying the
# rollout's routed experts in the actor (router_replay.mode=R3).
#
# Needs an MoE model (Qwen3-30B-A3B-class defaults), 8 GPUs, sglang>=0.5.16
# with native gRPC, and ai-dynamo with the SGLang sidecar (1.7.0.dev20261110+).
# PYTORCH_CUDA_ALLOC_CONF must be unset (torch_memory_saver).
python3 -c "import dynamo.sglang.sidecar, sglang" || {
  echo "dynamo.sglang.sidecar / sglang not importable" >&2
  exit 2
}
unset PYTORCH_CUDA_ALLOC_CONF || true

project_name=${PROJECT_NAME:-verl-dynamo}
exp_name=${EXP_NAME:-dynamo-v1-sglang-r3-smoke}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
TOTAL_STEPS=${TOTAL_STEPS:-3}
TRAIN_BATCH=${TRAIN_BATCH:-16}
ROLLOUT_N=${ROLLOUT_N:-4}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
# Any MoE checkpoint SGLang and Megatron both support; parallelism defaults fit 30B-A3B.
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen3-30B-A3B"}
# verl examples/data_preprocess/gsm8k.py output: short math prompts keep the smoke fast.
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/gsm8k/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/gsm8k/test.parquet"}
# Loading a 30B checkpoint from network storage can exceed the default 600 s
# startup budget (frontend health, control plane, sidecar gRPC deadline).
export VERL_DYNAMO_FE_READY_TIMEOUT=${VERL_DYNAMO_FE_READY_TIMEOUT:-1800}

export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe.dynamo.register}"
DYNAMO_CONFIG_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../config" && pwd)

python3 -m verl.trainer.main_ppo \
    --config-path "${DYNAMO_CONFIG_DIR}" \
    --config-name=dynamo_trainer_v1_colocate_sglang_r3 \
    "ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_USE_EXTERNAL_MODULES}'" \
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_DYNAMO_FE_READY_TIMEOUT='${VERL_DYNAMO_FE_READY_TIMEOUT}'" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
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
