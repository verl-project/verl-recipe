#!/usr/bin/env bash

set -xeuo pipefail

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

PROJECT_NAME=${PROJECT_NAME:-'reward_queue'}
EXP_NAME=${EXP_NAME:-'verl_async_train'}
MODEL_PATH=${MODEL_PATH:-'Qwen3-8B'}
TRAIN_FILE=${TRAIN_FILE:-'./gsm8k/train/gsm8k_tra.jsonl'}
VAL_FILE=${VAL_FILE:-'./gsm8k/eval/gsm8k_ev.jsonl'}
CKPTS_DIR=${CKPTS_DIR:-"./ckpts/${project_name}/${exp_name}"}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
N_SAMPLE=${N_SAMPLE:-8}
VAL_N_SAMPLE=${VAL_N_SAMPLE:-5}

MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1000}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2000}

LR=${LR:-1e-6}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
TEST_FREQ=${TEST_FREQ:-5}

ASYNC_STALENESS=${ASYNC_STALENESS:-0.3}
ASYNC_SYNC_STEP=${ASYNC_SYNC_STEP:-2}
ASYNC_REQUIRE_BATCHES=${ASYNC_REQUIRE_BATCHES:-4}

TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-8}

python -m recipe.reward_queue.main \
    --config-path=config \
    --config-name='fully_async' \
    algorithm.adv_estimator=grpo \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    rollout.nnodes=${NNODES} \
    rollout.n_gpus_per_node=${NGPUS_PER_NODE} \
    rollout.total_rollout_steps=${TOTAL_TRAINING_STEPS} \
    rollout.test_freq=${TEST_FREQ} \
    async_training.staleness_threshold=${ASYNC_STALENESS} \
    async_training.trigger_parameter_sync_step=${ASYNC_SYNC_STEP} \
    async_training.require_batches=${ASYNC_REQUIRE_BATCHES} \
    async_training.partial_rollout=true \
    async_training.use_trainer_do_validate=false \
    async_training.enable_reward_queue=true \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.n=${N_SAMPLE} \
    actor_rollout_ref.rollout.val_kwargs.n=${VAL_N_SAMPLE} \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${TRAIN_BATCH_SIZE} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.save_freq=${TEST_FREQ} \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    "$@"

