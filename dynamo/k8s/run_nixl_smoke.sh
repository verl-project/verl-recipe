#!/usr/bin/env bash
# NIXL refit smoke for the dynamo rollout backend — real training steps so
# update_weights exercises the CheckpointEngineWorker chain each step.
# Run from /workspace/verl on the Ray head pod.
#   NNODES=1 NGPUS_PER_NODE=8 bash recipe/dynamo/k8s/run_nixl_smoke.sh   # single-node
#   NNODES=2 NGPUS_PER_NODE=8 bash recipe/dynamo/k8s/run_nixl_smoke.sh   # multi-node
set -xuo pipefail

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
CE_BACKEND=${CE_BACKEND:-nixl}          # nixl | naive (regression)
TOTAL_STEPS=${TOTAL_STEPS:-3}
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen2.5-0.5B-Instruct}
TRAIN_FILE=${TRAIN_FILE:-/workspace/data_dapo/data/dapo-math-17k.parquet}
TEST_FILE=${TEST_FILE:-/workspace/data_aime/data/aime-2024.parquet}
BUCKET_MB=${BUCKET_MB:-1024}
EXP_NAME=${EXP_NAME:-nixl-smoke-${CE_BACKEND}-n${NNODES}}

export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
export HYDRA_FULL_ERROR=1

cd /workspace/verl

python3 -m recipe.dynamo.main_dynamo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=8 \
    data.max_prompt_length=512 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=False \
    data.truncation='left' \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.rollout.name=dynamo \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.calculate_log_probs=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=1024 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend="${CE_BACKEND}" \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${BUCKET_MB}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv \
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=false \
    trainer.logger='["console"]' \
    trainer.project_name=verl-dynamo-nixl \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    "$@"
RC=$?
echo "TRAIN_RC=${RC}"
exit ${RC}
