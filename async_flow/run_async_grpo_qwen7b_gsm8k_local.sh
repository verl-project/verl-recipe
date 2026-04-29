#!/bin/bash
set -x

#unset RAY_DEBUG_POST_MORTEM
# ray stop --force
project_name='AsyncFlow-GRPO'
exp_name='AsyncFlow-GRPO-Qwen2.5-7B-gsm8k-npu'

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-".model/Qwen2.5-7B-Instruct"}
CKPTS_DIR=${CKPTS_DIR:-".ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-".dataset/preprocessed_gsm8k/train.parquet"}
TEST_FILE=${TRAIN_FILE:-".dataset/preprocessed_gsm8k/test.parquet"}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-"rollout_data_dir"}

export PROMETHEUS_METRICS_PORT=9400
export PROMETHEUS_METRICS_ENABLE=true    # set this flag as true to enable prometheus metrics collection
export PROMETHEUS_MULTIPROC_DIR=/tmp/prom_metrics
rm -rf "$PROMETHEUS_MULTIPROC_DIR"
mkdir -p "$PROMETHEUS_MULTIPROC_DIR"

export VLLM_USE_V1=1
export VERL_CLUSTER_TRACE=1
export RAY_DEDUP_LOGS="0"
export LOGGING_LEVEL="DEBUG"
export ASCEND_LAUNCH_BLOCKING=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

python3 -m recipe.async_flow.grpo_main \
        algorithm.adv_estimator=grpo \
        data.max_prompt_length=1024 \
        data.max_response_length=3072 \
        data.train_batch_size=32 \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.rollout_n=8 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.response_length=3072 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
        trainer.use_legacy_worker_impl="disable" \
        trainer.project_name="${project_name}" \
        trainer.experiment_name="${exp_name}" \
        trainer.default_local_dir="${CKPTS_DIR}" \
        trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
        async_resources.actor_train.n_gpus_per_node=4 \
        2>&1 | tee "logs/${exp_name}_$(date +%Y%m%d_%H%M).log"

