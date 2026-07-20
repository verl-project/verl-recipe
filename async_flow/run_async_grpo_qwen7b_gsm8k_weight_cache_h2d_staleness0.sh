#!/bin/bash
set -x
export HCCL_INTRA_ROCE_ENABLE=1
export ASCEND_CONNECT_TIMEOUT=400000
export ASCEND_TRANSFER_TIMEOUT=400000
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1

#unset RAY_DEBUG_POST_MORTEM
# ray stop --force
project_name='AsyncFlow-GRPO'
exp_name='AsyncFlow-GRPO-Qwen2.5-7B-gsm8k-npu'
temperature=1

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
# export VERL_CLUSTER_TRACE=1
export RAY_DEDUP_LOGS="0"
export LOGGING_LEVEL="WARN"
export ASCEND_LAUNCH_BLOCKING=1
export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:128
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True

python3 -m recipe.async_flow.grpo_main \
        algorithm.adv_estimator=grpo \
        algorithm.use_kl_in_reward=False \
        \
        data.shuffle=False \
        data.validation_shuffle=False \
        data.train_files="${TRAIN_FILE}" \
        data.val_files="${TEST_FILE}" \
        data.train_batch_size=32 \
        data.max_prompt_length=1024 \
        data.max_response_length=1024 \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.seed=42 \
        \
        actor_rollout_ref.model.path="${MODEL_PATH}" \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        \
        actor_rollout_ref.actor.optim.lr=5e-8 \
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        \
        actor_rollout_ref.rollout.agent.num_workers=8 \
        \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
        actor_rollout_ref.rollout.enable_chunked_prefill=False \
        actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
        actor_rollout_ref.rollout.temperature=${temperature} \
        actor_rollout_ref.rollout.response_length=1024 \
        actor_rollout_ref.rollout.calculate_log_probs=True \
        \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
        actor_rollout_ref.ref.fsdp_config.param_offload=False \
        \
        async_flow.staleness=0 \
        \
        async_resources.actor_train.n_gpus_per_node=4 \
        async_resources.actor_fwd.n_gpus_per_node=2 \
        async_resources.ref_fwd.n_gpus_per_node=2 \
        \
        trainer.n_gpus_per_node=8 \
        trainer.use_legacy_worker_impl="disable" \
        trainer.logger=console \
        trainer.experiment_name="${exp_name}" \
        trainer.project_name="${project_name}" \
        trainer.save_freq=-1 \
        trainer.test_freq=-1 \
        trainer.total_training_steps=500 \
        actor_rollout_ref.rollout.checkpoint_engine.backend="flexfetch" \
        2>&1 | tee "logs/${exp_name}_$(date +%Y%m%d_%H%M).log"

