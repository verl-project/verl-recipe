#!/usr/bin/env bash
# Online Policy Distillation: Qwen3-8B student + Qwen3-32B teacher on GSM8K
#
# Prerequisites:
#   1. Start teacher server (see recipe/gkd/teacher/):
#      cd recipe/gkd/teacher
#      CUDA_VISIBLE_DEVICES=0,1,2,3 bash start_server.sh  # TP=4 for 32B
#
#   2. Prepare GSM8K data:
#      python3 -c "
#      from datasets import load_dataset; import pandas as pd
#      ds = load_dataset('openai/gsm8k', 'main', split='train')
#      records = [{'prompt': i['question'], 'data_source': 'gsm8k',
#                  'reward_model': {'ground_truth': i['answer']}} for i in ds]
#      pd.DataFrame(records).to_parquet('data/gsm8k_train.parquet')
#      "
#
# Usage:
#   bash recipe/opd/run_opd.sh [kl_loss_coef]
#   # kl_loss_coef: 0.0 (baseline), 0.001 (weak), 0.01 (medium, default)

set -xeuo pipefail

KL_COEF=${1:-0.01}
STUDENT_MODEL=${STUDENT_MODEL:-"Qwen/Qwen3-8B"}
TEACHER_IP=${TEACHER_IP:-"127.0.0.1"}
TEACHER_PORT=${TEACHER_PORT:-15555}
TRAIN_FILE=${TRAIN_FILE:-"data/gsm8k_train.parquet"}
TEST_FILE=${TEST_FILE:-"data/gsm8k_test.parquet"}
NGPUS=${NGPUS:-4}
TOTAL_STEPS=${TOTAL_STEPS:-200}

python3 -m recipe.opd.main_opd \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation=left \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.train_batch_size=8 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.model.path="${STUDENT_MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.grad_clip=1.0 \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    "++actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096" \
    reward.custom_reward_function.path=recipe/opd/reward_gsm8k.py \
    reward.custom_reward_function.name=compute_score \
    +teacher.server_ip="${TEACHER_IP}" \
    +teacher.server_port="${TEACHER_PORT}" \
    +teacher.kl_loss_coef="${KL_COEF}" \
    "trainer.logger=[console]" \
    trainer.project_name=verl_opd \
    trainer.experiment_name="opd-kl${KL_COEF}" \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.nnodes=1 \
    trainer.val_before_train=False \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.total_epochs=3 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1
