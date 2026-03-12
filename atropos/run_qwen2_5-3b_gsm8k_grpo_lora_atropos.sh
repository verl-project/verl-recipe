#!/usr/bin/env bash
set -euo pipefail
set -x

python3 recipe/atropos/launch_atropos_verl_services.py \
  --config recipe/atropos/config/atropos_grpo_small.yaml -- \
  algorithm.adv_estimator=grpo \
  trainer.project_name=verl_grpo_example_gsm8k \
  trainer.experiment_name=qwen2.5_3b_grpo_lora_atropos_small \
  trainer.atropos.api_url=http://localhost:9001 \
  trainer.atropos.environment=gsm8k \
  data.train_files=$HOME/data/gsm8k_chat/train.parquet \
  data.val_files=$HOME/data/gsm8k_chat/test.parquet \
  data.train_batch_size=2 \
  data.max_prompt_length=256 \
  data.max_response_length=256 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
  actor_rollout_ref.model.lora_rank=64 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.actor.optim.lr=3e-6 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  inference.port=8000 \
  inference.gpu_memory_utilization=0.2
