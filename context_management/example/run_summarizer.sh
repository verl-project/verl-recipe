#!/bin/bash
# Minimal GRPO example wiring the `naive_summarizer_agent` context-management loop.
# Set MODEL_PATH / TRAIN_FILES / VAL_FILES for your task, then: bash run_summarizer.sh
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen2.5-3B-Instruct"}
TRAIN_FILES=${TRAIN_FILES:-"$HOME/data/gsm8k/train.parquet"}
VAL_FILES=${VAL_FILES:-"$HOME/data/gsm8k/test.parquet"}

# Path to this recipe's agent-loop registry (relative to a verl checkout with the recipe submodule).
AGENT_LOOP_CONFIG=recipe/context_management/example/agent.yaml

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILES" \
    data.val_files="$VAL_FILES" \
    data.train_batch_size=128 \
    data.max_prompt_length=2048 \
    data.max_response_length=16384 \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$AGENT_LOOP_CONFIG" \
    actor_rollout_ref.rollout.agent.default_agent_loop=naive_summarizer_agent \
    trainer.logger='["console"]' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.total_epochs=15 \
    trainer.device=cuda "$@"
