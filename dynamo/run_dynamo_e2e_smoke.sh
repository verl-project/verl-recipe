#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# m2 end-to-end smoke training run: Qwen2.5-0.5B + dapo-math + recipe.dynamo
# rollout backend, 1 node, 3 training steps. Goal is *not* convergence — it's
# to confirm the recipe-side dynamo path doesn't regress vs canonical vllm
# anywhere in the trainer ↔ ServerAdapter ↔ HttpServer chain.
#
# At m2 the dynamo classes are thin subclasses of vllm classes (no Dynamo
# Frontend / runtime yet), so a successful run also implies vllm-equivalent
# behavior under the dynamo registry path.

set -xeuo pipefail

project_name='verl-dynamo'
exp_name='dynamo-m2-e2e-smoke'

# Ray / topology
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}

# Paths (resolved on the user's lustre tree)
RAY_DATA_HOME=${RAY_DATA_HOME:-"/lustre/fsw/general_sa/sopyang"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/multi-modal/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/datasets/dapo_data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/datasets/dapo_data/aime-2024.parquet"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpt/${project_name}/${exp_name}"}

# Tiny shapes — we just want the loop to not crash for a few iterations.
max_prompt_length=$((1024))
max_response_length=$((1024))
train_prompt_bsz=8
train_prompt_mini_bsz=4
n_resp_per_prompt=2

# Algorithm — plain GRPO/DAPO defaults; not chasing convergence.
adv_estimator=grpo
clip_ratio_low=0.2
clip_ratio_high=0.28
loss_agg_mode="token-mean"
temperature=1.0
top_p=1.0
top_k=-1

# 1 GPU per actor + 1 GPU per rollout (we have 8 GPUs, so this is fine).
sp_size=1
gen_tp=1
fsdp_size=${NGPUS_PER_NODE}
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))

cd /lustre/fsw/general_sa/sopyang/rl/verl_0211/verl

python3 recipe/dynamo/main_dynamo.py \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.rollout.name=dynamo \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    reward_model.reward_manager=naive \
    trainer.logger='["console"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=3 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=disable
