# Example DMPO training script for Qwen2.5-Math on OpenR1-Math.
set -x

export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}

train_files=${TRAIN_FILES:-"['$HOME/data/openr1_math/train.parquet']"}
test_files=${VAL_FILES:-"['$HOME/data/aime2024/test.parquet']"}
model_path=${MODEL_PATH:-"$HOME/models/Qwen2.5-Math-7B-16k-think"}
output_dir=${OUTPUT_DIR:-"$PWD/outputs/qwen2_5_math_grpo_dmpo_zero_openR1_46k_beta_2.0"}

python3 -m recipe.dmpo.main_dmpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=grpo_dmpo_zero \
    actor_rollout_ref.actor.policy_loss.dmpo_beta=2.0 \
    actor_rollout_ref.actor.policy_loss.dmpo_temperature=0.06666666666666667 \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=128 \
    data.val_batch_size=128 \
    data.truncation=error \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=8 \
    data.max_prompt_length=2048 \
    data.max_response_length=8192 \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=10240 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.response_length=8192 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=10240 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=10240 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_k=-1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    trainer.critic_warmup=0 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir="$output_dir" \
    trainer.rollout_data_dir="$output_dir/rollout" \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name=qwen2_5_math_openR1_46k \
    trainer.experiment_name=qwen2_5_math_grpo_dmpo_zero_openR1_46k_beta_2.0 \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=300 \
    trainer.total_epochs=1 \
    "$@"
