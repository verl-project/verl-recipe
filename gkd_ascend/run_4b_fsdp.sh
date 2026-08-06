set -x

export VERL_LOGGING_LEVEL=INFO


train_files=openai-gsm8k/train.parquet
test_files=openai-gsm8k/test.parquet

# model
HF_MODEL_PATH=/path/to/Qwen3-4B/

INFER_TP=1

# train config
scheduler=one_step_off
NODES=1

# 2. run the script
python3 -u -m main_gkd --config-path=config --config-name on_policy_distill_trainer \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.prompt_key="prompt" \
    data.train_batch_size=32 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    data.trust_remote_code=true \
    data.filter_overlong_prompts_workers=32 \
    data.return_raw_chat=true \
    +data.apply_chat_template_kwargs.enable_thinking=false \
    actor_rollout_ref.teacher.server_ip=127.0.0.1 \
    actor_rollout_ref.teacher.server_port=15555 \
    actor_rollout_ref.teacher.n_server_workers=1 \
    actor_rollout_ref.model.path=$HF_MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.model.use_remove_padding=false \
    +actor_rollout_ref.model.override_config.attn_implementation=eager \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=true \
    actor_rollout_ref.actor.fsdp_config.param_offload=false \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=false \
    actor_rollout_ref.actor.fsdp_config.use_orig_params=false \
    actor_rollout_ref.actor.optim.lr=4e-5 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.02 \
    actor_rollout_ref.actor.use_dynamic_bsz=false \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=false \
    actor_rollout_ref.actor.use_torch_compile=false \
    actor_rollout_ref.actor.checkpoint.save_contents=["model","optimizer","extra","hf_model"] \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.99 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$INFER_TP \
    actor_rollout_ref.rollout.load_format='auto' \
    actor_rollout_ref.rollout.free_cache_engine=false \
    actor_rollout_ref.rollout.enforce_eager=true \
    algorithm.use_kl_in_reward=false \
    trainer.device=npu \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name='verl_fsdp_examples' \
    trainer.experiment_name='qwen-4b-distill' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=$NODES \
    rollout.n_gpus_per_node=2 \
    rollout.nnodes=$NODES \
    trainer.balance_batch=false \
    trainer.scheduler=$scheduler \
    trainer.save_freq=5 \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.total_training_steps=10 \
    trainer.total_epochs=1 $@