ray stop --force

# Mindspeed-MM fsdp config
export NON_MEGATRON=true
export MULTI_STREAM_MEMORY_REUSE=2
export OMP_NUM_THREADS=1
export PYTORCH_NPU_ALLOC_CONF=garbage_collection_threshold:0.85

project_name='GRPO-Qwen3_5'
exp_name='GRPO-Qwen3_5-35b-npu'
gen_tp=8
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}

MODEL_PATH=./weights/Qwen3.5-35B-A3B
TRAIN_FILE=./datasets/geo3k/train.parquet
TEST_FILE=./datasets/geo3k/test.parquet
export MM_CONFIG_FILE=./examples/qwen3_5_35B_config.yaml

start_time=$(date +%Y%m%d)_$(date +%H%M%S)

mkdir -p logs
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=False \
    actor_rollout_ref.model.path=${MODEL_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode="FULL_DECODE_ONLY" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.ignore_eos=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.offload_policy=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.ref.use_torch_compile=False \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node=16 \
    trainer.nnodes=1 \
    trainer.balance_batch=False \
    trainer.resume_from_path=checkpoints/ \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.val_before_train=True \
    trainer.use_legacy_worker_impl=auto \
    actor_rollout_ref.rollout.enable_prefix_caching=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144 \
    trainer.total_epochs=10 $@ 2>&1 | tee logs/qwen3_5-35b-${start_time}.log