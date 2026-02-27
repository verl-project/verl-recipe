ray stop --force

# source /usr/local/Ascend/ascend-toolkit/set_env.sh

export CUSTOM_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=1200

PROJECT_DIR="$(pwd)"

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
export DISABLE_L2_CACHE=1

reward_model_path=/home/CKPT/HPSv3/HPSv3.safetensors
model_args_path=$PROJECT_DIR/recipe/mm_dance_grpo/examples/wan2.2/5B/t2v/output_args
# 用于在训练过程中，每隔n步进行一次相同prompt的推理，保存推理结果和评分
rollout_online_path=$PROJECT_DIR/rollout_online_save/$(date +%Y%m%d)_$(date +%H%M%S)/
# 间隔测试的prompt
rollout_online_prompt='The video presents a serene winter landscape that remains consistent throughout its duration. It begins showcasing breathtaking scenery that includes snow-capped mountains, a mirror-like body of water reflecting the mountains and sky, snowflakes falling gently, and coniferous trees lining the shoreline. The scene is marked by a tranquil ambiance, with no noticeable changes as the video progresses. The mountains, water, snowflakes, and trees maintain their picturesque and motionless state, creating a continuous and peaceful winter wonderland. The overall composition of the landscape evokes a sense of tranquility and natural beauty, with the elements of the scene—light, shadow, and color—remaining unchanged, reinforcing the stillness and serenity of the winter setting.'
# 用于单独进行推理测试，保存所有推理结果和评分
rollout_result_path=$PROJECT_DIR/rollout_result_save/$(date +%Y%m%d)_$(date +%H%M%S)/

train_data=$PROJECT_DIR/recipe/mm_dance_grpo/data/parquet/train.parquet
test_data=$PROJECT_DIR/recipe/mm_dance_grpo/data/parquet/test.parquet

logfile=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

python3 -m recipe.mm_dance_grpo.main_dance \
    --config-path=config \
    --config-name="dance_ppo_trainer" \
    algorithm.adv_estimator=grpo \
    data.train_files=$train_data \
    data.val_files=$test_data \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path='' \
    +actor_rollout_ref.model.reward_model="hpsv3" \
    +actor_rollout_ref.model.reward_model_path=$reward_model_path \
    actor_rollout_ref.actor.optim.lr=5e-8 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    +actor_rollout_ref.actor.optim.lr_scheduler_name='cosine' \
    +actor_rollout_ref.actor.optim.lr_scheduler_num_warmup_steps=1000 \
    +actor_rollout_ref.actor.optim.lr_scheduler_num_training_steps=10000 \
    +actor_rollout_ref.actor.optim.lr_scheduler_num_cycles=1 \
    +actor_rollout_ref.actor.ppo_adv_clip_max=10.0 \
    +actor_rollout_ref.actor.ppo_kl_coeff=1.0 \
    +actor_rollout_ref.actor.ppo_max_grad_norm=1.0 \
    +actor_rollout_ref.actor.clip_range=1e-4 \
    +actor_rollout_ref.actor.shift=1.0 \
    +actor_rollout_ref.actor.timestep_fraction=1 \
    +actor_rollout_ref.actor.sampling_steps=10 \
    +actor_rollout_ref.actor.model_args_path=$model_args_path \
    +actor_rollout_ref.actor.micro_batch_size=2 \
    +actor_rollout_ref.rollout.micro_batch_size=2 \
    +actor_rollout_ref.rollout.latent_w=128 \
    +actor_rollout_ref.rollout.latent_h=128 \
    +actor_rollout_ref.rollout.init_same_noise=True \
    +actor_rollout_ref.rollout.online.test=True \
    +actor_rollout_ref.rollout.online.step.interval=10 \
    +actor_rollout_ref.rollout.online.save.path=$rollout_online_path \
    "+actor_rollout_ref.rollout.online.prompt=\"$rollout_online_prompt\"" \
    +actor_rollout_ref.rollout.only=False \
    +actor_rollout_ref.rollout.result.save.path=$rollout_result_path \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
    actor_rollout_ref.rollout.n=8 \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.mode="sync" \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.project_name='wan2_2_project' \
    trainer.experiment_name='wan2_2_experiment' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=200 \
    trainer.device=npu \
    2>&1 | tee logs/train_${logfile}.log