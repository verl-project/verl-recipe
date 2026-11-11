# 数据集和模型路径,保持为空,不需要修改
data_path=""
model_path=""

# 参数校验，不需要修改
for para in $*
do
    if [[ $para == --data_path* ]];then
        data_path=`echo ${para#*=}`
    elif [[ $para == --model_path* ]];then
        model_path=`echo ${para#*=}`
    fi
done

ENGINE=vllm
export VLLM_USE_V1=1
export HCCL_CONNECT_TIMEOUT=5400

# Some models are optimized by vllm ascend. While in some case, e.g. rlhf training,
# the optimized model may not be suitable. In this case, set this value to 0 to disable the optimized model.
export USE_OPTIMIZED_MODEL=0

# prompt&response length
max_prompt_length=${max_prompt_length:=1024}
max_response_length=${max_response_length:=2048}
max_num_batched_tokens=8192

# vllm related params
free_cache_engine=True
gpu_memory_utilization=${gpu_memory_utilization:=0.5}
tensor_model_parallel_size=${tensor_model_parallel_size:=4}
enable_chunked_prefill=True
enforce_eager=${enforce_eager:=False}

# batch size
train_batch_size=${train_batch_size:=512}
ppo_mini_batch_size=32
ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu:=4}
log_prob_micro_batch_size_per_gpu=4
use_remove_padding=True
ignore_eos=False

# training params
enable_gradient_checkpointing=True
nnodes=2
n_gpus_per_node=8
save_freq=${save_freq:=20}
test_freq=${test_freq:=20}
val_before_train=${val_before_train:=False}
total_training_steps=${total_training_steps:=100}
sp_size=${sp_size:=1}

# DAPO feature
# 1. Clip Higher
clip_ratio_low=0.2
clip_ratio_high=0.28

# 2. Dynamic Sampling
filter_groups_enable=True
filter_groups_metric=acc
max_num_gen_batches=10
gen_batch_size=$((train_batch_size * 3))
rollout_n=${rollout_n:=16}

# 3. Token-Level Policy Gradient Loss
loss_agg_mode="token-mean"

# 4. Overlong Reward Shaping
overlong_buffer_enable=True
# overlong_buffer_len=$((1024 * 4))
overlong_buffer_len=${overlong_buffer_len:=4096}
overlong_buffer_penalty_factor=1.0

# remove kl divergence
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.7

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size * 2))
ref_log_prob_ppo_max_token_len=$(((max_prompt_length + max_response_length) / sp_size))
max_num_batched_tokens=$((max_prompt_length + max_response_length))
offload=True
max_num_seqs=128

echo "overlong_buffer_len: $overlong_buffer_len"
echo "actor_ppo_max_token_len: $actor_ppo_max_token_len"
echo "ref_log_prob_ppo_max_token_len: $ref_log_prob_ppo_max_token_len"

python3 -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    algorithm.filter_groups.enable=${filter_groups_enable} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    data.train_files=$data_path/train.parquet \
    data.val_files=$data_path/test.parquet \
    data.train_batch_size=$train_batch_size \
    data.gen_batch_size=$gen_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    data.shuffle=True \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=$use_remove_padding \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_torch_compile=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=$enable_gradient_checkpointing \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=False \
    actor_rollout_ref.ref.fsdp_config.forward_prefetch=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.ignore_eos=$ignore_eos \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.enforce_eager=$enforce_eager \
    actor_rollout_ref.rollout.free_cache_engine=$free_cache_engine \
    actor_rollout_ref.rollout.n=${rollout_n} \
    actor_rollout_ref.rollout.max_num_seqs=${max_num_seqs} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${overlong_buffer_enable} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_buffer_penalty_factor} \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name='verl_grpo_example_geo3k' \
    trainer.experiment_name='qwen3_vl_8b_function_rm' \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    trainer.total_epochs=25 \
    trainer.total_training_steps=100 \
    trainer.device=npu \
    trainer.val_before_train=$val_before_train \
    actor_rollout_ref.rollout.enable_chunked_prefill=$enable_chunked_prefill \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ref_log_prob_ppo_max_token_len} \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.max_num_batched_tokens=$max_num_batched_tokens
