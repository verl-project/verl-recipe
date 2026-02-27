#!/usr/bin/env bash
set -xeuo pipefail

# Example: DAPO training with NVFP4 QAT using Megatron backend
# This script demonstrates how to enable Quantization-Aware Training
#
# Environment:
#   Docker image: verlai/verl:vllm012.latest
#   Megatron-Bridge needs to be installed manually inside the container:
#     pip install --no-deps git+https://github.com/NVIDIA-NeMo/Megatron-Bridge@e940d997d7bdb7810f621f5b32bf70255b5aa2d9

ID=${1:-"dapo-qwen3-30b-a3b-B32-R20K-nvfp4-qat-megatron-4n"}
HOME_DIR=/apps

################################################### quick config ###################################################

project_name='VERL-NVFP4-QAT'
exp_name=$ID

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024))
max_response_length=$((1024 * 20))
enable_overlong_buffer=False
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=32
n_resp_per_prompt=16
train_prompt_mini_bsz=32
gen_prompt_bsz=$((train_prompt_bsz * 2))

# Ray
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
echo "WORKING_DIR: ${WORKING_DIR}"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
echo "RUNTIME_ENV: ${RUNTIME_ENV}"
NNODES=4

# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME_DIR}"}
MODEL_PATH="/apps/models/Qwen3-30B-A3B-Base"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k-one.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1

# Rollout Importance Sampling parameters
rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_token_veto_threshold=null

# QAT Configuration
qat_enable=True
qat_mode=w4a16    # w4a16 for weight-only FP4
qat_config_path="${qat_config_path:-"${WORKING_DIR}/recipe/qat/config/nvfp4_w4a16_megatron.json"}"

export VERL_LOGGING_LEVEL=INFO
export VLLM_CONFIGURE_LOGGING=1
export VLLM_USE_V1=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export WANDB_API_KEY=149737fd3c4537b349a37aab90b6fff96f385ebc

################################################### start of config ###################################################

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.truncation='left'
    data.return_raw_chat=True
    data.filter_overlong_prompts=True
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.gen_batch_size=${gen_prompt_bsz}
    data.train_batch_size=${train_prompt_bsz}
)

ALGORITHM=(
    algorithm.adv_estimator=${adv_estimator}
    algorithm.use_kl_in_reward=${use_kl_in_reward}
    algorithm.kl_ctrl.kl_coef=${kl_coef}
    algorithm.filter_groups.enable=${enable_filter_groups}
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches}
    algorithm.filter_groups.metric=${filter_groups_metric}
    algorithm.rollout_correction.rollout_is=${rollout_is}
    algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}
    algorithm.rollout_correction.rollout_rs=${rollout_rs}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
)

ACTOR=(
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=0
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=32
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.actor.megatron.context_parallel_size=8
    actor_rollout_ref.actor.megatron.sequence_parallel=False
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
)

QAT=(
    actor_rollout_ref.actor.qat.enable=${qat_enable}
    actor_rollout_ref.actor.qat.mode=${qat_mode}
    actor_rollout_ref.actor.qat.quantization_config_path="${qat_config_path}"
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.50
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( 1024 * 16 ))
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
)

PERF_OPT=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_enable_deepep=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=flex
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.use_arbitrary_attention_mask=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
)

REWARD_MODEL=(
    reward_model.reward_manager=dapo
    reward_model.overlong_buffer.enable=${enable_overlong_buffer}
    reward_model.overlong_buffer.len=${overlong_buffer_len}
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor}
    reward_model.overlong_buffer.log=False
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=8
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=False
    trainer.test_freq=10
    trainer.save_freq=5
    trainer.total_epochs=100
    trainer.total_training_steps=500
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
)

FORWARD_ONLY_SETS=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=1
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=1
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=32
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1
    actor_rollout_ref.ref.megatron.context_parallel_size=8
    actor_rollout_ref.ref.megatron.sequence_parallel=False
)

################################################### start script ###################################################

RAY_ADDRESS='http://127.0.0.1:8265' ray job submit --runtime-env=${RUNTIME_ENV} \
    -- python3 -m verl.trainer.main_ppo \
    --config-path="${WORKING_DIR}/recipe/qat/config" \
    --config-name='dapo_qat_megatron_trainer' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${QAT[@]}" \
    "${ROLLOUT[@]}" \
    "${PERF_OPT[@]}" \
    "${REWARD_MODEL[@]}" \
    "${TRAINER[@]}" \
    "${FORWARD_ONLY_SETS[@]}"
