#!/usr/bin/env bash
set -xeuo pipefail

# End-to-end MXFP8:
#   - Training side (actor / Megatron + Transformer Engine): MXFP8
#   - Rollout side (SGLang): MXFP8
# use docker: docker://verlai/verl:sgl0512.dev2
# and update the sglang in docker: pip install "sglang @
#   git+https://github.com/sgl-project/sglang.git@sglang-miles-v0.5.12-final#subdirectory=python"

ID=${1:-"dapo-qwen3-30b-megatron-sglang-mxfp8-e2e-20k-b128"}

project_name=${PROJECT_NAME:-VERL-MXFP8-RL}
exp_name=$ID

################################################### quick config ###################################################

rollout_mode="async"
rollout_name="sglang" # sglang or vllm
return_raw_chat="False"
if [ "$rollout_mode" = "async" ]; then
    return_raw_chat="True"
fi
dtype="bfloat16" # ["bfloat16", "float16"]

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

rollout_is=token
rollout_is_threshold=2.0
rollout_rs=null
rollout_rs_threshold=null

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10
train_prompt_bsz=128
gen_prompt_bsz=256
n_resp_per_prompt=16
train_prompt_mini_bsz=128

RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}
# Paths
HOME_DIR="${HOME}/verl"
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME_DIR}"}
MODEL_PATH=${RAY_DATA_HOME}/models/Qwen3-30B-A3B-Base
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM rollout
val_top_p=0.6

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=true
gen_tp=1
actor_lr_warmup_steps=10
# ---- Rollout / inference side MXFP8 ----
rollout_quantization=mxfp8
fp8_gemm_runner_backend=flashinfer_cutlass
moe_runner_backend=flashinfer_trtllm

# ---- Training side (actor / Megatron + Transformer Engine) MXFP8 ----
actor_fp8_recipe=mxfp8
actor_fp8_param=False

actor_fp8_param_gather=False
actor_reuse_grad_buf_for_mxfp8_param_ag=False


train_tp=4
train_pp=1
train_ep=8
train_etp=1
train_cp=1

################################################### start of config ###################################################

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.truncation='left'
    data.return_raw_chat=$return_raw_chat
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
)

PERF_OPT=(
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=False
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    actor_rollout_ref.model.use_fused_kernels=True
)

MXFP8_TRAINING=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8=e4m3
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe=${actor_fp8_recipe}
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_param=${actor_fp8_param}
    +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe=${actor_fp8_recipe}
    +actor_rollout_ref.actor.megatron.override_ddp_config.fp8_param_gather=${actor_fp8_param_gather}
    +actor_rollout_ref.actor.megatron.override_ddp_config.reuse_grad_buf_for_mxfp8_param_ag=${actor_reuse_grad_buf_for_mxfp8_param_ag}
    +actor_rollout_ref.actor.megatron.override_ddp_config.overlap_grad_reduce=True
    +actor_rollout_ref.actor.megatron.override_ddp_config.overlap_param_gather=False
    +actor_rollout_ref.actor.optim.override_optimizer_config.reuse_grad_buf_for_mxfp8_param_ag=${actor_reuse_grad_buf_for_mxfp8_param_ag}
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_param_gather=False
    actor_rollout_ref.actor.megatron.use_distributed_optimizer=True
)

ACTOR=(
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss}
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef}
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low}
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high}
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_warmup_steps=${actor_lr_warmup_steps}
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.optim.clip_grad=1.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    actor_rollout_ref.actor.megatron.override_transformer_config.attention_backend='flash'
    actor_rollout_ref.actor.optim.use_checkpoint_opt_param_scheduler=True
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}
)

ROLLOUT=(
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=$(( 1024 * 32 ))
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature}
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.enforce_eager=False
    +actor_rollout_ref.rollout.engine_kwargs.sglang.fp8_gemm_runner_backend=${fp8_gemm_runner_backend}
    +actor_rollout_ref.rollout.engine_kwargs.sglang.moe_runner_backend=${moe_runner_backend}
    actor_rollout_ref.rollout.quantization=${rollout_quantization}
)

FORWARD_ONLY_SETS=(
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
)

REWARD=(
    reward.reward_manager.name=dapo
    reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    reward.reward_kwargs.max_resp_len=${max_response_length}
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
    trainer.max_actor_ckpt_to_keep=5
    trainer.total_epochs=10
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
    trainer.total_training_steps=500
)

################################################### start script ###################################################
python3 -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name='dapo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${PERF_OPT[@]}" \
    "${MXFP8_TRAINING[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${FORWARD_ONLY_SETS[@]}" \
    "${REWARD[@]}" \
    "${TRAINER[@]}"
