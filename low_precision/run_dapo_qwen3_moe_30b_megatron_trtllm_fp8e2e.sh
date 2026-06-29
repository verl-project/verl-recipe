#!/usr/bin/env bash
set -xeuo pipefail

# Target: GB200 / Blackwell, CUDA 12.9 or higher.
# This is the TRT-LLM sibling of run_dapo_qwen3_moe_30b_megatron_fp8e2e.sh:
# Megatron FP8 training with a TRT-LLM FP8 rollout instead of vLLM.
# Use the TRT-LLM container image (tensorrt_llm 1.3.0rc15) or self build.
# Verified verl commit: 8ebf167e (branch fp8e2e).

ID=${1:-"dapo-qwen3-30b-a3b-B128-R20K-FP8E2E-TIS-trtllm-8n"}

# This env var is required for TE fp8 training.
# On Blackwell, TE blockwise FP8 is emulated via MX-FP8, which requires power-of-two
# scales, so we set this to 0 (the H100 recipe uses 1 for FP32 scales).
# For multi-node runs it must also be set in the Ray runtime env (see below).
export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=0

################################################### quick config ###################################################


rollout_mode="async"
rollout_name="trtllm"
return_raw_chat="True"
dtype="bfloat16" # ["bfloat16", "float16"]

project_name='VERL-FP8-RL'
exp_name=$ID

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))
enable_overlong_buffer=True
overlong_buffer_len=$((512))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"

train_prompt_bsz=128
n_resp_per_prompt=16
train_prompt_mini_bsz=128
gen_prompt_bsz=$((128 * 3))

# Ray
RAY_ADDRESS=${RAY_ADDRESS:-"http://127.0.0.1:8265"}
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-8}
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen/Qwen3-30B-A3B-Base"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1 # 0 for HF rollout, -1 for vLLM/TRT-LLM rollout
val_top_p=0.7
enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

# TRT-LLM rollout
KV_CACHE_DTYPE=fp8

# Performance Related Parameter
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 1))
offload=True
gen_tp=1
gen_ep=1
# PP=1 is required: PP=2 deadlocks the step-2 backward pass on a
# PIPELINE_MODEL_PARALLEL NCCL collective (hits the 30-min watchdog). PP=1 is
# memory-safe here thanks to param/optimizer offload + recompute-full.
train_tp=4
train_pp=1
train_ep=8
train_etp=1

# Reference model parallelism (mirrors the actor; PP=1 for the same reason).
ref_pp=1
ref_tp=4
ref_ep=8
ref_etp=1

# Rollout Correction parameters (token-level TIS)
rollout_is=token
rollout_is_threshold=2.0
rollout_is_batch_normalize=false


################################################### start of config ###################################################

FP8=(
    # According to our experiments, hybrid recipe works better than e4m3 and trains more stably
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8="hybrid"
    +actor_rollout_ref.actor.megatron.override_transformer_config.fp8_recipe="blockwise"
    +actor_rollout_ref.actor.optim.override_optimizer_config.fp8_recipe="blockwise"
    +actor_rollout_ref.rollout.quantization=fp8

    # Since Qwen3 MoE in the rollout uses a bf16 router, we don't set it to fp32 in MCore.
    # We have tried fp32 routers; it doesn't seem to bring any accuracy improvement.
    # +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.attention_dropout=0.0
    +actor_rollout_ref.actor.megatron.override_transformer_config.hidden_dropout=0.0
)

DATA=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.prompt_key=prompt
    data.return_raw_chat=$return_raw_chat
    data.truncation='left'
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.train_batch_size=${train_prompt_bsz}
    data.gen_batch_size=${gen_prompt_bsz}
)

REWARD_MODEL=(
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer}
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len}
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor}
    +reward_model.reward_kwargs.overlong_buffer_cfg.log=False
    +reward_model.reward_kwargs.max_resp_len=${max_response_length}
    reward_model.reward_manager=dapo
)

PERF_OPT=(
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    actor_rollout_ref.model.use_fused_kernels=False

    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=False

    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
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
    actor_rollout_ref.actor.optim.weight_decay=0.1
    actor_rollout_ref.actor.optim.clip_grad=1.0
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}
    actor_rollout_ref.actor.megatron.param_offload=${offload}
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload}
    actor_rollout_ref.actor.megatron.grad_offload=${offload}
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=False
)

REF=(
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${ref_pp}
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${ref_tp}
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${ref_ep}
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${ref_etp}
    actor_rollout_ref.ref.megatron.param_offload=${offload}
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${rollout_name}
    actor_rollout_ref.rollout.mode=${rollout_mode}
    actor_rollout_ref.rollout.dtype=${dtype}
    actor_rollout_ref.rollout.gpu_memory_utilization=0.581
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}
    actor_rollout_ref.rollout.expert_parallel_size=${gen_ep}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=True
    actor_rollout_ref.rollout.enforce_eager=false
    actor_rollout_ref.rollout.max_num_batched_tokens=42074
    actor_rollout_ref.rollout.max_num_seqs=1283
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096
    +actor_rollout_ref.rollout.moe_tensor_parallel_size=1
    +actor_rollout_ref.rollout.enable_sleep_mode=true
    # TRT-LLM engine-specific knobs
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.enable_attention_dp=false
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.cuda_graph_config.enable_padding=true
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.cuda_graph_config.max_batch_size=256
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_timeout_iters=0
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.batch_wait_max_tokens_ratio=0.845
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.kv_cache_config.dtype=${KV_CACHE_DTYPE}
    +actor_rollout_ref.rollout.engine_kwargs.trtllm.moe_config.backend=TRTLLM
    actor_rollout_ref.rollout.temperature=${temperature}
    actor_rollout_ref.rollout.top_p=${top_p}
    actor_rollout_ref.rollout.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p}
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k}
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.n=${n_resp_per_prompt}
)

TRAINER=(
    trainer.logger='["console","wandb"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node=4
    trainer.nnodes="${NNODES}"
    trainer.val_before_train=False
    trainer.test_freq=10
    trainer.save_freq=5
    trainer.max_actor_ckpt_to_keep=5
    trainer.total_epochs=100
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.resume_mode=auto
    trainer.log_val_generations=2
    trainer.use_legacy_worker_impl="disable"
)

FORWARD_ONLY_SETS=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz}
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
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
    algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize}
)

# Propagate the Blackwell FP8 scale setting to all Ray worker nodes.
RAY_RUNTIME_ENV=(
    +ray_kwargs.ray_init.runtime_env.env_vars.NVTE_FP8_BLOCK_SCALING_FP32_SCALES="0"
)
################################################### start script ###################################################
RAY_ADDRESS=$RAY_ADDRESS ray job submit --runtime-env="${RUNTIME_ENV}" \
    -- python3 -m recipe.dapo.main_dapo \
    --config-path=config \
    --config-name='dapo_megatron_trainer.yaml' \
    "${DATA[@]}" \
    "${ALGORITHM[@]}" \
    "${MODEL[@]}" \
    "${ROLLOUT[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${REWARD_MODEL[@]}" \
    "${FP8[@]}" \
    "${PERF_OPT[@]}" \
    "${TRAINER[@]}" \
    "${FORWARD_ONLY_SETS[@]}" \
    "${RAY_RUNTIME_ENV[@]}"
