#!/usr/bin/env bash

set -euo pipefail
set -x

PROJECT_NAME=${PROJECT_NAME:-"int8-w8a8-qat"}
EXP_NAME=${EXP_NAME:-"qwen3-8b-int8-w8a8-qat"}

MODEL_PATH=${MODEL_PATH:?"Set MODEL_PATH to the BF16 actor checkpoint."}
ROLLOUT_MODEL_PATH=${ROLLOUT_MODEL_PATH:?"Set ROLLOUT_MODEL_PATH to the Ascend W8A8 checkpoint."}
TRAIN_FILE=${TRAIN_FILE:?"Set TRAIN_FILE to the training parquet file."}
VAL_FILE=${VAL_FILE:?"Set VAL_FILE to the validation parquet file."}
CKPT_DIR=${CKPT_DIR:-"./checkpoints/${EXP_NAME}"}

N_GPUS_PER_NODE=${N_GPUS_PER_NODE:-8}
NNODES=${NNODES:-1}

# export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
export HYDRA_FULL_ERROR=1
export ASCEND_LAUNCH_BLOCKING=0
export VLLM_USE_V1=1
export USE_STOCHASTIC=${USE_STOCHASTIC:-1}

QAT=${QAT:-true}
QAT_W_BIT=${QAT_W_BIT:-8}
SCALE_SOURCE=${SCALE_SOURCE:-learned}

# Importance Sampling (IS) weights configuration
ROLLOUT_IS=${ROLLOUT_IS:-null}
ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-null}
ROLLOUT_IS_BATCH_NORMALIZE=${ROLLOUT_IS_BATCH_NORMALIZE:-null}

# Rejection Sampling (RS) configuration
ROLLOUT_RS=${ROLLOUT_RS:-null}
ROLLOUT_RS_THRESHOLD=${ROLLOUT_RS_THRESHOLD:-null}

# Algorithm
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-1.0}
TOP_K=${TOP_K:--1} # 0 for HF rollout, -1 for vLLM rollout

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=512 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=32 \
    data.truncation='left' \
    data.trust_remote_code=True \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_scheduler_type='constant' \
    actor_rollout_ref.actor.optim.lr_warmup_steps=3 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.clip_grad=1.0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.qat="${QAT}" \
    actor_rollout_ref.actor.qat_w_bit="${QAT_W_BIT}" \
    actor_rollout_ref.actor.scale_source="${SCALE_SOURCE}" \
    actor_rollout_ref.actor.fsdp_config.dtype="bfloat16" \
    actor_rollout_ref.actor.fsdp_config.model_dtype="bfloat16" \
    actor_rollout_ref.ref.fsdp_config.dtype="bfloat16" \
    actor_rollout_ref.ref.fsdp_config.model_dtype="bfloat16" \
    actor_rollout_ref.rollout.dtype="bfloat16" \
    actor_rollout_ref.rollout.quantization="ascend" \
    actor_rollout_ref.rollout.model_path="${ROLLOUT_MODEL_PATH}" \
    actor_rollout_ref.rollout.load_format="auto" \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.n=5 \
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS}" \
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}" \
    algorithm.rollout_correction.rollout_is_batch_normalize="${ROLLOUT_IS_BATCH_NORMALIZE}" \
    algorithm.rollout_correction.rollout_rs="${ROLLOUT_RS}" \
    algorithm.rollout_correction.rollout_rs_threshold="${ROLLOUT_RS_THRESHOLD}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.top_k="${TOP_K}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0 \
    trainer.critic_warmup=0 \
    trainer.logger='["console","tensorboard"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.nnodes="${NNODES}" \
    trainer.save_freq=200 \
    trainer.test_freq=5 \
    trainer.total_epochs=1 \
    trainer.device=npu \
    "$@"
