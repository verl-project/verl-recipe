#!/usr/bin/env bash
# ==============================================================================
# veRL GRPO — tau2-bench airline × Qwen2.5-7B-Instruct, multi-turn agent loop
# GPU1 = policy (LoRA GRPO + colocated vLLM rollout); GPU0 = usersim server.
# Built on the WORKING GSM8K single-GPU config (sdpa / enforce_eager /
# free_cache_engine=False / use_v1=False / LoRA), plus the multi-turn agent-loop
# wiring for the custom tau2_agent.
#
# Prereqs:
#   1) usersim up:      bash serve_usersim_7b.sh          (on GPU0)
#   2) data built:      python data_prep_airline.py
#   3) offline test OK: python test_tau2_loop_offline.py  (validates loop on CPU)
#
# Usage:  bash run_tau2_grpo_7b.sh
# ==============================================================================
set -xeuo pipefail

# ---- paths: all overridable; defaults work from a plain checkout ----
# ROOT  = workspace holding models/, data/ and all caches (keep off the OS disk)
# INTEG = the recipe dir holding tau2_agent_loop.py; goes on PYTHONPATH so hydra
#         can resolve _target_ in agent_loop_config.yaml
# VENV  = optional; leave empty if verl is already importable in your env
ROOT=${ROOT:-$PWD}
INTEG=${INTEG:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
VENV=${VENV:-}

# ---- caches/tmp on /data, never root disk ----
export TMPDIR=$ROOT/tmp
export HF_HOME=$ROOT/.hfhome
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=$ROOT/.mscache
export VLLM_CACHE_ROOT=$ROOT/.vllmcache
export TRITON_CACHE_DIR=$ROOT/.triton
export TORCHINDUCTOR_CACHE_DIR=$ROOT/.inductor
export XDG_CACHE_HOME=$ROOT/.xdgcache
export RAY_TMPDIR=$ROOT/tmp
export VLLM_USE_V1=1
# NOTE: do NOT set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True here —
# vLLM V1's CUDA memory pool asserts against it (pytorch#147851). The long-
# sequence log_prob OOM is instead solved by use_fused_kernels below (chunked
# CE avoids materializing the full [seq x vocab] logits).
mkdir -p "$TMPDIR" "$HF_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$XDG_CACHE_HOME"

# ---- policy trains on GPU1 by default ----
export CUDA_VISIBLE_DEVICES=${GPU:-1}

# ---- custom agent loop + bridge on PYTHONPATH so hydra can import it ----
export PYTHONPATH="$INTEG:${PYTHONPATH:-}"

# ---- tau2 bridge config (read by tau2_agent_loop._build_bridge) ----
export TAU2_DOMAIN=airline
export TAU2_USER_LLM=openai/usersim
export TAU2_USER_API_BASE=${TAU2_USER_API_BASE:-http://127.0.0.1:18001/v1}
export TAU2_USER_API_KEY=dummy
export TAU2_USER_TEMPERATURE=0.0
export TAU2_EVAL_TYPE=all
export TAU2_MAX_ERRORS=10

# ---- precheck: policy GPU must be mostly free ----
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${CUDA_VISIBLE_DEVICES} | tr -d " ")
echo "[precheck] policy GPU ${CUDA_VISIBLE_DEVICES} used ${used} MiB"
# PRECHECK_MAX high when the local usersim SHARES this GPU (single-card 80GB).
if [ "${used}" -gt "${PRECHECK_MAX:-6000}" ]; then
  echo "[ABORT] GPU ${CUDA_VISIBLE_DEVICES} busy (${used}MiB > ${PRECHECK_MAX:-6000})."; exit 1
fi
# ---- precheck: usersim endpoint must answer ----
if ! curl -sf "${TAU2_USER_API_BASE%/v1}/v1/models" >/dev/null 2>&1; then
  echo "[ABORT] usersim endpoint ${TAU2_USER_API_BASE} not reachable. Start serve_usersim_7b.sh first."; exit 1
fi

[[ -n "$VENV" ]] && source "$VENV/bin/activate"
# An editable verl checkout wants its own cwd; a pip-installed verl does not care.
[[ -d "$ROOT/verl" ]] && cd "$ROOT/verl"

# Policy model is overridable so we can validate the loop with 1.5B first
# (fits easily) before scaling to 7B (needs memory tuning: param_offload etc.).
MODEL_PATH=${POLICY_MODEL:-$ROOT/models/qwen25-7b-instruct}
TRAIN_FILE=$ROOT/data/tau2_airline/train.parquet
TEST_FILE=$ROOT/data/tau2_airline/test.parquet

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$TEST_FILE" \
    data.train_batch_size=${TRAIN_BS:-16} \
    data.max_prompt_length=${MAX_PROMPT:-6144} \
    data.max_response_length=${MAX_RESP:-3072} \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI:-8} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-False} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPT_OFFLOAD:-False} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=20 \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.multi_turn.max_tool_response_length=1024 \
    actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable \
    actor_rollout_ref.rollout.agent.default_agent_loop=tau2_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="$INTEG/agent_loop_config.yaml" \
    actor_rollout_ref.rollout.agent.num_workers=8 \
    actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN:-10240} \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL:-0.35} \
    actor_rollout_ref.rollout.n=${ROLLOUT_N:-8} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.use_v1=False \
    trainer.critic_warmup=0 \
    trainer.logger=[console,tensorboard] \
    trainer.val_before_train=True \
    trainer.resume_mode=${RESUME_MODE:-disable} \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.project_name=verl_tau2 \
    trainer.experiment_name=${EXP_NAME:-tau2_airline_qwen25_7b_grpo} \
    trainer.default_local_dir=$ROOT/ckpts/${EXP_NAME:-tau2_airline_7b} \
    trainer.save_freq=${SAVE_FREQ:-20} \
    trainer.test_freq=${TEST_FREQ:-10} \
    trainer.total_epochs=${EPOCHS:-3} \
    2>&1 | tee $ROOT/${EXP_NAME:-tau2_airline_7b_grpo}.log
