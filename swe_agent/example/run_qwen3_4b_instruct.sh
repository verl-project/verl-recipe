#!/usr/bin/env bash
# =============================================================================
# SWE Agent VERL Training Script - Quick Test Version
#
# Quick test version based on run_qwen3_4b_instaruction.sh
# Uses small data to verify the refactored code works correctly
# =============================================================================

set -xeuo pipefail

# ================= Work directories =================
# Set WORK_BASE to control where all cache/tmp files go.
# Override with: export WORK_BASE=/your/path before running this script.
WORK_BASE=${WORK_BASE:-/data1/lmy/workspace}
export TMPDIR=$WORK_BASE/tmp
export TEMP=$WORK_BASE/tmp
export TMP=$WORK_BASE/tmp
export RAY_TMPDIR=$WORK_BASE/ray_tmp
export TRITON_CACHE_DIR=$WORK_BASE/triton_cache
export TORCH_EXTENSIONS_DIR=$WORK_BASE/torch_extensions
export HF_HOME=$WORK_BASE/hf_cache
export XDG_CACHE_HOME=$WORK_BASE/cache
mkdir -p $TMPDIR $RAY_TMPDIR $TRITON_CACHE_DIR $TORCH_EXTENSIONS_DIR $HF_HOME $XDG_CACHE_HOME

# ================= cluster topology =================
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export RAY_NUM_NODES=$NNODES

echo "=========================================="
echo "SWE Agent Quick Test - Using run_qwen3_4b.sh Config"
echo "=========================================="
echo "==========================================="
echo "Using $NNODES nodes and $GPUS_PER_NODE GPUs per node..."
echo "==========================================="

# ================= paths =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERL_ROOT="$(cd "$RECIPE_DIR/../.." && pwd)"

# ========== Model config ==========
model_path=${MODEL_PATH:-/data1/models/Qwen/Qwen3-4B-Instruct-2507}

# ========== Test data config (using small dataset) ==========
DATA_DIR=$VERL_ROOT/data/swe_agent_test
train_files=$DATA_DIR/train.parquet
test_files=$DATA_DIR/test.parquet

if [ ! -f "$train_files" ]; then
    echo "[ERROR] Test data not found at $train_files"
    echo "Run: python3.12 recipe/swe_agent/prepare/prepare_data.py --mode simple --train_size 8 --test_size 2 --output_dir data/swe_agent_test"
    exit 1
fi

# ========== Agent Loop config ==========
agent_loop_config_path=recipe/swe_agent/config/swe_agent_config.yaml

# =================== wandb ===================
project_name=swe_agent_test
experiment_name=qwen3-4b-swe-train-v1
default_local_dir=$WORK_BASE/checkpoints/$experiment_name

# ================= algorithm =================
adv_estimator=grpo
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=false
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

# ========== Training parameters ==========
max_turns=15              # Give agent enough turns to complete task+submit (incl. ~5 format retries)
max_prompt_length=4096    # Actual prompt ~300 tokens, but padding upper bound needs headroom
max_response_length=16384 # Multi-turn dialogue accumulates long responses (incl. retries), 16k reduces truncation
actor_lr=5e-6             # GRPO typically needs higher LR than PPO to learn signal

train_batch_size=8        # Should match agent_loop_workers count
ppo_mini_batch_size=4     # mini_batch < batch: multiple gradient updates per step for better data utilization
n_resp_per_prompt=1
n_resp_per_prompt_val=1

# =================== logging ===================
export RAY_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

# ================= performance =================
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# Fix /dev/shm space issue - use alternative communication methods
export NCCL_SHM_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SOCKET_IFNAME=lo

# Key optimization: use all GPUs for TP/SP to distribute memory pressure
infer_tp=8  # vLLM tensor parallel - use all 8 GPUs to reduce per-GPU memory
train_sp=8  # Ulysses sequence parallel - must match GPU count

# ================= FSDP Optimization =================
fsdp_strategy=fsdp2
offload_policy=true
param_offload=false
optimizer_offload=false

# ================= vLLM Memory Optimization =================
# In async mode, actor model is initialized first, leaving limited free memory.
# vLLM must fit in remaining space. Use conservative settings.
gpu_memory_utilization=0.4   # Per-GPU 24GB, training peak ~13.5GB, vLLM can use ~40%
# max_model_len: vLLM max sequence length, must be >= prompt + response generation length
max_model_len=16384          # 16k supports longer multi-turn interactions

# prompt_length is used for rollout padding, must be <= max_model_len
# actor_max_token_len_per_gpu must be >= (prompt_length + response_length)
rollout_prompt_length=$max_prompt_length
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))   # = 24576
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))              # = 49152

train_files="['$train_files']"
test_files="['$test_files']"

echo "=========================================="
echo "Configuration:"
echo "  Model: $model_path"
echo "  Train data: $train_files"
echo "  Test data: $test_files"
echo "  Agent config: $agent_loop_config_path"
echo "  Max turns: $max_turns"
echo "  Batch size: $train_batch_size"
echo "  TP: $infer_tp, SP: $train_sp"
echo "=========================================="

# ================= Pre-install SWE-Agent tools (for local mode only) =================
# NOTE: In Docker mode (default), tools are installed inside containers, so this step
# only helps when running in local deployment mode to prevent FileExistsError.
# In Docker mode with --network host, this step can be safely skipped.
SWE_AGENT_TOOLS_SRC="$VERL_ROOT/../SWE-agent/tools"
SWE_AGENT_TOOLS_DST="/root/tools"

echo "Checking SWE-Agent tools pre-installation (for local mode)..."
if [ -d "$SWE_AGENT_TOOLS_SRC" ]; then
    # Create destination directory if it doesn't exist
    if ! mkdir -p "$SWE_AGENT_TOOLS_DST" 2>/dev/null; then
        echo "⚠ Warning: Cannot create $SWE_AGENT_TOOLS_DST (permission denied?)"
        echo "  This is fine if using Docker mode (default)."
    else
        # Clean existing tools to avoid conflicts
        if [ -d "$SWE_AGENT_TOOLS_DST" ] && [ "$(ls -A "$SWE_AGENT_TOOLS_DST" 2>/dev/null)" ]; then
            rm -rf "$SWE_AGENT_TOOLS_DST"/* 2>/dev/null || true
        fi
        
        # Copy tools
        if cp -r "$SWE_AGENT_TOOLS_SRC"/* "$SWE_AGENT_TOOLS_DST"/ 2>/dev/null; then
            echo "✓ SWE-Agent tools pre-installed to $SWE_AGENT_TOOLS_DST"
        else
            echo "⚠ Warning: Failed to copy SWE-Agent tools."
            echo "  This is fine if using Docker mode (default)."
        fi
    fi
else
    echo "⚠ SWE-Agent tools source not found at $SWE_AGENT_TOOLS_SRC"
    echo "  This is fine if using Docker mode (tools will be installed in containers)."
fi
echo ""

python3.12 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.return_raw_chat=true \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=true \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.enable_activation_offload=true \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.strategy=$fsdp_strategy \
    actor_rollout_ref.actor.fsdp_config.offload_policy=$offload_policy \
    actor_rollout_ref.actor.fsdp_config.param_offload=$param_offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$optimizer_offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=$agent_loop_config_path \
    actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
    actor_rollout_ref.rollout.prompt_length=$rollout_prompt_length \
    actor_rollout_ref.rollout.max_model_len=$max_model_len \
    actor_rollout_ref.rollout.max_num_seqs=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    custom_reward_function.path="${RECIPE_DIR}/reward/compute_score.py" \
    custom_reward_function.name=compute_score \
    trainer.logger='["console"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.val_before_train=false \
    trainer.log_val_generations=10 \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=5 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.test_freq=5 \
    trainer.total_epochs=2 "$@"
