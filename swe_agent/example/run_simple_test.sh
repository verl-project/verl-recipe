#!/usr/bin/env bash
# =============================================================================
# SWE-Agent Simple Test Training
#
# Self-contained script: generates synthetic data and launches FSDP training.
# Designed for quick validation on a single node (e.g. 8× RTX 3090).
#
# Usage:
#   bash run_simple_test.sh                          # default: 10 epochs
#   bash run_simple_test.sh trainer.total_epochs=2   # quick 2-epoch run
#
# Key hyperparameters (override via environment or Hydra CLI):
#   MODEL_PATH       Model directory         (default: Qwen3-4B-Instruct-2507)
#   WORK_BASE        Workspace root          (default: ~/workspace)
#   TRAIN_BATCH_SIZE Prompts per rollout      (default: 8)
#   GPUS_PER_NODE    Number of GPUs           (default: 8)
# =============================================================================

set -xeuo pipefail

# ================= Work directories =================
WORK_BASE=${WORK_BASE:-$HOME/workspace}
export TMPDIR=$WORK_BASE/tmp  TEMP=$WORK_BASE/tmp  TMP=$WORK_BASE/tmp
export RAY_TMPDIR=$WORK_BASE/ray_tmp
export TRITON_CACHE_DIR=$WORK_BASE/triton_cache
export TORCH_EXTENSIONS_DIR=$WORK_BASE/torch_extensions
export HF_HOME=$WORK_BASE/hf_cache
export XDG_CACHE_HOME=$WORK_BASE/cache
mkdir -p "$TMPDIR" "$RAY_TMPDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$HF_HOME" "$XDG_CACHE_HOME"

# ================= Cluster topology =================
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NNODES=${NNODES:-1}
export RAY_NUM_NODES=$NNODES

# ================= Paths =================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECIPE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VERL_ROOT="$(cd "$RECIPE_DIR/../.." && pwd)"

model_path=${MODEL_PATH:-/path/to/model}

# ================= Data =================
DATA_DIR=$VERL_ROOT/data/swe_agent_test
train_files=$DATA_DIR/train.parquet
test_files=$DATA_DIR/test.parquet

train_size=${TRAIN_SIZE:-12}
test_size=${TEST_SIZE:-4}
if [ ! -f "$train_files" ]; then
    echo "[INFO] Generating synthetic test data (train=$train_size, test=$test_size)..."
    python3 "$RECIPE_DIR/prepare/prepare_data.py" \
        --mode simple --train_size "$train_size" --test_size "$test_size" \
        --output_dir "$DATA_DIR"
fi

# ================= Experiment =================
agent_loop_config_path=recipe/swe_agent/config/swe_agent_config.yaml
project_name=swe_agent_test
experiment_name=${EXPERIMENT_NAME:-qwen3-4b-simple-v1}
default_local_dir=$WORK_BASE/checkpoints/$experiment_name

rollout_data_dir=$WORK_BASE/trajectories/$experiment_name/rollout
validation_data_dir=$WORK_BASE/trajectories/$experiment_name/validation
mkdir -p "$rollout_data_dir" "$validation_data_dir"

export VERL_FILE_LOGGER_PATH=$WORK_BASE/logs/${experiment_name}_metrics.jsonl
mkdir -p "$(dirname "$VERL_FILE_LOGGER_PATH")"

# ================= Algorithm =================
adv_estimator=grpo
use_kl_in_reward=false
kl_coef=0.0
use_kl_loss=true
kl_loss_coef=0.001
clip_ratio_low=0.1
clip_ratio_high=0.15

# ================= Training parameters =================
max_turns=5
max_prompt_length=4096
max_response_length=4096
actor_lr=5e-6
ppo_epochs=4

train_batch_size=${TRAIN_BATCH_SIZE:-8}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}
n_resp_per_prompt=${N_RESP_PER_PROMPT:-4}
n_resp_per_prompt_val=1

# ================= Logging =================
export RAY_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1

# ================= Performance =================
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

if [ "$NNODES" -gt 1 ]; then
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp96s0f0}
else
    unset NCCL_SHM_DISABLE 2>/dev/null || true
    unset NCCL_P2P_DISABLE 2>/dev/null || true
    unset NCCL_SOCKET_IFNAME 2>/dev/null || true
fi

# ================= Parallelism =================
infer_tp=$GPUS_PER_NODE
train_sp=$GPUS_PER_NODE

# ================= FSDP =================
fsdp_strategy=fsdp2
offload_policy=true
param_offload=false
optimizer_offload=false

# ================= vLLM =================
gpu_memory_utilization=0.5
max_model_len=4096
rollout_prompt_length=$max_prompt_length
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * n_resp_per_prompt ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

train_files="['$train_files']"
test_files="['$test_files']"

echo "=========================================="
echo "SWE-Agent Simple Test Training"
echo "  Model:              $model_path"
echo "  Experiment:         $experiment_name"
echo "  GPUs:               $GPUS_PER_NODE × $NNODES node(s)"
echo "  batch / n_resp:     $train_batch_size × $n_resp_per_prompt"
echo "  ppo_epochs:         $ppo_epochs"
echo "  actor_lr:           $actor_lr"
echo "  TP=$infer_tp  SP=$train_sp"
echo "=========================================="

python3 -m verl.trainer.main_ppo \
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
    actor_rollout_ref.actor.ppo_epochs=$ppo_epochs \
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
    actor_rollout_ref.rollout.max_num_seqs=16 \
    actor_rollout_ref.rollout.max_num_batched_tokens=16384 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    custom_reward_function.path="${RECIPE_DIR}/reward/reward.py" \
    custom_reward_function.name=compute_score \
    trainer.logger='["console","file"]' \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node="$GPUS_PER_NODE" \
    trainer.val_before_train=true \
    trainer.log_val_generations=10 \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=999 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.test_freq=2 \
    trainer.total_epochs=10 \
    trainer.rollout_data_dir="$rollout_data_dir" \
    trainer.validation_data_dir="$validation_data_dir" "$@"
