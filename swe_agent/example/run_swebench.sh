#!/usr/bin/env bash
# =============================================================================
# SWE-bench VERL Training (Single-Node / Multi-Node)
#
# Trains on real SWE-bench instances with per-instance Docker sandboxes.
# Defaults to single-node (8 GPU); set NNODES>1 for multi-node.
#
# Data preparation (run once):
#   cd verl && python3 recipe/swe_agent/prepare/prepare_data.py \
#       --mode swebench \
#       --swebench_train /path/to/swe_bench_train.json \
#       --swebench_test  /path/to/swe_bench_test.json \
#       --output_dir data/swe_bench
#
# Usage:
#   bash run_swebench.sh                                # single-node, 8 GPU
#   NNODES=4 bash run_swebench.sh                       # 4-node, Ray cluster
#   bash run_swebench.sh trainer.total_epochs=5          # override via Hydra
#
# Environment overrides:
#   MODEL_PATH, WORK_BASE, GPUS_PER_NODE, NNODES, TRAIN_BATCH_SIZE,
#   EXPERIMENT_NAME, NCCL_SOCKET_IFNAME, MASTER_ADDR
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
DATA_DIR=${DATA_DIR:-$VERL_ROOT/data/swe_bench}
train_files=$DATA_DIR/train.parquet
test_files=$DATA_DIR/test.parquet

if [ ! -f "$train_files" ]; then
    echo "[ERROR] SWE-bench data not found at $train_files"
    echo ""
    echo "Prepare data first:"
    echo "  cd $VERL_ROOT"
    echo "  python3 recipe/swe_agent/prepare/prepare_data.py \\"
    echo "      --mode swebench \\"
    echo "      --swebench_train /path/to/swe_bench_train.json \\"
    echo "      --swebench_test  /path/to/swe_bench_test.json \\"
    echo "      --output_dir data/swe_bench"
    exit 1
fi

# ================= Experiment =================
agent_loop_config_path=recipe/swe_agent/config/swe_agent_config_swebench.yaml
project_name=swe_bench_training
experiment_name=${EXPERIMENT_NAME:-qwen3-4b-swebench-v1}
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
use_kl_loss=false
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# ================= Training parameters =================
max_turns=10
max_prompt_length=8192
max_response_length=4096
actor_lr=5e-6

train_batch_size=${TRAIN_BATCH_SIZE:-8}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-4}
n_resp_per_prompt=1
n_resp_per_prompt_val=1

# ================= Logging =================
export RAY_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=1
export RAY_memory_usage_threshold=0.98

# ================= Performance =================
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

if [ "$NNODES" -gt 1 ]; then
    export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp96s0f0}
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-enp96s0f0}
    export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-enp96s0f0}
    export MASTER_ADDR=${MASTER_ADDR:?"MASTER_ADDR must be set for multi-node (e.g. export MASTER_ADDR=192.168.1.100)"}
    export RAY_ADDRESS=${RAY_ADDRESS:-auto}
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
gpu_memory_utilization=0.6
max_model_len=12288
rollout_prompt_length=$max_prompt_length
actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 2 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 2 ))

# ================= Safety checks =================
num_agent_workers=$(( GPUS_PER_NODE * NNODES ))
if (( train_batch_size % num_agent_workers != 0 )); then
    echo "[ERROR] train_batch_size ($train_batch_size) must be divisible by total agent workers ($num_agent_workers = GPUS_PER_NODE * NNODES)"
    echo "        Suggested values: $num_agent_workers, $((num_agent_workers * 2)), ..."
    exit 1
fi

if (( ppo_mini_batch_size > train_batch_size )); then
    echo "[ERROR] ppo_mini_batch_size ($ppo_mini_batch_size) must be <= train_batch_size ($train_batch_size)"
    exit 1
fi

if (( max_prompt_length + max_response_length > max_model_len )); then
    echo "[ERROR] Invalid token budget: max_prompt_length ($max_prompt_length) + max_response_length ($max_response_length) > max_model_len ($max_model_len)"
    exit 1
fi

train_files="['$train_files']"
test_files="['$test_files']"

echo "=========================================="
echo "SWE-bench Training ($NNODES node(s), ${GPUS_PER_NODE} GPUs each)"
echo "  Model:          $model_path"
echo "  Experiment:     $experiment_name"
echo "  Data:           $DATA_DIR"
echo "  batch_size:     $train_batch_size"
echo "  max_turns:      $max_turns"
echo "  TP=$infer_tp  SP=$train_sp"
echo "=========================================="

# ================= Build Hydra overrides =================
MULTI_NODE_ARGS=()
if [ "$NNODES" -gt 1 ]; then
    MULTI_NODE_ARGS=(
        "+ray_kwargs.ray_init.runtime_env.env_vars.NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
        "+ray_kwargs.ray_init.runtime_env.env_vars.GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME"
        "+ray_kwargs.ray_init.runtime_env.env_vars.TP_SOCKET_IFNAME=$TP_SOCKET_IFNAME"
        "+ray_kwargs.ray_init.runtime_env.env_vars.MASTER_ADDR=$MASTER_ADDR"
    )
fi

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
    trainer.val_before_train=false \
    trainer.log_val_generations=10 \
    trainer.nnodes="$NNODES" \
    trainer.save_freq=5 \
    trainer.default_local_dir="$default_local_dir" \
    trainer.test_freq=5 \
    trainer.total_epochs=2 \
    trainer.rollout_data_dir="$rollout_data_dir" \
    trainer.validation_data_dir="$validation_data_dir" \
    "${MULTI_NODE_ARGS[@]}" "$@"
