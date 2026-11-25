#!/usr/bin/env bash
# DeepSWE | Qwen3.5-397B-A17B | R2E-Gym + ReAct (uni-agent) | TorchTitan trainer + FP8 vLLM rollout | TPU v7x
#
# Defaults match our validated 384-chip TPU v7x run (32 hosts / 128 chips / 256 TensorCores trainer with
# TP=1, CP=4 Ulysses, DP_shard=64, EP=8, static RoE dispatcher, 8 token chunks, paired grouped MM, bf16 params +
# fp32 masters + CPU optimizer offload, bf16 gradient reduce-scatter; 64 x 2x2x1 rollout replicas = 256 chips / 512 TensorCores with on-load
# online FP8 MoE quantization and dummy weight load), 64k context (8k prompt + 56k response), RLOO, wait off-policy,
# lookahead sandbox warming, and 0.992_1.008 rollout correction band.
#
# Delegates launcher wiring and Hydra overrides to run_qwen3_5_35b_deepswe.sh.

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model & experiment
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-397B-A17B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_397b_deepswe_r2e_gym}"

# Trainer topology: 32 hosts x 8 TensorCores = 256 TensorCores (128 TPU v7x chips), 4x4x8 slice.
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-tpu7x}"
export NNODES_TRAINER="${NNODES_TRAINER:-32}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-8}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-4}"
export CONTEXT_PARALLEL_STRATEGY="${CONTEXT_PARALLEL_STRATEGY:-ulysses}"
export DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-$((NNODES_TRAINER * N_CHIPS_TRAINER / TENSOR_PARALLEL_SIZE / CONTEXT_PARALLEL_SIZE))}"
export EXPERT_PARALLEL_SIZE="${EXPERT_PARALLEL_SIZE:-8}"
export FUSE_QKV="${FUSE_QKV:-True}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"

# Trainer memory & EP compute
export USE_GRAPH_TRAINER_OPT="${USE_GRAPH_TRAINER_OPT:-False}"
export PARAM_DTYPE="${PARAM_DTYPE:-bfloat16}"
export FP32_MASTER="${FP32_MASTER:-True}"
# dtype of the FSDP gradient reduce-scatter. The parameters are bf16 already (the fp32 masters stay on the host), so
# bf16 halves the reduce-scatter traffic across the DP shards; set float32 for torchtitan's default.
MIXED_PRECISION_REDUCE="${MIXED_PRECISION_REDUCE:-bfloat16}"
export OPTIMIZER_HOST_OFFLOAD="${OPTIMIZER_HOST_OFFLOAD:-True}"
export OPTIM_IMPL="${OPTIM_IMPL:-fused}"
# verl-hardware-plugin#187 replaced ep_dispatcher with ep_transport (default "auto" -> Ring-of-Experts when EP <= top_k).
export EP_DISPATCHER="${EP_DISPATCHER:-}"
export EP_TRANSPORT="${EP_TRANSPORT:-auto}"
export EP_TOKEN_CHUNKS="${EP_TOKEN_CHUNKS:-8}"
export EP_PAIRED_GROUPED_MM="${EP_PAIRED_GROUPED_MM:-True}"
export AC_SAVE_SPLASH_AND_ROUTING="${AC_SAVE_SPLASH_AND_ROUTING:-True}"
export SPARSECORE_OFFLOAD="${SPARSECORE_OFFLOAD:-True}"
export TRAINER_SCOPED_VMEM_LIMIT_KIB="${TRAINER_SCOPED_VMEM_LIMIT_KIB-65472}"
# SPARSECORE_OFFLOAD=True appends the SparseCore + HLO sharding v3=false flags in run_qwen3_5_torchtitan.sh;
# keeping TRAINER_EXTRA_LIBTPU_INIT_ARGS empty avoids duplicating them.
export TRAINER_EXTRA_LIBTPU_INIT_ARGS="${TRAINER_EXTRA_LIBTPU_INIT_ARGS-}"
export SPLASH_MAX_BLOCK_SIZE="${SPLASH_MAX_BLOCK_SIZE-1024}"
export FROZEN_PARAM_REGEX="${FROZEN_PARAM_REGEX-moe.router.gate}"
export GDN_ROW_SLOT_LEN="${GDN_ROW_SLOT_LEN:-}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"

# Sampler topology & online FP8 MoE quantization: 64 x 2x2x1 replicas (512 TensorCores = 256 TPU v7x chips).
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-64}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-8}"
export ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-8}"
export ROLLOUT_EXPERT_PARALLEL_SIZE="${ROLLOUT_EXPERT_PARALLEL_SIZE:-8}"
export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-dummy}"
export QUANTIZE_ON_LOAD_PREFIXES="${QUANTIZE_ON_LOAD_PREFIXES:-experts}"
# vllm-torchtpu quantizes the experts to FP8 at load time in fp32 chunks of this many bytes (its default is 1 GiB).
# On a one-host 2x2x1 replica that default leaves almost no HBM headroom at startup, so use 256 MiB chunks.
export TPU_MOE_REQUANT_CHUNK_BYTES="${TPU_MOE_REQUANT_CHUNK_BYTES:-268435456}"
export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY:-QUANTIZE_ON_LOAD_PREFIXES,TPU_MOE_REQUANT_CHUNK_BYTES}"
export ROLLOUT_FP8_MOE="${ROLLOUT_FP8_MOE:-True}"

# 64k context window (8k prompt + 56k response) & static compiled micro-batch shape
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-57344}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
# With USE_DYNAMIC_BSZ=True a micro-batch is packed up to MAX_TOKEN_LEN_PER_DEVICE x TP x CP tokens, so the per-device
# budget is divided by TP x CP to keep a micro-batch at MICRO_BATCH_SIZE full-length sequences (the compiled shape).
if [[ "${USE_DYNAMIC_BSZ,,}" == "true" ]]; then
  export MAX_TOKEN_LEN_PER_DEVICE="${MAX_TOKEN_LEN_PER_DEVICE:-$((MICRO_BATCH_SIZE * MAX_MODEL_LEN / (TENSOR_PARALLEL_SIZE * CONTEXT_PARALLEL_SIZE)))}"
else
  export MAX_TOKEN_LEN_PER_DEVICE="${MAX_TOKEN_LEN_PER_DEVICE:-$((MICRO_BATCH_SIZE * MAX_MODEL_LEN))}"
fi
export STATIC_MICRO_BATCH_LEN="${STATIC_MICRO_BATCH_LEN-$((MICRO_BATCH_SIZE * MAX_MODEL_LEN))}"
export STATIC_MAX_RESPONSE_LEN="${STATIC_MAX_RESPONSE_LEN-${MAX_RESPONSE_LEN}}"
export PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-4096}"

# Batch size, optimization & schedule
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export LEARNING_RATE="${LEARNING_RATE:-2e-6}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-5}"
export CLIP_GRAD="${CLIP_GRAD:-0.0625}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-40}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-70}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-70}"
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

# Sampler decoding, overlap & sync
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-4096}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.75}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-True}"
export VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE-512}"
export VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING:-False}"
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-2}"
export MAX_OFF_POLICY_STRATEGY="${MAX_OFF_POLICY_STRATEGY:-wait}"
export MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-3}"
export VERL_SANDBOX_LOOKAHEAD="${VERL_SANDBOX_LOOKAHEAD:-1}"
export TPU_MOE_SKIP_PADDED_TOKENS="${TPU_MOE_SKIP_PADDED_TOKENS:-1}"
export TPU_ENABLE_GDN_DYNAMIC_TILING="${TPU_ENABLE_GDN_DYNAMIC_TILING:-1}"
export USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-1}"
export TPU_SYNC_VERIFY_PARITY="${TPU_SYNC_VERIFY_PARITY:-norm}"
export TPU_SYNC_RELEASE_BUFFERS="${TPU_SYNC_RELEASE_BUFFERS:-True}"

# RL algorithm & rollout correction
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo_loo}"
export ROLLOUT_IS="${ROLLOUT_IS:-token}"
export ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-5.0}"
export ROLLOUT_RS="${ROLLOUT_RS:-seq_mean_k1,seq_mean_k2}"
export ROLLOUT_RS_THRESHOLD="${ROLLOUT_RS_THRESHOLD:-0.98_1.02,0.80}"
# BYPASS_MODE=False (default): decoupled PPO, the trainer recomputes old_log_prob with a forward-only pass.
# BYPASS_MODE=True: 2-policy bypass using rollout log-probs directly (with BYPASS_LOSS_TYPE=reinforce or ppo_clip).
export BYPASS_MODE="${BYPASS_MODE:-False}"
export BYPASS_LOSS_TYPE="${BYPASS_LOSS_TYPE:-reinforce}"
# Optional, BYPASS_MODE=False only: static micro-batch length in tokens for the forward-only old_log_prob pass
# (for example 131072, twice the training micro-batch). Empty uses STATIC_MICRO_BATCH_LEN.
export STATIC_LOG_PROB_MICRO_BATCH_LEN="${STATIC_LOG_PROB_MICRO_BATCH_LEN:-}"

# Agent & sandbox concurrency (1024 concurrent R2E-Gym trajectories = 64 x 16)
export TASK_CONFIG="${TASK_CONFIG:-${SCRIPT_DIR}/task_config_64k.yaml}"
export GATEWAY_COUNT="${GATEWAY_COUNT:-16}"
export CONCURRENCY="${CONCURRENCY:-1024}"
export MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-False}"

exec bash "${SCRIPT_DIR}/run_qwen3_5_35b_deepswe.sh" \
  "++actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.ep_transport=${EP_TRANSPORT}" \
  "++actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.rollout_fp8_moe=${ROLLOUT_FP8_MOE}" \
  "++actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.mixed_precision_reduce=${MIXED_PRECISION_REDUCE}" \
  "actor_rollout_ref.actor.optim.clip_grad=${CLIP_GRAD}" \
  "actor_rollout_ref.rollout.top_k=-1" \
  "actor_rollout_ref.rollout.val_kwargs.top_k=-1" \
  "++actor_rollout_ref.rollout.engine_kwargs.vllm.max_logprobs=0" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.QUANTIZE_ON_LOAD_PREFIXES='${QUANTIZE_ON_LOAD_PREFIXES}'" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.TPU_MOE_REQUANT_CHUNK_BYTES='${TPU_MOE_REQUANT_CHUNK_BYTES}'" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VLLM_RAY_EXTRA_ENV_VARS_TO_COPY='${VLLM_RAY_EXTRA_ENV_VARS_TO_COPY}'" \
  "$@"
