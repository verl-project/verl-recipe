#!/usr/bin/env bash
# GRPO training with SGLang rollouts served through Dynamo's SGLang sidecar.
#
# Each rollout shard runs stock `sglang.launch_server` plus `dynamo.sglang.sidecar`.
# Turns go to the Dynamo frontend's SGLang-native /generate. Weight sync and
# sleep/wake use SGLang's own HTTP routes. See README "SGLang sidecar mode".
#
# Prerequisites, on every node:
#   * a verl checkout with this recipe at recipe/dynamo, run from the verl root
#   * sglang with the native gRPC server (0.5.16+; 0.5.20 tested)
#   * ai-dynamo with the SGLang sidecar (1.7.0.dev20261110+), plus nixl for PD_PREFILL
#   * etcd and nats-server on PATH
#
# Usage (1 node x 8 GPUs, two TP4 rollout shards):
#   MODEL_PATH=/models/Qwen3-30B-A3B-Instruct TRAIN_FILE=train.parquet TEST_FILE=test.parquet \
#     bash recipe/dynamo/run_sglang_sidecar_rl.sh
#   DP_ATTN=4    bash recipe/dynamo/run_sglang_sidecar_rl.sh   # DP-attention 4 / EP4 in each shard
#   PD_PREFILL=1 bash recipe/dynamo/run_sglang_sidecar_rl.sh   # shard 0 prefill, shard 1 decode
#   R3=1         bash recipe/dynamo/run_sglang_sidecar_rl.sh   # Megatron + rollout routing replay
# Extra arguments are Hydra overrides, appended last.
#
# Knobs: NNODES=1 NGPUS_PER_NODE=8 ROLLOUT_TP=4 TRAIN_BATCH=16 ROLLOUT_N=8
# TOTAL_STEPS=100 MAX_PROMPT_LENGTH=1024 MAX_RESPONSE_LENGTH=2048 GPU_MEM_UTIL=0.6
#
# To check rollout/training agreement, watch training/rollout_actor_probs_pearson_corr.
# On 8xH100 with Qwen3-Coder-30B-A3B (FSDP), verl's native SGLang rollout gave 0.9975.
# The sidecar with TP4 shards gave 0.9982, with DP_ATTN=4 0.9976, and with PD_PREFILL=1
# 0.9980 (against 0.9976 native in that run).
set -xeuo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
TRAIN_FILE=${TRAIN_FILE:?set TRAIN_FILE (parquet)}
TEST_FILE=${TEST_FILE:-${TRAIN_FILE}}
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
ROLLOUT_TP=${ROLLOUT_TP:-4}
TRAIN_BATCH=${TRAIN_BATCH:-16}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_STEPS=${TOTAL_STEPS:-100}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-2048}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.6}
DP_ATTN=${DP_ATTN:-1}
PD_PREFILL=${PD_PREFILL:-0}
R3=${R3:-0}

python3 -c "import dynamo.sglang.sidecar, sglang" || {
  echo "dynamo.sglang.sidecar / sglang not importable" >&2
  exit 2
}
# torch_memory_saver (SGLang sleep/wake) refuses a custom CUDA allocator config.
unset PYTORCH_CUDA_ALLOC_CONF PYTORCH_ALLOC_CONF || true
export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe.dynamo.register}"
# Loading a 30B checkpoint from network storage can outlast the default 600 s.
export VERL_DYNAMO_FE_READY_TIMEOUT=${VERL_DYNAMO_FE_READY_TIMEOUT:-1800}
CONFIG_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/config" && pwd)

ARGS=(
  ++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.launch_mode=sidecar
  ++actor_rollout_ref.rollout.engine_kwargs.dynamo.generate_api=generate
  # The engine binds its ports only after loading weights; use stable per-shard ports.
  ++actor_rollout_ref.rollout.engine_kwargs.dynamo.stable_kv_event_ports=true
)
if [[ "${R3}" == "1" ]]; then
  # Megatron actor with router_replay.mode=R3; the config sets the rollout side.
  CONFIG_NAME=dynamo_trainer_v1_colocate_sglang_r3
else
  CONFIG_NAME=dynamo_trainer_v1_colocate_sglang
  # Fits a 30B-A3B FSDP actor next to two TP4 engines on 8x80 GB.
  ARGS+=(
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.use_kl_loss=False
  )
fi
if [[ "${DP_ATTN}" != "1" ]]; then
  ARGS+=("++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.extra_args=[\"--dp-size\",\"${DP_ATTN}\",\"--enable-dp-attention\",\"--ep-size\",\"${DP_ATTN}\"]")
fi
if [[ "${PD_PREFILL}" != "0" ]]; then
  ARGS+=(++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.disaggregation.prefill_shards="${PD_PREFILL}")
fi

python3 -m verl.trainer.main_ppo \
  --config-path "${CONFIG_DIR}" \
  --config-name="${CONFIG_NAME}" \
  "ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_USE_EXTERNAL_MODULES}'" \
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_DYNAMO_FE_READY_TIMEOUT='${VERL_DYNAMO_FE_READY_TIMEOUT}'" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.train_batch_size="${TRAIN_BATCH}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH}" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192 \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.multi_turn.enable=False \
  trainer.logger=console \
  trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
  trainer.nnodes="${NNODES}" \
  trainer.val_before_train=False \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.test_freq=-1 \
  trainer.save_freq=-1 \
  "${ARGS[@]}" \
  "$@"
