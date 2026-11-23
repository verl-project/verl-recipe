#!/usr/bin/env bash
# DeepSWE | Qwen3.5-35B-A3B | R2E-Gym + ReAct (uni-agent) | TorchTitan trainer + vLLM rollout | TPU v7x
#
# Defaults are tuned for TPU v7x (2x2x2 FSDP16 trainer + 3 x 2x2x1 EP8 rollout replicas, 32k context,
# 16 issues x 16 attempts = 256 trajectories/step, prefix caching + 512-token KV blocks, fused EP MoE
# kernel, on-device fp32 AdamW, Splash block size 1024, frozen MoE router gate, 0.998_1.002 rollout
# correction band, vLLM async_scheduling disabled for tpu-sync stability).
#
# Preprocess R2E-Gym into non-overlapping train/eval Parquet splits beforehand:
#   PYTHONPATH=uni-agent python3 deepswe_tpu/prepare_data.py --local-save-dir "${RAY_DATA_HOME}/data/r2e_gym"

set -xeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${VERL_ROOT:-}" ]]; then
  if [[ -f "${SCRIPT_DIR}/../../examples/tpu/grpo/run_qwen3_5_torchtitan.sh" ]]; then
    VERL_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
  elif [[ -f "/workspace/verl/examples/tpu/grpo/run_qwen3_5_torchtitan.sh" ]]; then
    VERL_ROOT="/workspace/verl"
  elif [[ -f "${HOME}/verl/examples/tpu/grpo/run_qwen3_5_torchtitan.sh" ]]; then
    VERL_ROOT="${HOME}/verl"
  else
    echo "ERROR: Set VERL_ROOT to your verl checkout containing examples/tpu/grpo/run_qwen3_5_torchtitan.sh" >&2
    exit 1
  fi
fi

if [[ -f "${VERL_ROOT}/recipe/deepswe_tpu/deepswe.py" ]]; then
  DEEPSWE_MODULE="${DEEPSWE_MODULE:-recipe.deepswe_tpu.deepswe}"
else
  DEEPSWE_MODULE="${DEEPSWE_MODULE:-deepswe_tpu.deepswe}"
fi

# Topology: trainer 2 hosts x 8 TensorCores (2x2x2), FSDP16; 3 rollout replicas of 8 chips (2x2x1), TP8/EP8.
export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-tpu7x}"
export NNODES_TRAINER="${NNODES_TRAINER:-2}"
export N_CHIPS_TRAINER="${N_CHIPS_TRAINER:-8}"
export NNODES_ROLLOUT="${NNODES_ROLLOUT:-3}"
export N_CHIPS_ROLLOUT="${N_CHIPS_ROLLOUT:-8}"
export ROLLOUT_TENSOR_PARALLEL_SIZE="${ROLLOUT_TENSOR_PARALLEL_SIZE:-8}"
export ROLLOUT_EXPERT_PARALLEL_SIZE="${ROLLOUT_EXPERT_PARALLEL_SIZE:-8}"
export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-dummy}"
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_SIZE="${CONTEXT_PARALLEL_SIZE:-1}"
export CONTEXT_PARALLEL_STRATEGY="${CONTEXT_PARALLEL_STRATEGY:-all_gather}"
export DATA_PARALLEL_SHARD_SIZE="${DATA_PARALLEL_SHARD_SIZE:-$((NNODES_TRAINER * N_CHIPS_TRAINER / TENSOR_PARALLEL_SIZE / CONTEXT_PARALLEL_SIZE))}"
export EXPERT_PARALLEL_SIZE="${EXPERT_PARALLEL_SIZE:-1}"
export FUSE_QKV="${FUSE_QKV:-True}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"

# Trainer memory/compute: on-device fp32 parameters + fused AdamW, no host offload, SparseCore FSDP collectives.
export PARAM_DTYPE="${PARAM_DTYPE:-float32}"
export OPTIMIZER_HOST_OFFLOAD="${OPTIMIZER_HOST_OFFLOAD:-False}"
export OPTIM_IMPL="${OPTIM_IMPL:-fused}"
export SPARSECORE_OFFLOAD="${SPARSECORE_OFFLOAD:-False}"
export TRAINER_SCOPED_VMEM_LIMIT_KIB="${TRAINER_SCOPED_VMEM_LIMIT_KIB-65472}"
_DEFAULT_TRAINER_EXTRA_LIBTPU_INIT_ARGS="--xla_tpu_enable_latency_hiding_layer_scheduler=true --xla_enable_async_all_gather=true --xla_tpu_prefer_async_allgather_to_allreduce=true --xla_tpu_enable_sparse_core_collective_aggregator=true --xla_tpu_enable_sparse_core_offload_queuing_in_lhs=true --xla_tpu_enable_sparse_core_collective_offload_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_2d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_3d_all_gather=true --xla_tpu_enable_sparse_core_collective_offload_reduce_scatter=true --xla_tpu_enable_sparse_core_reduce_scatter_v2=true --xla_tpu_use_single_sparse_core_for_reduce_scatter_offload=false --xla_tpu_enable_layer_scheduler_for_dependent_collectives=true"
export TRAINER_EXTRA_LIBTPU_INIT_ARGS="${TRAINER_EXTRA_LIBTPU_INIT_ARGS-${_DEFAULT_TRAINER_EXTRA_LIBTPU_INIT_ARGS}}"
export AC_SAVE_SPLASH_AND_ROUTING="${AC_SAVE_SPLASH_AND_ROUTING:-True}"
export USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"

RAY_DATA_HOME="${RAY_DATA_HOME:-/data}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
export MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/${MODEL_NAME}}"
export TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/r2e_gym/train.parquet}"
export TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/r2e_gym/test.parquet}"
export PROJECT_NAME="${PROJECT_NAME:-verl-tpu-deepswe}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_35b_deepswe_r2e_gym}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-${TRAIN_BATCH_SIZE}}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-100}"
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"

# 32k context; one 32768-token micro-batch per rank, padded to one compiled shape.
export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-24576}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
MAX_TOKEN_LEN_PER_DEVICE="${MAX_TOKEN_LEN_PER_DEVICE:-$((MICRO_BATCH_SIZE * MAX_MODEL_LEN))}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-${MAX_TOKEN_LEN_PER_DEVICE}}"
export STATIC_MICRO_BATCH_LEN="${STATIC_MICRO_BATCH_LEN-${MAX_TOKEN_LEN_PER_DEVICE}}"
export PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-2048}"

# Sampler, prefix caching, and overlap.
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-True}"
VLLM_BLOCK_SIZE="${VLLM_BLOCK_SIZE-512}"
# vLLM 0.32 enables async scheduling by default. With tpu-sync's in-flight weight updates (abort + pause + resume),
# the scheduler and model runner can desync and gpu_model_runner._update_states raises
# KeyError(req_data.all_token_ids[req_id]) -> EngineDeadError. Keep it off by default here.
VLLM_ASYNC_SCHEDULING="${VLLM_ASYNC_SCHEDULING:-False}"
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-2}"
# A prompt group sampled at step s that was dispatched at step d has span s - d + 1. With
# max_off_policy_strategy=drop, groups whose span exceeds MAX_OFF_POLICY_THRESHOLD are dropped instead of trained
# on ("wait" blocks sampling for them instead). NUM_WARMUP_BATCHES prefetched batches give spans up to
# NUM_WARMUP_BATCHES + 1, so that is the default threshold: nothing is dropped unless rollouts fall further behind.
MAX_OFF_POLICY_STRATEGY="${MAX_OFF_POLICY_STRATEGY:-drop}"
MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-$((NUM_WARMUP_BATCHES + 1))}"
if (( NUM_WARMUP_BATCHES > 0 )); then
  export VERL_REFILL_AFTER_SAMPLE="${VERL_REFILL_AFTER_SAMPLE:-1}"
else
  export VERL_REFILL_AFTER_SAMPLE="${VERL_REFILL_AFTER_SAMPLE:-0}"
fi
export VERL_SANDBOX_LOOKAHEAD="${VERL_SANDBOX_LOOKAHEAD:-1}"
export SANDBOX_EVAL_REPLICAS="${SANDBOX_EVAL_REPLICAS:-1}"
# A per-run id lets the trainer's warm pools serve the rollout workers' sandboxes, keeps them separate from other jobs
# in the namespace, and is what the reaper selects on after a crash (see README).
# Lowercase letters, digits and "-" only: it becomes part of Kubernetes resource names.
export SANDBOX_RUN_ID="${SANDBOX_RUN_ID:-deepswe-$(date +%m%d-%H%M%S)}"

# vLLM-TorchTPU kernel flags, TorchTitan TPU options, tpu-sync options, and rollout correction band.
TPU_MOE_SKIP_PADDED_TOKENS="${TPU_MOE_SKIP_PADDED_TOKENS:-1}"
TPU_ENABLE_GDN_DYNAMIC_TILING="${TPU_ENABLE_GDN_DYNAMIC_TILING:-1}"
USE_MOE_FUSED_EP_KERNEL="${USE_MOE_FUSED_EP_KERNEL:-1}"
SPLASH_MAX_BLOCK_SIZE="${SPLASH_MAX_BLOCK_SIZE-1024}"
FROZEN_PARAM_REGEX="${FROZEN_PARAM_REGEX-moe.router.gate}"
# Opt-in: set GDN_ROW_SLOT_LEN=32768 when using an image built with google-pytorch/torchtitan#142.
GDN_ROW_SLOT_LEN="${GDN_ROW_SLOT_LEN:-}"
TPU_SYNC_VERIFY_PARITY="${TPU_SYNC_VERIFY_PARITY:-off}"
TPU_SYNC_RELEASE_BUFFERS="${TPU_SYNC_RELEASE_BUFFERS:-True}"
# Advantage estimator: rloo (LOO mean, unscaled) or grpo_loo (LOO mean + LOO std scale, registered in deepswe.py).
export ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo_loo}"
# Rollout correction: token-level TIS paired with geometric-mean (seq_mean_k1) and symmetric K2 (seq_mean_k2)
# sequence rejection. Quote ROLLOUT_RS when non-null so comma-separated filters do not trigger Hydra multirun sweeps.
ROLLOUT_IS="${ROLLOUT_IS:-token}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-5.0}"
ROLLOUT_RS="${ROLLOUT_RS:-seq_mean_k1,seq_mean_k2}"
ROLLOUT_RS_THRESHOLD="${ROLLOUT_RS_THRESHOLD:-0.998_1.002,0.25}"
BYPASS_MODE="${BYPASS_MODE:-False}"
BYPASS_LOSS_TYPE="${BYPASS_LOSS_TYPE:-reinforce}"
STATIC_LOG_PROB_MICRO_BATCH_LEN="${STATIC_LOG_PROB_MICRO_BATCH_LEN:-}"
ROLLOUT_CORRECTION_OVERRIDES=(
  "algorithm.rollout_correction.rollout_is=${ROLLOUT_IS}"
)
if [[ "${ROLLOUT_RS}" == "null" ]]; then
  ROLLOUT_CORRECTION_OVERRIDES+=("algorithm.rollout_correction.rollout_rs=null")
else
  ROLLOUT_CORRECTION_OVERRIDES+=(
    "algorithm.rollout_correction.rollout_rs='${ROLLOUT_RS}'"
    "algorithm.rollout_correction.rollout_rs_threshold='${ROLLOUT_RS_THRESHOLD}'"
  )
fi
if [[ "${ROLLOUT_IS}" != "null" ]]; then
  ROLLOUT_CORRECTION_OVERRIDES+=("algorithm.rollout_correction.rollout_is_threshold='${ROLLOUT_IS_THRESHOLD}'")
fi

# Agent, tools, prompt and sandbox options (image mirror, network policy mode): see the file. Every Ray node reads it.
TASK_CONFIG="${TASK_CONFIG:-${SCRIPT_DIR}/task_config.yaml}"
GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
# One agent session per decode slot (3 replicas x 128).
CONCURRENCY="${CONCURRENCY:-384}"
DISPATCH_MODE="${DISPATCH_MODE:-inline_async}"
TRAJECTORY_SELECTION="${TRAJECTORY_SELECTION:-all}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-/tmp/uni_agent_logs/${EXPERIMENT_NAME}}"
MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-False}"

VERL_EXT_MODULES="${VERL_EXT_MODULES:-verl_hardware_plugin,${DEEPSWE_MODULE}}"
# With lookahead on, the trainers registered by ${DEEPSWE_MODULE} warm the sandboxes of each batch one step ahead.
# Without that module nothing is warmed and every sandbox cold-starts, so treat it as a configuration error.
if [[ ! "${VERL_SANDBOX_LOOKAHEAD,,}" =~ ^(0|false|no)$ && ",${VERL_EXT_MODULES}," != *",${DEEPSWE_MODULE},"* ]]; then
  echo "ERROR: VERL_SANDBOX_LOOKAHEAD=${VERL_SANDBOX_LOOKAHEAD} needs ${DEEPSWE_MODULE} in VERL_EXT_MODULES" \
    "(got '${VERL_EXT_MODULES}'); add it or set VERL_SANDBOX_LOOKAHEAD=0." >&2
  exit 1
fi
AGENT_OVERRIDES=(
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_EXT_MODULES}'"
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_REFILL_AFTER_SAMPLE='${VERL_REFILL_AFTER_SAMPLE}'"
  "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_SANDBOX_LOOKAHEAD='${VERL_SANDBOX_LOOKAHEAD}'"
  "++ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_EVAL_REPLICAS='${SANDBOX_EVAL_REPLICAS}'"
  "++actor_rollout_ref.rollout.multi_turn.format=qwen3_coder"
  "++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter"
  "++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT}"
  "++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR}"
  "++actor_rollout_ref.rollout.custom.agent_framework.allowed_request_sampling_param_keys=[temperature,top_p,top_k]"
  "++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=${MASK_UNFINISHED_EPISODE}"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=${DISPATCH_MODE}"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${CONCURRENCY}"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=${TRAJECTORY_SELECTION}"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG}"
  "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=${MODEL_NAME}"
  "reward.reward_manager.name=dapo"
  "reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner"
  "reward.custom_reward_function.name=score_from_runner_result"
)
# Only forward optional sandbox/node-selector env vars when set, so unset vars do not arrive as "".
for _opt_var in SANDBOX_NAMESPACE SANDBOX_RUN_ID NODE_SELECTOR_KEY NODE_SELECTOR_VAL; do
  if [[ "${!_opt_var:-}" == *"'"* ]]; then
    echo "ERROR: ${_opt_var} must not contain a single quote (it is passed as a quoted Hydra override)." >&2
    exit 1
  fi
  if [[ -n "${!_opt_var:-}" ]]; then
    AGENT_OVERRIDES+=("++ray_kwargs.ray_init.runtime_env.env_vars.${_opt_var}='${!_opt_var}'")
  fi
done

OPTIMIZATION_OVERRIDES=(
  "++ray_kwargs.ray_init.runtime_env.env_vars.TPU_MOE_SKIP_PADDED_TOKENS='${TPU_MOE_SKIP_PADDED_TOKENS}'"
  "++ray_kwargs.ray_init.runtime_env.env_vars.TPU_ENABLE_GDN_DYNAMIC_TILING='${TPU_ENABLE_GDN_DYNAMIC_TILING}'"
  "++ray_kwargs.ray_init.runtime_env.env_vars.USE_MOE_FUSED_EP_KERNEL='${USE_MOE_FUSED_EP_KERNEL}'"
  "++actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling=${VLLM_ASYNC_SCHEDULING}"
  "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.verify_parity=${TPU_SYNC_VERIFY_PARITY}"
  "++actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.tpu-sync.release_buffers_after_sync=${TPU_SYNC_RELEASE_BUFFERS}"
)
# Clear verl main's hard-coded no_enable_prefix_caching=True via null (dropped by vllm_async_server.py),
# which works both before and after verl#8614 lands.
if [[ "${ENABLE_PREFIX_CACHING}" == "True" ]]; then
  OPTIMIZATION_OVERRIDES+=(
    "actor_rollout_ref.rollout.enable_prefix_caching=True"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching=null"
  )
else
  OPTIMIZATION_OVERRIDES+=(
    "actor_rollout_ref.rollout.enable_prefix_caching=False"
    "++actor_rollout_ref.rollout.engine_kwargs.vllm.no_enable_prefix_caching=True"
  )
fi
if [[ -n "${VLLM_BLOCK_SIZE}" ]]; then
  OPTIMIZATION_OVERRIDES+=("++actor_rollout_ref.rollout.engine_kwargs.vllm.block_size=${VLLM_BLOCK_SIZE}")
fi
if [[ -n "${SPLASH_MAX_BLOCK_SIZE}" ]]; then
  OPTIMIZATION_OVERRIDES+=("++actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.splash_max_block_size=${SPLASH_MAX_BLOCK_SIZE}")
fi
if [[ -n "${FROZEN_PARAM_REGEX}" ]]; then
  OPTIMIZATION_OVERRIDES+=("++actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.frozen_param_regex=${FROZEN_PARAM_REGEX}")
fi
if [[ -n "${GDN_ROW_SLOT_LEN}" ]]; then
  OPTIMIZATION_OVERRIDES+=("++actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.gdn_row_slot_len=${GDN_ROW_SLOT_LEN}")
fi

MODE_OVERRIDES=(
  "algorithm.rollout_correction.bypass_mode=False"
  "actor_rollout_ref.actor.policy_loss.loss_mode=vanilla"
)
if [[ "${BYPASS_MODE,,}" == "true" ]]; then
  MODE_OVERRIDES=(
    "algorithm.rollout_correction.bypass_mode=True"
    "algorithm.rollout_correction.loss_type=${BYPASS_LOSS_TYPE}"
    "actor_rollout_ref.actor.policy_loss.loss_mode=bypass_mode"
  )
elif [[ -n "${STATIC_LOG_PROB_MICRO_BATCH_LEN}" ]]; then
  MODE_OVERRIDES+=(
    "++actor_rollout_ref.actor.torchtitan.engine_kwargs.tpu.static_log_prob_micro_batch_len=${STATIC_LOG_PROB_MICRO_BATCH_LEN}"
  )
fi

exec bash "${VERL_ROOT}/examples/tpu/grpo/run_qwen3_5_torchtitan.sh" \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.use_kl_in_reward=False \
    "${MODE_OVERRIDES[@]}" \
    "${ROLLOUT_CORRECTION_OVERRIDES[@]}" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.optim.clip_grad=0.125 \
    "actor_rollout_ref.actor.optim.betas=[0.9,0.999]" \
    actor_rollout_ref.actor.optim.min_lr_factor=1.0 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_DEVICE}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_DEVICE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_DEVICE}" \
    ++data.max_length="${MAX_MODEL_LEN}" \
    ++data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_DEVICE}" \
    data.return_raw_chat=True \
    data.truncation=error \
    actor_rollout_ref.rollout.mode=async \
    "${AGENT_OVERRIDES[@]}" \
    "${OPTIMIZATION_OVERRIDES[@]}" \
    trainer.v1.sampler.max_off_policy_threshold="${MAX_OFF_POLICY_THRESHOLD}" \
    trainer.v1.sampler.max_off_policy_strategy="${MAX_OFF_POLICY_STRATEGY}" \
    "$@"
