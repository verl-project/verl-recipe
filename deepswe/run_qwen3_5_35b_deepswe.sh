#!/usr/bin/env bash
# DeepSWE | Qwen3.5-35B-A3B | R2E-Gym + ReAct (uni-agent) | TorchTitan trainer + vLLM rollout | TPU
#
# Preprocess R2E-Gym dataset beforehand with uni-agent:
#   python3 -m uni_agent.tasks.r2e_gym.preprocess --local-save-dir "${RAY_DATA_HOME}/data/r2e_gym_easy"

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

if [[ -f "${VERL_ROOT}/recipe/deepswe/deepswe.py" ]]; then
  DEEPSWE_MODULE="${DEEPSWE_MODULE:-recipe.deepswe.deepswe}"
else
  DEEPSWE_MODULE="${DEEPSWE_MODULE:-deepswe.deepswe}"
fi

export TPU_ACCELERATOR_TYPE="${TPU_ACCELERATOR_TYPE:-tpu7x}"
export USE_KL_LOSS="${USE_KL_LOSS:-False}"
export PARAM_DTYPE="${PARAM_DTYPE:-bfloat16}"
export FP32_MASTER="${FP32_MASTER:-True}"
export OPTIMIZER_HOST_OFFLOAD="${OPTIMIZER_HOST_OFFLOAD:-True}"
export OPTIM_IMPL="${OPTIM_IMPL:-fused}"
export ROLLOUT_LOAD_FORMAT="${ROLLOUT_LOAD_FORMAT:-dummy}"

RAY_DATA_HOME="${RAY_DATA_HOME:-/data}"
export MODEL_NAME="${MODEL_NAME:-Qwen3.5-35B-A3B}"
export MODEL_PATH="${MODEL_PATH:-${RAY_DATA_HOME}/assets/hf/${MODEL_NAME}}"
export TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/data/r2e_gym_easy/train.parquet}"
export TEST_FILE="${TEST_FILE:-${RAY_DATA_HOME}/data/r2e_gym_easy/test.parquet}"
export PROJECT_NAME="${PROJECT_NAME:-verl-tpu-deepswe}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_5_35b_deepswe_r2e_gym}"

export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
export ROLLOUT_N="${ROLLOUT_N:-16}"
export LEARNING_RATE="${LEARNING_RATE:-1e-6}"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-50}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-70}"
export TEST_FREQ="${TEST_FREQ:-10}"
export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

export MAX_PROMPT_LEN="${MAX_PROMPT_LEN:-8192}"
export MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-57344}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
MAX_TOKEN_LEN_PER_DEVICE="${MAX_TOKEN_LEN_PER_DEVICE:-${MAX_MODEL_LEN}}"
export PAD_TO_LENGTH_BUCKET="${PAD_TO_LENGTH_BUCKET:-4096}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
export MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"
export NUM_WARMUP_BATCHES="${NUM_WARMUP_BATCHES:-1}"
# max_off_policy_strategy=wait blocks sampling while any in-flight prompt group has
# (global_steps - dispatch_step + 1) >= MAX_OFF_POLICY_THRESHOLD. With 1, every step waits for all in-flight
# groups, including the warmup prefetch, so the prefetch never overlaps training. With a prefetch (staleness 1,
# MLPerf MAX_STALENESS=1) use 2: a step only waits for the groups dispatched in the previous step.
if awk -v w="${NUM_WARMUP_BATCHES}" 'BEGIN { exit !(w > 0) }'; then
  MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-2}"
  export VERL_REFILL_AFTER_SAMPLE="${VERL_REFILL_AFTER_SAMPLE:-1}"
else
  MAX_OFF_POLICY_THRESHOLD="${MAX_OFF_POLICY_THRESHOLD:-1}"
  export VERL_REFILL_AFTER_SAMPLE="${VERL_REFILL_AFTER_SAMPLE:-0}"
fi
export VERL_SANDBOX_LOOKAHEAD="${VERL_SANDBOX_LOOKAHEAD:-1}"
export SANDBOX_EVAL_REPLICAS="${SANDBOX_EVAL_REPLICAS:-1}"

UNI_AGENT_HOME="${UNI_AGENT_HOME:-/workspace/uni-agent}"
TASK_CONFIG="${TASK_CONFIG:-${UNI_AGENT_HOME}/examples/quickstart/training/task_config_react_gke.yaml}"
GATEWAY_COUNT="${GATEWAY_COUNT:-4}"
CONCURRENCY="${CONCURRENCY:-32}"
DISPATCH_MODE="${DISPATCH_MODE:-inline_async}"
TRAJECTORY_SELECTION="${TRAJECTORY_SELECTION:-all}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-/tmp/uni_agent_logs/${EXPERIMENT_NAME}}"
MASK_UNFINISHED_EPISODE="${MASK_UNFINISHED_EPISODE:-False}"

VERL_EXT_MODULES="${VERL_EXT_MODULES:-verl_hardware_plugin,${DEEPSWE_MODULE}}"
AGENT_OVERRIDES=(
  "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_EXT_MODULES}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_REFILL_AFTER_SAMPLE='${VERL_REFILL_AFTER_SAMPLE}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.VERL_SANDBOX_LOOKAHEAD='${VERL_SANDBOX_LOOKAHEAD}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_EVAL_REPLICAS='${SANDBOX_EVAL_REPLICAS}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_NAMESPACE='${SANDBOX_NAMESPACE:-}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_RUN_ID='${SANDBOX_RUN_ID:-}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_IMAGE_MAP='${SANDBOX_IMAGE_MAP:-}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_POD_ANNOTATIONS='${SANDBOX_POD_ANNOTATIONS:-}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_NETWORK_POLICY_MANAGEMENT='${SANDBOX_NETWORK_POLICY_MANAGEMENT:-}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.NODE_SELECTOR_KEY='${NODE_SELECTOR_KEY:-}'"
  "+ray_kwargs.ray_init.runtime_env.env_vars.NODE_SELECTOR_VAL='${NODE_SELECTOR_VAL:-}'"
  "++actor_rollout_ref.rollout.multi_turn.format=qwen3_coder"
  "++actor_rollout_ref.rollout.agent.agent_loop_manager_class=${DEEPSWE_MODULE}.DeepSWERolloutAdapter"
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

export ADV_ESTIMATOR="${ADV_ESTIMATOR:-grpo_loo}"

exec bash "${VERL_ROOT}/examples/tpu/grpo/run_qwen3_5_torchtitan.sh" \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.use_kl_in_reward=False \
    algorithm.rollout_correction.bypass_mode=False \
    algorithm.rollout_correction.rollout_is=null \
    algorithm.rollout_correction.rollout_rs=seq_mean_k1 \
    "algorithm.rollout_correction.rollout_rs_threshold='0.999_1.002'" \
    actor_rollout_ref.actor.policy_loss.loss_mode=vanilla \
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
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_DEVICE}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${MICRO_BATCH_SIZE:-1}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_DEVICE}" \
    ++data.max_length="${MAX_MODEL_LEN}" \
    ++data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_DEVICE}" \
    data.return_raw_chat=True \
    data.truncation=error \
    actor_rollout_ref.rollout.mode=async \
    "${AGENT_OVERRIDES[@]}" \
    trainer.v1.sampler.max_off_policy_threshold="${MAX_OFF_POLICY_THRESHOLD}" \
    trainer.v1.sampler.max_off_policy_strategy=wait \
    "$@"
