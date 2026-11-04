#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ -n "${VERL_ROOT:-}" ]]; then
  VERL_ROOT="$(cd -- "${VERL_ROOT}" && pwd -P)"
elif [[ -f "${SCRIPT_DIR}/../../verl/trainer/main_ppo.py" ]]; then
  VERL_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
else
  echo "ERROR: set VERL_ROOT to the patched VERL source checkout." >&2
  exit 2
fi

if [[ -z "${UNIAGENT_ROOT:-}" ]]; then
  for candidate in "${SCRIPT_DIR}/../../uni-agent"; do
    if [[ -f "${candidate}/uni_agent/framework/entry.py" ]]; then
      UNIAGENT_ROOT="${candidate}"
      break
    fi
  done
fi
: "${UNIAGENT_ROOT:?Clone Uni-Agent beside the Recipe or set UNIAGENT_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH to the local Qwen3 model directory}"
TRAIN_FILE="${TRAIN_FILE:-${HOME}/data/tpr/swe_rebench_pelita875.parquet}"

TASK_CONFIG="${TASK_CONFIG:-${SCRIPT_DIR}/task_config_swe_rebench_docker.yaml}"

[[ -f "${VERL_ROOT}/verl/trainer/main_ppo.py" ]] || { echo "ERROR: invalid VERL_ROOT: ${VERL_ROOT}" >&2; exit 2; }
[[ -f "${UNIAGENT_ROOT}/uni_agent/framework/entry.py" ]] || { echo "ERROR: invalid UNIAGENT_ROOT: ${UNIAGENT_ROOT}" >&2; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "ERROR: missing model config: ${MODEL_PATH}/config.json" >&2; exit 2; }
[[ -f "${TRAIN_FILE}" ]] || { echo "ERROR: missing TRAIN_FILE: ${TRAIN_FILE}" >&2; exit 2; }
[[ -f "${TASK_CONFIG}" ]] || { echo "ERROR: missing TASK_CONFIG: ${TASK_CONFIG}" >&2; exit 2; }

# Resolve paths before changing to the VERL checkout.
VERL_ROOT="$(cd -- "${VERL_ROOT}" && pwd -P)"
UNIAGENT_ROOT="$(cd -- "${UNIAGENT_ROOT}" && pwd -P)"
MODEL_PATH="$(cd -- "${MODEL_PATH}" && pwd -P)"
TRAIN_FILE="$(cd -- "$(dirname -- "${TRAIN_FILE}")" && pwd -P)/$(basename -- "${TRAIN_FILE}")"
TASK_CONFIG="$(cd -- "$(dirname -- "${TASK_CONFIG}")" && pwd -P)/$(basename -- "${TASK_CONFIG}")"
TEST_FILE="${TEST_FILE:-${TRAIN_FILE}}"
TEST_FILE="$(cd -- "$(dirname -- "${TEST_FILE}")" && pwd -P)/$(basename -- "${TEST_FILE}")"
[[ -f "${TEST_FILE}" ]] || { echo "ERROR: missing TEST_FILE: ${TEST_FILE}" >&2; exit 2; }

normalize_bool() {
  case "$1" in
    True|true|TRUE|1) printf 'True' ;;
    False|false|FALSE|0) printf 'False' ;;
    *) echo "Expected boolean True/False, got: $1" >&2; return 2 ;;
  esac
}

NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES="${RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES:-1}"
TRAIN_TP="${TRAIN_TP:-2}"
TRAIN_CP="${TRAIN_CP:-4}"
# PP and SP are fixed in this Recipe; TP and CP are the exposed parallelism settings.
TRAIN_PP=1
TRAIN_SP=True
TPR_ENABLED="$(normalize_bool "${TPR_ENABLED:-True}")"

WORLD_SIZE=$((NNODES * N_GPUS_PER_NODE))
PARALLEL_PRODUCT=$((TRAIN_TP * TRAIN_PP * TRAIN_CP))
if (( PARALLEL_PRODUCT <= 0 || WORLD_SIZE % PARALLEL_PRODUCT != 0 )); then
  echo "ERROR: total NPU count must be divisible by TP*PP*CP." >&2
  exit 2
fi
TRAIN_DP=$((WORLD_SIZE / PARALLEL_PRODUCT))

# TRAIN_SP is passed through to native Megatron; TPR imposes no SP/CP topology gate.

N="${N:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16384}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}"
ROLLOUT_TP="${ROLLOUT_TP:-2}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
CONCURRENCY="${CONCURRENCY:-8}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-1}"
GATEWAY_COUNT="${GATEWAY_COUNT:-1}"
TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-$TRAIN_DP}"
GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-$TRAIN_DP}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$TRAIN_DP}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-10}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-$TOTAL_TRAINING_STEPS}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
TOOL_PARSER="${TOOL_PARSER:-hermes}"

if [[ -z "${VLLM_ASCEND_ROOT:-}" ]]; then
  if [[ -d /workspace/vllm-ascend ]]; then
    VLLM_ASCEND_ROOT=/workspace/vllm-ascend
  elif [[ -d /vllm-ascend ]]; then
    VLLM_ASCEND_ROOT=/vllm-ascend
  else
    VLLM_ASCEND_ROOT=""
  fi
fi

# Some Ascend images ship Megatron-Bridge as source rather than a pip package.
# VERL imports MindSpeed before Megatron so NPU TE compatibility patches are available.
if [[ -z "${MEGATRON_BRIDGE_SRC:-}" ]]; then
  for candidate in /workspace/Megatron-Bridge/src /Megatron-Bridge/src; do
    if [[ -f "${candidate}/megatron/bridge/__init__.py" ]]; then
      MEGATRON_BRIDGE_SRC="${candidate}"
      break
    fi
  done
fi
if [[ -n "${MEGATRON_BRIDGE_SRC:-}" && ! -f "${MEGATRON_BRIDGE_SRC}/megatron/bridge/__init__.py" ]]; then
  echo "ERROR: invalid MEGATRON_BRIDGE_SRC: ${MEGATRON_BRIDGE_SRC}" >&2
  exit 2
fi
export PYTHONPATH="${MEGATRON_BRIDGE_SRC:+${MEGATRON_BRIDGE_SRC}:}${SCRIPT_DIR}:${VERL_ROOT}:${UNIAGENT_ROOT}${VLLM_ASCEND_ROOT:+:${VLLM_ASCEND_ROOT}}${PYTHONPATH:+:${PYTHONPATH}}"

RUN_NAME="${RUN_NAME:-tpr-swe-rebench-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-${VERL_ROOT}/outputs/tpr/${RUN_NAME}}"
mkdir -p "${RUN_ROOT}"
RUN_ROOT="$(cd -- "${RUN_ROOT}" && pwd -P)"
LOG_FILE="${RUN_ROOT}/launcher.log"
if [[ -e "${LOG_FILE}" ]]; then
  echo "ERROR: refusing to overwrite existing log: ${LOG_FILE}" >&2
  echo "Set a new RUN_NAME or RUN_ROOT." >&2
  exit 2
fi
AGENT_LOG_DIR="${RUN_ROOT}/agent"
CKPTS_DIR="${RUN_ROOT}/ckpts"
mkdir -p "${AGENT_LOG_DIR}" "${CKPTS_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "MEGATRON_BRIDGE_SRC=${MEGATRON_BRIDGE_SRC:-<not set>}"
echo "VERL_ROOT=${VERL_ROOT}"
echo "UNIAGENT_ROOT=${UNIAGENT_ROOT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "TRAIN_FILE=${TRAIN_FILE}"
echo "TASK_CONFIG=${TASK_CONFIG}"
echo "TPR_ENABLED=${TPR_ENABLED}"
echo "Topology: NNODES=${NNODES} N_GPUS_PER_NODE=${N_GPUS_PER_NODE} TP=${TRAIN_TP} CP=${TRAIN_CP} DP=${TRAIN_DP}"
echo "Sequence: prompt=${MAX_PROMPT_LENGTH} response=${MAX_RESPONSE_LENGTH} max_model_len=${MAX_MODEL_LEN}"
echo "RUN_ROOT=${RUN_ROOT}"

# Import the package from this Recipe checkout, not from another installed copy.
TPR_RECIPE_DIR="${SCRIPT_DIR}" python - <<'PY'
import os
from pathlib import Path
import verl_tpr
expected = Path(os.environ["TPR_RECIPE_DIR"]).resolve()
actual = Path(verl_tpr.__file__).resolve()
if expected not in actual.parents:
    raise SystemExit(f"wrong verl_tpr import: {actual}; expected under {expected}")
from verl_tpr.adapter import install_tpr_module_spec, run_tpr_forward_backward_batch
print(f"TPR runtime import OK: {actual}")
PY

# Check the same import ordering used by VERL on NPU: MindSpeed patches first,
# Megatron-Bridge second. A plain Bridge import can fail without TE shims.
python - <<'PY'
import mindspeed.megatron_adaptor
from verl.models.mcore.bridge import AutoBridge
print(f"Megatron-Bridge import OK after MindSpeed patches: {AutoBridge}")
PY

cd "${VERL_ROOT}"
python -m verl.trainer.main_ppo \
  --config-name=ppo_megatron_trainer \
  trainer.use_v1=True \
  trainer.v1.trainer_mode=colocate_async \
  trainer.v1.colocate_async.num_warmup_batches=0 \
  transfer_queue.enable=True \
  transfer_queue.backend.SimpleStorage.total_storage_size=100000 \
  trainer.device=npu \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  actor_rollout_ref.hybrid_engine=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=True \
  +actor_rollout_ref.model.override_config.model_config.max_position_embeddings="${MAX_MODEL_LEN}" \
  "data.train_files=['${TRAIN_FILE}']" \
  "data.val_files=['${TEST_FILE}']" \
  data.prompt_key=prompt \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
  data.gen_batch_size="${GEN_BATCH_SIZE}" \
  data.return_raw_chat=True \
  data.trust_remote_code=True \
  data.dataloader_num_workers=0 \
  algorithm.adv_estimator=grpo \
  algorithm.filter_groups.enable=False \
  algorithm.use_kl_in_reward=False \
  actor_rollout_ref.actor.optim.use_precision_aware_optimizer=True \
  actor_rollout_ref.actor.optim.main_grads_dtype=bf16 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
  +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0 \
  +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_MODEL_LEN}" \
  actor_rollout_ref.actor.checkpoint.strict=False \
  actor_rollout_ref.actor.megatron.use_mbridge=True \
  actor_rollout_ref.actor.megatron.use_dist_checkpointing=False \
  actor_rollout_ref.actor.megatron.param_offload=True \
  actor_rollout_ref.actor.megatron.optimizer_offload=True \
  actor_rollout_ref.actor.megatron.tensor_model_parallel_size="${TRAIN_TP}" \
  actor_rollout_ref.actor.megatron.pipeline_model_parallel_size="${TRAIN_PP}" \
  actor_rollout_ref.actor.megatron.context_parallel_size="${TRAIN_CP}" \
  actor_rollout_ref.actor.megatron.sequence_parallel="${TRAIN_SP}" \
  ++actor_rollout_ref.actor.megatron.override_transformer_config.sequence_parallel="${TRAIN_SP}" \
  actor_rollout_ref.actor.megatron.use_remove_padding=True \
  +actor_rollout_ref.actor.megatron.tpr_enabled="${TPR_ENABLED}" \
  algorithm.rollout_correction.bypass_mode=True \
  algorithm.rollout_correction.loss_type=ppo_clip \
  actor_rollout_ref.rollout.n="${N}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.enforce_eager=True \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
  actor_rollout_ref.rollout.prompt_length="${MAX_PROMPT_LENGTH}" \
  actor_rollout_ref.rollout.response_length="${MAX_RESPONSE_LENGTH}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.max_num_seqs="${CONCURRENCY}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=True \
  actor_rollout_ref.rollout.disable_log_stats=False \
  +actor_rollout_ref.rollout.enable_sleep_mode=True \
  actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
  actor_rollout_ref.rollout.top_p="${TOP_P}" \
  actor_rollout_ref.rollout.top_k="${TOP_K}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
  ++actor_rollout_ref.rollout.multi_turn.format="${TOOL_PARSER}" \
  actor_rollout_ref.rollout.agent.num_workers="${NUM_AGENT_WORKERS}" \
  ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter \
  ++actor_rollout_ref.rollout.custom.agent_framework.gateway_count="${GATEWAY_COUNT}" \
  ++actor_rollout_ref.rollout.custom.agent_framework.log_dir="${AGENT_LOG_DIR}" \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=uni_agent.framework.task_runner.run_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=ray_task \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions="${CONCURRENCY}" \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.trajectory_selection=all \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path="${TASK_CONFIG}" \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name="${MODEL_NAME}" \
  ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True \
  ++actor_rollout_ref.rollout.custom.agent_framework.mask_unfinished_episode=False \
  reward.reward_manager.name=dapo \
  reward.custom_reward_function.path=pkg://uni_agent.framework.task_runner \
  reward.custom_reward_function.name=score_from_runner_result \
  'trainer.logger=["console"]' \
  trainer.project_name=uniagent-rebench-tpr \
  trainer.experiment_name="${RUN_NAME}" \
  trainer.val_before_train=False \
  trainer.resume_mode=disable \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.total_epochs="${TOTAL_EPOCHS}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
  trainer.default_local_dir="${CKPTS_DIR}" \
  skip.rollout_tq.enable=False \
  "$@"
