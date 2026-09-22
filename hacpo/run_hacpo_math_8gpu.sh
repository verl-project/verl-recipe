#!/usr/bin/env bash

set -euo pipefail

: "${AGENT_A_MODEL:?Set AGENT_A_MODEL to the first policy checkpoint}"
: "${AGENT_B_MODEL:?Set AGENT_B_MODEL to the second policy checkpoint}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the 7.5k MATH training parquet}"
: "${VAL_PATHS:?Set VAL_PATHS to colon-separated evaluation parquets}"

AGENT_A_ID=${AGENT_A_ID:-agent_a}
AGENT_B_ID=${AGENT_B_ID:-agent_b}
PROJECT_NAME=${PROJECT_NAME:-hacpo_math_paper}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-hacpo_${AGENT_A_ID}_${AGENT_B_ID}_seed42}
SEED=${SEED:-42}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
VAL_BATCH_SIZE=${VAL_BATCH_SIZE:-128}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-64}
ROLLOUT_N=${ROLLOUT_N:-8}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-1024}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-4096}
TARGET_MAX_PROMPT_LENGTH=${TARGET_MAX_PROMPT_LENGTH:-1536}
TARGET_MAX_RESPONSE_LENGTH=${TARGET_MAX_RESPONSE_LENGTH:-${MAX_RESPONSE_LENGTH}}
PPO_MAX_TOKEN_LEN_PER_GPU=${PPO_MAX_TOKEN_LEN_PER_GPU:-8192}
TRAJECTORY_OVERFLOW=${TRAJECTORY_OVERFLOW:-truncate}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-8192}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
TEST_FREQ=${TEST_FREQ:-3}
SAVE_FREQ=${SAVE_FREQ:-1000000000}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-true}
VAL_DO_SAMPLE=${VAL_DO_SAMPLE:-false}
VAL_TEMPERATURE=${VAL_TEMPERATURE:-0.0}
VAL_TOP_P=${VAL_TOP_P:-1.0}
VAL_TOP_K=${VAL_TOP_K:--1}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-false}
ENABLE_THINKING=${ENABLE_THINKING:-false}
VAL_ONLY=${VAL_ONLY:-false}
VAL_ROLLOUT_N=${VAL_ROLLOUT_N:-1}
VALIDATION_AGENT_IDS=${VALIDATION_AGENT_IDS:-null}
RESUME_MODE=${RESUME_MODE:-disable}
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}

CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints/hacpo/${EXPERIMENT_NAME}}
HYDRA_RUN_DIR=${HYDRA_RUN_DIR:-outputs/hacpo/${EXPERIMENT_NAME}}
TENSORBOARD_DIR=${TENSORBOARD_DIR:-tensorboard_log/${PROJECT_NAME}/${EXPERIMENT_NAME}}
REWARD_FILE=${REWARD_FILE:-recipe/hacpo/math_reward.py}
PYTHON_BIN=${PYTHON_BIN:-python3}

IFS=: read -r -a val_path_array <<< "${VAL_PATHS}"
if (( ${#val_path_array[@]} == 0 )); then
  echo "VAL_PATHS must contain at least one parquet" >&2
  exit 2
fi

if [[ ${PYTHON_BIN} == */* ]]; then
  [[ -x ${PYTHON_BIN} ]] || { echo "Python is not executable: ${PYTHON_BIN}" >&2; exit 1; }
else
  command -v "${PYTHON_BIN}" >/dev/null || { echo "Python is not on PATH: ${PYTHON_BIN}" >&2; exit 1; }
fi

for required_path in \
  "${AGENT_A_MODEL}/config.json" \
  "${AGENT_B_MODEL}/config.json" \
  "${TRAIN_FILE}" \
  "${REWARD_FILE}" \
  "${val_path_array[@]}"; do
  if [[ ! -e ${required_path} ]]; then
    echo "Required path does not exist: ${required_path}" >&2
    exit 1
  fi
done

val_files='['
for val_path in "${val_path_array[@]}"; do
  [[ ${val_files} == '[' ]] || val_files+=','
  val_files+="'${val_path}'"
done
val_files+=']'

mkdir -p "${CHECKPOINT_DIR}" "${HYDRA_RUN_DIR}" "${TENSORBOARD_DIR}"
export TENSORBOARD_DIR
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export RAY_DEDUP_LOGS=0
export RAY_ADDRESS=local
unset WANDB_RUN_ID WANDB_PROJECT WANDB_NAME

echo "HACPO policies: ${AGENT_A_ID}=${AGENT_A_MODEL} <-> ${AGENT_B_ID}=${AGENT_B_MODEL}"
echo "Train: ${TRAIN_FILE}"
printf 'Validation: %s\n' "${val_path_array[@]}"
echo "Schedule: epochs=${TOTAL_EPOCHS}, steps=${TOTAL_TRAINING_STEPS}, batch=${TRAIN_BATCH_SIZE}, mini_batch=${PPO_MINI_BATCH_SIZE}, n=${ROLLOUT_N}, test_freq=${TEST_FREQ}, save_freq=${SAVE_FREQ}"
echo "Lengths: prompt=${MAX_PROMPT_LENGTH}, response=${MAX_RESPONSE_LENGTH}, target_prompt=${TARGET_MAX_PROMPT_LENGTH}, target_response=${TARGET_MAX_RESPONSE_LENGTH}, overflow=${TRAJECTORY_OVERFLOW}"
echo "Validation sampling: n=1, do_sample=${VAL_DO_SAMPLE}, temperature=${VAL_TEMPERATURE}, top_p=${VAL_TOP_P}, top_k=${VAL_TOP_K}"
echo "Chat template: enable_thinking=${ENABLE_THINKING}; val_only=${VAL_ONLY}"
echo "TensorBoard: ${TENSORBOARD_DIR}"

exec "${PYTHON_BIN}" -m recipe.hacpo.main_hacpo \
  "hacpo.agents.0.id=${AGENT_A_ID}" \
  "hacpo.agents.0.model_path=${AGENT_A_MODEL}" \
  "hacpo.agents.1.id=${AGENT_B_ID}" \
  "hacpo.agents.1.model_path=${AGENT_B_MODEL}" \
  "data.train_files=['${TRAIN_FILE}']" \
  "data.val_files=${val_files}" \
  "data.train_batch_size=${TRAIN_BATCH_SIZE}" \
  "data.gen_batch_size=${TRAIN_BATCH_SIZE}" \
  "data.val_batch_size=${VAL_BATCH_SIZE}" \
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}" \
  "data.max_response_length=${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  data.dataloader_num_workers=0 \
  data.validation_shuffle=false \
  data.return_raw_chat=true \
  "data.seed=${SEED}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  algorithm.filter_groups.enable=false \
  "+data.apply_chat_template_kwargs.enable_thinking=${ENABLE_THINKING}" \
  "+actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}" \
  "actor_rollout_ref.model.use_remove_padding=${USE_REMOVE_PADDING}" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.actor.ppo_epochs=1 \
  actor_rollout_ref.actor.shuffle=false \
  actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
  actor_rollout_ref.actor.calculate_entropy=false \
  actor_rollout_ref.actor.optim.lr=1.0e-6 \
  actor_rollout_ref.actor.fsdp_config.param_offload=true \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.calculate_log_probs=false \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.rollout.multi_turn.enable=false \
  "actor_rollout_ref.rollout.n=${ROLLOUT_N}" \
  "actor_rollout_ref.rollout.seed=${SEED}" \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.top_k=-1 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.enforce_eager=true \
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}" \
  "actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}" \
  "actor_rollout_ref.rollout.max_model_len=${MAX_MODEL_LEN}" \
  actor_rollout_ref.rollout.agent.num_workers=8 \
  "actor_rollout_ref.rollout.val_kwargs.n=${VAL_ROLLOUT_N}" \
  "actor_rollout_ref.rollout.val_kwargs.do_sample=${VAL_DO_SAMPLE}" \
  "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}" \
  "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}" \
  "actor_rollout_ref.rollout.val_kwargs.top_k=${VAL_TOP_K}" \
  "hacpo.validation_agent_ids=${VALIDATION_AGENT_IDS}" \
  "hacpo.loss.self_clip_low=0.0003" \
  "hacpo.loss.self_clip_high=0.0004" \
  "hacpo.loss.cross_clip_delta=0.2" \
  "hacpo.loss.cross_clip_step=0.025" \
  "hacpo.loss.alpha=1.0" \
  "hacpo.capability.window_size=5" \
  "hacpo.trajectory.target_max_prompt_length=${TARGET_MAX_PROMPT_LENGTH}" \
  "hacpo.trajectory.target_max_response_length=${TARGET_MAX_RESPONSE_LENGTH}" \
  "hacpo.trajectory.overflow=${TRAJECTORY_OVERFLOW}" \
  reward.num_workers=8 \
  reward.reward_model.enable=false \
  "reward.custom_reward_function.path=${REWARD_FILE}" \
  reward.custom_reward_function.name=compute_score \
  trainer.n_gpus_per_node=8 \
  trainer.nnodes=1 \
  "trainer.total_epochs=${TOTAL_EPOCHS}" \
  "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}" \
  "trainer.val_before_train=${VAL_BEFORE_TRAIN}" \
  "trainer.val_only=${VAL_ONLY}" \
  "trainer.test_freq=${TEST_FREQ}" \
  "trainer.save_freq=${SAVE_FREQ}" \
  "trainer.resume_mode=${RESUME_MODE}" \
  "trainer.resume_from_path=${RESUME_FROM_PATH}" \
  "trainer.default_local_dir=${CHECKPOINT_DIR}" \
  "trainer.project_name=${PROJECT_NAME}" \
  "trainer.experiment_name=${EXPERIMENT_NAME}" \
  trainer.logger="['console','tensorboard']" \
  trainer.log_val_generations=0 \
  "hydra.run.dir=${HYDRA_RUN_DIR}" \
  "$@"
