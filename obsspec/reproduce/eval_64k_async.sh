#!/usr/bin/env bash
# Terminal async evaluation: pre-expanded local rollout rows, 256 in flight by default.
# Set POLICY_MODEL_PATH, SPECULATOR_MODEL_PATH, EXPERIMENT_NAME; ENABLE_SPECULATION=false for baseline.
set -euo pipefail

RECIPE_DIR=recipe/obsspec
DATA_DIR="${DATA_DIR:-${HOME}/data/terminal_obsspec}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/outputs}"
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:?set POLICY_MODEL_PATH}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?set EXPERIMENT_NAME}"
ENABLE_SPECULATION="${ENABLE_SPECULATION:-true}"
EVAL_REPORT_LEVEL="${EVAL_REPORT_LEVEL:-basic}"

VAL_FILES="${VAL_FILES:-${DATA_DIR}/endless-terminals-async.parquet}"
TRAIN_FILES="${TRAIN_FILES:-${DATA_DIR}/endless-terminals-val.parquet}"
PROJECT_NAME="${PROJECT_NAME:-terminal-obsspec}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/endless_terminals}"
POLICY_INFER_TP="${POLICY_INFER_TP:-2}"
MAX_INFLIGHT_ROLLOUTS="${MAX_INFLIGHT_ROLLOUTS:-256}"
ROLLOUTS_PER_GROUP="${ROLLOUTS_PER_GROUP:-16}"
PROMPT_GROUPS_PER_BATCH="${PROMPT_GROUPS_PER_BATCH:-16}"
ASYNC_EVAL_POLL_INTERVAL_SEC="${ASYNC_EVAL_POLL_INTERVAL_SEC:-0.05}"

SPECULATION_OVERRIDES=()
if [[ "${ENABLE_SPECULATION}" == "true" ]]; then
    SPECULATOR_MODEL_PATH="${SPECULATOR_MODEL_PATH:?set SPECULATOR_MODEL_PATH when speculation is enabled}"
else
    SPECULATOR_MODEL_PATH="${SPECULATOR_MODEL_PATH:-${POLICY_MODEL_PATH}}"
    SPECULATION_OVERRIDES+=(world_model_actor.enable=False)
fi
if [[ "${EVAL_REPORT_LEVEL}" != "basic" && "${EVAL_REPORT_LEVEL}" != "detailed" ]]; then
    echo "Unknown EVAL_REPORT_LEVEL=${EVAL_REPORT_LEVEL}; expected basic or detailed" >&2
    exit 2
fi

DATASET=endless_terminals \
MODEL_PATH="${POLICY_MODEL_PATH}" \
WM_MODEL_PATH="${SPECULATOR_MODEL_PATH}" \
TRAIN_FILES="${TRAIN_FILES}" \
VAL_FILES="${VAL_FILES}" \
PROJECT_NAME="${PROJECT_NAME}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CHECKPOINT_DIR="${OUTPUT_DIR}/unused_checkpoints" \
ENABLE_SPECULATION="${ENABLE_SPECULATION}" \
SPECULATE_DURING_VALIDATION=true \
WORLD_MODEL_WARMUP_STEPS=0 \
bash "${RECIPE_DIR}/train.sh" \
    data.val_batch_size=1 \
    data.validation_shuffle=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${POLICY_INFER_TP}" \
    trainer.validation_data_dir="${OUTPUT_DIR}/trajectories" \
    +trainer.eval_report_level="${EVAL_REPORT_LEVEL}" \
    +trainer.async_rollout_eval.enabled=true \
    +trainer.async_rollout_eval.max_inflight_rollouts="${MAX_INFLIGHT_ROLLOUTS}" \
    +trainer.async_rollout_eval.rollouts_per_group="${ROLLOUTS_PER_GROUP}" \
    +trainer.async_rollout_eval.prompt_groups_per_batch="${PROMPT_GROUPS_PER_BATCH}" \
    +trainer.async_rollout_eval.poll_interval_sec="${ASYNC_EVAL_POLL_INTERVAL_SEC}" \
    trainer.logger='["console"]' \
    trainer.val_only=True \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=1 \
    "${SPECULATION_OVERRIDES[@]}" \
    "$@"
