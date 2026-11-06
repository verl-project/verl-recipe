#!/usr/bin/env bash
# Main evaluation: 16 batches x 16 task slots x 16 rollouts = 4,096 rollouts.
# DATASET=endless_terminals/dapo_tir; set POLICY_MODEL_PATH, SPECULATOR_MODEL_PATH, EXPERIMENT_NAME.
# ENABLE_SPECULATION=false runs the baseline without a WM.
set -euo pipefail

RECIPE_DIR=recipe/obsspec
DATA_DIR="${DATA_DIR:-${HOME}/data/terminal_obsspec}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/outputs}"
POLICY_MODEL_PATH="${POLICY_MODEL_PATH:?set POLICY_MODEL_PATH}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?set EXPERIMENT_NAME}"
DATASET="${DATASET:-endless_terminals}"
ENABLE_SPECULATION="${ENABLE_SPECULATION:-true}"
EVAL_REPORT_LEVEL="${EVAL_REPORT_LEVEL:-detailed}"

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

case "${DATASET}" in
    endless_terminals)
        VAL_FILES="${VAL_FILES:-${DATA_DIR}/endless-terminals-val.parquet}"
        EVAL_MANIFEST_PATH="${EVAL_MANIFEST_PATH:-${DATA_DIR}/endless_terminals_eval_manifest.json}"
        MAX_TOKENS_PER_GENERATION="${MAX_TOKENS_PER_GENERATION:-2048}"
        ;;
    dapo_tir)
        VAL_FILES="${VAL_FILES:-${DATA_DIR}/dapo-math-val.parquet}"
        EVAL_MANIFEST_PATH="${EVAL_MANIFEST_PATH:-${DATA_DIR}/dapo_tir_eval_manifest.json}"
        MAX_TOKENS_PER_GENERATION="${MAX_TOKENS_PER_GENERATION:-16384}"
        ;;
    *)
        echo "Unknown DATASET=${DATASET}; expected endless_terminals or dapo_tir" >&2
        exit 2
        ;;
esac

PROJECT_NAME="${PROJECT_NAME:-terminal-obsspec}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/${DATASET}}"
POLICY_INFER_TP="${POLICY_INFER_TP:-2}"

DATASET="${DATASET}" \
MODEL_PATH="${POLICY_MODEL_PATH}" \
WM_MODEL_PATH="${SPECULATOR_MODEL_PATH}" \
TRAIN_FILES="${VAL_FILES}" \
VAL_FILES="${VAL_FILES}" \
PROJECT_NAME="${PROJECT_NAME}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
CHECKPOINT_DIR="${OUTPUT_DIR}/unused_checkpoints" \
ENABLE_SPECULATION="${ENABLE_SPECULATION}" \
SPECULATE_DURING_VALIDATION=true \
WORLD_MODEL_WARMUP_STEPS=0 \
MAX_TOKENS_PER_GENERATION="${MAX_TOKENS_PER_GENERATION}" \
bash "${RECIPE_DIR}/train.sh" \
    data.val_batch_size=16 \
    data.validation_shuffle=False \
    data.custom_cls.path="${RECIPE_DIR}/reproduce/eval_manifest_dataset.py" \
    data.custom_cls.name=EvalManifestDataset \
    +data.eval_manifest_path="${EVAL_MANIFEST_PATH}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${POLICY_INFER_TP}" \
    trainer.validation_data_dir="${OUTPUT_DIR}/trajectories" \
    +trainer.eval_report_level="${EVAL_REPORT_LEVEL}" \
    trainer.logger='["console"]' \
    trainer.val_only=True \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.total_training_steps=1 \
    "${SPECULATION_OVERRIDES[@]}" \
    "$@"
