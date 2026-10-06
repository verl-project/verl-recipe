#!/usr/bin/env bash
# Convert missing rollout batches, then train the offline WM. Existing converted steps are reused.
# SOURCE_EXPERIMENT=... EXPERIMENT_NAME=... END_STEP=500 MODEL_PATH=... bash recipe/obsspec/reproduce/train_offline_wm.sh
set -euo pipefail

unset RAY_ADDRESS
unset RAY_JOB_ID
unset RAY_RUNTIME_ENV_URI
unset RAY_SESSION_NAME
unset RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES

RECIPE_DIR=recipe/obsspec
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/outputs}"
SOURCE_EXPERIMENT="${SOURCE_EXPERIMENT:?set SOURCE_EXPERIMENT}"
DATASET_DIR="${DATASET_DIR:-${OUTPUT_ROOT}/terminal-obsspec/${SOURCE_EXPERIMENT}/sft_batches}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"

PROJECT_NAME="${PROJECT_NAME:-terminal-obsspec}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?set EXPERIMENT_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/checkpoints}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-18432}"
LR="${LR:-1e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:--1}"

SOURCE_BATCH_DIR="${SOURCE_BATCH_DIR:-${OUTPUT_ROOT}/terminal-obsspec/${SOURCE_EXPERIMENT}/rollout_batches}"
START_STEP="${START_STEP:-1}"
END_STEP="${END_STEP:-500}"

python3 "${RECIPE_DIR}/reproduce/offline_rollout_dataset.py" \
    --source-dir "${SOURCE_BATCH_DIR}" \
    --output-dir "${DATASET_DIR}" \
    --start-step "${START_STEP}" \
    --end-step "${END_STEP}"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${N_GPUS_PER_NODE}" \
    -m verl.trainer.sft_trainer \
    data.train_files="${DATASET_DIR}" \
    data.val_files=null \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.num_workers=0 \
    data.pad_mode=no_padding \
    data.use_dynamic_bsz=true \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.custom_cls.path="${RECIPE_DIR}/reproduce/offline_rollout_dataset.py" \
    data.custom_cls.name=OfflineRolloutDataset \
    +data.shuffle=false \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=true \
    model.enable_gradient_checkpointing=true \
    optim.lr="${LR}" \
    optim.lr_scheduler_type=constant \
    optim.lr_warmup_steps="${LR_WARMUP_STEPS}" \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    engine=fsdp \
    engine.strategy=fsdp2 \
    engine.param_offload=true \
    engine.optimizer_offload=true \
    checkpoint.save_contents='["hf_model"]' \
    checkpoint.load_contents='[]' \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.logger='["console","wandb"]' \
    trainer.total_epochs=1 \
    trainer.save_freq=100 \
    trainer.test_freq=-1 \
    trainer.max_ckpt_to_keep=null \
    trainer.resume_mode=disable \
    "$@"
