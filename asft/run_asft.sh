#!/usr/bin/env bash
# Unified 8-GPU launcher for SFT-family fine-tuning on the bio/med dataset.
#
# Supported loss modes (matches fsdp_sft_trainer.py):
#   sft, dft, asft
#
# Defaults: 8 GPUs, train -> save every N steps -> med eval after each run.
#
# Examples:
#   # Single mode
#   LOSS_MODES=asft bash verl/recipe/asft/run_asft.sh
#
#   # Sweep all three (default)
#   bash verl/recipe/asft/run_asft.sh
#
#   # Smaller smoke test
#   EPOCHS=1 GLOBAL_BSZ=16 SAVE_EVERY=10 LOSS_MODES="sft" \
#     bash verl/recipe/asft/run_asft.sh

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"  # .../verl
REPO_DIR="$(cd "${PROJECT_DIR}/.." && pwd)"                         # repo root
ROOT_DIR="$(cd "${REPO_DIR}/.." && pwd)"                            # parent (where models/ lives)
cd "${PROJECT_DIR}"

# -------- Model / data --------
MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-7B}"
MODEL_PATH="${MODEL_PATH:-${ROOT_DIR}/models/Qwen2.5-7B-base}"
DATA_DIR="${DATA_DIR:-${PROJECT_DIR}/recipe/asft/data}"
RUN_ROOT="${RUN_ROOT:-${ROOT_DIR}/checkpoints/asft_qwen2.5-7b-base}"
TEST_DATA_DIR="${TEST_DATA_DIR:-${PROJECT_DIR}/eval/medeval/test_data}"
MEDEVAL_RUNNER="${MEDEVAL_RUNNER:-${PROJECT_DIR}/eval/medeval/run_med_eval.py}"

# -------- GPU layout (8 GPUs total: 7 train + 1 eval) --------
NUM_GPUS="${NUM_GPUS:-7}"
CUDA_TRAIN="${CUDA_TRAIN:-0,1,2,3,4,5,6}"
CUDA_EVAL="${CUDA_EVAL:-7}"
EVAL_TP="${EVAL_TP:-1}"

# -------- Training hyperparams --------
EPOCHS="${EPOCHS:-3}"
GLOBAL_BSZ="${GLOBAL_BSZ:-64}"
MICRO_BSZ="${MICRO_BSZ:-2}"
LR="${LR:-2e-5}"
MAX_LEN="${MAX_LEN:-512}"
SEED="${SEED:-42}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:-10000}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-10000}"

# -------- Loss-mode-specific knobs (only used when applicable) --------
ASFT_KL_COEF="${ASFT_KL_COEF:-0.1}"

# -------- Eval cadence --------
# SAVE_EVERY/TEST_EVERY: in steps. Default = once per epoch.
SAVE_EVERY="${SAVE_EVERY:-0}"   # 0 -> auto = steps_per_epoch
TEST_EVERY="${TEST_EVERY:-0}"

# -------- Modes to run --------
LOSS_MODES="${LOSS_MODES:-sft dft asft}"

mkdir -p "${DATA_DIR}" "${RUN_ROOT}" "${ROOT_DIR}/models"

# Prepare data + model
python "${PROJECT_DIR}/recipe/asft/prepare_data.py" \
  --output_dir "${DATA_DIR}" --dataset med --max_samples "${TRAIN_MAX_SAMPLES}"

if ! ls "${MODEL_PATH}"/*.safetensors "${MODEL_PATH}"/pytorch_model*.bin >/dev/null 2>&1; then
  mkdir -p "${MODEL_PATH}"
  huggingface-cli download "${MODEL_ID}" --local-dir "${MODEL_PATH}"
fi

TRAIN_PARQUET="${DATA_DIR}/med/train.parquet"
VAL_PARQUET="${DATA_DIR}/med/val.parquet"

DATA_SIZE=$(python - <<PY
import pandas as pd
print(len(pd.read_parquet("${TRAIN_PARQUET}")))
PY
)
STEPS_PER_EPOCH=$(( DATA_SIZE / GLOBAL_BSZ ))
[[ "${STEPS_PER_EPOCH}" -lt 1 ]] && { echo "Invalid steps/epoch=${STEPS_PER_EPOCH}"; exit 1; }
TOTAL_STEPS=$(( STEPS_PER_EPOCH * EPOCHS ))
[[ "${SAVE_EVERY}" -eq 0 ]] && SAVE_EVERY="${STEPS_PER_EPOCH}"
[[ "${TEST_EVERY}" -eq 0 ]] && TEST_EVERY="${STEPS_PER_EPOCH}"

echo "data_size=${DATA_SIZE} steps/epoch=${STEPS_PER_EPOCH} total=${TOTAL_STEPS}"
echo "save_every=${SAVE_EVERY} test_every=${TEST_EVERY}"
echo "modes: ${LOSS_MODES}"

for MODE in ${LOSS_MODES}; do
  EXP_NAME="qwen2.5-7b-base_bio10k_notemplate_${MODE}"
  CKPT_DIR="${RUN_ROOT}/${MODE}"
  LOG_PATH="${CKPT_DIR}/train_${MODE}.log"
  mkdir -p "${CKPT_DIR}"

  echo "==== Train: ${MODE} -> ${CKPT_DIR} ===="
  CUDA_VISIBLE_DEVICES="${CUDA_TRAIN}" torchrun --standalone --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    -m recipe.asft.fsdp_sft_trainer_asft \
    data.train_files="${TRAIN_PARQUET}" \
    data.val_files="${VAL_PARQUET}" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.multiturn.enable=false \
    data.use_chat_template=false \
    data.max_length="${MAX_LEN}" \
    data.truncation=right \
    data.train_batch_size="${GLOBAL_BSZ}" \
    data.micro_batch_size_per_gpu="${MICRO_BSZ}" \
    data.train_max_samples="${TRAIN_MAX_SAMPLES}" \
    data.val_max_samples="${VAL_MAX_SAMPLES}" \
    model.partial_pretrain="${MODEL_PATH}" \
    model.strategy=fsdp \
    model.fsdp_config.model_dtype=bf16 \
    model.enable_gradient_checkpointing=true \
    model.lora_rank=0 \
    use_remove_padding=false \
    ulysses_sequence_parallel_size=1 \
    optim.lr="${LR}" \
    optim.lr_scheduler=cosine \
    optim.lr_warmup_steps_ratio=0.03 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.project_name=asft-bio \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.total_epochs="${EPOCHS}" \
    trainer.total_training_steps=null \
    trainer.seed="${SEED}" \
    trainer.resume_mode=disable \
    trainer.save_freq="${SAVE_EVERY}" \
    trainer.test_freq="${TEST_EVERY}" \
    trainer.logger=['console'] \
    trainer.loss_mode="${MODE}" \
    trainer.asft_kl_coef="${ASFT_KL_COEF}" \
    trainer.benchmark_eval_dir="${TEST_DATA_DIR}" \
    trainer.checkpoint.save_contents=[model,optimizer,extra,hf_model] \
    trainer.checkpoint.load_contents=[model,optimizer,extra,hf_model] \
    trainer.max_ckpt_to_keep=3 \
    2>&1 | tee "${LOG_PATH}"

  LATEST_CKPT="$(ls -d "${CKPT_DIR}"/global_step_* | sort -V | tail -n 1)"
  HF_MODEL_DIR="${LATEST_CKPT}/huggingface"
  EVAL_JSON="${CKPT_DIR}/medeval_${MODE}.json"
  EVAL_LOG="${CKPT_DIR}/medeval_${MODE}.log"

  echo "==== Med eval: ${MODE} -> ${EVAL_JSON} ===="
  CUDA_VISIBLE_DEVICES="${CUDA_EVAL}" python "${MEDEVAL_RUNNER}" \
    --model "${HF_MODEL_DIR}" \
    --test_data_dir "${TEST_DATA_DIR}" \
    --tensor_parallel_size "${EVAL_TP}" \
    --output_json "${EVAL_JSON}" \
      2>&1 | tee "${EVAL_LOG}"

  echo "==== Done: ${MODE} ===="
done

echo "All runs completed."
