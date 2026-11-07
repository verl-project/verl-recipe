#!/bin/bash
set -e
set -x

################################################################################
# SFT Training Script
################################################################################
# Usage:
#   bash recipe/humanlm/train_sft.sh <gpu_list> <dataset_name> <thinking_mode> [percent] [model_path] [extra hydra overrides...]
#
# Examples:
#   bash recipe/humanlm/train_sft.sh "0,1,2,3,4,5,6,7" amazon no_thinking
#   bash recipe/humanlm/train_sft.sh "0,1,2,3,4,5,6,7" amazon thinking
#   bash recipe/humanlm/train_sft.sh "0,1,2,3" reddit no_thinking 10
#   bash recipe/humanlm/train_sft.sh "0,1,2,3" reddit thinking 5 Qwen/Qwen3-8B optim.lr=2e-5
################################################################################

PROJECT_DIR="$(pwd)"
export VLLM_USE_V1=1
# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

GPU_LIST=${1:?"Error: GPU list required (e.g., '0,1,2,3')"}
DATASET_NAME=${2:?"Error: Dataset name required (reddit|medium|youtube|amazon|wildchat_english|enron)"}
THINKING_MODE=${3:?"Error: thinking_mode required (thinking|no_thinking)"}
shift 3

# Optional args:
# - percent is optional and must look like an integer (optionally with trailing "p")
# - model_path is optional and must NOT look like a hydra override (no '=')
PERCENT=""
MODEL_PATH="Qwen/Qwen3-8B"
if [[ $# -ge 1 ]]; then
  if [[ "$1" =~ ^[0-9]+([pP])?$ ]]; then
    PERCENT="$1"
    shift 1
  fi
fi
if [[ $# -ge 1 ]]; then
  if [[ "$1" != *"="* ]]; then
    MODEL_PATH="$1"
    shift 1
  fi
fi

export CUDA_VISIBLE_DEVICES="$GPU_LIST"
export NUM_GPUS
NUM_GPUS=$(echo "$GPU_LIST" | awk -F',' '{print NF}')

################################################################################
# Default percent if not specified
################################################################################
if [[ -z "$PERCENT" ]]; then
  case "$DATASET_NAME" in
    amazon)   PERCENT=100 ;;
    reddit)   PERCENT=50  ;;
    medium)   PERCENT=25  ;;
    youtube)  PERCENT=5   ;;
    wildchat_english) PERCENT=100 ;;
    enron)    PERCENT=100 ;;
    *)
      echo "Error: Invalid dataset '$DATASET_NAME' (use: reddit|medium|youtube|amazon|wildchat_english|enron)" >&2
      exit 1
      ;;
  esac
fi

# Normalize percent: allow "100" or "100p"
PERCENT="${PERCENT%p}"
PERCENT="${PERCENT%P}"

################################################################################
# Dataset path + chat template selection
################################################################################
# NOTE: this script is wired to the "dup" processed data layout (dedup datasets)
export DATASET_DIR="$PROJECT_DIR/llm_twin/processed_data"
DATASET_REPO_DIR="${DATASET_NAME}_processed_dataset_by_post_dedup"

# assert THINKING_MODE in thinking or no_thinking
if [[ "$THINKING_MODE" != "thinking" && "$THINKING_MODE" != "no_thinking" ]]; then
  echo "Error: THINKING_MODE must be 'thinking' or 'no_thinking', got '$THINKING_MODE'" >&2
  exit 1
fi

if [[ "$THINKING_MODE" == "thinking" ]]; then
  MODE_DIR="thinking_sft"
  STATE_DIR="think_r"
  ENABLE_THINKING=true
else
  MODE_DIR="sft"
  STATE_DIR="r_no_tag"
  ENABLE_THINKING=false
fi

CHAT_TEMPLATE="recipe/humanlm/chat_templates/qwen3_multi_role_template_think.jinja"
DATA_PATH="$DATASET_DIR/$DATASET_REPO_DIR/$MODE_DIR/$STATE_DIR/${PERCENT}p"
EXP_NAME="sft_${DATASET_NAME}_${THINKING_MODE}_${STATE_DIR}_${PERCENT}p"

################################################################################
# Fixed paths + caches
################################################################################
VERL_PATH="./"
# Override by exporting OUTPUT_ROOT before running the script if you want a custom path.
OUTPUT_ROOT_DEFAULT="//llm_twin/${USER}/outputs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_ROOT_DEFAULT}"
OUTPUT_DIR="${OUTPUT_ROOT}/${EXP_NAME}"
CACHE_DIR="//llm_twin/verl_cache"
export NEW_HF_CACHE=//llm_twin/hf-cache/$USER

export WANDB_ENTITY=dsp-team
export HF_HOME="$NEW_HF_CACHE"
export HUGGINGFACE_HUB_CACHE="$NEW_HF_CACHE/hub"
export TRANSFORMERS_CACHE="$NEW_HF_CACHE/hub"
export HF_DATASETS_CACHE="$NEW_HF_CACHE/datasets"
export XDG_CACHE_HOME="$NEW_HF_CACHE"
export VLLM_DOWNLOAD_DIR="$NEW_HF_CACHE/hub"
export VERL_CACHE_DIR="$NEW_HF_CACHE/verl-cache"

################################################################################
# Display summary
################################################################################
cat <<EOF
================================================================================
                          SFT Configuration Summary
================================================================================
GPUs:                 $GPU_LIST ($NUM_GPUS GPUs)
Dataset:              $DATASET_NAME
Thinking Mode:        $THINKING_MODE
Percent:              ${PERCENT}p
Data Path:            $DATA_PATH
Chat Template:        $CHAT_TEMPLATE
Enable Thinking:      $ENABLE_THINKING
Model Path:           $MODEL_PATH
Experiment Name:      $EXP_NAME
Output Dir:           $OUTPUT_DIR
================================================================================
EOF

mkdir -p "$OUTPUT_DIR"

################################################################################
# Launch training
################################################################################
python3 -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  -m verl.trainer.fsdp_sft_trainer \
  data.train_files="$DATA_PATH/train.parquet" \
  data.val_files="$DATA_PATH/val.parquet" \
  +data.kwargs.multirole_chat_template_path="$CHAT_TEMPLATE" \
  +data.apply_chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
  data.multiturn.enable=false \
  data.max_length=8196 \
  +data.dataset=$DATASET_NAME \
  data.truncation=right \
  data.train_batch_size=128 \
  data.micro_batch_size_per_gpu=2 \
  data.prompt_key=prompt \
  data.response_key=generation \
  model.partial_pretrain="$MODEL_PATH" \
  model.fsdp_config.model_dtype=bfloat16 \
  model.enable_gradient_checkpointing=true \
  optim.lr=1e-6 \
  optim.warmup_steps_ratio=0.1 \
  optim.lr_scheduler=cosine \
  +trainer.val_before_train=true \
  trainer.total_epochs=2 \
  trainer.project_name=humanlm \
  trainer.experiment_name="$EXP_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.save_freq=300 \
  trainer.test_freq=20 \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  "$@"

echo "Training completed"
echo "Model saved to: $OUTPUT_DIR"
