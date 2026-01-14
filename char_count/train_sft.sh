#!/usr/bin/env bash
set -xeuo pipefail

ENTRYPOINT=${ENTRYPOINT:-"-m verl.trainer.sft_trainer"}

TRAIN_FILES=${TRAIN_FILES:-$HOME/data/char_count/sft/train.parquet}
TEST_FILES=${TEST_FILES:-$HOME/data/char_count/sft/test.parquet}

backend=${BACKEND:-fsdp}

project_name=char_count-sft

RESUME_MODE=auto
MODEL_ID=${MODEL_ID:-HuggingFaceTB/SmolLM2-135M-Instruct}

SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-1}
FSDP_STRATEGY=${FSDP_STRATEGY:-"fsdp2"}

TP_SIZE=${TP_SIZE:-1}
PP_SIZE=${PP_SIZE:-1}
VPP_SIZE=${VPP_SIZE:-null}
CP_SIZE=${CP_SIZE:-1}

PAD_MODE=${PAD_MODE:-no_padding}

USE_REMOVE_PADDING=${USE_REMOVE_PADDING:-True}

FSDP_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.min_lr_ratio=0.1 \
    optim.warmup_style=cosine \
    engine.ulysses_sequence_parallel_size=${SP_SIZE} \
    engine.strategy=${FSDP_STRATEGY} \
    engine.fsdp_size=${FSDP_SIZE}"


MEGATRON_ENGINE_CONFIG="\
    engine=${backend} \
    optim=${backend} \
    optim.lr=2e-5 \
    optim.lr_warmup_steps_ratio=0.01 \
    optim.weight_decay=0.1 \
    optim.betas="[0.9,0.95]" \
    optim.clip_grad=1.0 \
    optim.lr_warmup_init=0 \
    optim.lr_decay_style=cosine \
    optim.min_lr=2e-6 \
    engine.tensor_model_parallel_size=${TP_SIZE} \
    engine.pipeline_model_parallel_size=${PP_SIZE} \
    engine.virtual_pipeline_model_parallel_size=${VPP_SIZE} \
    engine.context_parallel_size=${CP_SIZE} \
    engine.use_mbridge=False"

if [ "$backend" = "fsdp" ]; then
    ENGINE_CONFIG="$FSDP_ENGINE_CONFIG"
    echo "Using fsdp engine"
    exp_name=char_count-sft-SmolLM2-135M-Instruct-fsdp
else
    ENGINE_CONFIG="$MEGATRON_ENGINE_CONFIG"
    echo "Using megatron engine"
    exp_name=char_count-sft-SmolLM2-135M-Instruct-megatron
fi

CKPT_HOME=${CKPT_HOME:-$HOME/experiments/char_count/models/sft/$backend}
mkdir -p "${CKPT_HOME}"

torchrun --standalone --nnodes=1 --nproc-per-node=${NUM_TRAINERS:-1} \
    ${ENTRYPOINT} \
    data.train_files="${TRAIN_FILES}" \
    data.train_batch_size=64 \
    data.val_files="${TEST_FILES}" \
    data.max_length=256 \
    data.pad_mode=${PAD_MODE} \
    data.truncation=error \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu=1792 \
    data.messages_key=messages \
    model.path=$MODEL_ID \
    model.use_remove_padding=${USE_REMOVE_PADDING} \
    ${ENGINE_CONFIG} \
    trainer.test_freq=-1 \
    trainer.save_freq=70 \
    trainer.logger=['console'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPT_HOME}" \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_ckpt_to_keep=5 \
    checkpoint.save_contents=[model,optimizer,extra]