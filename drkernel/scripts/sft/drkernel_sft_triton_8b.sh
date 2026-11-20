#!/bin/bash
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# DR.Kernel SFT (multi-turn) launcher — cold-start on Triton trajectories.
#
# Mirrors the original DR.Kernel SFT recipe (qwen3-8b-base coldstart) but
# routes through verl's modern SFT entry point
# (`verl.trainer.sft_trainer`, config: sft_trainer_engine), using the
# `MultiTurnSFTDataset` path that consumes a `messages` column directly.
#
# Dataset: triton_sft_trajectories_8k.parquet
#   Columns: original_python_code (str), messages (list<{role,content}>), id (str)
#   Each sample is a full 16-turn (8 user/assistant) conversation between the
#   model and the KernelGym feedback environment, which matches the multi-turn
#   RL rollout format and serves as the SFT cold start before drkernel RL.
#
# Required env (must be set):
#   MODEL_PATH         path to the base model on the NPU cluster
#                      (e.g. /data/nfs/model/Qwen3-8B)
#   TRAIN_FILES        path to the SFT parquet
#                      (e.g. /data/nfs/ahmad/dataset/thinking/triton_sft_trajectories_8k.parquet)
#
# Optional env (defaults shown):
#   VAL_FILES=null              parquet for validation; null skips val
#   NNODES=1                    total nodes (must match torchrun --nnodes)
#   NGPUS_PER_NODE=16           NPUs per node (Ascend default)
#   NODE_RANK=0                 rank of this node (override per node)
#   MASTER_ADDR=127.0.0.1       head node hostname (e.g. npu-3 on the cluster)
#   MASTER_PORT=29500           torchrun rendezvous port
#
#   TRAIN_BATCH_SIZE=64         global batch size
#   MICRO_BATCH_SIZE_PER_GPU=2  per-device micro batch (static-bsz)
#   USE_DYNAMIC_BSZ=False       True -> use max_token_len_per_gpu instead
#   MAX_TOKEN_LEN_PER_GPU=36864 only used when USE_DYNAMIC_BSZ=True
#   MAX_LENGTH=18432            sequence truncation length
#   TOTAL_EPOCHS=4
#   SAVE_FREQ=50
#   TEST_FREQ=-1                set >0 when VAL_FILES is provided
#   LEARNING_RATE=2e-5
#   LR_WARMUP_RATIO=0.1
#   SP_SIZE=4                   engine.ulysses_sequence_parallel_size
#   STRATEGY=fsdp2              fsdp | fsdp2
#   OFFLOAD_POLICY=True         FSDP2-only: CPU-offload params/grad/optimizer.
#                               On FSDP1, param_offload/optimizer_offload must
#                               be set jointly (both True or both False); the
#                               engine refuses mixed offload.
#   TRUNCATION=right
#
#   PROJECT_NAME=drkernel-sft
#   EXP_NAME=drkernel-8b-coldstart-triton
#
# Example (single 16-NPU node, dense Qwen3-8B):
#   MODEL_PATH=/data/nfs/model/Qwen3-8B \
#   TRAIN_FILES=/data/nfs/ahmad/dataset/thinking/triton_sft_trajectories_8k.parquet \
#   bash recipe/drkernel/scripts/sft/drkernel_sft_triton_8b.sh
#
# Multi-node (run on each node after syncing repo; head is rank 0):
#   NNODES=4 NODE_RANK=<i> MASTER_ADDR=npu-3 MASTER_PORT=29500 \
#   MODEL_PATH=... TRAIN_FILES=... \
#   bash recipe/drkernel/scripts/sft/drkernel_sft_triton_8b.sh

set -x
set -eo pipefail

# ---------- Ascend env (no-op off-NPU) ----------
export HYDRA_FULL_ERROR=1
if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi
if [ -f /usr/local/Ascend/nnal/atb/set_env.sh ]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi

# ---------- Required ----------
MODEL_PATH="${MODEL_PATH:-/home/a00927464/models/Qwen3-8B}"
TRAIN_FILES="${TRAIN_FILES:-/home/a00927464/dataset/sft-kernel/triton_sft_trajectories_8k.parquet}"
VAL_FILES="${VAL_FILES:-null}"

# ---------- Distributed defaults ----------
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-12}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

# ---------- SFT hyper-params (DR.Kernel 8b-coldstart defaults) ----------
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-2}"
USE_DYNAMIC_BSZ="${USE_DYNAMIC_BSZ:-False}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-36864}"
MAX_LENGTH="${MAX_LENGTH:-18432}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-4}"
SAVE_FREQ="${SAVE_FREQ:-50}"
TEST_FREQ="${TEST_FREQ:--1}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.1}"
SP_SIZE="${SP_SIZE:-4}"
STRATEGY="${STRATEGY:-fsdp2}"
OFFLOAD_POLICY="${OFFLOAD_POLICY:-True}"
TRUNCATION="${TRUNCATION:-right}"

PROJECT_NAME="${PROJECT_NAME:-drkernel-sft}"
EXP_NAME="${EXP_NAME:-drkernel-8b-coldstart-triton}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-checkpoints/${PROJECT_NAME}/${EXP_NAME}}"

# ---------- Resolve repo root + PYTHONPATH ----------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
cd "${REPO_ROOT}"

mkdir -p "logs/${PROJECT_NAME}"

# ---------- Launch ----------
torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NGPUS_PER_NODE}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    -m verl.trainer.sft_trainer \
    \
    `# data (MultiTurnSFTDataset reads the messages column directly)` \
    data.train_files="${TRAIN_FILES}" \
    data.val_files="${VAL_FILES}" \
    data.messages_key=messages \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    data.max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    data.max_length="${MAX_LENGTH}" \
    data.truncation="${TRUNCATION}" \
    data.pad_mode=no_padding \
    `# Qwen3 thinking template injects <think></think> only on the final turn` \
    `# when tokenized whole, which disagrees with the per-turn-then-concat` \
    `# path MultiTurnSFTDataset uses. The trajectories were generated against` \
    `# the per-turn representation, so accept it as the source of truth.` \
    data.ignore_input_ids_mismatch=True \
    \
    `# model` \
    model.path="${MODEL_PATH}" \
    model.use_remove_padding=True \
    model.enable_gradient_checkpointing=True \
    model.trust_remote_code=True \
    \
    `# engine (FSDP2 + Ulysses SP; offload_policy is the FSDP2 way to do` \
    `# what DR.Kernel called cpu_offload — params/grad/optimizer all go to CPU` \
    `# during train and are gathered for forward. On FSDP1, swap this for` \
    `# engine.param_offload + engine.optimizer_offload set jointly.)` \
    engine.strategy="${STRATEGY}" \
    engine.ulysses_sequence_parallel_size="${SP_SIZE}" \
    engine.model_dtype=bf16 \
    engine.offload_policy="${OFFLOAD_POLICY}" \
    \
    `# optim` \
    optim.lr="${LEARNING_RATE}" \
    optim.lr_warmup_steps_ratio="${LR_WARMUP_RATIO}" \
    optim.lr_scheduler_type=cosine \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    \
    `# trainer` \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.device=npu \
    trainer.logger='["console","tensorboard"]' \
    \
    "$@" 2>&1 | tee "logs/${PROJECT_NAME}/run_${EXP_NAME}.log"
