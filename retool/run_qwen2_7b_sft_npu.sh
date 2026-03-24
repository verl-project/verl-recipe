#!/bin/bash
set -x

nnodes=1
nproc_per_node=8

project_name=retool_sft
experiment_name=multiturn-sft-qwen-2.5-7b-instruct

TRAIN_DATA=PATH/TO/ReTool-SFT/data/train-00000-of-00001.parquet
EVAL_DATA=PATH/TO/ReTool-SFT/data/train-00000-of-00001.parquet
MODEL_PATH=PATH/TO/Qwen2.5-7B-Instruct
SAVE_PATH=PATH/TO/checkpoint/$experiment_name

torchrun --nnodes=$nnodes \
     --nproc_per_node=$nproc_per_node \
     -m verl.trainer.sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$EVAL_DATA \
    data.max_length=16384 \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu=8 \
    model.path=$MODEL_PATH \
    engine=fsdp \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    data.ignore_input_ids_mismatch=True \
    trainer.logger='["console"]' \
    trainer.total_epochs=6 \
    trainer.save_freq=100 \
    trainer.device=npu \
    engine.ulysses_sequence_parallel_size=2 \
    model.use_remove_padding=true
