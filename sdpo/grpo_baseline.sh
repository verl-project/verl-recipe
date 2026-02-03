#!/usr/bin/env bash
# GRPO Baseline Training Script (for comparison with SDPO)

export VLLM_USE_V1=1
export VERL_LOGGING_LEVEL=INFO

python3 -m verl.trainer.main_ppo \
    --config-dir "${PWD}/recipe/sdpo/config" \
    --config-name "qwen3_vl_grpo.yaml" \
    2>&1 | tee "qwen3_vl_grpo_baseline.log"