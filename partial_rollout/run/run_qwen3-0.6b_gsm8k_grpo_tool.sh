#!/usr/bin/env bash
set -xeo pipefail

export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
export PYTHONUNBUFFERED=1
export HTTP_PROXY=
export HTTPS_PROXY=

# Tool-calling variant of run_qwen3-0.6b_gsm8k_grpo.sh.
# Differences vs. the single-turn script:
#   - default_agent_loop=tool_agent (multi-turn agent loop with tool calls)
#   - rollout.multi_turn.{enable,tool_config_path,max_assistant_turns} set
#   - data.return_raw_chat=True so messages list survives to the agent loop
#   - max_prompt_length / max_response_length doubled to give room for
#     tool-response turns within the same response budget
#
# Dataset note: generate the multi-turn-with-tool dataset before running:
#   python3 examples/data_preprocess/gsm8k_multiturn_w_tool.py \
#       --local_save_dir $HOME/data/gsm8k_tool
# Kept separate from $HOME/data/gsm8k/ so the single-turn run script can
# coexist without one preprocessor overwriting the other's parquet.

# Anchor paths to the script's directory so tool config and log file stay
# co-located with the script, regardless of where the launcher's $(pwd) is.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_CONFIG_PATH="$SCRIPT_DIR/gsm8k_tool_config.yaml"

python3 -m recipe.partial_rollout.main_ppo \
    trainer.project_name="partial_rollout" \
    trainer.experiment_name="partial_rollout_tool" \
    trainer.logger=[console,swanlab] \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10000 \
    trainer.resume_mode=disable \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 \
    data.train_files=$HOME/data/gsm8k_tool/train.parquet \
    data.val_files=$HOME/data/gsm8k_tool/test.parquet \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.train_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=3584 \
    data.return_raw_chat=True \
    +async_training.partial_rollout=True \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.rollout_correction.rollout_is=token \
    algorithm.rollout_correction.rollout_is_threshold="2.0" \
    actor_rollout_ref.model.path=Qwen/Qwen3-0.6B \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=k3+ \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.fsdp_config.offload_policy=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.offload_policy=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=5 \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG_PATH" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_model_len=4608 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    critic.model.path=Qwen/Qwen3-0.6B \
    critic.optim.lr=1e-5 \
    critic.ppo_micro_batch_size_per_gpu=2 \
    2>&1 | tee "$SCRIPT_DIR/verl_partial_rollout_tool.log"
