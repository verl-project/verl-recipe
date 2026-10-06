#!/usr/bin/env bash
# Train a Terminal or TIR policy; defaults: 1 node, 8 GPUs/node.
# Set MODEL_PATH, EXPERIMENT_NAME, and NNODES for the run.
set -euo pipefail

RECIPE_DIR=recipe/obsspec
DATA_DIR="${DATA_DIR:-${HOME}/data/terminal_obsspec}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PWD}/outputs}"
CHAT_TEMPLATE="$(python3 -c 'import json, pathlib, sys; print(json.dumps(pathlib.Path(sys.argv[1]).read_text()))' "$RECIPE_DIR/qwen3_xml_tool_calling.jinja")"

DATASET="${DATASET:-endless_terminals}"
NNODES="${NNODES:-1}"
case "${DATASET}" in
    endless_terminals)
        TRAIN_FILES="${TRAIN_FILES:-${DATA_DIR}/endless-terminals-train.parquet}"
        VAL_FILES="${VAL_FILES:-${DATA_DIR}/endless-terminals-val.parquet}"
        MAX_TOKENS_PER_GENERATION="${MAX_TOKENS_PER_GENERATION:-2048}"
        ;;
    dapo_tir)
        TRAIN_FILES="${TRAIN_FILES:-${DATA_DIR}/dapo-math-train.parquet}"
        VAL_FILES="${VAL_FILES:-${DATA_DIR}/dapo-math-val.parquet}"
        MAX_TOKENS_PER_GENERATION="${MAX_TOKENS_PER_GENERATION:-16384}"
        ;;
    *) echo "Unknown DATASET=${DATASET}; expected endless_terminals or dapo_tir" >&2; exit 2 ;;
esac

MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
PROJECT_NAME="${PROJECT_NAME:-terminal-obsspec}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:?set EXPERIMENT_NAME}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/checkpoints}"

ENABLE_THINKING="${ENABLE_THINKING:-false}"
SANDBOX_PROVIDER="${SANDBOX_PROVIDER:-modal}"
MODAL_APP_NAME="${MODAL_APP_NAME:-verl-sandbox}"
MODAL_SANDBOX_TIMEOUT="${MODAL_SANDBOX_TIMEOUT:-540.0}"
MAX_TURNS="${MAX_TURNS:-32}"
MAX_TERMINAL_OUTPUT_CHARS="${MAX_TERMINAL_OUTPUT_CHARS:-10000}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-120.0}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-360.0}"
VERIFIER_TIMEOUT="${VERIFIER_TIMEOUT:-120.0}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
POLICY_INFER_TP="${POLICY_INFER_TP:-2}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
ROLLOUT_N="${ROLLOUT_N:-16}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-16}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-2048}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-16384}"
MAX_TOKEN_LEN_PER_GPU=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))
ROLLOUT_BATCH_DATA_DIR="${ROLLOUT_BATCH_DATA_DIR:-${OUTPUT_ROOT}/${PROJECT_NAME}/${EXPERIMENT_NAME}/rollout_batches}"

python3 -m verl.trainer.main_ppo \
    "+ray_kwargs.ray_init.runtime_env.env_vars.ENABLE_SPECULATION='false'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.SANDBOX_PROVIDER='${SANDBOX_PROVIDER}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MODAL_APP_NAME='${MODAL_APP_NAME}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MODAL_SANDBOX_TIMEOUT='${MODAL_SANDBOX_TIMEOUT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.ENABLE_THINKING='${ENABLE_THINKING}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MAX_TURNS='${MAX_TURNS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MAX_TOKENS_PER_GENERATION='${MAX_TOKENS_PER_GENERATION}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.MAX_TERMINAL_OUTPUT_CHARS='${MAX_TERMINAL_OUTPUT_CHARS}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.STARTUP_TIMEOUT='${STARTUP_TIMEOUT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.AGENT_TIMEOUT='${AGENT_TIMEOUT}'" \
    "+ray_kwargs.ray_init.runtime_env.env_vars.VERIFIER_TIMEOUT='${VERIFIER_TIMEOUT}'" \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="['${TRAIN_FILES}']" \
    data.val_files="['${VAL_FILES}']" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.custom_chat_template="${CHAT_TEMPLATE}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${POLICY_INFER_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
    actor_rollout_ref.rollout.multi_turn.tool_config_path="${RECIPE_DIR}/tools.yaml" \
    actor_rollout_ref.rollout.multi_turn.format=qwen3_coder \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${RECIPE_DIR}/agent_loop.yaml" \
    actor_rollout_ref.rollout.agent.default_agent_loop=terminal_obsspec_agent \
    actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS}" \
    world_model_actor.enable=False \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.rollout_batch_data_dir="${ROLLOUT_BATCH_DATA_DIR}" \
    trainer.logger='["console","wandb"]' \
    trainer.use_v1=True \
    trainer.v1.trainer_mode=sync \
    trainer.nnodes="${NNODES}" \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
    trainer.val_before_train=True \
    trainer.save_freq=100 \
    trainer.test_freq=20 \
    trainer.total_training_steps=500 \
    "$@"
