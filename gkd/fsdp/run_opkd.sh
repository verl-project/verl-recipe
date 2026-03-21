set -x

# 0. download the config
# only need to download the `configuration_deepseek.py`, `config.json`, `tokenizer_config.json`, `tokenizer.json` and `generation_config.json`
# remove the `quantization_config` in the `config.json`
# set `num_nextn_predict_layers=0` to disable MTP, which is not currently supported

# huggingface-cli download deepseek-ai/DeepSeek-V3-0324 configuration_deepseek.py config.json

# 1. download the dist_ckpt format model from https://huggingface.co/BearBiscuit05/dpsk-v3-671B-BF16-dist_ckpt/tree/main
# change the HF_MODEL_PATH and DIST_CKPT_PATH to your own path

# Please set your own paths
STUDENT_MODEL_PATH=~/models/Qwen3-4B
TEACHER_MODEL_PATH=~/models/Qwen3-32B


SAVE_CKPT_PATH=~/qwen3_4B_kl_exp1
train_files=~/datasets/100k_general7_mathcode3.parquet
test_files=~/datasets/100k_general7_mathcode3.parquet

function now() {
    date '+%Y-%m-%d-%H-%M'
}

WORKING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/config/runtime_env.yaml"}
ray job submit --no-wait --runtime-env="${RUNTIME_ENV}" \
    --working-dir "${WORKING_DIR}" \
    -- python3 -m main_opkd --config-name opkd_trainer \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=32 \
    data.return_raw_chat=True \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=192 \
    data.truncation='error' \
    data.trust_remote_code=True \
    +data.apply_chat_template_kwargs='{enable_thinking:false}' \
    +data.seed=1234 \
    actor_rollout_ref.hybrid_engine=True \
    actor_rollout_ref.model.path=$STUDENT_MODEL_PATH \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    +actor_rollout_ref.actor.topk=256 \
    +actor_rollout_ref.actor.kd_loss_type=forward_kl \
    +actor_rollout_ref.actor.jsd_alpha=0.5 \
    +actor_rollout_ref.actor.use_power_weighting=True \
    +actor_rollout_ref.actor.power_alpha=1.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=256 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.ref.teacher_path=$TEACHER_MODEL_PATH \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=512 \
    trainer.logger='[console, wandb]' \
    trainer.project_name='OPKD' \
    trainer.experiment_name="qwen3_4B_kl_exp1_$(now)" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    +trainer.ref_n_gpus_per_node=8 \
    +trainer.ref_nnodes=2 \
    trainer.save_freq=150 \
    trainer.test_freq=3000 \
    trainer.default_local_dir=$SAVE_CKPT_PATH \
    trainer.total_epochs=3 
    $@