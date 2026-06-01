# ray stop --force

export CUSTOM_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HYDRA_FULL_ERROR=1
export HCCL_HOST_SOCKET_PORT_RANGE="auto"
export HCCL_NPU_SOCKET_PORT_RANGE="auto"
mkdir -p logs
export WORKING_DIR=${WORKING_DIR:-"${PWD}"}

TRAIN_DATA=${TRAIN_DATA:-"./recipe/dance_grpo_mm/data/pickapic_single.json"}
VAL_DATA=${VAL_DATA:-"$TRAIN_DATA"}
MODEL_PATH=${MODEL_PATH:-"path_to_mm2"}
REWARD_MODEL_PATH=${REWARD_MODEL_PATH:-"path_to_hpsv3/HPSv3.safetensors"}

LOG_PROB_BATCH_SIZE=${LOG_PROB_BATCH_SIZE:-8}
PPO_BATCH_SIZE=${PPO_BATCH_SIZE:-8}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
ROLLOUT_N=${ROLLOUT_N:-8}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-500}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-500}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-512}

# Checkpoint controls (saved under: ${CHECKPOINT_ROOT_DIR}/global_step_<step>/actor/actor_model.pt)
CHECKPOINT_ROOT_DIR=${CHECKPOINT_ROOT_DIR:-"${WORKING_DIR}/checkpoints/mammothmoda2_project/mammothmoda2_experiment"}
SAVE_FREQ=-1
SAVE_FREQ=${SAVE_FREQ:-50}
MAX_ACTOR_CKPT_TO_KEEP=${MAX_ACTOR_CKPT_TO_KEEP:-3}
RESUME_MODE=${RESUME_MODE:-disable}  # disable | auto | resume_path
RESUME_FROM_PATH=${RESUME_FROM_PATH:-""}  # e.g. ${CHECKPOINT_ROOT_DIR}/global_step_200

ROLLOUT_HEIGHT=${ROLLOUT_HEIGHT:-512}
ROLLOUT_WIDTH=${ROLLOUT_WIDTH:-512}
ROLLOUT_STEPS=${ROLLOUT_STEPS:-40}
ROLLOUT_ETA=${ROLLOUT_ETA:-0.3}
ROLLOUT_INIT_SAME_NOISE=${ROLLOUT_INIT_SAME_NOISE:-true}
ROLLOUT_CFG_SCALE=${ROLLOUT_CFG_SCALE:-3.0}
ROLLOUT_TEXT_CFG_SCALE=${ROLLOUT_TEXT_CFG_SCALE:-3.0}
TRAIN_CFG_SCALE=${TRAIN_CFG_SCALE:-3.0}
ROLLOUT_VAE_SCALE_FACTOR=${ROLLOUT_VAE_SCALE_FACTOR:-16}
ROLLOUT_OUTPUT_DIR=${ROLLOUT_OUTPUT_DIR:-"${WORKING_DIR}/logs"}

ACTOR_TIMESTEP_FRACTION=${ACTOR_TIMESTEP_FRACTION:-0.6}
ACTOR_SAMPLING_STEPS=${ROLLOUT_STEPS}
ACTOR_CLIP_RANGE=${ACTOR_CLIP_RANGE:-0.0001}
ACTOR_ADV_CLIP_MAX=${ACTOR_ADV_CLIP_MAX:-5.0}
ACTOR_KL_COEFF=${ACTOR_KL_COEFF:-0.0}
ACTOR_MAX_GRAD_NORM=${ACTOR_MAX_GRAD_NORM:-1.0}
ACTOR_LR=${ACTOR_LR:-5e-6}
ACTOR_WEIGHT_DECAY=${ACTOR_WEIGHT_DECAY:-0.01}
ACTOR_FREEZE_NON_ATTN_FFN=${ACTOR_FREEZE_NON_ATTN_FFN:-False}

model_name=$(basename "$MODEL_PATH")
model_name=${model_name//./_}
model_name=${model_name//-/_}
log=logs/debug_dance_${model_name}_${CUSTOM_TIMESTAMP}_2222.log

RESUME_ARGS=""
if [[ "$RESUME_MODE" == "resume_path" && -n "$RESUME_FROM_PATH" ]]; then
    RESUME_ARGS="trainer.resume_from_path=$RESUME_FROM_PATH"
fi

python3 -m recipe.dance_grpo.main_dance  \
    --config-path=config \
    --config-name="dance_ppo_trainer" \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$MAX_PROMPT_LENGTH \
    data.max_response_length=128 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.reward_model_path=$REWARD_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=$ACTOR_LR \
    actor_rollout_ref.actor.optim.weight_decay=$ACTOR_WEIGHT_DECAY \
    +actor_rollout_ref.actor.optim.lr_scheduler.name='constant' \
    +actor_rollout_ref.actor.optim.lr_scheduler.num_warmup_steps=0 \
    +actor_rollout_ref.actor.optim.lr_scheduler.num_training_steps=10000 \
    +actor_rollout_ref.actor.optim.lr_scheduler.num_cycles=1 \
    +actor_rollout_ref.actor.ppo_adv_clip_max=$ACTOR_ADV_CLIP_MAX \
    +actor_rollout_ref.actor.ppo_clip_range=$ACTOR_CLIP_RANGE \
    +actor_rollout_ref.actor.ppo_kl_coeff=$ACTOR_KL_COEFF \
    +actor_rollout_ref.actor.ppo_max_grad_norm=$ACTOR_MAX_GRAD_NORM \
    +actor_rollout_ref.actor.train_cfg_scale=$TRAIN_CFG_SCALE \
    +actor_rollout_ref.actor.freeze_non_attn_ffn=$ACTOR_FREEZE_NON_ATTN_FFN \
    +actor_rollout_ref.actor.timestep_fraction=$ACTOR_TIMESTEP_FRACTION \
    +actor_rollout_ref.actor.sampling_steps=$ACTOR_SAMPLING_STEPS \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$LOG_PROB_BATCH_SIZE \
    +actor_rollout_ref.rollout.height=$ROLLOUT_HEIGHT \
    +actor_rollout_ref.rollout.width=$ROLLOUT_WIDTH \
    +actor_rollout_ref.rollout.num_inference_steps=$ROLLOUT_STEPS \
    +actor_rollout_ref.rollout.eta=$ROLLOUT_ETA \
    +actor_rollout_ref.rollout.cfg_scale=$ROLLOUT_CFG_SCALE \
    +actor_rollout_ref.rollout.text_cfg_scale=$ROLLOUT_TEXT_CFG_SCALE \
    +actor_rollout_ref.rollout.vae_scale_factor=$ROLLOUT_VAE_SCALE_FACTOR \
    +actor_rollout_ref.rollout.init_same_noise=$ROLLOUT_INIT_SAME_NOISE \
    +actor_rollout_ref.rollout.output_dir=$ROLLOUT_OUTPUT_DIR \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_BATCH_SIZE \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    algorithm.use_kl_in_reward=False \
    actor_rollout_ref.rollout.mode="sync" \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.val_before_train=False \
    trainer.project_name='mammothmoda2_project' \
    trainer.experiment_name='mammothmoda2_experiment' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$CHECKPOINT_ROOT_DIR \
    trainer.save_freq=$SAVE_FREQ \
    trainer.max_actor_ckpt_to_keep=$MAX_ACTOR_CKPT_TO_KEEP \
    trainer.resume_mode=$RESUME_MODE \
    $RESUME_ARGS \
    trainer.test_freq=1 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.total_training_steps=$TOTAL_TRAINING_STEPS \
    trainer.device=npu \
    ++trainer.fixed_eval=False \
    2>&1 | tee $log
