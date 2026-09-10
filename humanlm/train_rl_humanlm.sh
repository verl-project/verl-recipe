set -e
set -x

CONFIG_FILE="${CLUSTER_CONFIG:-$(dirname "$0")/cluster_config.sh}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "Error: cluster config not found at $CONFIG_FILE" >&2
  exit 1
fi
source "$CONFIG_FILE"

set -a
source "$ENV_FILE"
set +a

# ARGUMENTS
GPU_LIST=${1:?"Error: GPU list required (e.g., '0,1,2,3')"}
DATASET_NAME=${2:?"Error: Dataset name required (reddit|medium|youtube|amazon|wildchat|enron)"}
RUN_MODE=${3:?"Error: mode required (eval_only|train_humanlm)"}

RESUME_PATH=${4:-""}   
MODEL_TYPE=${5:-base}

shift 5

# ENVIRONMENT SETUP 
export CUDA_VISIBLE_DEVICES="$GPU_LIST"
NUM_GPUS=$(echo "$GPU_LIST" | awk -F',' '{print NF}')
GPU_MEMORY_UTILIZATION=0.4

# RUN-MODE CONFIGURATION
CHAT_TEMPLATE="./recipe/humanlm/chat_templates/qwen3_multi_role_template_think.jinja"
ENABLE_HETERO_THINK=False
STRICT_FORMAT=True
VAL_DATA_FILE=val
VAL_BEFORE_TRAIN=False
VAL_METRICS='{response:{state_reward:{weight:1.0,kwargs:{model:"anthropic/claude-haiku-4-5",temperature:0}}}}'
TRAIN_EPOCHS=1
MAX_GEN_LENGTH=512
SAVE_FREQ=25
ADDITIONAL_GENERATION_PROMPT=''

# These are set per mode
RESUME_MODE='auto'
NO_REPEAT_NGRAM_SIZE=0
ENABLE_THINKING=False
SEPARATE_GENERATION=False
USE_DIFF_H_SYS_PROMPTS=False
ENABLE_STATE=False
IDENTIFIER=""
CONFIG=""
STATE_CONFIG=""
export VLLM_USE_V1=1

case "$RUN_MODE" in
  eval_only)
    export VLLM_USE_V1=1
    VAL_DATA_FILE=test
    VAL_BEFORE_TRAIN=True
    STRICT_FORMAT=False
    VAL_METRICS='{response:{state_reward_on_response:{weight:1.0,kwargs:{model:"anthropic/claude-haiku-4-5",temperature:0,config_path:"./recipe/humanlm/state_config/sebvgc.json"}},state_reward:{weight:1.0,kwargs:{model:"anthropic/claude-haiku-4-5",temperature:0}}}}'
    ENABLE_STATE=True
    TRAIN_EPOCHS=0
    CONFIG="r"
    STATE_CONFIG="./recipe/humanlm/state_config/$CONFIG.json"
    RESUME_MODE='resume_path'
    NO_REPEAT_NGRAM_SIZE=4

    case "$MODEL_TYPE" in
      base)
        ENABLE_THINKING=False
        ;;
      base-think)
        ENABLE_THINKING=True
        ;;
      humanlm)
        ENABLE_THINKING=True
        ENABLE_HETERO_THINK=True
        MAX_GEN_LENGTH=1024
        ;;
      *)
        echo "Error: Unknown MODEL_TYPE for eval_only: $MODEL_TYPE" >&2
        exit 1
        ;;
    esac

    IDENTIFIER="eval_qwen3_${MODEL_TYPE}"
    SEPARATE_GENERATION=False
    USE_DIFF_H_SYS_PROMPTS=False
    ;;
    
  train_humanlm)
    CONFIG="sebvgcr"
    IDENTIFIER="humanlm_grpo"
    ENABLE_THINKING=True
    ENABLE_HETERO_THINK=True
    MAX_GEN_LENGTH=1024
    SEPARATE_GENERATION=True
    USE_DIFF_H_SYS_PROMPTS=True
    ENABLE_STATE=False
    STATE_CONFIG="./recipe/humanlm/state_config/sebvgcr.json"
    ;;
    
  *)
    echo "Error: Invalid mode '$RUN_MODE'. Use eval_only|train_humanlm" >&2
    exit 1
    ;;
esac

if [[ "$ENABLE_THINKING" == "True" && "$MAX_GEN_LENGTH" -lt 1024 ]]; then
  MAX_GEN_LENGTH=1024
fi

################################################################################
# DATASET CONFIGURATION 
################################################################################
FILTER_OVERLONG_PROMPTS=True
DATA_CONFIG_FOR_PATH="${DATA_CONFIG:-$CONFIG}"

BATCH_SIZE=$((32 / NUM_GPUS * NUM_GPUS))
case "$DATASET_NAME" in
  reddit)   FILTER_OVERLONG_PROMPTS=False; MAX_LENGTH=5120; DATA_PATH="$DATASET_DIR/reddit_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/50p" ;;
  medium)   MAX_LENGTH=7168;    DATA_PATH="$DATASET_DIR/medium_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/25p" ;;
  youtube)  MAX_LENGTH=5120;    DATA_PATH="$DATASET_DIR/youtube_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/5p" ;;
  amazon)   MAX_LENGTH=7168;    DATA_PATH="$DATASET_DIR/amazon_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/100p" ;;
  wildchat) MAX_LENGTH=7168;    DATA_PATH="$DATASET_DIR/wildchat_english_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/100p" ;;
  enron)    MAX_LENGTH=5120;    DATA_PATH="$DATASET_DIR/enron_processed_dataset_by_post_dedup/rl/$DATA_CONFIG_FOR_PATH/100p" ;;
  *)
    echo "Error: Invalid dataset '$DATASET_NAME' (reddit|medium|youtube|amazon|wildchat|enron)" >&2
    exit 1
    ;;
esac

if [[ "$DATASET_NAME" == "reddit" ]]; then
  if [[ "$RUN_MODE" == "eval_only" ]]; then VAL_SIZE=5000; else VAL_SIZE=1500; fi
else
  if [[ "$RUN_MODE" == "eval_only" ]]; then VAL_SIZE=2000; else VAL_SIZE=500; fi
fi

EXP_NAME="${IDENTIFIER}_${DATASET_NAME}"

# DISPLAY CONFIGURATION SUMMARY
cat <<EOF
================================================================================
                          Configuration Summary
================================================================================
GPUs:                 $GPU_LIST ($NUM_GPUS GPUs)
GPU Memory Util:      $GPU_MEMORY_UTILIZATION
Model Type:           $MODEL_TYPE
Model Path:           $MODEL_PATH
Dataset:              $DATASET_NAME
Max Length:           $MAX_LENGTH
Batch Size:           $BATCH_SIZE
Data Path:            $DATA_PATH
Experiment Name:      $EXP_NAME
Resume Path:          $RESUME_PATH
Training Epochs:      $TRAIN_EPOCHS
================================================================================
EOF

# COMPUTE STOP SEQUENCES FROM STATE CONFIG
if [[ -n "$STATE_CONFIG" && -f "$STATE_CONFIG" ]]; then
    STOP_SEQUENCES=$(python3 -c "import json; print(json.dumps([f'</{h}>' for h in json.load(open('$STATE_CONFIG')).keys()]))" 2>/dev/null || echo '[]')
else
    STOP_SEQUENCES='["</response>"]'
fi

export VERL_STOP_SEQUENCES="$STOP_SEQUENCES"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=$VAL_BEFORE_TRAIN \
    '+trainer.load_state_map=False' \
    algorithm.use_kl_in_reward=False \
    reward.reward_manager.source=importlib \
    reward.reward_manager.name=HumanLMRewardManager \
    reward.reward_manager.module.path="./recipe/humanlm/reward_function.py" \
    custom_reward_function.path="./recipe/humanlm/reward_function.py" \
    custom_reward_function.name="compute_reward" \
    +reward_model.reward_kwargs.enable_state=$ENABLE_STATE \
    '+reward_model.reward_kwargs.fetch_global_best_state=True' \
    +reward_model.reward_kwargs.separate_generation=$SEPARATE_GENERATION \
    +reward_model.reward_kwargs.enable_thinking=$ENABLE_THINKING \
    +reward_model.reward_kwargs.eval_push_to_hub="snap-stanford/$EXP_NAME-split_$VAL_DATA_FILE" \
    +reward_model.reward_kwargs.state_config=$STATE_CONFIG \
    +reward_model.reward_kwargs.strict_format=$STRICT_FORMAT \
    +reward_model.reward_kwargs.val_metrics=$VAL_METRICS \
    +reward_model.reward_kwargs.n_rollouts=4 \
    data.train_files=$DATA_PATH/train.parquet \
    data.val_files=$DATA_PATH/$VAL_DATA_FILE.parquet \
    +data.cache_dir=$CACHE_DIR \
    data.train_batch_size=$BATCH_SIZE \
    data.val_batch_size=256 \
    +data.kwargs.multirole_chat_template_path="$CHAT_TEMPLATE" \
    '+data.seed=0' \
    data.max_response_length=$MAX_GEN_LENGTH \
    data.max_prompt_length=$MAX_LENGTH \
    data.filter_overlong_prompts=$FILTER_OVERLONG_PROMPTS \
    data.truncation='error' \
    data.filter_overlong_prompts_workers=128 \
    +data.state_config_path=$STATE_CONFIG \
    +data.enable_hetero_think=$ENABLE_HETERO_THINK \
    +data.augment_with_states=$USE_DIFF_H_SYS_PROMPTS \
    +data.val_size=$VAL_SIZE \
    +data.dataset=$DATASET_NAME \
    +data.custom_cls.path="./recipe/humanlm/state_dataset.py" \
    +data.custom_cls.name='StateDataset' \
    +reward_model.reward_kwargs.additional_generation_prompt=$ADDITIONAL_GENERATION_PROMPT \
    +data.additional_generation_prompt=$ADDITIONAL_GENERATION_PROMPT \
    +data.apply_chat_template_kwargs.enable_thinking=$ENABLE_THINKING \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=4096 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24000 \
    +actor_rollout_ref.model.custom_chat_template="$CHAT_TEMPLATE" \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.rollout.engine_kwargs.dtype=bfloat16 \
    actor_rollout_ref.rollout.agent.default_agent_loop=humanlm_agent \
    +actor_rollout_ref.rollout.agent.agent_loop_manager_class="recipe.humanlm.humanlm_agent_loop_worker.HumanLMAgentLoopManager" \
    +actor_rollout_ref.rollout.agent.agent_loop_config_path="./recipe/humanlm/configs/humanlm_agent_loop_config.yaml" \
    "+actor_rollout_ref.rollout.engine_kwargs={stop: $STOP_SEQUENCES}" \
    +actor_rollout_ref.no_repeat_ngram_size=0 \
    trainer.resume_mode='auto' \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name='humanlm' \
    trainer.experiment_name="$EXP_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.nnodes=1 \
    trainer.save_freq=$SAVE_FREQ \
    trainer.test_freq=$SAVE_FREQ \
    trainer.default_hdfs_dir=null \
    trainer.log_val_generations=20 \
    trainer.resume_from_path=$RESUME_PATH \
    trainer.total_epochs=$TRAIN_EPOCHS $@