# This should have your llm judge api keys, as well as wandb api key
export ENV_FILE="$PROJECT_DIR/.env"

# Paths for RL training
export PROJECT_DIR="/path/to/shared/project"
export SCRATCH_DIR="/path/to/your/scratch/$USER"

export DATASET_DIR="$PROJECT_DIR/llm_twin/processed_data"
export MODEL_PATH="$PROJECT_DIR/llm_twin/models/Qwen3-8B"
export CACHE_DIR="$PROJECT_DIR/llm_twin/verl_cache"
export OUTPUT_DIR="$SCRATCH_DIR/humanlm_outputs/$EXP_NAME"

# Set Cache directories
export HF_HOME="$SCRATCH_DIR/hf"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export XDG_CACHE_HOME="$HF_HOME/xdg"