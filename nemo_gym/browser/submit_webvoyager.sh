#!/bin/bash
#SBATCH --job-name=verl-nemogym-browser-grpo-8b
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --partition=your_partition
#SBATCH --account=your_account
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

GPUS_PER_NODE=8

source "${SLURM_SUBMIT_DIR}/config.env"

MODEL_PATH="/path/to/Qwen3-8B"
TRAIN_FILE="/path/to/webvoyager_train.jsonl"   # produced by prepare_webvoyager_data.py
TEST_FILE="${TRAIN_FILE}"
CKPTS_DIR="${RESULTS_ROOT}/grpo-qwen3-8b-browser"

CONTAINER="verlai/verl:vllm018.latest"
MOUNTS="/lustre:/lustre"

mkdir -p "${CKPTS_DIR}"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')

RAY_PORT=6379
ip_head="${head_node_ip}:${RAY_PORT}"
echo "Head node: ${head_node} (${head_node_ip})"

SRUN_ARGS="--no-container-mount-home --container-image=${CONTAINER} --container-mounts=${MOUNTS} --container-workdir=${VERL_ROOT}"

echo "Starting Ray head on ${head_node}..."
srun --nodes=1 --ntasks=1 -w "${head_node}" ${SRUN_ARGS} --container-name=ray-head \
    env -u ROCR_VISIBLE_DEVICES WANDB_API_KEY="${WANDB_API_KEY}" ray start --head \
        --node-ip-address="${head_node_ip}" \
        --port=${RAY_PORT} \
        --num-gpus="${GPUS_PER_NODE}" \
        --block &
sleep 10

worker_num=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting Ray worker ${i} on ${node_i}..."
    srun --nodes=1 --ntasks=1 -w "${node_i}" ${SRUN_ARGS} \
        env -u ROCR_VISIBLE_DEVICES WANDB_API_KEY="${WANDB_API_KEY}" ray start \
            --address="${ip_head}" \
            --num-gpus="${GPUS_PER_NODE}" \
            --block &
    sleep 5
done

CONTAINER_DIR="/raid/enroot/data/user-${UID}/pyxis_${SLURM_JOB_ID}_ray-head"
echo "Waiting for ray-head container at ${CONTAINER_DIR}..."
elapsed=0
while [[ ! -d "${CONTAINER_DIR}" && ${elapsed} -lt 300 ]]; do
    sleep 5
    elapsed=$((elapsed + 5))
done
if [[ ! -d "${CONTAINER_DIR}" ]]; then
    echo "ERROR: ray-head container never appeared after 300s"
    exit 1
fi
echo "Container ready. Waiting 90s for all Ray workers to connect..."
sleep 90

echo "Installing nemo-gym..."
srun --overlap --nodes=1 --ntasks=1 -w "${head_node}" \
    --no-container-mount-home --container-mounts=${MOUNTS} \
    --container-name=ray-head \
    bash -c "echo 'blinker==1.4' > /tmp/constraints.txt && pip install -q uv && pip install -q -e ${NEMO_GYM_ROOT} -c /tmp/constraints.txt"

echo "Starting the interactive_browser resources server on ${head_node}..."
srun --overlap --nodes=1 --ntasks=1 -w "${head_node}" \
    --no-container-mount-home --container-mounts=${MOUNTS} \
    --container-workdir="${NEMO_GYM_ROOT}" --container-name=ray-head \
    env LEXMOUNT_API_KEY="${LEXMOUNT_API_KEY:-}" \
        LEXMOUNT_PROJECT_ID="${LEXMOUNT_PROJECT_ID:-}" \
        LEXMOUNT_BASE_URL="${LEXMOUNT_BASE_URL:-}" \
    gym env start --resources-server "${BROWSER_RESOURCES_SERVER}" --no-agent --no-model &
sleep 30

echo "Launching training on ${head_node}..."
PYTHONUNBUFFERED=1 srun --overlap --nodes=1 --ntasks=1 -w "${head_node}" \
    --no-container-mount-home --container-mounts=${MOUNTS} \
    --container-workdir=${VERL_ROOT} --container-name=ray-head \
    env -u ROCR_VISIBLE_DEVICES \
        WANDB_API_KEY="${WANDB_API_KEY}" \
        HF_HOME="${HF_HOME}" \
        HF_HUB_CACHE="${HF_HOME}/hub" \
        RAY_ADDRESS="auto" \
        VLLM_USE_V1=1 \
        NEMO_GYM_ROOT="${NEMO_GYM_ROOT}" \
        NEMO_GYM_BROWSER_URL="${NEMO_GYM_BROWSER_URL}" \
        JUDGE_BASE_URL="${JUDGE_BASE_URL}" \
        JUDGE_API_KEY="${JUDGE_API_KEY}" \
        JUDGE_MODEL="${JUDGE_MODEL}" \
        PYTHONPATH="${VERL_ROOT}/recipe/nemo_gym/browser:${VERL_ROOT}" \
        NEMO_GYM_BROWSER_DROP_INVALID=1 \
        VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    python3 -m verl.trainer.main_ppo \
            data.train_files="${TRAIN_FILE}" \
            data.val_files="${TEST_FILE}" \
            +data.custom_cls.path="${VERL_ROOT}/recipe/nemo_gym/browser/dataset.py" \
            +data.custom_cls.name=BrowserJSONLDataset \
            data.truncation=left \
            data.train_batch_size=8 \
            data.max_prompt_length=8192 \
            data.max_response_length=16384 \
            actor_rollout_ref.rollout.n=8 \
            algorithm.adv_estimator=grpo \
            algorithm.use_kl_in_reward=False \
            actor_rollout_ref.model.path="${MODEL_PATH}" \
            actor_rollout_ref.actor.optim.lr=1e-6 \
            actor_rollout_ref.actor.ppo_mini_batch_size=8 \
            actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
            actor_rollout_ref.actor.use_kl_loss=True \
            actor_rollout_ref.actor.kl_loss_coef=0.001 \
            actor_rollout_ref.actor.kl_loss_type=low_var_kl \
            actor_rollout_ref.actor.entropy_coeff=0 \
            actor_rollout_ref.rollout.name=vllm \
            actor_rollout_ref.rollout.mode=async \
            actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
            actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
            actor_rollout_ref.rollout.temperature=1.0 \
            actor_rollout_ref.rollout.multi_turn.enable=True \
            actor_rollout_ref.rollout.multi_turn.format=hermes \
            actor_rollout_ref.rollout.multi_turn.max_assistant_turns=20 \
            actor_rollout_ref.rollout.multi_turn.tool_config_path="${VERL_ROOT}/recipe/nemo_gym/browser/configs/browser_tool_config.yaml" \
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.enable-auto-tool-choice=true' \
            '+actor_rollout_ref.rollout.engine_kwargs.vllm.tool-call-parser=hermes' \
            actor_rollout_ref.rollout.agent.default_agent_loop=nemo_gym_browser_agent \
            'trainer.logger=["console","wandb"]' \
            trainer.project_name=verl-nemogym-browser \
            trainer.experiment_name=grpo-qwen3-8b-webvoyager \
            trainer.n_gpus_per_node=${GPUS_PER_NODE} \
            trainer.nnodes=${SLURM_JOB_NUM_NODES} \
            trainer.val_before_train=False \
            trainer.save_freq=20 \
            trainer.total_training_steps=60 \
            trainer.default_local_dir="${CKPTS_DIR}" \
    2>&1
