#!/bin/bash
#SBATCH --account=coreai_dlalgo_llm
#SBATCH --job-name=verl-dynamo-30b
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00
#SBATCH --output=/lustre/fsw/portfolios/coreai/users/sopyang/dynamorl_workspace/slurm/logs/train_30b_dynamo_%j.log
#SBATCH --error=/lustre/fsw/portfolios/coreai/users/sopyang/dynamorl_workspace/slurm/logs/train_30b_dynamo_%j.err
#
# Dynamo-backed retool: same workload as recipe/retool/train_30b_rl.sh
# but with rollout.name=dynamo + KV-router Frontend.
# Multi-node (2x8 H100) — uses the new recipe.dynamo (Route B: subprocess
# + watchdog), see recipe/dynamo/dynamo_design_0507.md.
#
# Differences vs the original dynamo_bk version of this script:
#   * ai-dynamo pinned to 1.0.2 (1.1.0 needs vLLM with MultiModalUUIDDict,
#     which the vllm017 container doesn't have)
#   * pip-install nixl (transitive dep of dynamo.common.multimodal)
#   * download static etcd + nats-server binaries to a cached /workspace dir
#   * set VERL_DYNAMO_LOG_DIR to surface subprocess logs out of /tmp
#   * drop VERL_DYNAMO_ENABLE_RUNTIME / VERL_DYNAMO_ENABLE_FRONTEND env-var
#     patches — those were Route A (m3/m4) toggles; Route B is unconditional.

set -x
WORKSPACE=/lustre/fsw/portfolios/coreai/users/sopyang/
CONTAINER="/lustre/fsw/portfolios/coreai/users/sopyang/images/vllm017_latest.sqsh"

echo "=== Qwen3-30B Retool / Dynamo backend (Route B) — multi-node (2x8 H100) ==="
echo "Node: $(hostname), $(date -Iseconds)"

nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=( $nodes )
node_1=${nodes_array[0]}
node_2=${nodes_array[1]}
ip=$node_1
port=6379
ip_head=$ip:$port
export ip_head
echo "Ray head at $ip_head; worker on $node_2"

# Shared bootstrap — verl + retool patches + ai-dynamo SDK + etcd/nats.
read -r -d '' bootstrap <<EOF
set -x
export PIP_CACHE_DIR=/workspace/hf_models/.cache/pip
export HF_HOME=/workspace/hf_models/.cache/huggingface
# Use cached model files; avoid HF API calls (transformers' model_info check
# hits the API even for cached models and gets 429-rate-limited from a
# shared SLURM IP).
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset ROCR_VISIBLE_DEVICES 2>/dev/null
mkdir -p "\$PIP_CACHE_DIR" "\$HF_HOME"

pip install verl 2>&1 | tail -3
pip install setuptools sandbox-fusion 2>&1 | tail -1
# Pin ai-dynamo to 1.0.2: 1.1.0 introduced multimodal_utils.protocol that
# requires MultiModalUUIDDict from vllm.inputs (vLLM 0.18+), incompatible
# with the vllm017 container.
DYNAMO_VER=\${DYNAMO_VER:-1.0.2}
pip install --pre --no-deps "ai-dynamo==\${DYNAMO_VER}" "ai-dynamo-runtime==\${DYNAMO_VER}" 2>&1 | tail -5 || {
    echo "[fatal] ai-dynamo install failed; dynamo backend needs the SDK" >&2
    exit 2
}
python3 -c "import dynamo, dynamo.runtime, dynamo.llm; print('dynamo SDK loaded')"
pip install pyzmq 2>&1 | tail -1
python3 -c "import nixl" 2>/dev/null || pip install nixl 2>&1 | tail -3
python3 -c "import aiohttp" 2>/dev/null || pip install aiohttp 2>&1 | tail -1

# Static etcd + nats-server binaries cached in /workspace/dynamo_bin so this
# pulls only once per cluster lifetime.
DYN_BIN_DIR=/workspace/dynamo_bin
mkdir -p "\$DYN_BIN_DIR"
ETCD_VER=v3.5.21
NATS_VER=v2.10.22
if [[ ! -x "\$DYN_BIN_DIR/etcd" ]]; then
    cd /tmp
    curl -sSL "https://github.com/etcd-io/etcd/releases/download/\${ETCD_VER}/etcd-\${ETCD_VER}-linux-amd64.tar.gz" | tar xz
    install -m 755 "etcd-\${ETCD_VER}-linux-amd64/etcd"     "\$DYN_BIN_DIR/etcd"
    install -m 755 "etcd-\${ETCD_VER}-linux-amd64/etcdctl"  "\$DYN_BIN_DIR/etcdctl"
    rm -rf "etcd-\${ETCD_VER}-linux-amd64"*
    cd -
fi
if [[ ! -x "\$DYN_BIN_DIR/nats-server" ]]; then
    cd /tmp
    curl -sSL "https://github.com/nats-io/nats-server/releases/download/\${NATS_VER}/nats-server-\${NATS_VER}-linux-amd64.tar.gz" | tar xz
    install -m 755 "nats-server-\${NATS_VER}-linux-amd64/nats-server" "\$DYN_BIN_DIR/nats-server"
    rm -rf "nats-server-\${NATS_VER}-linux-amd64"*
    cd -
fi
export PATH="\$DYN_BIN_DIR:\$PATH"
etcd --version | head -1
nats-server --version

VERL_DIR=\$(python3 -c "import verl,os; print(os.path.dirname(verl.__file__))")
grep -rl "bucket_size_mb" "\$VERL_DIR" --include="*.py" | xargs sed -i "s/bucket_size_mb=2048/bucket_size_mb=4096/g; s/bucket_size_mb: int = 512/bucket_size_mb: int = 4096/g"

ROLLOUT_PY="\$VERL_DIR/workers/rollout/vllm_rollout/vllm_rollout.py"
test -f "\$ROLLOUT_PY" && grep -q "update_weights_bucket_megabytes" "\$ROLLOUT_PY" && sed -i "s/bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes/bucket_size_mb = max(4096, self.config.checkpoint_engine.update_weights_bucket_megabytes)/" "\$ROLLOUT_PY"

for SF_PY in "\$VERL_DIR/tools/sandbox_fusion_tools.py"; do
  if [ -f "\$SF_PY" ] && ! grep -q "lifetime=\"detached\"" "\$SF_PY"; then
    sed -i "s|TokenBucketWorker\.options(name=\"rate-limiter\", get_if_exists=True)|TokenBucketWorker.options(name=\"rate-limiter\", get_if_exists=True, lifetime=\"detached\", namespace=\"verl_sandbox\")|g" "\$SF_PY"
  fi
done
cd /tmp && rm -rf symeval && git clone -q https://github.com/tongyx361/symeval.git && cd symeval
sed -i "s/from pkg_resources import parse_version/from packaging.version import parse as parse_version/" setup.py
pip install . 2>&1 | tail -1
pip install "antlr4-python3-runtime==4.9.3" 2>&1 | tail -1

# Surface dynamo subprocess logs out of the container's /tmp.
export VERL_DYNAMO_LOG_DIR=/workspace/dynamorl_workspace/slurm/dynamo_logs/\${SLURM_JOB_ID:-manual}
mkdir -p "\$VERL_DYNAMO_LOG_DIR"
EOF

# 1. Ray worker on node_2 (background, joins head)
srun --overlap --nodes=1 --ntasks=1 -w "$node_2" \
  --container-image="$CONTAINER" --container-mounts="${WORKSPACE}:/workspace" \
  bash -c "$bootstrap
ray start --address=$ip_head --block" &
sleep 30

# 2. Head + driver in the same srun on node_1.
read -r -d '' driver_cmd <<EOF
$bootstrap
ray start --head --node-ip-address=$node_1 --port=$port
sleep 5
ray status

python3 -c "import verl; print(f'verl {verl.__version__}')"
python3 -c "import vllm; print(f'vllm {vllm.__version__}')"
python3 -c "import dynamo, dynamo.runtime, dynamo.llm; print('dynamo SDK loaded')"

echo "=== Setting up sandbox-fusion server ==="
cd /tmp && rm -rf SandboxFusion
git clone --depth 1 https://github.com/bytedance/SandboxFusion.git 2>&1 | tail -2
cd SandboxFusion
pip install tenacity structlog psutil aiofiles aiohttp "databases[aiomysql,aiosqlite]" 2>&1 | tail -3
python3 /workspace/dynamorl_workspace/verl/recipe/retool/patch_sf_runner.py /tmp/SandboxFusion
export PYTHONPATH=/tmp/SandboxFusion:\${PYTHONPATH:-}

mkdir -p /workspace/dynamorl_workspace/slurm
SANDBOX_LOG=/workspace/dynamorl_workspace/slurm/sandbox_fusion_\${SLURM_JOB_ID:-manual}.log
nohup python3 -m uvicorn sandbox.server.server:app --host 0.0.0.0 --port 8080 > "\$SANDBOX_LOG" 2>&1 &
disown
sleep 8
curl -sf http://localhost:8080/v1/ping && echo " Sandbox-Fusion OK" || echo "Sandbox-Fusion FAILED"

cd /workspace/dynamorl_workspace/verl
export PYTHONPATH=/workspace/dynamorl_workspace/verl:/tmp/SandboxFusion:\${PYTHONPATH:-}

project_name=retool_30b_dynamo_2node_routeB
experiment_name=routeB_30b_dynamo_2node
default_local_dir=/workspace/dynamorl_workspace/ckpt/\$experiment_name
mkdir -p "\$default_local_dir"

mkdir -p /workspace/dynamorl_workspace/slurm
export VERL_ROLLOUT_PROMPT_LOG_PATH=/workspace/dynamorl_workspace/slurm/rollout_prompt_response_\${experiment_name}_\${SLURM_JOB_ID:-manual}.jsonl
rm -f "\$VERL_ROLLOUT_PROMPT_LOG_PATH"

if [ -n "\${WANDB_API_KEY:-}" ]; then
  TRAINER_LOGGER="[\"console\",\"wandb\"]"
else
  TRAINER_LOGGER="[\"console\"]"
fi

dapo_math_17k=/workspace/dynamorl_workspace/datasets/DAPO-Math-17k
aime_2025=/workspace/dynamorl_workspace/datasets/aime_2025

export RAY_ADDRESS=$ip_head

python3 recipe/dynamo/main_dynamo.py \\
    algorithm.adv_estimator=grpo \\
    algorithm.use_kl_in_reward=False \\
    algorithm.kl_ctrl.kl_coef=0.0 \\
    data.train_files="[\"\$dapo_math_17k\"]" \\
    data.val_files="[\"\$aime_2025\"]" \\
    data.return_raw_chat=True \\
    data.train_batch_size=16 \\
    data.max_prompt_length=2048 \\
    data.max_response_length=16384 \\
    data.filter_overlong_prompts=True \\
    data.truncation=error \\
    data.custom_cls.path=/workspace/dynamorl_workspace/verl/recipe/retool/retool.py \\
    data.custom_cls.name=CustomRLHFDataset \\
    custom_reward_function.path=/workspace/dynamorl_workspace/verl/recipe/retool/retool.py \\
    custom_reward_function.name=compute_score \\
    actor_rollout_ref.model.path=/workspace/RL/hf_models/Qwen3-30B-A3B \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.model.enable_gradient_checkpointing=True \\
    actor_rollout_ref.actor.use_kl_loss=False \\
    actor_rollout_ref.actor.kl_loss_coef=0.0 \\
    actor_rollout_ref.actor.clip_ratio_low=0.2 \\
    actor_rollout_ref.actor.clip_ratio_high=0.28 \\
    actor_rollout_ref.actor.clip_ratio_c=10.0 \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.actor.use_dynamic_bsz=True \\
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \\
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=18432 \\
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=4 \\
    actor_rollout_ref.actor.fsdp_config.param_offload=True \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \\
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=73728 \\
    actor_rollout_ref.rollout.name=dynamo \\
    actor_rollout_ref.rollout.mode=async \\
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.return_tokens_as_token_ids=true \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \\
    actor_rollout_ref.rollout.multi_turn.enable=True \\
    actor_rollout_ref.rollout.multi_turn.max_user_turns=16 \\
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=16 \\
    actor_rollout_ref.rollout.multi_turn.tool_config_path=/workspace/dynamorl_workspace/verl/recipe/retool/sandbox_fusion_tool_config.yaml \\
    actor_rollout_ref.rollout.multi_turn.format=hermes \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \\
    actor_rollout_ref.rollout.n=8 \\
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \\
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \\
    actor_rollout_ref.rollout.val_kwargs.n=30 \\
    trainer.logger="\$TRAINER_LOGGER" \\
    trainer.project_name=\$project_name \\
    trainer.experiment_name=\$experiment_name \\
    trainer.n_gpus_per_node=8 \\
    trainer.val_before_train=False \\
    trainer.nnodes=2 \\
    trainer.save_freq=10 \\
    trainer.default_local_dir=\$default_local_dir \\
    trainer.resume_mode=auto \\
    trainer.test_freq=999 \\
    trainer.total_training_steps=1000

echo ">>> Sandbox-Fusion log (last 20 lines):"
tail -20 "\$SANDBOX_LOG" 2>/dev/null
EOF

srun --overlap --nodes=1 --ntasks=1 -w "$node_1" \
  --container-image="$CONTAINER" --container-mounts="${WORKSPACE}:/workspace" \
  bash -c "$driver_cmd"

echo "=== Done $(date -Iseconds) ==="
