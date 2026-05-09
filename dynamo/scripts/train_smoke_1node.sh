#!/bin/bash
#SBATCH --account=general_sa
#SBATCH --job-name=verl-dynamo-train-smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/general_sa/sopyang/rl/verl_0211/slurm/logs/train_smoke_1node_%j.log
#SBATCH --error=/lustre/fsw/general_sa/sopyang/rl/verl_0211/slurm/logs/train_smoke_1node_%j.err
#
# 1-node training smoke for recipe/dynamo (Route B).
#   * Qwen2.5-0.5B-Instruct on gsm8k
#   * GRPO, 2-3 steps total
#   * TP=1, n_gpus_per_node=2, single node
#
# Pass criterion: at least 1 successful weight update step (training metrics
# logged to console). This is the first end-to-end exercise of the weight
# update control sidecar — collective_rpc('update_weights_from_ipc')
# bridged from the actor through ZMQ to dynamo.vllm subprocess, picked up
# by the verl WorkerExtension, weights flow via CUDA IPC + ZMQ to vLLM.

set -x
WORKSPACE=/lustre/fsw/general_sa/sopyang/
CONTAINER="/lustre/fsw/general_sa/sopyang/images/vllm017_latest.sqsh"

echo "=== Dynamo Route B 1-node training smoke (Qwen2.5-0.5B + gsm8k) ==="
echo "Node: $(hostname), $(date -Iseconds)"

read -r -d '' inner <<EOF
set -euo pipefail
set -x
export PIP_CACHE_DIR=/workspace/hf_models/.cache/pip
export HF_HOME=/workspace/hf_models/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
unset ROCR_VISIBLE_DEVICES 2>/dev/null || true
mkdir -p "\$PIP_CACHE_DIR" "\$HF_HOME"

VERL_REPO=/workspace/rl/verl_0211/verl
export PYTHONPATH="\${VERL_REPO}:\${PYTHONPATH:-}"
cd "\$VERL_REPO"

# Pin ai-dynamo to 1.0.2 (1.1.0 needs vLLM 0.18+; container has vllm017).
DYNAMO_VER=\${DYNAMO_VER:-1.0.2}
pip install --pre --no-deps "ai-dynamo==\${DYNAMO_VER}" "ai-dynamo-runtime==\${DYNAMO_VER}" 2>&1 | tail -3
pip install pyzmq 2>&1 | tail -1
python3 -c "import nixl" 2>/dev/null || pip install nixl 2>&1 | tail -3
python3 -c "import dynamo, dynamo.runtime, dynamo.llm; print('dynamo SDK loaded')"
python3 -c "import verl; print('verl', verl.__version__)"

# Static etcd + nats-server binaries (cached in /workspace/dynamo_bin).
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

export VERL_DYNAMO_LOG_DIR=/workspace/rl/verl_0211/slurm/dynamo_logs/\${SLURM_JOB_ID:-manual}
mkdir -p "\$VERL_DYNAMO_LOG_DIR"
echo "[inner] dynamo subprocess logs -> \$VERL_DYNAMO_LOG_DIR"

# 1-node Ray.
HEAD_IP=\$(hostname -I | awk '{print \$1}')
ray stop --force >/dev/null 2>&1 || true
ray start --head --node-ip-address="\$HEAD_IP" --port=6379
sleep 5
ray status

# Drive training. Use main_dynamo (recipe/dynamo) which is just main_ppo +
# rollout.name=dynamo. A handful of overrides shrink the workload to a
# couple of steps on Qwen2.5-0.5B + gsm8k so the smoke completes < 30 min.
cd /workspace/rl/verl_0211/verl
export PYTHONPATH=/workspace/rl/verl_0211/verl:\${PYTHONPATH:-}
export RAY_ADDRESS="\$HEAD_IP:6379"

set +e
python3 recipe/dynamo/main_dynamo.py \\
    algorithm.adv_estimator=grpo \\
    algorithm.use_kl_in_reward=False \\
    algorithm.kl_ctrl.kl_coef=0.0 \\
    data.train_files=/workspace/datasets/gsm8k/train.parquet \\
    data.val_files=/workspace/datasets/gsm8k/test.parquet \\
    data.train_batch_size=8 \\
    data.max_prompt_length=512 \\
    data.max_response_length=512 \\
    data.filter_overlong_prompts=True \\
    data.truncation=error \\
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \\
    actor_rollout_ref.model.use_remove_padding=True \\
    actor_rollout_ref.actor.use_kl_loss=False \\
    actor_rollout_ref.actor.kl_loss_coef=0.0 \\
    actor_rollout_ref.actor.optim.lr=1e-6 \\
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \\
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \\
    actor_rollout_ref.actor.fsdp_config.param_offload=False \\
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \\
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \\
    actor_rollout_ref.ref.fsdp_config.param_offload=True \\
    actor_rollout_ref.rollout.name=dynamo \\
    actor_rollout_ref.rollout.mode=async \\
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \\
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \\
    actor_rollout_ref.rollout.enforce_eager=True \\
    actor_rollout_ref.rollout.enable_chunked_prefill=False \\
    actor_rollout_ref.rollout.max_model_len=2048 \\
    actor_rollout_ref.rollout.max_num_batched_tokens=2048 \\
    actor_rollout_ref.rollout.n=2 \\
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \\
    trainer.critic_warmup=0 \\
    trainer.logger='["console"]' \\
    trainer.project_name=verl-dynamo-smoke \\
    trainer.experiment_name=routeB_train_smoke_0p5b \\
    trainer.n_gpus_per_node=2 \\
    trainer.nnodes=1 \\
    trainer.save_freq=999 \\
    trainer.test_freq=999 \\
    trainer.val_before_train=False \\
    trainer.total_training_steps=2 \\
    trainer.default_local_dir=/workspace/rl/verl_0211/ckpt/routeB_train_smoke_0p5b
TRAIN_RC=\$?
set -e

ray stop --force >/dev/null 2>&1 || true
echo "[inner] training exit code: \$TRAIN_RC"
exit \$TRAIN_RC
EOF

srun --container-image="$CONTAINER" \
     --container-mounts="${WORKSPACE}:/workspace" \
     bash -c "$inner"
