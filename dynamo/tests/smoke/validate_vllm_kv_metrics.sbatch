#!/bin/bash
#SBATCH --job-name=kvval_vllm
#SBATCH --account=CHANGE_ME
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --exclusive
#SBATCH --cpus-per-task=112
#SBATCH --time=02:30:00
#SBATCH --output=/workspace/logs/kvval_%j.out
#
# Launcher for train_30b_rl_dynamo_kv_metrics.sh, which is an inner training
# command only -- it has no #SBATCH header, no container and no ray bring-up, so
# README's `sbatch train_30b_rl_dynamo_kv_metrics.sh` cannot work. Orchestration
# here is lifted from retool_ab.sbatch, which is the proven path on this cluster.
#
# Usage: CONTAINER=/path/to/retool.sqsh sbatch --export=ALL,STEPS=3 recipe/dynamo/tests/smoke/validate_vllm_kv_metrics.sbatch
set -x
B=/workspace
V=$B/verl_dynamo/verl
: "${CONTAINER:?Set CONTAINER to a prepared ReTool container image (.sqsh)}"
STEPS=${STEPS:-3}

nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
node_1=${nodes[0]}
port=$(( 6500 + SLURM_JOB_ID % 1000 ))
ip_head=$node_1:$port
RUN_DIR=$B/logs/kvval_$SLURM_JOB_ID
mkdir -p "$RUN_DIR"
export ARM=dynamo_vllm

# kv_metrics.sh defaults its data to ${RAY_DATA_HOME}/data/*.parquet, which does
# not exist here: retool's CustomRLHFDataset needs HF-dataset DIRECTORIES, and on
# this cluster they live under $B/datasets/local. Pass them explicitly.
#
# ORDERING IS LOAD-BEARING (fixed 2026-08-31). This block must be emitted BEFORE
# `ray start` in both worker.sh and head.sh. Ray workers are forked by the raylet,
# so they inherit the raylet's environment, not the environment of whatever runs
# after it. With the export placed after `ray start` (the previous layout),
# VERL_DEFER_OPTIMIZER_LOAD never reached the training workers: job 17015763 ran
# with the Adam-offload patch silently disabled and OOMed in step-2 backward, and
# not one "Before/After deferred load_fsdp_optimizer" probe appeared in its log.
# The probe lines are the check that it actually took effect -- their absence is
# the symptom to grep for.
COMMON="export RAY_DATA_HOME=$B
export MODEL_PATH=$B/hf_models/Qwen3-30B-A3B-Base
export TRAIN_FILE=$B/datasets/local/dapo-math-17k
export TEST_FILE=$B/datasets/local/aime-2024
export METRICS_DIR=$RUN_DIR/metrics
export CKPTS_DIR=$RUN_DIR/ckpt
export NNODES=4
export EXP_NAME=kvval_vllm_$SLURM_JOB_ID
export VERL_DEFER_OPTIMIZER_LOAD=1"

{ echo "export ARM=$ARM"; cat "$V/recipe/dynamo/tests/smoke/container_bootstrap.sh"
  echo "cd $V"
  echo "$COMMON"
  echo "ray start --address=$ip_head --block"; } > "$RUN_DIR/worker.sh"

{ echo "export ARM=$ARM"; cat "$V/recipe/dynamo/tests/smoke/container_bootstrap.sh"
  echo "cd $V"
  echo "$COMMON"
  echo "ray start --head --node-ip-address=$node_1 --port=$port"
  echo "sleep 10"; echo "ray status"; echo "export RAY_ADDRESS=$ip_head"
  # prometheus off: kv_metrics.sh sets prometheus.enable=True but omits the
  # disable_log_stats=False that dynamo_agent_loop.py:131 requires alongside it --
  # every other script in this directory pairs them. This run only validates the
  # rollout refactor, so switch the metrics off rather than repair a config we are
  # not testing.
  # fused kernels: kv_metrics.sh predates the later OOM findings and
  # sets neither eager experts nor fused kernels, so step 1 forward would OOM at
  # its 18432-token budget. Override rather than edit the script under test.
  # 2026-08-31: dropped the gpu_memory_utilization=0.35 and
  # ref.log_prob_max_token_len_per_gpu=2304 overrides added on 08-30. 0.35 is below
  # the TP2 weight fraction (30.5/79.11 = 0.386), so vLLM has no memory left for KV
  # blocks and EngineCore fails at startup (job 17014039).
  echo "bash recipe/dynamo/train_30b_rl_dynamo_kv_metrics.sh \
    trainer.total_training_steps=$STEPS \
    actor_rollout_ref.model.use_fused_kernels=True \
    ++actor_rollout_ref.model.fused_kernel_options.impl_backend=torch \
    actor_rollout_ref.rollout.prometheus.enable=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=2304 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=8"
  echo "echo KVVAL_DONE rc=\$?"; } > "$RUN_DIR/head.sh"

# One ray worker per non-head node. This loop replaced a hardcoded single
# `-w "$node_2"` srun, which silently left nodes 2..3 out of the ray cluster.
for ((i = 1; i < ${#nodes[@]}; i++)); do
  srun --overlap --nodes=1 --ntasks=1 -w "${nodes[$i]}" \
    --container-image="$CONTAINER" --container-mounts="$B:$B" --no-container-mount-home \
    bash "$RUN_DIR/worker.sh" &
done
sleep 90

srun --overlap --nodes=1 --ntasks=1 -w "$node_1" \
  --container-image="$CONTAINER" --container-mounts="$B:$B" --no-container-mount-home \
  bash "$RUN_DIR/head.sh"
echo "=== kvval done ==="
