#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
#SBATCH --account=general_sa
#SBATCH --partition=batch
#SBATCH --job-name=general_sa-dynamo.m2_e2e
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --output=/lustre/fsw/general_sa/sopyang/dynamo_smoke_logs/m2_e2e_%j.out
#SBATCH --error=/lustre/fsw/general_sa/sopyang/dynamo_smoke_logs/m2_e2e_%j.err
#
# m2 end-to-end training smoke. Single node, 8 GPUs, ~3 training steps.
# Uses vllm017_latest.sqsh + pip-installs verl source and ai-dynamo on top
# (so the container is forward-compatible with m3 dynamo runtime work).

set -x

CONTAINER="/lustre/fsw/general_sa/sopyang/images/vllm017_latest.sqsh"
HOME_DIR="/lustre/fsw/general_sa/sopyang"
VERL_REPO="${HOME_DIR}/rl/verl_0211/verl"
MOUNTS="${HOME_DIR}:${HOME_DIR}"

mkdir -p "${HOME_DIR}/dynamo_smoke_logs"
JOBLOG="${HOME_DIR}/dynamo_smoke_logs/m2_e2e_${SLURM_JOB_ID}.train.log"
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} started" | tee -a "$JOBLOG"

echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=( $nodes )
node_1=${nodes_array[0]}
ip=$node_1
port=6379
ip_head=$ip:$port
export ip_head

echo "Ray head at $ip_head" | tee -a "$JOBLOG"

# Single-node Ray cluster. We do head + train in one srun --overlap pair so the
# Ray daemon stays alive while the training driver runs.
srun --nodes=1 --ntasks=1 -w "$node_1" \
    --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    bash -c "
        set -x
        # verl source (editable install, --no-deps to keep container's vllm/torch).
        pip install --no-deps --quiet -e ${VERL_REPO}

        # Forward-compat: install ai-dynamo so m3+ has the SDK available.
        # --no-deps avoids clobbering the container's pinned vllm.
        pip install --prerelease=allow --no-deps --quiet ai-dynamo || \
            echo '[warn] ai-dynamo install failed; m2 does not require it, continuing'

        # verl recipes need a couple extras.
        pip install --quiet cupy-cuda12x pyzmq

        export PYTHONPATH=${VERL_REPO}:\${PYTHONPATH:-}
        ray start --head --node-ip-address=$ip --port=$port --block
    " &>> "$JOBLOG" &
sleep 20

# Driver: connect to Ray head, run the training script.
read -r -d '' cmd <<EOF
set -x
ray status
export PYTHONPATH=${VERL_REPO}:\${PYTHONPATH:-}
export HF_HOME=${HOME_DIR}/.hf_cache
export TOKENIZERS_PARALLELISM=true
cd ${VERL_REPO}
bash recipe/dynamo/run_dynamo_e2e_smoke.sh
EOF

srun --overlap --nodes=1 --ntasks=1 -w "$node_1" \
    --container-image="$CONTAINER" --container-mounts="$MOUNTS" \
    bash -c "$cmd" &>> "$JOBLOG"

rc=$?
echo "$(date '+%Y-%m-%d %H:%M:%S') Job ${SLURM_JOB_ID} finished rc=$rc" | tee -a "$JOBLOG"
exit $rc
