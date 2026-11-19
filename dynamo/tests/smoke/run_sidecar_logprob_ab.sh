#!/usr/bin/env bash
# Sidecar training A/B on one node: the V1 colocate SGLang smoke run once per arm on
# the same config, with the Dynamo arms gated against verl's native SGLang rollout.
#
#   native   rollout.name=sglang -- the baseline; runs first
#   tp4      Dynamo SGLang sidecar + native /generate
#   dpattn4  the same with DP-attention 4 / EP4
#
# An arm is name:rollout:dp_attn[:mode]. mode is how a dynamo arm runs its shards:
# sidecar (default) or worker (dynamo.sglang, aggregated), e.g.
# ARMS="native:sglang:1 tp4:dynamo:1 agg:dynamo:1:worker".
#
# A Dynamo arm passes when its step-mean rollout/actor logprob pearson is at most
# MAX_PEARSON_DROP (0.01) below native, its probs_diff_mean at most
# MAX_DIFF_MEAN_RISE (0.002) above, no logprob fallback fired, and advantages were
# nonzero (otherwise the weights never move and weight sync goes untested).
# Validated on 8xH100 with Qwen3-Coder-30B-A3B-Instruct, FSDP, DAPO-Math-17k:
# native 0.9975 / 0.0046, tp4 0.9982 / 0.0050, dpattn4 0.9976 / 0.0045.
#
# Needs a prepared verl environment with this recipe at recipe/, sglang with native
# gRPC (0.5.20 tested), ai-dynamo with the SGLang sidecar (1.7.0.dev20261110+),
# and etcd and nats-server on PATH. Without TRAIN_FILE it downloads DAPO-Math-17k
# from the Hugging Face Hub and keeps DATA_ROWS distinct problems (fixed seed).
#
#   MODEL_PATH=/models/Qwen3-Coder-30B-A3B-Instruct bash run_sidecar_logprob_ab.sh
#
# Extra arguments are Hydra overrides appended to every arm. Each arm writes
# $OUT/<arm>.stdout, the gated log $OUT/<arm>.log, Dynamo logs and a copy of Ray's
# session logs (they live under /tmp and vanish with a container); $OUT/summary.txt
# has one verdict per arm. The exit status is nonzero if any arm failed.
set -euo pipefail

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the model directory}
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SMOKE=${SMOKE:-$HERE/smoke_dynamo_v1_colocate_sglang.sh}
OUT=${OUT:-${TMPDIR:-/tmp}/sidecar_logprob_ab_$(date +%Y%m%d_%H%M%S)}
ARMS=${ARMS:-"native:sglang:1 tp4:dynamo:1 dpattn4:dynamo:4"}
REF_SMOKE_LOG=${REF_SMOKE_LOG:-$OUT/native.log}
DATA_DIR=${DATA_DIR:-$OUT/data}
DATA_ROWS=${DATA_ROWS:-2000}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
ROLLOUT_TP=${ROLLOUT_TP:-4}

mkdir -p "$OUT"
echo "logs: $OUT"

command -v etcd >/dev/null && command -v nats-server >/dev/null || {
    echo "etcd and nats-server must be on PATH" >&2
    exit 2
}
python3 -c '
import importlib.metadata as md
import dynamo.sglang.sidecar, verl.trainer.ppo.v1.trainer_colocate_async  # noqa: F401
print({p: md.version(p) for p in ("sglang", "ai-dynamo", "ai-dynamo-runtime", "torch")}, flush=True)
' || { echo "environment check failed: see the import error above" >&2; exit 2; }

if [[ -z "${TRAIN_FILE:-}" ]]; then
    # The published DAPO-Math-17k file repeats each problem ~100x.
    if [[ ! -f "$DATA_DIR/train.parquet" ]]; then
        mkdir -p "$DATA_DIR"
        python3 - "$DATA_DIR" "$DATA_ROWS" <<'PY'
import sys
import pandas as pd
from huggingface_hub import hf_hub_download
out, rows = sys.argv[1], int(sys.argv[2])
src = hf_hub_download("BytedTsinghua-SIA/DAPO-Math-17k", "data/dapo-math-17k.parquet", repo_type="dataset")
df = pd.read_parquet(src)
df = df.loc[~df["prompt"].map(lambda m: m[0]["content"]).duplicated()]
df = df.sample(n=min(rows, len(df)), random_state=0).reset_index(drop=True)
df.to_parquet(f"{out}/train.parquet")
df.head(16).to_parquet(f"{out}/test.parquet")
print("dapo rows", len(df), flush=True)
PY
    fi
    TRAIN_FILE=$DATA_DIR/train.parquet
    TEST_FILE=${TEST_FILE:-$DATA_DIR/test.parquet}
fi
TEST_FILE=${TEST_FILE:-$TRAIN_FILE}

export VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:-recipe.dynamo.register,recipe.dynamo.verl_logging_compat}
export VERL_DYNAMO_FE_READY_TIMEOUT=${VERL_DYNAMO_FE_READY_TIMEOUT:-1800}
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF
ulimit -l unlimited 2>/dev/null || true

# Memory settings that fit a 30B-A3B actor next to TP4 engines on 8x80 GB.
# model_dtype=bf16: the FSDP engine otherwise loads fp32 weights on every rank
# and exhausts host memory before sharding.
OVERRIDES=(
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=8192
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=8192
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.rollout.enforce_eager=True
    algorithm.use_kl_in_reward=False
    data.filter_overlong_prompts=True
    "++ray_kwargs.ray_init.runtime_env.env_vars.VERL_DYNAMO_FE_READY_TIMEOUT='${VERL_DYNAMO_FE_READY_TIMEOUT}'"
)

failed=0
for arm in $ARMS; do
    IFS=: read -r name rollout dp mode <<< "$arm"
    if [[ $rollout == dynamo ]]; then
        case ${mode:-sidecar} in
            sidecar) arm_env=(SIDECAR=1) ;;
            worker) arm_env=(SIDECAR=0) ;;
            *) echo "arm $arm: unknown mode $mode" >&2; exit 2 ;;
        esac
        arm_env+=(REF_SMOKE_LOG="$REF_SMOKE_LOG")
    else
        # The baseline only has to be sane; its own numbers set the Dynamo bar.
        arm_env=(SIDECAR=0 MIN_PEARSON="${BASELINE_MIN_PEARSON:-0.9}")
    fi
    echo "=== arm $name (rollout=$rollout DP_ATTN=$dp) $(date -Is)"
    rc=0
    env "${arm_env[@]}" ROLLOUT_NAME="$rollout" DP_ATTN="$dp" GATE=1 \
        ROLLOUT_TP="$ROLLOUT_TP" NGPUS_PER_NODE="$NGPUS_PER_NODE" \
        TRAIN_BATCH="${TRAIN_BATCH:-16}" ROLLOUT_N="${ROLLOUT_N:-8}" TOTAL_STEPS="${TOTAL_STEPS:-3}" \
        GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}" \
        MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}" MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}" \
        MODEL_PATH="$MODEL_PATH" TRAIN_FILE="$TRAIN_FILE" TEST_FILE="$TEST_FILE" \
        EXP_NAME="sidecar-ab-$name" SMOKE_LOG="$OUT/$name.log" VERL_DYNAMO_LOG_DIR="$OUT/$name-dynamo" \
        bash "$SMOKE" "${OVERRIDES[@]}" "$@" > "$OUT/$name.stdout" 2>&1 || rc=$?
    if [[ $rc == 0 ]]; then verdict=PASS; else verdict=FAIL; failed=1; fi
    echo "$name $verdict rc=$rc $(grep -a -o -E '(PASS|FAIL): .*' "$OUT/$name.stdout" | tail -1)" | tee -a "$OUT/summary.txt"
    if [[ -d /tmp/ray/session_latest/logs ]]; then
        cp -rL /tmp/ray/session_latest/logs "$OUT/$name-ray-logs" 2>/dev/null || true
    fi
    # The next arm needs the GPUs back: stop this arm's Ray cluster and its engines.
    if command -v ray >/dev/null; then
        ray stop --force >/dev/null 2>&1 || true
        sleep "${ARM_COOLDOWN_S:-30}"
    fi
done
exit "$failed"
