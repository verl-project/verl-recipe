#!/usr/bin/env bash
set -xeuo pipefail

# V1 colocate_async smoke on the SGLANG engine: gate/abort/sleep-wake/weight
# injection + real training steps on a tiny model.
#
# Environment prerequisites (see README "Required versions"):
#   pip install "ai-dynamo[sglang]"  + the #11640 incremental-logprobs backport
#   PYTORCH_CUDA_ALLOC_CONF must be unset (torch_memory_saver).
#
# SIDECAR=1 runs each shard as stock SGLang behind dynamo.sglang.sidecar and sends
# rollouts through the frontend's native /generate (sglang.launch_mode=sidecar,
# generate_api=generate); it needs ai-dynamo with the sidecar (1.7.0.dev20261110+).
# DP_ATTN=N adds --dp-size N --enable-dp-attention --ep-size N to every engine
# (rollout TP must be a multiple of N).
# PD_PREFILL=K disaggregates the pool (worker mode only): the first K shards run
# SGLang prefill, the rest decode, KV moves over NIXL and the frontend's prefill
# router splits every request. Weight refit and sleep/wake go to both roles.
#
# ROLLOUT_NAME=sglang runs the same config on verl's native SGLang rollout instead
# of Dynamo: the baseline the Dynamo numbers are judged against.
#
# GATE=1 tees the run to SMOKE_LOG and fails it, even on exit 0, unless:
#   * the mean training/rollout_actor_probs_pearson_corr over all steps >= MIN_PEARSON, and
#   * the mean training/rollout_probs_diff_mean over all steps <= MAX_PROBS_DIFF_MEAN, and
#   * no logprob fallback fired (text re-encode, padded or unmappable logprobs), and
#   * with SIDECAR=1, at least one sidecar shard was started.
# With REF_SMOKE_LOG (a gated baseline run's SMOKE_LOG) both thresholds become
# relative: pearson >= baseline mean - MAX_PEARSON_DROP and
# diff_mean <= baseline mean + MAX_DIFF_MEAN_RISE. An MoE model without routing
# replay sits well below a dense model's pearson, so an absolute bar does not transfer.
ROLLOUT_NAME=${ROLLOUT_NAME:-dynamo}
SIDECAR=${SIDECAR:-0}
DP_ATTN=${DP_ATTN:-1}
PD_PREFILL=${PD_PREFILL:-0}
GATE=${GATE:-0}
MIN_PEARSON=${MIN_PEARSON:-0.99}
MAX_PROBS_DIFF_MEAN=${MAX_PROBS_DIFF_MEAN:-0.02}
REF_SMOKE_LOG=${REF_SMOKE_LOG:-}
MAX_PEARSON_DROP=${MAX_PEARSON_DROP:-0.01}
MAX_DIFF_MEAN_RISE=${MAX_DIFF_MEAN_RISE:-0.002}

if [[ "${ROLLOUT_NAME}" != "dynamo" ]]; then
  [[ "${SIDECAR}" != "1" && "${DP_ATTN}" == "1" && "${PD_PREFILL}" == "0" ]] || {
    echo "SIDECAR=1, DP_ATTN and PD_PREFILL need ROLLOUT_NAME=dynamo" >&2
    exit 2
  }
  python3 -c "import sglang" || { echo "sglang not importable" >&2; exit 2; }
elif [[ "${PD_PREFILL}" != "0" && "${SIDECAR}" == "1" ]]; then
  echo "PD_PREFILL needs the dynamo.sglang worker (SIDECAR=0)" >&2
  exit 2
elif [[ "${SIDECAR}" == "1" ]]; then
  python3 -c "import dynamo.sglang.sidecar" || {
    echo "dynamo.sglang.sidecar not importable — install ai-dynamo 1.7.0.dev20261110 or newer" >&2
    exit 2
  }
else
  python3 -c "import dynamo.sglang" || {
    echo "dynamo.sglang not importable — install ai-dynamo[sglang] first" >&2
    exit 2
  }
fi
unset PYTORCH_CUDA_ALLOC_CONF || true

project_name=${PROJECT_NAME:-verl-dynamo}
exp_name=${EXP_NAME:-dynamo-v1-sglang-smoke}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
TOTAL_STEPS=${TOTAL_STEPS:-2}
TRAIN_BATCH=${TRAIN_BATCH:-2}
ROLLOUT_N=${ROLLOUT_N:-1}
ROLLOUT_TP=${ROLLOUT_TP:-1}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.5}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
SMOKE_LOG=${SMOKE_LOG:-"${TMPDIR:-/tmp}/smoke_dynamo_v1_colocate_sglang_$$.log"}

export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe.dynamo.register}"
if [[ "${ROLLOUT_NAME}" != "dynamo" ]]; then
  # recipe.dynamo.register swaps verl's LLMServerManager for the Dynamo one, which
  # launches Dynamo whatever rollout.name says. The baseline must not load it.
  VERL_USE_EXTERNAL_MODULES=$(tr ',' '\n' <<< "${VERL_USE_EXTERNAL_MODULES}" |
    { grep -v -x -F recipe.dynamo.register || true; } | paste -s -d, -)
  VERL_USE_EXTERNAL_MODULES=${VERL_USE_EXTERNAL_MODULES:-recipe.dynamo.verl_logging_compat}
fi
DYNAMO_CONFIG_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../config" && pwd)

ENGINE_ARGS=()
if [[ "${ROLLOUT_NAME}" != "dynamo" ]]; then
  ENGINE_ARGS+=(actor_rollout_ref.rollout.name="${ROLLOUT_NAME}")
fi
if [[ "${SIDECAR}" == "1" ]]; then
  ENGINE_ARGS+=(
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.launch_mode=sidecar
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.generate_api=generate
    # A port picked before a large model finishes loading is often taken by then.
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.stable_kv_event_ports=true
  )
fi
if [[ "${PD_PREFILL}" != "0" ]]; then
  ENGINE_ARGS+=(++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.disaggregation.prefill_shards="${PD_PREFILL}")
fi
if [[ "${DP_ATTN}" != "1" ]]; then
  ENGINE_ARGS+=(
    "++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.extra_args=[\"--dp-size\",\"${DP_ATTN}\",\"--enable-dp-attention\",\"--ep-size\",\"${DP_ATTN}\"]"
  )
fi

run() {
  python3 -m verl.trainer.main_ppo \
    --config-path "${DYNAMO_CONFIG_DIR}" \
    --config-name=dynamo_trainer_v1_colocate_sglang \
    "ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_USE_EXTERNAL_MODULES}'" \
    algorithm.adv_estimator=grpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.train_batch_size="${TRAIN_BATCH}" \
    data.val_batch_size=1 \
    data.train_max_samples=$((TRAIN_BATCH * TOTAL_STEPS)) \
    data.val_max_samples=1 \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${TRAIN_BATCH}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.multi_turn.enable=False \
    trainer.logger=console \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.total_training_steps="${TOTAL_STEPS}" \
    trainer.test_freq=-1 \
    trainer.save_freq=-1 \
    ${ENGINE_ARGS[@]+"${ENGINE_ARGS[@]}"} \
    "$@"
}

if [[ "${GATE}" != "1" ]]; then
  run "$@"
  exit
fi

run "$@" 2>&1 | tee "${SMOKE_LOG}"

# Exit 0 is not enough: every logprob incident so far exited 0 with a full metric set.
set +x
fail() {
  echo "FAIL: $* (log: ${SMOKE_LOG})" >&2
  exit 1
}
# Mean of one metric over every step line in a log; empty when it never appears.
# Single steps swing by ~0.01 in pearson on a 30B MoE, so gates compare means.
mean_metric() {
  grep -a -o "$2:[-0-9.eE+naif]*" "$1" | cut -d: -f2 |
    awk '{ s += $1; n++ } END { if (n) printf "%.6f\n", s / n }' || true
}
pearson=$(mean_metric "${SMOKE_LOG}" training/rollout_actor_probs_pearson_corr)
diff_mean=$(mean_metric "${SMOKE_LOG}" training/rollout_probs_diff_mean)
[[ -n "${pearson}" ]] || fail "no training/rollout_actor_probs_pearson_corr in the log"
[[ -n "${diff_mean}" ]] || fail "no training/rollout_probs_diff_mean in the log"
if [[ -n "${REF_SMOKE_LOG}" ]]; then
  ref_pearson=$(mean_metric "${REF_SMOKE_LOG}" training/rollout_actor_probs_pearson_corr)
  ref_diff_mean=$(mean_metric "${REF_SMOKE_LOG}" training/rollout_probs_diff_mean)
  [[ -n "${ref_pearson}" && -n "${ref_diff_mean}" ]] || fail "baseline ${REF_SMOKE_LOG} has no logprob metrics"
  MIN_PEARSON=$(awk -v r="${ref_pearson}" -v d="${MAX_PEARSON_DROP}" 'BEGIN { printf "%.6f", r - d }')
  MAX_PROBS_DIFF_MEAN=$(awk -v r="${ref_diff_mean}" -v d="${MAX_DIFF_MEAN_RISE}" 'BEGIN { printf "%.6f", r + d }')
  echo "baseline: pearson=${ref_pearson} probs_diff_mean=${ref_diff_mean} (${REF_SMOKE_LOG})"
fi
awk -v p="${pearson}" -v m="${MIN_PEARSON}" 'BEGIN { exit !(p + 0 >= m + 0) }' ||
  fail "rollout/actor logprob pearson (mean over steps)=${pearson} < ${MIN_PEARSON}"
awk -v d="${diff_mean}" -v m="${MAX_PROBS_DIFF_MEAN}" 'BEGIN { exit !(d + 0 <= m + 0) }' ||
  fail "rollout/actor probs diff mean (mean over steps)=${diff_mean} > ${MAX_PROBS_DIFF_MEAN}"
# All-zero advantages mean no gradient: the weights never change, so the weight
# sync is never tested (every GRPO group all-correct or all-wrong on easy data).
nonzero_adv=$(grep -a -o -E "critic/advantages/(max|min):[-0-9.eE+naif]*" "${SMOKE_LOG}" |
  cut -d: -f2 | awk '$1 + 0 != 0 { n++ } END { print n + 0 }')
[[ "${nonzero_adv}" != "0" ]] || fail "advantages are 0 at every step: no gradient, use harder data or a larger ROLLOUT_N"
# Fallbacks that keep a run going on data the engine did not produce.
for pattern in "re-encoding the response text" "usable logprobs" "Cannot map logprob token"; do
  hits=$(grep -a -c -F "${pattern}" "${SMOKE_LOG}" || true)
  [[ "${hits}" == "0" ]] || fail "logprob fallback fired ${hits}x: '${pattern}'"
done
if [[ "${SIDECAR}" == "1" ]]; then
  grep -a -q -F "starting sglang sidecar shard" "${SMOKE_LOG}" || fail "SIDECAR=1 but no sidecar shard was started"
fi
if [[ "${PD_PREFILL}" != "0" ]]; then
  for role in prefill decode; do
    grep -a -q -F -e "--disaggregation-mode ${role}" "${SMOKE_LOG}" || fail "PD_PREFILL=${PD_PREFILL} but no ${role} shard was started"
  done
fi
echo "PASS: V1 colocate SGLang smoke (rollout=${ROLLOUT_NAME}, sidecar=${SIDECAR}, dp_attn=${DP_ATTN}, pd_prefill=${PD_PREFILL}, pearson=${pearson} >= ${MIN_PEARSON}, probs_diff_mean=${diff_mean} <= ${MAX_PROBS_DIFF_MEAN})"
