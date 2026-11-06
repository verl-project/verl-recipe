#!/usr/bin/env bash
set -xeuo pipefail

# Smoke for the Dynamo × SGLang rollout path.
#
# Two stages, selected by STAGE:
#   STAGE=gen   (default) generation-only — brings up etcd + nats + dynamo.sglang
#               + frontend and serves one completion. Proves M1.
#   STAGE=train 2-step GRPO — additionally exercises the weight-sync path
#               (control/update_weights_from_tensor) and sleep/wake via
#               release/resume_memory_occupation. Proves M2. Fails unless the
#               rollout-vs-actor logprob agreement (pearson) reaches MIN_PEARSON.
#
# Env: MODEL_PATH / TRAIN_FILE / TEST_FILE / RAY_DATA_HOME; NNODES / NGPUS_PER_NODE / TP
#      (NGPUS_PER_NODE=2 TP=1 puts two shards behind one frontend, so the KV router
#      and session affinity actually have something to route); DYNAMO_SRC (Dynamo
#      source overlay, see below); MIN_PEARSON (default 0.99); SMOKE_LOG.
#
# Prerequisite the script checks for you: `python -m dynamo.sglang` must be
# importable. Install it per-job from the local wheel (see below) — no dynamo
# patching is needed, base64 on the wire is SGLang's own contract.

project_name=${PROJECT_NAME:-verl-dynamo-sglang}
exp_name=${EXP_NAME:-dynamo-sglang-smoke}
STAGE=${STAGE:-gen}

max_prompt_length=${MAX_PROMPT_LENGTH:-512}
max_response_length=${MAX_RESPONSE_LENGTH:-512}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-1}
TP=${TP:-1}
RAY_DATA_HOME=${RAY_DATA_HOME:-"${HOME}/verl"}
MODEL_PATH=${MODEL_PATH:-"${RAY_DATA_HOME}/models/Qwen2.5-0.5B-Instruct"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/aime-2024.parquet"}
MIN_PEARSON=${MIN_PEARSON:-0.99}
SMOKE_LOG=${SMOKE_LOG:-"${TMPDIR:-/tmp}/smoke_dynamo_sglang_${STAGE}_$$.log"}

# --- Dynamo source overlay --------------------------------------------------
# The prebuilt wheel (ai_dynamo 1.3.0 @ 94accc7389) predates upstream #11640: its
# common/backend/logprobs.py slices sglang's per-chunk logprobs as if they were
# cumulative, so every chunk after the first carries none and the recipe fills
# them in ("[logprobs] engine returned N/M usable logprobs" in the log). The run
# still exits 0 with a full metric set; only rollout_actor_probs_pearson_corr
# (0.07-0.15 instead of >0.999) gives it away -- which is why STAGE=train now
# gates on it. Point DYNAMO_SRC at a checkout carrying the fix and it is put in
# front of the wheel, the same way train_qwen3_30b_sglang.sh does.
DYNAMO_SRC=${DYNAMO_SRC:-}
if [[ -n "${DYNAMO_SRC}" ]]; then
    [[ -d "${DYNAMO_SRC}/components/src/dynamo" ]] || {
        echo "FAIL: DYNAMO_SRC=${DYNAMO_SRC} has no components/src/dynamo" >&2
        exit 1
    }
    export PYTHONPATH="${DYNAMO_SRC}/components/src${PYTHONPATH:+:${PYTHONPATH}}"
fi

# --- preflight -------------------------------------------------------------
# NB: install from the LOCAL wheel, not PyPI — ai_dynamo==1.3.0 is not published
# there (only 1.3.0.post1 / 1.3.1 / dev builds), so `pip install ai_dynamo[sglang]`
# fails with "No matching distribution found".
#
# WARNING: do NOT bake this into the .sqsh. Installing sglang downgrades vLLM's
# guided-decoding stack (llguidance, outlines_core, xgrammar) and two kernel
# packages (tilelang, tokenspeed-mla). `import vllm` still succeeds, so the damage
# only surfaces at runtime. Install it per-job, only for sglang runs.
DYNAMO_WHEEL_DIR=${DYNAMO_WHEEL_DIR:-"/workspace/dynamo_wheels_1.3.0_94accc7389d4"}
# blake3: ai_dynamo runtime dep skipped by --no-deps installs; `import dynamo.sglang`
# alone cannot detect it (lazy handler imports) -- the worker dies at startup instead.
python3 -c "import blake3" 2>/dev/null || pip install blake3 2>&1 | tail -1
python3 -c "import dynamo.sglang" 2>/dev/null || {
    cat >&2 <<MSG
FAIL: 'import dynamo.sglang' failed. Install it for THIS JOB ONLY with:

  pip install "\$DYNAMO_WHEEL_DIR/ai_dynamo-1.3.0-py3-none-any.whl[sglang]" \\
              "\$DYNAMO_WHEEL_DIR/ai_dynamo_runtime-1.3.0-cp310-abi3-manylinux_2_39_x86_64.whl"

(DYNAMO_WHEEL_DIR currently = ${DYNAMO_WHEEL_DIR})
Do not pre-install it into the container image.
MSG
    exit 1
}
# Say which logprobs.py this run is about to trust (wheel vs. overlay).
python3 -c 'import dynamo.common.backend.logprobs as m; print("dynamo.common.backend.logprobs:", m.__file__)'

export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:-recipe.dynamo.register}"

# Argument set mirrors the proven vLLM dynamo runs in
# recipe/dynamo/train_30b_rl_dynamo_kv_metrics.sh rather than being
# derived from first principles — deriving it independently already cost one
# failed run (missing ppo_micro_batch_size_per_gpu). Two things it gets right that
# the naive version did not:
#   * use_dynamic_bsz=True, which is why the proven scripts never need
#     ppo_micro_batch_size{,_per_gpu} at all;
# Backend selection is pure config, the way verl does every other backend:
#   rollout.name=dynamo  +  engine_kwargs.dynamo.engine=sglang
# dynamo_rollout.ServerAdapter dispatches on that key, so there is no sglang-specific
# registry name, trainer yaml or entry point.
#
# Entry point is verl.trainer.main_ppo with trainer.use_v1=False: this smoke exercises
# the legacy V0 agent-loop path (dynamo_trainer preset, DynamoAgentLoopManager), the
# same for both engines. The V1 unified trainer has its own sglang smokes
# (smoke_dynamo_v1_colocate_sglang.sh, smoke_dynamo_v1_separate_sglang.sh) and
# refuses DynamoAgentLoopManager.
COMMON_ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    data.train_files="[\"${TRAIN_FILE}\"]"
    data.val_files="[\"${TEST_FILE}\"]"
    data.return_raw_chat=True
    data.train_batch_size=1
    data.max_prompt_length="${max_prompt_length}"
    data.max_response_length="${max_response_length}"
    data.filter_overlong_prompts=True
    data.truncation=error
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_mini_batch_size=1
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.rollout.name=dynamo
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.rollout.tensor_model_parallel_size="${TP}"
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length))
    actor_rollout_ref.rollout.n=1
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.multi_turn.enable=False
    actor_rollout_ref.rollout.agent.num_workers=1
    # '++' not '+': config/dynamo_trainer.yaml already defines this key, so a
    # bare '+' (append) errors with "An item is already at ...".
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.dynamo.dynamo_agent_loop.DynamoAgentLoopManager
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.engine=sglang
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_engine_data=true
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_session_affinity_ttl_secs=600
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.thunderagent.enabled=false
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_worker_system_metrics=true
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_timeout_s=1800
    ++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.enable_rl=true
    # piecewise cuda-graph compile CUBLAS-fails on this stack even with
    # --disable-cuda-graph (separate switch); the 30B launcher disables it too.
    '++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.extra_args=["--disable-piecewise-cuda-graph"]'
    trainer.logger='["console"]'
    trainer.project_name="${project_name}"
    trainer.experiment_name="${exp_name}"
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}"
    trainer.nnodes="${NNODES}"
    trainer.save_freq=-1
)

if [[ "${STAGE}" == "gen" ]]; then
    python3 -m verl.trainer.main_ppo \
        --config-path ../../recipe/dynamo/config --config-name dynamo_trainer \
        "ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_USE_EXTERNAL_MODULES}'" \
        trainer.use_v1=False \
        "${COMMON_ARGS[@]}" \
        trainer.val_before_train=True \
        trainer.val_only=True \
        trainer.total_training_steps=1 \
        "$@"
    echo "PASS: Dynamo x SGLang generation smoke completed"
else
    python3 -m verl.trainer.main_ppo \
        --config-path ../../recipe/dynamo/config --config-name dynamo_trainer \
        "ray_kwargs.ray_init.runtime_env.env_vars.VERL_USE_EXTERNAL_MODULES='${VERL_USE_EXTERNAL_MODULES}'" \
        trainer.use_v1=False \
        "${COMMON_ARGS[@]}" \
        actor_rollout_ref.rollout.free_cache_engine=True \
        ++actor_rollout_ref.rollout.engine_kwargs.dynamo.free_engine_on_train=true \
        ++actor_rollout_ref.rollout.engine_kwargs.dynamo.sglang.verify_weight_sync=true \
        trainer.val_before_train=False \
        trainer.total_training_steps=2 \
        "$@" 2>&1 | tee "${SMOKE_LOG}"
    # Exit code alone is not enough: both logprob incidents exited 0. Gate on the
    # last reported rollout-vs-actor pearson (native sglang and a correct dynamo
    # stack both give >0.999; the broken wheel gives ~0.1).
    pearson=$(grep -a -o "training/rollout_actor_probs_pearson_corr:[0-9.eE+-]*" "${SMOKE_LOG}" | tail -1 | cut -d: -f2 || true)
    if [[ -z "${pearson}" ]]; then
        echo "FAIL: no training/rollout_actor_probs_pearson_corr in ${SMOKE_LOG} (calculate_log_probs must stay on)" >&2
        exit 1
    fi
    if ! awk -v p="${pearson}" -v m="${MIN_PEARSON}" 'BEGIN { exit !(p + 0 >= m + 0) }'; then
        echo "FAIL: rollout/actor logprob pearson=${pearson} < MIN_PEARSON=${MIN_PEARSON}." >&2
        echo "      The rollout logprobs do not match the trainer's. Known cause: dynamo wheel without" >&2
        echo "      upstream #11640 (set DYNAMO_SRC to a fixed checkout). Log: ${SMOKE_LOG}" >&2
        exit 1
    fi
    echo "PASS: Dynamo x SGLang 2-step training smoke completed (pearson=${pearson} >= ${MIN_PEARSON})"
fi
