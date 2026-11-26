#!/usr/bin/env bash
# Uni-Agent + Rust ThunderAgent. Defaults live in config/dynamo_uniagent_vllm.yaml.
set -euo pipefail
: "${VERL_SRC:?Set VERL_SRC to the prepared verl checkout (with this repo at recipe/ or on PYTHONPATH)}"
: "${MODEL_PATH:?Set MODEL_PATH to the local model directory}"
: "${TRAIN_FILE:?Set TRAIN_FILE to the workload training parquet}"
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runner_args=("actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=${AGENT_RUNNER_FQN:-uni_agent.framework.task_runner.run_task}")
if [[ ${AGENT_RUNNER_FQN:-uni_agent.framework.task_runner.run_task} == uni_agent.framework.task_runner.run_task ]]; then
    : "${TASK_CONFIG:?Set TASK_CONFIG to your Uni-Agent task YAML}"
else
    runner_args+=(
        "~actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs"
        "+actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs={}"
    )
fi

export RUN_NAME=${RUN_NAME:-uniagent-rustta-smoke}
export OUTPUT_DIR=${OUTPUT_DIR:-$VERL_SRC/outputs/$RUN_NAME}
export PYTHONPATH="${DYNAMO_SRC:+$DYNAMO_SRC/components/src:}$VERL_SRC${UNIAGENT_ROOT:+:$UNIAGENT_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
for module in recipe.dynamo.register recipe.dynamo.verl_logging_compat; do
    case ",${VERL_USE_EXTERNAL_MODULES:-}," in
        *,"$module",*) ;;
        *) export VERL_USE_EXTERNAL_MODULES="${VERL_USE_EXTERNAL_MODULES:+$VERL_USE_EXTERNAL_MODULES,}$module" ;;
    esac
done
export VERL_DYNAMO_LOG_DIR="$OUTPUT_DIR/dynamo"
export VERL_DYNAMO_FE_READY_TIMEOUT=${VERL_DYNAMO_FE_READY_TIMEOUT:-1800}
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTORCH_ALLOC_CONF PYTORCH_CUDA_ALLOC_CONF RAY_ADDRESS ROCR_VISIBLE_DEVICES

# main_ppo passes runtime_env to Ray before resolving the full config. Keep
# these values literal; oc.env interpolations here would reach Ray unresolved.
runtime_args=()
for key in VERL_USE_EXTERNAL_MODULES PYTHONPATH PATH VERL_DYNAMO_LOG_DIR VERL_DYNAMO_FE_READY_TIMEOUT \
    TOKENIZERS_PARALLELISM OMP_NUM_THREADS OPENBLAS_NUM_THREADS \
    MODEL_PATH TRAIN_FILE TEST_FILE TASK_CONFIG RUN_NAME OUTPUT_DIR TOTAL_STEPS \
    GATEWAY_COUNT AGENT_CONCURRENCY AGENT_RUNNER_FQN RAY_CLUSTER_ADDRESS RAY_NUM_CPUS; do
    if [[ -n ${!key:-} ]]; then
        runtime_args+=("++ray_kwargs.ray_init.runtime_env.env_vars.$key='${!key}'")
    fi
done
cd "$VERL_SRC"
exec python3 -m verl.trainer.main_ppo \
    --config-path "$SCRIPT_DIR/config" --config-name dynamo_uniagent_vllm \
    "${runtime_args[@]}" "${runner_args[@]}" "$@"
