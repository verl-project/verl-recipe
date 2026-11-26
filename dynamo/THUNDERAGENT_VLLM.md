# Uni-Agent + Rust ThunderAgent + vLLM

Use [one launcher](run_uniagent_thunderagent_vllm.sh) and
[one training preset](config/dynamo_uniagent_vllm.yaml). The two mode fragments
only describe placement and synchronization:

| Mode | Configuration | Default resources |
| --- | --- | --- |
| `uniagent_mode=colocate` (default) | [colocate.yaml](config/uniagent_mode/colocate.yaml) | 8 shared GPUs, two TP4 engines |
| `uniagent_mode=separate` | [separate.yaml](config/uniagent_mode/separate.yaml) | 8 training + 8 rollout GPUs, two TP4 engines |

The [fully async script](run_uniagent_thunderagent_vllm_fully_async.sh) is a
short alias for the second mode. Both entry points use the vLLM 0.29 environment
below; they do not install or patch dependencies.

The defaults target Qwen3-Coder-30B-A3B on H100 80GB with FSDP2, the
`qwen3_coder` tool parser and a 32K context budget (16K prompt + 16K response). A default run takes two
training steps with two prompts × two responses per batch. This is a smoke
configuration; production batch size, context budget and concurrency depend on
the workload. CUDA graphs are disabled by default to simplify startup.

Allow sufficient **host RAM**, in addition to GPU memory. With FP32 training
parameters and CPU offload, the two-step colocate smoke peaked at **1278 GiB**
of container memory on an 8×H100 node. It completed with a 1800 GiB limit; an
earlier 1100 GiB limit caused a host OOM during training. This is a measurement
for this preset and image, not a general minimum for the model.

On 2026-10-19, both entry points completed two training steps and exited with
code 0 using the pinned environment below and real Uni-Agent Claude Code tasks.
The separate run's second step had a nonzero gradient norm of 1.468 and completed
its NCCL weight update. The colocate run's equal rewards produced zero GRPO
advantages; an earlier colocate run also exercised a nonzero gradient.
The image passed 220 CPU tests. This validates the short training flow, including
subagent program identity and final requests; it is not a throughput benchmark.

## Environment

Prepare a Linux Python 3.12 CUDA environment on every training/rollout node:

- The image's unmodified verl **0.10.0.dev0**, with this repository available as
  `$VERL_SRC/recipe` or its parent directory on `PYTHONPATH`.
- Uni-Agent at the immutable revision in [requirements-uniagent.txt](requirements-uniagent.txt).
  It provides `AgentContext`; older Uni-Agent checkouts cannot load this Gateway.
- vLLM **0.29.0**, PyTorch **2.13.0+cu130**, Transformers **5.12.1**,
  and the matching FlashAttention build from the official uv image.
- Matching Dynamo **1.6.0** Python and Rust wheels (build `b1429b8`) containing the
  **builtin ThunderAgent plugin**. See the [source installation](INSTALL.md#source-install)
  for the pinned revision and build features.
- TransferQueue for V1 training; CuPy matching CUDA for separate-pool NCCL weight
  synchronization. Keep `etcd` and `nats-server` on `PATH`.
- The agent's dependencies, sandbox provider, model checkpoint, training parquet
  and task configuration. Claude Code must be available inside its sandbox.

```bash
# Inside verlai/verl:uv.cu130.dev4, materialize its locked dependencies locally.
cd /workspace/verl
uv sync --offline --frozen --all-packages --extra vllm --extra fsdp
source /workspace/verl/.venv/bin/activate

export VERL_SRC=/workspace/verl
uv pip install --no-deps -r "$VERL_SRC/recipe/dynamo/requirements-uniagent.txt"
```

The amd64 image is pinned as
`verlai/verl@sha256:aaf5271484842d12a40de5e92ea0ce5beb3b478f6c7988e330baebb170d383c2`.
Its [Dockerfile](https://github.com/verl-project/verl/blob/263653e749306287a5032140cce3d7833c1959d0/docker/Dockerfile.uv.cu130)
bakes a uv cache and creates the selected environment at startup. Keep
`/workspace/verl` available when mounting data; a volume over `/workspace` hides
the image's project. After installing Uni-Agent and matching Dynamo dependencies,
run the launcher from this environment rather than syncing the image project again.
verl 0.9.1 requires Transformers `<5.11` and cannot be substituted into this
5.12.1 environment without changing its dependency constraints.

Use the image digest above to reproduce the validated engine stack. Install
matching Dynamo Python/native wheels and verify their imports before training.
Image tags can move; check the installed versions instead of inferring them
from a tag name.

`DYNAMO_SRC` and `UNIAGENT_ROOT` are optional: set them only to load Python code
from prepared source checkouts. Otherwise the launcher uses installed packages.
Avoid an old checkout shadowing the pinned Uni-Agent.

## Workload and launch

Use a current Uni-Agent training parquet with serialized Task Configs in
`extra_info.tools_kwargs.task`. Configure the corresponding task YAML, sandbox
and reward implementation. For Claude Code, set `agent.name: claude_code` and
`agent.enable_subagents: true`. See the [Claude configuration](README.md#subagent-programs-claude-code)
for the optional `SubagentStop` hook and parent/child program mapping.

```bash
export MODEL_PATH=/models/Qwen3-Coder-30B-A3B-Instruct
export TRAIN_FILE=/data/train.parquet
export TASK_CONFIG=/data/task_config_claude_code.yaml

# Resolve the actual training configuration without allocating GPUs.
bash "$VERL_SRC/recipe/dynamo/run_uniagent_thunderagent_vllm.sh" --cfg job --resolve

# Run in an allocated 8-GPU environment.
bash "$VERL_SRC/recipe/dynamo/run_uniagent_thunderagent_vllm.sh"
```

`TEST_FILE` defaults to `TRAIN_FILE`; validation and checkpoint saving are disabled
for this smoke. `OUTPUT_DIR` defaults to `$VERL_SRC/outputs/$RUN_NAME`.

For separate pools, first start Ray on two allocated 8-GPU nodes. Make model,
data, task configuration, packages and recipe available at the same paths:

```bash
# Head node; HEAD_IP is the routable IP of this allocated node.
ray start --head --node-ip-address="$HEAD_IP" --port=6379 \
    --num-gpus=8 --num-cpus=64 --object-store-memory=8589934592 \
    --include-dashboard=false
# Second node:
ray start --address="$HEAD_IP:6379" --num-gpus=8 --num-cpus=64 \
    --object-store-memory=8589934592

# On the head node:
export RAY_CLUSTER_ADDRESS="$HEAD_IP:6379"
export RUN_NAME=uniagent-rustta-separate-smoke
bash "$VERL_SRC/recipe/dynamo/run_uniagent_thunderagent_vllm_fully_async.sh" --cfg job --resolve
bash "$VERL_SRC/recipe/dynamo/run_uniagent_thunderagent_vllm_fully_async.sh"
```

Alternatively use the main launcher with `uniagent_mode=separate`. The script
connects to Ray; it does not allocate nodes. Submit it inside a persistent
server tmux session or a cluster job when disconnecting the laptop.
Use a fresh container/job for each mode. When reusing an allocation, verify
that the previous serving processes have exited: `ray stop` alone left vLLM
TP children holding GPU memory in the sequential smoke experiment.
The example reserves 8 GiB per node for Ray's object store; provide at least
that much `/dev/shm` space. Explicit sizing avoids deriving a large object
store from the host RAM reserved for training offload.

## Defaults and overrides

The shared YAML enables Rust TA, KV routing, subagent programs, prefix caching,
Gateway lifecycle management and V1 TransferQueue integration. It selects the
upstream `uni_agent.framework.task_runner.run_task` and `trajectory_selection: longest`.
The runner's `model_name` follows `actor_rollout_ref.model.path` automatically.
Set `trajectory_selection: all` to train on every agent chain.
The preset also enables strict refit checks: an unreachable weight-update
control channel fails startup before training proceeds.

Only workload paths and cluster address are required from the user. Existing
`RUN_NAME`, `OUTPUT_DIR`, `TOTAL_STEPS`, `GATEWAY_COUNT`, `AGENT_CONCURRENCY` and
`RAY_NUM_CPUS` environment overrides remain supported. All training options can
also be overridden with Hydra, for example:

```bash
bash "$VERL_SRC/recipe/dynamo/run_uniagent_thunderagent_vllm.sh" \
    trainer.total_training_steps=40 \
    data.train_batch_size=8 \
    actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=16
```

The actor mini-batch follows `data.train_batch_size`. Separate mode defaults to
one parameter sync per batch; if changing that interval, keep
`train_batch_size = parameter_sync_step × ppo_mini_batch_size`.

Gateway count, agent session concurrency and rollout engine `max_num_seqs` are
different limits. Raising an engine's capacity alone does not increase the
number of scheduled agent episodes. The YAML owns `max_num_seqs`; `extra_args`
no longer silently overrides it with a second value.

The launcher only prepares environment variables and calls `verl.trainer.main_ppo`.
It forwards workload variables to Ray because the worker resolves their YAML
interpolations. The recipe's scoped logging compatibility module handles verl's
separate trainer logger without a verl source patch. Neither verl nor Uni-Agent
source is changed by these launchers.
