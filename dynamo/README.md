# Dynamo rollout backend for verl

This recipe plugs [NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) into verl
as a first-class **async rollout backend**, alongside the built-in `vllm`,
`sglang`, and `trtllm` backends. Turning it on is a one-line config change
(`actor_rollout_ref.rollout.name=dynamo`); everything Dynamo-specific
(KV-aware routing, disaggregated frontend/worker topology, KV-cache offload)
is driven from `rollout.engine_kwargs.dynamo.*`.

Dynamo owns request routing behind a single logical frontend, so its
**KV-cache-aware router** can raise the prefix-cache hit rate across a rollout
step. Weight updates flow through verl's colocated CUDA-IPC path when trainer
and `dynamo.vllm` workers share GPUs (`colocate_async` / legacy V0), and
through a two-hop checkpoint-engine path (nccl across pools → node-local
CUDA-IPC) for the standalone rollout pool in `separate_async`.

## How it works

The Dynamo backend keeps verl's AgentLoop execution model but replaces the
rollout server with a Dynamo deployment. A single Ray actor per node
(`DynamoHttpServer`) supervises the whole Dynamo stack as subprocesses; it
reserves **no** GPUs of its own — the colocated trainer workers already own
them, and the actor only forwards `CUDA_VISIBLE_DEVICES` into the
`dynamo.vllm` shards.

```
 verl trainer (colocated, owns GPUs)
        │  HTTP chat/completions            control RPC (sleep / wake /
        │  (per-rank ServerAdapter)         update_weights) via Ray + ZMQ
        ▼                                             │
 ┌─────────────────── DynamoHttpServer (Ray actor, 1 / node) ──────────────────┐
 │  supervises + watchdogs subprocesses, forwards CUDA_VISIBLE_DEVICES          │
 │                                                                             │
 │   dynamo.frontend ──► KV-aware router ──► dynamo.vllm × N  (one / DP shard) │
 │        ▲                                        │                            │
 │        │                                   CUDA IPC + ZMQ weight receiver    │
 │   etcd + nats-server (service discovery / messaging)                        │
 │                                                                             │
 │   optional: KV-cache offload (mooncake / flexkv), per-worker metrics sidecar │
 └─────────────────────────────────────────────────────────────────────────────┘
```

Request routing happens inside Dynamo's KV router, **not** in verl's
`GlobalRequestLoadBalancer` — verl only ever talks to the one shared frontend.

### Key files


| File                                                           | Role                                                                                                                                          |
| -------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| `[register.py](register.py)`                                   | Registers `dynamo` in verl's rollout registries; loaded via `VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register`.                               |
| `[main_dynamo.py](main_dynamo.py)`                             | Training entry point — `main_ppo` with the `dynamo_trainer` config.                                                                           |
| `[config/dynamo_trainer.yaml](config/dynamo_trainer.yaml)`     | Hydra config: inherits `ppo_trainer`, sets `rollout.name=dynamo`, `rollout.mode=async`.                                                       |
| `[dynamo_async_server.py](dynamo_async_server.py)`             | `DynamoReplica` / `DynamoHttpServer` — spawns and watchdogs etcd, nats-server, `dynamo.vllm` workers, and `dynamo.frontend`.                  |
| `[dynamo_rollout.py](dynamo_rollout.py)`                       | `ServerAdapter` — per-rank client; HTTP generation via the frontend, control RPCs (sleep/wake/`update_weights`) to the shared per-node actor. |
| `[dynamo_agent_loop.py](dynamo_agent_loop.py)`                 | `DynamoServerManager` / `DynamoLLMServerManager` — talk to the single shared frontend instead of load-balancing across replicas.              |
| `[dynamo_worker_extension.py](dynamo_worker_extension.py)`     | vLLM `worker_extension_cls` that maps each DP shard to a node-global rank so trainer and engine agree on the CUDA-IPC socket path.            |
| `[_dynamo_vllm_with_control.py](_dynamo_vllm_with_control.py)` | Private ZMQ control sidecar that bridges verl's `collective_rpc` into the `dynamo.vllm` subprocess.                                           |
| `[metrics_sidecar.py](metrics_sidecar.py)`                     | Optional per-worker system-status / metrics scraper.                                                                                          |


Enable the backend by pointing verl at the recipe's registration module:

```bash
export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
```



## Configuration

Everything Dynamo-specific lives under
`actor_rollout_ref.rollout.engine_kwargs.dynamo`. All keys are optional; sane
defaults are applied in `DynamoHttpServer`.


| Key                                                    | Values / example                                         | Purpose                                                                                                  |
| ------------------------------------------------------ | -------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `router_mode`                                          | `kv` (default), `round-robin`, `random`, `least-loaded`  | Dynamo request-routing policy; `kv` enables KV-cache-aware routing.                                      |
| `kv_offload_backend`                                   | `none` (default), `mooncake`, `flexkv`                   | Where to offload the KV cache between steps.                                                             |
| `kv_offload_reset_timeout_s`                           | `300`                                                    | Deadline for the external KV store to flush on weight update (fail-closed when offload is on).           |
| `frontend_http_port` / `etcd_port` / `nats_port`       | `0` = auto-assign                                        | Fixed ports if you need them.                                                                            |
| `served_model_name`                                    | falls back to `model_config.local_path`                  | Model name the frontend advertises.                                                                      |
| `request_engine_data` / `request_completion_token_ids` | `true` / `false`                                         | Ask the frontend to return `nvext.engine_data` / raw `completion_token_ids` (token-in/token-out for RL). |
| `return_tokens_as_token_ids`                           | `true` / `false`                                         | Emit token ids instead of detokenized text.                                                              |
| `request_timeout_s`                                    | `600` (default; scripts use `1800`)                      | Per-request timeout.                                                                                     |
| `free_engine_on_train`                                 | mirrors `rollout.free_cache_engine`                      | DEPRECATED as an independent switch: now follows `rollout.free_cache_engine`; an explicit value that contradicts it fails at startup. |
| `enable_worker_system_metrics`                         | `true` / `false`                                         | Expose the per-worker system-status / metrics port (paired with `metrics_sidecar.py`).                   |
| `extra_args`                                           | `["--generation-config","vllm","--stream-interval=100"]` | Extra CLI args forwarded verbatim to `dynamo.vllm`.                                                      |




## Quick start



### 1. Generation-only smoke

Verifies the full Dynamo stack (etcd + nats + workers + frontend) can serve a
completion, no training loop. Passes when the log prints `PASS:`.

```bash
bash recipe/dynamo/scripts/smoke_dynamo_v1.sh          # Qwen2.5-0.5B-Instruct, 1 GPU
```



### 2. One-node training smoke

```bash
export VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register
python3 recipe/dynamo/main_dynamo.py \
    algorithm.adv_estimator=grpo \
    data.train_files=.../gsm8k/train.parquet \
    data.val_files=.../gsm8k/test.parquet \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
    actor_rollout_ref.rollout.name=dynamo \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv \
    trainer.n_gpus_per_node=2 trainer.nnodes=1 \
    trainer.total_training_steps=2
```



### 3. Multi-node 30B RL

`Qwen3-30B-A3B-Base` RL runs (KV router and metrics
variants) live in the top-level scripts:

```bash
sbatch recipe/dynamo/train_30b_rl_dynamo_kv_metrics.sh     # KV router + metrics
```



## KV-aware routing result

The matched comparison below keeps only Dynamo KV routing with
`stream-interval=100` and the native vLLM baseline. Lower `ms/token` is better;
the similar response lengths are a sanity check that generation behavior stayed
comparable.


| Backend                           | ms/token | Mean response length | KV-cache hits / queries | KV-cache hit rate |
| --------------------------------- | -------- | -------------------- | ----------------------- | ----------------- |
| Dynamo KV (`stream-interval=100`) | 1.5956   | 876.1                | 2,248,368 / 2,520,362   | 89.21%            |
| vLLM baseline                     | 1.7220   | 872.3                | 1,860,816 / 2,432,064   | 76.51%            |


Per-step timing per token from the KV-aware router comparison

Dynamo KV is approximately **7.3% faster per generated token** in this
comparison and improves the KV-cache hit rate by **12.70 percentage points**.

## Running a full RL run (V1 trainer — the main path)

The recipe now targets verl's **V1 unified trainer** (see `REQUIRED_VERL.txt`
for the tested pin). Two entry configs ship ready to run:

| Entry config | Mode | Placement |
| --- | --- | --- |
| `--config-name=dynamo_trainer_v1_colocate` | `colocate_async` | trainer + rollout share GPUs; replicas abort + sleep every train step |
| `--config-name=dynamo_trainer_v1_separate` | `separate_async` | standalone rollout pool (`rollout.nnodes × n_gpus_per_node`); weights flow nccl → node-local CUDA-IPC |

Smoke them end-to-end (real training steps, tiny model):

```bash
bash recipe/dynamo/smoke_dynamo_v1_colocate.sh
bash recipe/dynamo/smoke_dynamo_v1_separate.sh   # needs >= 2 GPUs and cupy (nccl backend)
```

Under V1, leave `agent.agent_loop_manager_class` at `null` (the presets do):
verl's `AgentLoopManagerTQ` drives the loop and writes TransferQueue, and
`DynamoLLMServerManager` upgrades the client to a partial-rollout-aware
`FullyAsyncLLMServerClient` (with ThunderAgent affinity when enabled).
Multi-turn agent frameworks (e.g. uni-agent) plug in unchanged via their own
`agent_loop_manager_class` — validated with the uni-agent mem-agent recipe by
only switching `rollout.name=vllm` → `dynamo`.

ThunderAgent under V1: programs are keyed by the caller's stable `request_id`
and auto-finalized per generate call (`thunderagent.auto_finalize`, default
true). Multi-turn callers that want cross-turn affinity set
`auto_finalize: false` and call the client's `finalize_program(session_id)`
from their trajectory-end hook. Program tables are frontend-local and
separate_async re-routes aborted retries across pools, so the client records
**every server that serves a generation attempt** and finalizes each of them
(each finalize RPC bounded by `thunderagent.finalize_timeout_s`, default
60 s); cleanup counts as confirmed only when all ack, and unconfirmed
cleanups count toward `thunderagent.finalize_leak_threshold`.

### Legacy V0 path (compatibility only)

The original PR #110/#126 flow — `--config-name=dynamo_trainer` (which pins
`trainer.use_v1=false`), colocated `hybrid_engine=True`, and the legacy
`DynamoAgentLoopManager` — still works but is a **compatibility path**:
upstream has deprecated the V0 trainer (removal planned in v0.9.0), and the
legacy manager must NOT be combined with `trainer.use_v1=true` (the entry
point fails fast on that combination because it does not write TransferQueue).

```bash
actor_rollout_ref.rollout.mode=async \
actor_rollout_ref.rollout.multi_turn.enable=True \
actor_rollout_ref.rollout.agent.num_workers=64 \
actor_rollout_ref.rollout.agent.agent_loop_config_path=/path/to/agent_config.yaml \
actor_rollout_ref.rollout.agent.default_agent_loop=<your_loop_name> \
+actor_rollout_ref.rollout.agent.agent_loop_manager_class=recipe.dynamo.dynamo_agent_loop.DynamoAgentLoopManager
```

### Recommended `engine_kwargs.dynamo` for RL

Token-in/token-out generation (so the trainer scores the exact tokens the engine
produced), KV-aware routing, and freeing engine memory during the training
phase. See the Configuration table above for every key.

```bash
++actor_rollout_ref.rollout.engine_kwargs.dynamo.router_mode=kv \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_engine_data=true \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_completion_token_ids=true \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.return_tokens_as_token_ids=false \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.request_timeout_s=1800 \
++actor_rollout_ref.rollout.engine_kwargs.dynamo.enable_worker_system_metrics=true \
'++actor_rollout_ref.rollout.engine_kwargs.dynamo.extra_args=["--generation-config","vllm","--stream-interval=100"]'
```



### Optional: KV-metrics sidecar

With `enable_worker_system_metrics=true`, each `dynamo.vllm` worker writes an
`.endpoints` file under `$VERL_DYNAMO_WORKER_METRICS_DIR`. Run the sidecar
alongside training to scrape those `/metrics` endpoints into JSONL (KV-cache hit
rate, queue depth, …):

```bash
python3 recipe/dynamo/metrics_sidecar.py \
    --endpoints-glob "$VERL_DYNAMO_WORKER_METRICS_DIR/*.endpoints" \
    --output /path/to/logs/kv_metrics.jsonl \
    --label dynamo_kv --interval 30 &
```



## ThunderAgent extension

This recipe extends the Dynamo rollout backend End-to-end ThunderAgent from
[verl-recipe PR #110](https://github.com/verl-project/verl-recipe/pull/110)
with program-aware scheduling for multi-turn agent trajectories. It does not
modify core `verl` or Dynamo.

### Required versions

- Dynamo: PyPI `ai-dynamo>=1.3.0.post1` (ships `dynamo.vllm`,
`dynamo.frontend`, and `dynamo.thunderagent_router`, including
[PR #11185](https://github.com/ai-dynamo/dynamo/pull/11185)). Supersedes the
previous source build at commit `59d614641837e593f0567b79d75394aae5f864e0`.
- verl: pinned commit in [REQUIRED_VERL.txt](REQUIRED_VERL.txt).
- `separate_async` additionally needs `cupy-cuda12x` (verl's nccl
checkpoint-engine backend registers only when cupy imports); the V1 trainer
itself needs `TransferQueue` (tested with `0.1.9`).



### Topology

With ThunderAgent enabled, verl launches processes in this order:

```text
etcd -> NATS -> Dynamo vLLM workers -> ThunderAgent router -> frontend
```

The frontend uses round-robin only to reach the registered ThunderAgent
handler. ThunderAgent owns the internal KV router and forwards to the worker
endpoint `<namespace>.backend.generate`. Shutdown reverses the consumer side:
frontend, ThunderAgent, workers, NATS, then etcd.

### Configuration

The default recipe enables ThunderAgent:

```yaml
actor_rollout_ref:
  rollout:
    agent:
      agent_loop_manager_class: recipe.dynamo.dynamo_agent_loop.DynamoAgentLoopManager
    engine_kwargs:
      dynamo:
        thunderagent:
          enabled: true
          router_block_size: 16
```

`router_block_size` is applied to both vLLM and ThunderAgent. Pass scheduler CLI options without hard-coding them in the recipe:

```yaml
thunderagent:
  enabled: true
  router_block_size: 16
  extra_args:
    - --pause-threshold
    - "0.95"
```

Optional finalization controls are `finalize_max_attempts` (default `3`) and `finalize_retry_delay_s` (default `0.1`). Set `thunderagent.enabled=false` for
the PR #110 KV-router baseline.

### Run

From a verl checkout containing this repository at `recipe/`:

```bash
VERL_USE_EXTERNAL_MODULES=recipe.dynamo.register \
python -m recipe.dynamo.main_dynamo \
  actor_rollout_ref.model.path=/path/to/model \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
```

Supply the remaining dataset, trainer, and resource overrides required by the standard verl PPO configuration.

### UniAgent variants

`[run_uniagent_variant.sh](run_uniagent_variant.sh)` is a concise UniAgent training example. Select one rollout path with `VARIANT`:

- `ta` (default): Dynamo with ThunderAgent enabled.
- `dynamo`: the native Dynamo KV-router baseline with ThunderAgent disabled.
- `global`: the native verl vLLM rollout baseline, bypassing Dynamo. The
historical name does not mean that this script explicitly configures a
GlobalLoadBalancer.

Run it from the verl root:

```bash
VARIANT=ta RAY_DATA_HOME=/path/to/verl-data \
bash recipe/dynamo/run_uniagent_variant.sh

VARIANT=dynamo RAY_DATA_HOME=/path/to/verl-data \
bash recipe/dynamo/run_uniagent_variant.sh

VARIANT=global RAY_DATA_HOME=/path/to/verl-data \
bash recipe/dynamo/run_uniagent_variant.sh
```

### End-to-end ThunderAgent result

The ThunderAgent comparison used the following matched setup:

- Uni-Agent × verl synchronous end-to-end GRPO, including rollout,
reward/advantage, old log-probability, actor update, and weight
synchronization.
- Training and inference colocated and time-multiplexed on 8 × NVIDIA H20-3e
GPUs (140.4 GiB/GPU), with two TP4 replicas.
- Qwen3-Coder-30B-A3B-Instruct.
- The only backend change was verl Global LB versus Dynamo ThunderAgent.

Here, `rollout.mode=async` only enables concurrent agent requests within the
rollout phase; the RL algorithm remains synchronous and does not use a
`staleness_threshold`. Rollout throughput is generated response tokens divided
by rollout wall time. Full-step throughput also includes reward/advantage,
old-log-probability, actor-update, and weight-synchronization time.

<img src="assets/thunderagent_full_step_throughput.png" alt="ThunderAgent and Global LB complete synchronous RL-step throughput" width="800">

<img src="assets/thunderagent_rollout_throughput.png" alt="ThunderAgent and Global LB rollout-phase generated-token throughput" width="800">

At concurrency 64–256, ThunderAgent and Global LB are near parity. At
concurrency 384, ThunderAgent reaches **1.94× rollout-phase speedup** and
**1.39× observed full-step speedup**; at concurrency 512, the speedups reach
**2.40×** and **1.60×**, respectively.

<img src="assets/thunderagent_speedup.png" alt="ThunderAgent speedup over Global LB" width="800">
