# VeRL Tinker Server

`verl_tinker` runs a local FastAPI/Ray Serve HTTP server that exposes
Tinker-compatible endpoints backed by VeRL actors. Use it when you want to run
Tinker or Tinker Cookbook client code against your own VeRL workers and GPU
capacity.

The server owns one global model, session, and sampler state. It is intended for
one active training client at a time, not isolated multi-client sessions.

## News

- 2026-07-03: Released `verl-tinker` 🚀

## Install

Run this command from the root of `verl-recipe` in a Python 3.12 environment.
The combined server environment includes VeXact, which currently requires
Python `>=3.12,<3.13`.

```bash
# Install the full server runtime for this recipe.
./install_verl.sh --recipe verl_tinker
```

`install_verl.sh` reads `verl_tinker/REQUIRED_VERL.txt` and runs the recorded
install command. For this recipe, that command is `uv sync --project
verl_tinker --python python3.12`, so the server package and its runtime
dependencies are resolved together. Use `--show` to inspect it first:

```bash
./install_verl.sh --recipe verl_tinker --show
```

The `verl_tinker` environment includes the pinned core `verl` commit with the
`vllm` extra, plus VeOmni, VeXact from its GitHub source, Ray Serve, FastAPI,
Tinker, and the server package itself. Server-side Tinker is currently pinned
to `0.19.0`: newer Tinker releases changed the forward request types from
Pydantic models to stdlib dataclasses, which are not compatible with Ray
Serve's FastAPI ingress serialization path. Transformers is pinned to `5.9.0`,
which is the API level VeOmni expects for the VeRL engine registration path.
The quick-start rollout configs use vLLM, so keeping these dependencies in one
`uv` solve avoids pip reinstalling incompatible versions later. You can use
`actor.yaml` without vLLM at runtime, but vLLM is still installed as part of
this recipe environment.

## Choose A Config

Quick-start configs are under `verl_tinker/configs/quick_start/`.

- `actor.yaml`: actor only. Use this for SFT or optimizer-only workflows that do
  not call `asample`.
- `actor_rollout.yaml`: actor + rollout. Use this when clients call `asample`.
- `actor_rollout_ref.yaml`: actor + rollout + reference model. Use this for
  KL-enabled RL workloads.

Common environment overrides:

```bash
export TINKER_SERVER_MODEL=Qwen/Qwen3-1.7B
export TINKER_SERVER_N_GPUS_PER_NODE=8
export TINKER_SERVER_PORT=8000
export TINKER_CHECKPOINT_DIR=/tmp/tinker-checkpoints
```

For an existing Ray cluster, set `RAY_ADDRESS`; otherwise the server starts a
local Ray runtime.

## Validate a Config

Validate a config and print the final processed YAML without starting Ray or
requesting GPU resources:

```bash
cd verl_tinker
python -m verl_tinker.config_utils --config configs/quick_start/actor.yaml
```

The command exits successfully after printing the config, or raises an error
when loading, interpolation, processing, or validation fails.

## Start The Server

Start the server from inside the `verl_tinker` recipe directory. The config
paths below are relative to that directory:

```bash
cd verl_tinker
```

Actor + rollout:

```bash
python -m verl_tinker.start --config configs/quick_start/actor_rollout.yaml
```

Actor only:

```bash
python -m verl_tinker.start --config configs/quick_start/actor.yaml
```

Actor + rollout + reference:

```bash
python -m verl_tinker.start --config configs/quick_start/actor_rollout_ref.yaml
```

Wait for readiness:

```bash
curl http://127.0.0.1:8000/api/v1/healthz
```

Expected ready response:

```json
{"status": "ready"}
```

Stop the server:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/shutdown
```

## Torch Profiler

`verl_tinker` reuses verl's worker profiler. To capture a Chrome trace for
actor `forward_backward`, set `global_profiler.tool=torch`, choose the Tinker
training request numbers to profile, and configure the actor profiler contents:

```yaml
global_profiler:
  tool: torch
  steps: [1]
  save_path: outputs/profile

actor_rollout_ref:
  actor:
    profiler:
      tool_config:
        torch:
          contents: [cuda, cpu]
```

If `global_profiler.tool` is set and `actor_rollout_ref.actor.profiler.enable`
is not explicitly configured, the server enables actor profiling during config
processing. Trace files are written under `global_profiler.save_path`.

## Use A Tinker Client

Point the client process at the server:

```bash
export TINKER_BASE_URL=http://127.0.0.1:8000/
export TINKER_API_KEY=tml-verl-tinker-local
```

The API key is a compatibility value; the current server accepts keys that start
with `tml`.

Then run normal Tinker or Tinker Cookbook code. For example:

```python
import os
import tinker

os.environ["TINKER_BASE_URL"] = "http://127.0.0.1:8000/"
os.environ["TINKER_API_KEY"] = "tml-verl-tinker-local"

client = tinker.ServiceClient(base_url=os.environ["TINKER_BASE_URL"])
```

## Run The Included Client Examples

The examples intentionally use a separate client environment, because real
Tinker clients do not need the server package or core VeRL installed. Open a
separate shell, change into the client examples directory, and run the client
commands from there:

```bash
cd verl_tinker/client_examples
uv sync

uv run run_single_test.py \
  --base-url http://127.0.0.1:8000/ \
  --test-name sft_tulu3
```

Available `--test-name` values are documented in
`verl_tinker/client_examples/README.md`.

## API Surface

Compatibility and lifecycle:

- `GET /api/v1/healthz`
- `POST /api/v1/shutdown`
- `GET|POST /api/v1/get_server_capabilities`
- `POST /api/v1/client/config`
- `POST /api/v1/telemetry`

Session/model metadata:

- `POST /api/v1/get_info`
- `POST /api/v1/create_session`
- `GET /api/v1/sessions/{session_id}`
- `POST /api/v1/sessions`
- `POST /api/v1/create_model`
- `POST /api/v1/create_sampling_session`
- `GET /api/v1/samplers/{sampler_id}`
- `POST /api/v1/session_heartbeat`

Training, sampling, and checkpoint operations:

- `POST /api/v1/forward`
- `POST /api/v1/forward_backward`
- `POST /api/v1/optim_step`
- `POST /api/v1/save_weights_for_sampler`
- `POST /api/v1/save_weights`
- `POST /api/v1/load_weights`
- `POST /api/v1/weights_info`
- `POST /api/v1/export_model`
- `POST /api/v1/asample`
- `POST /api/v1/retrieve_future`

Most long-running operations return a `request_id`. Poll
`/api/v1/retrieve_future` with that ID until the result is available.

## Limitations

- Critic and reward model serving are not supported.
- Frozen teacher models may be configured through VERL's `distillation`
  section. They run on dedicated GPUs and support sampling and prompt
  log-probabilities, but not training or checkpoint mutation.
- Multi-teacher deployments can set `distillation.dedicated_resource_pools=true`
  to strictly pack each teacher into its own Ray placement group. Teachers are
  allocated largest-first; each teacher must fit on a single node.
- LoRA training is not supported. Some LoRA-shaped metadata is returned for
  Tinker Cookbook compatibility, but the backend trains full model weights.
- Multiple clients are not isolated: they share one model state, optimizer
  state, and sampler state.

 ## Acknowledgement

Developed by the ByteDance AML/Seed Team.

Contributors: [Tianle Zhong](https://luosuu.github.io/)\*,
[Huaye Zeng](https://www.wyett-zeng.com/)\*, [Xibin Wu](https://github.com/wuxibin89/), Siping Tao, [Peng Wu](https://www.linkedin.com/in/pengwu22/), [Yifan Pi](https://www.linkedin.com/in/yifan-pi-519971187/), and [Xiao Yu](https://www.linkedin.com/in/fishx/).
