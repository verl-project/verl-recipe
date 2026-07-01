# VeRL Tinker Server

This package runs a local FastAPI/Ray Serve HTTP server that exposes
Tinker-compatible endpoints backed by VeRL actors. It is intended for users who
have their own GPU capacity and want to run Tinker client code against a local
or self-managed VeRL deployment.

This server is designed for a single active training client. Unlike the public
Tinker service from Thinking Machines Lab, it does not provide isolated sessions
or independent model state per client. All compatible session and model APIs
operate on the same underlying VeRL actor state.

The README is scoped to `verl_recipes.verl_tinker_server` only.

## Installation

Install ModelChef normally and include `vllm` when you need rollout/sampling
support:

```bash
bash setup_venv.sh gpu dev vllm
```

Run Python commands from the ModelChef virtual environment:

```bash
source .venv/bin/activate
```

## Configuration

The server starts from a YAML config that follows the usual VeRL config shape,
with an additional top-level `server` section.

Quick-start configs launch a Qwen3-1.7B base model by default. Override it with
`TINKER_SERVER_MODEL` or by editing `actor_rollout_ref.model.path`.

- `config/quick_start/actor_rollout.yaml`: actor + vLLM rollout. Use this when
    Tinker code needs `asample`.
- `config/quick_start/actor_rollout_ref.yaml`: actor + vLLM rollout + reference
    model. Use this when KL-enabled loss requires reference log probabilities.
- `config/quick_start/actor.yaml`: actor only. Rollout is disabled, so sampling
    is unavailable; use this for forward/backward and optimizer-only workflows.

## Start the Server

From the repository root:

```bash
python -m verl_recipes.verl_tinker_server.start \
  --config verl-recipes/src/verl_recipes/verl_tinker_server/config/quick_start/actor_rollout.yaml
```

Use the SFT config for actor-only mode:

```bash
python -m verl_recipes.verl_tinker_server.start \
  --config verl-recipes/src/verl_recipes/verl_tinker_server/config/quick_start/actor.yaml
```

Use the actor-rollout-ref config when the workload enables KL loss:

```bash
python -m verl_recipes.verl_tinker_server.start \
  --config verl-recipes/src/verl_recipes/verl_tinker_server/config/quick_start/actor_rollout_ref.yaml
```

The process initializes Ray, deploys a single Ray Serve replica, and loads the
VeRL backend asynchronously. Check readiness with:

```bash
curl http://127.0.0.1:8000/api/v1/healthz
```

## Tinker Client Setup

Point Tinker clients at the server:

```python
import os

os.environ["TINKER_BASE_URL"] = "http://127.0.0.1:8000/"
os.environ["TINKER_API_KEY"] = "tml-verl-remote-actor-local"
```

The current server accepts API keys that start with `tml`. If your client code
also takes an explicit base URL, pass the same URL there.

## ByteDance Merlin Users

Arnold launch helpers are available under:

```text
verl-recipes/examples/tinker_server/tinker_server_merlin_launch_script
```

Use those scripts when launching the server as a Merlin/Arnold job instead of a
local process.

## Config Reference

Common `server` settings:

- `host`, `port`: HTTP bind address. Defaults are `0.0.0.0:8000`.
- `ray_address`: `local` or an existing Ray cluster address.
- `checkpoint_dir`: local root for `save_weights` / `load_weights` state.
- `server_max_runtime`: optional maximum uptime in seconds. `null` means no
    time limit.
- `max_concurrent_samples`: cap for concurrent `asample` requests.

Required VeRL fields include `actor_rollout_ref.model.path`,
`actor_rollout_ref.actor`, `algorithm`, `trainer.nnodes`, and
`trainer.n_gpus_per_node`.

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

Custom export example:

```json
POST /api/v1/export_model
{"save_path": "/shared/exports/qwen3-1.7b-step-100"}
```

## Operational Notes

- The server owns one global model, session, and sampling session for
    compatibility with Tinker client flows.
- `forward`, `forward_backward`, `optim_step`, checkpoint, and weight update
    operations are scheduled exclusively. `asample` may run concurrently up to
    `server.max_concurrent_samples`.
- `save_weights` writes a VeRL checkpoint under `server.checkpoint_dir` and
    returns a `tinker://...` URI. `load_weights` resolves that URI back to local
    checkpoint state.
- `export_model` writes the current actor weights to a server-side `save_path`
    as a Hugging Face `save_pretrained` directory. The path must be absolute and
    empty or not yet exist, and its parent directory must exist. This is the
    portable export API for loading the model in other applications.
- `save_weights_for_sampler` updates rollout weights from the actor. In rollout
    mode this is required before sampling from newly trained weights.

## Current Limitations

- Critic, reward model, and teacher model serving are not supported (may be added in the future).
- LoRA training is not supported. Some LoRA-shaped metadata is returned for
    Tinker cookbook compatibility, but the backend trains full model weights.
- Session and client creation are compatibility no-ops. Multiple clients are not
    isolated: they share one model state, optimizer state, and sampler state, so
    updates from one client will impact all others.
- Creating a new session or client does not snapshot model weights. Save and
    reload checkpoints explicitly if you need to return to an earlier state.
- `load_weights` currently restores optimizer state even when optimizer loading
    is disabled in the request. For a clean optimizer reset, save model weights as
    safetensors and restart the server with `actor_rollout_ref.model.path`
    pointing to that safetensors checkpoint.
