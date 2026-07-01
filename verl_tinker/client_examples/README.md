# Tinker Server Client Examples

This directory contains client-side smoke recipes for the VeRL-backed Tinker
server implemented in `verl_recipes.verl_tinker_server.tinker_router`. The
launch scripts under `tinker_server_merlin_launch_script/` start the server on
Merlin or Arnold; the files under `tasks/` exercise that server through the
Tinker client APIs.

The client code intentionally runs in a small, separate `uv` environment defined
by this directory's `pyproject.toml`. This keeps the smoke tests close to a real
Tinker user workflow: the client does not own model workers or GPU resources. It
discovers an existing Ray Serve endpoint, sets `TINKER_BASE_URL`, waits for
`/api/v1/healthz` to return `ready`, runs one cookbook workload, and calls
`/api/v1/shutdown` when the test exits.

## Available Client Tests

`tasks/run_single_test.py` selects the workload from `TEST_NAME`. Supported
values are:

- `sft_tulu3`
- `sft_norobot`
- `sft_norobot_no_rollout`
- `sdft_single_task`
- `rl_gsm8k`
- `sft_rl_gsm8k`
- `opd_deepmath`

If `TEST_NAME` is not set, the runner defaults to `sft_tulu3`.

## Prerequisites

- ModelChef is installed and available in the environment that launches the
    server.
- A VeRL Tinker server is already running locally or has been launched through
    Merlin or Arnold.
- For remote Ray Serve endpoints, `SERVER_RAY_SERVE_PROXY_PSM` is set to the
    server PSM. If it is unset, the client assumes local development and uses
    `http://127.0.0.1:8000/`.

## Expected Flow

1. Launch a server config that matches the workload.
1. Wait for the server PSM to become discoverable.
1. Run the client from this directory or launch the matching CI client config.
1. The client resolves the Ray Serve URL, polls `/api/v1/healthz`, runs the
    recipe, and shuts the server down.

## Launching the Server (ByteDance Merlin Users)

### For Interactive Debugging

Start the server directly from a GPU worker (we recommend using tmux or screen for prolonged session):

```bash
bash verl-recipes/examples/tinker_server/tinker_server_merlin_launch_script/entrypoint_qwen3_1b7_actor_rollout.sh
```

Then connect to the same worker and run the client:

```bash
mlx worker list
mlx worker login <worker_id>
bash verl-recipes/examples/tinker_server/tasks/run_single_test.sh
```

This mode is convenient for debugging, but the client shares the same node
resources as the server.

### For Longer or Heavier Runs

Launch one of the server configs under `verl-recipes/ci/nightly/` as an Arnold
job. Each YAML config defines a Ray Serve proxy PSM near the bottom, for
example:

```yaml
BYTED_RAY_SERVE_PROXY_PSM: mlsys.verl_tinker.qwen3_1b7_grpo
```

Use that value as `SERVER_RAY_SERVE_PROXY_PSM` when running the client from a dev
box or another Merlin or Arnold worker.

### Resolving a Ray Serve URL from PSM

The client runner resolves a remote URL through BytedRay. The core logic is:

```python
import time
from ray.serve import get_serve_http_client


def wait_for_url(psm: str, max_wait_time: int = 28800):
    if psm is None or len(psm) == 0:
        return "http://127.0.0.1:8000/"

    print(f"waiting for url from psm: {psm}")
    url = ""
    ct = 0
    start_time = time.monotonic()

    while url == "":
        elapsed = time.monotonic() - start_time
        if elapsed > max_wait_time:
            raise TimeoutError(f"Timed out waiting for server URL after {max_wait_time} seconds for psm={psm}")

        url = get_serve_http_client(psm=psm).get_one_request_url()

        if url == "":
            ct += 1
            if ct % 12 == 0:
                print(f"Server not ready yet, waited: {ct // 12} minutes")
            time.sleep(5)

    return url
```

The returned URL is usable from dev boxes and other Merlin or Arnold GPU nodes
that can access the Ray Serve proxy.

## Server Configuration

The example entrypoints default to Qwen3-1.7B:

- Actor + rollout server config:
    `verl-recipes/src/verl_recipes/verl_tinker_server/config/quick_start/actor_rollout.yaml`
- Actor-only server config:
    `verl-recipes/src/verl_recipes/verl_tinker_server/config/quick_start/actor.yaml`
- Model path: `/mnt/hdfs/mlsys/models/Qwen3-1.7B`

Override these defaults with environment variables before launching the server:

```bash
export TINKER_SERVER_MODEL=/path/to/model
export TINKER_SERVER_NNODES=1
export TINKER_SERVER_N_GPUS_PER_NODE=8
```

The server owns the model, Ray cluster, Ray Serve proxy, worker roles, and GPU
resources. Training settings such as mini-batch size, optimizer parameters,
entropy or KL coefficients, and loss-specific options belong in the Tinker
client workload.

## Nightly CI Configs

The nightly configs under `verl-recipes/ci/nightly/` use a two-job layout: one
GPU server job and one CPU-only client job. They are the best reference when the
example YAML files in this directory drift.

| Workload         | Server config                                                     | Client config                                                     |
| ---------------- | ----------------------------------------------------------------- | ----------------------------------------------------------------- |
| SFT Tulu 3       | `launch_verl_tinker_server_qwen3_1b7_sft_vexact_tulu3.yaml`       | `launch_verl_tinker_client_qwen3_1b7_sft_vexact_tulu3.yaml`       |
| SFT No Robots    | `launch_verl_tinker_server_qwen3_1b7_sft_no_rollout_norobot.yaml` | `launch_verl_tinker_client_qwen3_1b7_sft_no_rollout_norobot.yaml` |
| SFT + RL GSM8K   | `launch_verl_tinker_server_qwen3_1b7_sft_rl_gsm8k.yaml`           | `launch_verl_tinker_client_qwen3_1b7_sft_rl_gsm8k.yaml`           |
| RL GSM8K         | `launch_verl_tinker_server_qwen3_1b7_rl_gsm8k.yaml`               | `launch_verl_tinker_client_qwen3_1b7_rl_gsm8k.yaml`               |
| SDFT Single Task | `launch_verl_tinker_server_qwen3_1b7_sdft_single_task.yaml`       | `launch_verl_tinker_client_qwen3_1b7_sdft_single_task.yaml`       |

Run the server config first, then the matching client config:

```bash
python verl-recipes/tasks/arnold_launch.py --config verl-recipes/ci/nightly/launch_verl_tinker_server_qwen3_1b7_sft_vexact_tulu3.yaml
python verl-recipes/tasks/arnold_launch.py --config verl-recipes/ci/nightly/launch_verl_tinker_client_qwen3_1b7_sft_vexact_tulu3.yaml
```

## Notes

The YAML files in this example directory are not covered by nightly CI and may
fall behind active server changes. If they fail, compare them with the
`launch_verl_tinker_server_*.yaml` configs under `verl-recipes/ci/nightly/`,
which are kept up to date by CI.
