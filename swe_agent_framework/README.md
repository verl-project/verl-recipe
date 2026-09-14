# SWE-Agent VERL Recipe

Train language models to solve real-world software engineering tasks using reinforcement learning. This recipe integrates [SWE-agent](https://github.com/SWE-agent/SWE-agent) as the agent framework with VERL's GRPO trainer, enabling models to learn from interactive coding feedback in Docker-sandboxed environments.

## Overview

The training loop works as follows:

1. **Data**: Each training sample contains a problem statement (e.g. "fix the bug in calculator.py") and a reference patch.
2. **Rollout**: For each sample, a SWE-Agent subprocess is launched inside a Docker container. The agent interacts with a codebase by reading files, editing code, and running commands.
3. **Model Proxy**: A lightweight HTTP server intercepts the agent's LLM API calls and routes them through VERL's vLLM rollout engine, so every token the agent generates is on-policy.
4. **Reward**: After the agent finishes (or hits the turn limit), its generated patch is compared against the reference patch to produce a 0–1 reward signal.
5. **Training**: VERL applies GRPO policy gradient updates using the collected trajectories and rewards.

```
┌─────────────────────────────────────────────────────┐
│               VERL GRPO Trainer                     │
│  (actor, ref model, vLLM rollout, reward scoring)   │
└──────────────────────┬──────────────────────────────┘
                       │  per-episode
          ┌────────────┴────────────┐
          │   SWEAgentLoop.run()    │
          └────────────┬────────────┘
                       │
     ┌─────────────────┼─────────────────┐
     │                 │                 │
     ▼                 ▼                 ▼
┌──────────┐   ┌─────────────┐   ┌──────────────┐
│ TempRepo │   │ ModelProxy  │   │ sweagent run  │
│ (git)    │   │ (HTTP)      │◄──│ (subprocess)  │
└──────────┘   └──────┬──────┘   └──────────────┘
                      │
                      ▼
              ┌───────────────┐
              │ vLLM generate │
              │ (on-policy)   │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │ compute_score │
              │ (patch diff)  │
              └───────────────┘
```

## Directory Structure

```
recipe/swe_agent/
├── swe_agent_loop.py              # Core agent loop (registered as "swe_agent")
├── model_proxy.py                 # HTTP proxy: SWE-Agent ↔ vLLM (mimics OpenAI API)
├── subprocess_runner.py           # Runs `sweagent run` subprocess + Docker container cleanup
├── reward.py                      # Patch-based reward function (0–1 scoring)
├── patch_extractor.py             # Extract patches from .patch files or git diff
├── config.py                      # Runtime config dataclass + per-instance merge logic
├── config/
│   └── swe_agent_config.yaml      # Agent config for simple/synthetic tasks
├── prepare/
│   ├── prepare_data.py            # Dataset generator: simple synthetic tasks
│   ├── simple_cases_train.json    # Training data for simple synthetic tasks
│   ├── simple_cases_val.json      # Validation data for simple synthetic tasks
│   └── preinstall_swerex.sh       # Pre-install swe-rex in SWE-bench eval images
├── docker/
│   └── Dockerfile.preinstalled    # Extends swerex-python:3.11 with tree-sitter pre-installed
├── example/
│   └── run_simple_test.sh         # Single-node quick test (synthetic data, auto-generated)
└── README.md                      # This file
```

## Prerequisites

### Hardware

- 8× NVIDIA GPUs per node (tested on RTX 3090 24GB; A100 / H100 also work)
- Sufficient disk space for model checkpoints (~50GB per checkpoint)
- For multi-node: RDMA/IB or high-speed TCP networking between nodes

### Runtime Environment

VERL can run either **directly on the host** or **inside a Docker container**. Set `docker_mode` in the YAML config to match your setup:

| `docker_mode` | When to use | How sandbox reaches ModelProxy |
|---------------|-------------|-------------------------------|
| `host` (default) | VERL runs on bare metal | Sandbox uses `--network host`, proxy at `127.0.0.1` |
| `dind` | VERL runs inside a container (Docker-in-Docker) | Outer launcher must expose the host-side proxy so sandbox containers can still reach it at `127.0.0.1` |

**Option A: Bare metal (`docker_mode: host`)**

VERL runs directly on the host. Docker must be installed for spawning sandbox containers.

```yaml
sandbox_config:
  docker_mode: host   # default — sandbox containers share host network
```

**Option B: Docker-in-Docker (`docker_mode: dind`)**

VERL runs inside a container that can create sandbox containers. The outer launcher is expected to expose the host-side ModelProxy into the sandbox network namespace, so sandbox containers still use `127.0.0.1` rather than `host.docker.internal`.

```yaml
sandbox_config:
  docker_mode: dind   # outer launcher exposes proxy so sandbox still uses 127.0.0.1
```

Required `docker run` flags for the outer container:

| Flag | Purpose |
|------|---------|
| `--network host` | ModelProxy must be reachable by sandbox containers |
| `-v /var/run/docker.sock:/var/run/docker.sock` | Allow creating sandbox containers from inside |
| `-v /usr/bin/docker:/usr/bin/docker:ro` | Make Docker CLI available inside the container |
| `--gpus all` | GPU access ([NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)) |
| `--shm-size=32g` | Shared memory for NCCL communication |

Example:

```bash
docker run -it \
  --gpus all \
  --network host \
  --shm-size=32g \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v /usr/bin/docker:/usr/bin/docker:ro \
  -v /path/to/data:/data \
  -v /path/to/models:/models \
  --entrypoint /bin/bash \
  --name verl_swe_train \
  <your-verl-image>
```

### Software Dependencies

```bash
# 1. VERL framework (already included in this repo)
pip install -e .   # from the verl root

# 2. SWE-agent CLI
pip install sweagent
which sweagent   # verify

# 3. Docker sandbox image
docker pull swerex-python:3.11
docker images swerex-python:3.11   # verify

# 4. (Optional) Pre-installed image to avoid tree-sitter install timeouts
docker build -t swerex-python:3.11-preinstalled -f recipe/swe_agent/docker/Dockerfile.preinstalled .

# 5. Model weights
ls /path/to/models/Qwen/Qwen3-4B-Instruct-2507/config.json   # or your model
```

### Pre-flight Check

```bash
nvidia-smi -L | wc -l                                        # expect: 8
python3 -c "import socket; print(socket.gethostbyname(socket.gethostname()))"
#   ^ Must print real IP, NOT 127.0.x.x (needed for host networking)
docker images swerex-python:3.11 --format '{{.Repository}}'  # swerex-python
docker ps                                                     # verify Docker access
```

## Quick Start

### 1. Simple Test (Synthetic Data)

The quickest way to validate the full pipeline. Generates synthetic tasks automatically and runs single-node FSDP training:

```bash
cd /path/to/agentic-rl/verl

# Quick 2-epoch validation
bash recipe/swe_agent/example/run_simple_test.sh trainer.total_epochs=2

# Full 10-epoch run (default)
bash recipe/swe_agent/example/run_simple_test.sh
```

`run_simple_test.sh` will:
1. Auto-generate 8 training + 2 test synthetic tasks (rename, create file, fix bug, etc.)
2. Launch GRPO training with `n_resp_per_prompt=4` (group sampling)
3. Save metrics to JSONL, trajectories to workspace

### 2. Monitor Training

```bash
# Ray dashboard (available on head node)
# http://<HEAD_IP>:8265

# Metrics JSONL (for run_simple_test.sh)
tail -f ~/workspace/logs/qwen3-4b-simple-v1_metrics.jsonl

# Inspect trajectories
ls ~/workspace/trajectories/qwen3-4b-simple-v1/rollout/
ls ~/workspace/trajectories/qwen3-4b-simple-v1/validation/
```

## Training Scripts

### `run_simple_test.sh` — Quick Pipeline Validation

Self-contained script for synthetic tasks. Generates data automatically if missing.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `GPUS_PER_NODE` | 8 | GPUs per node |
| `TRAIN_BATCH_SIZE` | 8 | Prompts per rollout batch |
| `N_RESP_PER_PROMPT` | 4 | GRPO group size (responses per prompt) |
| `MODEL_PATH` | Qwen3-4B-Instruct-2507 | Model directory |
| `EXPERIMENT_NAME` | qwen3-4b-simple-v1 | Experiment name (affects log/checkpoint paths) |
| `max_turns` | 5 | Max agent interaction turns |
| `max_prompt_length` | 4096 | Max prompt tokens |
| `max_response_length` | 4096 | Max response tokens |
| `actor_lr` | 5e-6 | Actor learning rate |
| `ppo_epochs` | 4 | PPO/GRPO update epochs per batch |
| `total_epochs` | 10 | Training epochs (override via Hydra CLI) |

## Configuration

All default values live in the `SWEAgentRuntimeConfig` dataclass (`config.py`). The YAML config files only need to specify values that differ from those defaults — no value is defined in two places.

### ⚠️ Important: RL Training & History Processors

When running PPO/GRPO with this recipe, **do not use `last_n_observations` or `closed_window`** history processors in your SWE-Agent config.

These processors dynamically rewrite and fold past environment observations (e.g., replacing them with *"Old environment output: (N lines omitted)"*). Because RL algorithms require strict alignment between the trajectories sampled during rollout and the token sequences evaluated during the PPO update phase, any retroactive modification of the agent's history context will cause severe state/policy mismatches. This misalignment will lead to KL divergence explosion and failure to converge.

If you need to manage context length, adjust the `max_observation_length` parameter instead, as it truncates text *before* it enters the history, preserving the integrity of the RL trajectory.

### Config hierarchy

```
SWEAgentRuntimeConfig (dataclass defaults)
  └── YAML config file (deployment overrides)
       └── extra_info per instance (runtime overrides, data-affine fields only)
```

### Key config fields

| Field | Category | Description |
|-------|----------|-------------|
| `sandbox_config.docker_mode` | Infrastructure | `"host"` (bare metal) or `"dind"` (Docker-in-Docker) |
| `sandbox_config.swe_agent_timeout` | Infrastructure | Total execution time limit (seconds) |
| `sandbox_config.docker_memory_limit` | Infrastructure | Container memory limit |
| `sandbox_config.max_parallel_tasks_per_worker` | Infrastructure | Concurrency limit per node |
| `sandbox_config.max_model_calls_per_instance` | Data-affine | Max model call count per instance |
| `sandbox_config.docker_image` | Data-affine | Sandbox Docker image |
| `agent.templates` | Data-affine | System/instance/next-step prompt templates |
| `agent.tools` | Data-affine | Tool bundles, parse function type |

Data-affine fields can be overridden per instance via `extra_info.sandbox_overrides` and `extra_info.agent_overrides` (set during data preparation).

### Config file

- `swe_agent_config.yaml`: Default config for simple/synthetic tasks.

## Key Components

### SWEAgentLoop (`swe_agent_loop.py`)

The core agent loop, registered with VERL as `"swe_agent"`. For each episode:

1. Parses `extra_info` to get the problem statement and repo content
2. Merges per-instance overrides with config defaults (`apply_data_overrides`)
3. Creates a temporary git repo on disk (inline utility)
4. Starts a `ModelProxy` HTTP server
5. Launches `sweagent run` as a subprocess pointing at the proxy
6. Intercepts each agent API call, sends to vLLM for on-policy generation
7. Extracts the final patch and returns it as `AgentLoopOutput`

### ModelProxy (`model_proxy.py`)

A lightweight HTTP server that mimics the OpenAI Chat Completions API. SWE-Agent sends requests to this proxy thinking it's an LLM API. The proxy:
- Queues requests for VERL to consume
- Blocks until VERL's vLLM engine generates a response
- Returns the response to SWE-Agent

Port assignment: `port=0` (default) lets the OS assign an available port per worker. If a fixed port is set (`port > 0`), ModelProxy auto-increments on conflict.

### SubprocessRunner (`subprocess_runner.py`)

Manages the `sweagent run` subprocess lifecycle:
- Launches with proper CLI arguments and environment
- Handles timeouts (SIGTERM → SIGKILL escalation)
- Captures logs for debugging
- Uses `PatchExtractor` to extract the generated patch

### Reward Function (`reward.py`)

Computes a patch-based reward with additional tool-use shaping for `swe_agent` data sources. Alignment failures are treated as invalid training trajectories and receive `0.0` reward.

| Condition | Score |
|-----------|-------|
| Exact patch match | `1.0` |
| Partial patch match | `0.10`–`0.85` |
| Patch generated but wrong files | `0.05` |
| No patch, but edited correct file | `0.05` |
| No patch, but ran tests / python verification | `0.03` |
| No patch, but edited wrong file | `0.02` |
| No patch, but explored correct file | `0.02` |
| No patch, but explored code | `0.01` |
| Alignment failed / 0 turns / timeout | `0.0` |
| Long and fruitless (>=10 turns, no edits) | `-0.05` |
| Premature failure with little tool use (<=2 turns) | `-0.1` |

### Data Preparation (`prepare/prepare_data.py`)

Mode:
- **`simple`**: Generates synthetic tasks (rename file, create file, fix bug, etc.) with distinct train/val task pools (no overlap)

Output parquet fields:
- `prompt`: Minimal chat-formatted problem description
- `reward_model.ground_truth`: Expected patch for reward computation
- `extra_info`: Problem statement, repo content, per-instance overrides
- `agent_name`: `"swe_agent"`

## WandB Monitoring

### Setup

```bash
pip install wandb
wandb login

export WANDB_API_KEY="your-api-key-here"
export WANDB_PROJECT="swe_agent_training"
export WANDB_ENTITY="your-username"  # Optional
```

### Launch Training with WandB

```bash
bash recipe/swe_agent/example/run_simple_test.sh
```

WandB will track training metrics (rewards, policy loss, KL divergence), system metrics (GPU utilization, throughput), and validation performance. View results at https://wandb.ai in your project.

To disable WandB, simply `unset WANDB_API_KEY`.

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Training exits at 100% immediately | Old checkpoint matches `total_epochs` | `rm -rf $WORK_BASE/checkpoints/$EXPERIMENT_NAME/` |
| Proxy port conflict (`port > 0`) | Multiple workers on same port | Keep `port: 0` (recommended) or increase `max_port_retries` |
| SWE-Agent TimeoutError | Docker container startup timeout | Pre-pull image: `docker pull swerex-python:3.11` |
| OOM during rollout | Too many concurrent Docker containers | Reduce `train_batch_size` or `docker_memory_limit` |
| No patch found | Agent didn't run `submit` | Increase `max_model_calls_per_instance` or improve system prompt |
| CUDA driver error: invalid device ordinal | `actor_max_token_len_per_gpu` too small for group sampling | Increase to `(max_prompt_length + max_response_length) × n_resp_per_prompt` |
| GPU hardware error (Unknown Error) | PCIe link failure or driver issue | Reboot node, check `nvidia-smi -q` for link errors, use fewer GPUs |

### Emergency Cleanup

```bash
# Stop all SWE-Agent Docker containers
docker ps --filter "ancestor=swerex-python:3.11" -q | xargs -r docker stop

# Force stop Ray
ray stop --force

# Kill training process
pkill -9 -f main_ppo
```

## Extending

### Custom Tasks

Create your own training data by adding new task generators in `prepare/prepare_data.py`. Each task needs:
- `problem_statement`: Natural language description
- `repo_content`: Dict mapping file paths to content (the starting codebase)
- `expected_patch`: The correct unified diff

### Custom Reward Functions

Replace or extend `reward.py`. The function signature is:

```python
def compute_score(solution_str, ground_truth, extra_info=None, **kwargs):
    """Returns a float reward in [0, 1]."""
```

### Custom Templates

Override prompt templates per-instance via `extra_info.agent_overrides.templates` or globally in `swe_agent_config.yaml`.
