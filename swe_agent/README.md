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
├── config/
│   ├── __init__.py                # Exports: SWEAgentRuntimeConfig, SWEAgentYAMLBuilder, etc.
│   ├── runtime_config.py          # Runtime config dataclass + per-instance merge logic
│   ├── swe_agent_config.yaml      # Agent config for simple/synthetic tasks
│   └── swe_agent_config_swebench.yaml  # Agent config for SWE-bench (higher timeouts, per-instance images)
├── runtime/
│   ├── __init__.py                # Exports: ModelProxy, execute_swe_agent, cleanup_instance_containers
│   ├── model_proxy.py             # HTTP proxy: SWE-Agent ↔ vLLM (mimics OpenAI API)
│   ├── subprocess_runner.py       # Runs `sweagent run` subprocess with timeout handling
│   └── container_cleanup.py       # Docker container cleanup per instance
├── reward/
│   ├── __init__.py                # Exports: compute_score, compare_patches_simple, normalize_patch
│   └── reward.py                  # Patch-based reward function (0–1 scoring)
├── prepare/
│   └── prepare_data.py            # Dataset generator: simple synthetic tasks or SWE-bench
├── utils/
│   ├── __init__.py                # Exports: PatchExtractor, normalize_openai_messages, etc.
│   ├── message_utils.py           # OpenAI message normalization for chat templates
│   ├── repo_manager.py            # Temporary git repo creation/cleanup
│   └── patch_extractor.py         # Extract patches from .patch files or git diff
├── docker/
│   └── Dockerfile.preinstalled    # Extends swerex-python:3.11 with tree-sitter pre-installed
├── example/
│   ├── run_simple_test.sh         # Single-node quick test (synthetic data, auto-generated)
│   ├── run_swebench.sh            # Single/multi-node SWE-bench training
│   ├── run_swebench_4node.sh      # 4-node wrapper (sets defaults, delegates to run_swebench.sh)
│   ├── setup_ray_cluster.sh       # Start 4-node Ray cluster
│   └── stop_ray_cluster.sh        # Stop Ray cluster + cleanup containers
└── README.md                      # This file
```

## Prerequisites

### Hardware

- 8× NVIDIA GPUs per node (tested on RTX 3090 24GB; A100 / H100 also work)
- Sufficient disk space for model checkpoints (~50GB per checkpoint)
- For multi-node: RDMA/IB or high-speed TCP networking between nodes

### Runtime Environment (Docker-in-Docker)

This recipe runs inside a VERL Docker container. Since training spawns SWE-Agent sandbox containers via Docker, the host container must support **Docker-in-Docker (DinD)** with **host networking**.

Required `docker run` flags:

| Flag | Purpose |
|------|---------|
| `--network host` | ModelProxy HTTP server must be reachable by sandbox containers |
| `-v /var/run/docker.sock:/var/run/docker.sock` | Allow creating/managing sandbox containers from inside |
| `-v /usr/bin/docker:/usr/bin/docker:ro` | Make Docker CLI available inside the container |
| `--gpus all` | GPU access (requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)) |
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

### 2. SWE-bench Training (Real Tasks)

#### Prepare Data

```bash
cd /path/to/agentic-rl/verl

# SWE-bench Lite / Verified
python3 recipe/swe_agent/prepare/prepare_data.py \
    --mode swebench \
    --swebench_train /path/to/swe_bench_train.json \
    --swebench_test  /path/to/swe_bench_test.json \
    --output_dir data/swe_bench
```

#### Single-Node (8 GPU)

```bash
bash recipe/swe_agent/example/run_swebench.sh
```

#### Multi-Node (4 nodes × 8 GPU)

```bash
# Step 1: Start Ray cluster
bash recipe/swe_agent/example/setup_ray_cluster.sh

# Step 2: Launch training
bash recipe/swe_agent/example/run_swebench_4node.sh

# Step 3: Cleanup after training
bash recipe/swe_agent/example/stop_ray_cluster.sh
```

### 3. Monitor Training

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

### `run_swebench.sh` — SWE-bench Training

Unified script for single-node or multi-node SWE-bench training.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NNODES` | 1 | Number of nodes (>1 enables multi-node) |
| `GPUS_PER_NODE` | 8 | GPUs per node |
| `TRAIN_BATCH_SIZE` | 8 | Training batch size |
| `DATA_DIR` | `data/swe_bench` | Path to SWE-bench parquet data |
| `EXPERIMENT_NAME` | qwen3-4b-swebench-v1 | Experiment name |
| `max_turns` | 10 | Max agent interaction turns |
| `max_prompt_length` | 8192 | Max prompt tokens |
| `max_response_length` | 8192 | Max response tokens |
| `total_epochs` | 2 | Training epochs |

### `run_swebench_4node.sh` — 4-Node Wrapper

Thin wrapper that sets 4-node defaults (`NNODES=4`, `TRAIN_BATCH_SIZE=40`) and delegates to `run_swebench.sh`. All Hydra overrides pass through.

## Configuration

### Agent Config (`config/swe_agent_config.yaml`)

The YAML config defines the SWE-Agent behavior and has two categories of fields:

**Infrastructure fields** (fixed per deployment):
- `proxy_config`: ModelProxy port, timeout, retry settings
- `sandbox_config.swe_agent_timeout`: Total execution time limit
- `sandbox_config.docker_memory_limit`: Container memory limit
- `sandbox_config.max_parallel_tasks_per_worker`: Concurrency limit per node

**Data-affine fields** (can be overridden per instance via `extra_info`):
- `sandbox_config.max_turns`: Max interaction turns
- `sandbox_config.max_steps`: Max model call count
- `sandbox_config.docker_image`: Sandbox Docker image
- `agent.templates`: System/instance/next-step prompt templates
- `agent.tools`: Tool bundles, parse function type

Per-instance overrides are applied at runtime via `extra_info.sandbox_overrides` and `extra_info.agent_overrides` (set during data preparation).

### Two Config Variants

| Config | Use Case | Key Differences |
|--------|----------|-----------------|
| `swe_agent_config.yaml` | Simple/synthetic tasks | Lower timeouts, generic Docker image |
| `swe_agent_config_swebench.yaml` | SWE-bench instances | Higher timeouts (install scripts), per-instance Docker image override, SWE-bench template |

## Key Components

### SWEAgentLoop (`swe_agent_loop.py`)

The core agent loop, registered with VERL as `"swe_agent"`. For each episode:

1. Parses `extra_info` to get the problem statement and repo content
2. Merges per-instance overrides with config defaults (`apply_data_overrides`)
3. Creates a temporary git repo on disk (`utils.repo_manager`)
4. Starts a `ModelProxy` HTTP server
5. Launches `sweagent run` as a subprocess pointing at the proxy
6. Intercepts each agent API call, sends to vLLM for on-policy generation
7. Extracts the final patch and returns it as `AgentLoopOutput`

### ModelProxy (`runtime/model_proxy.py`)

A lightweight HTTP server that mimics the OpenAI Chat Completions API. SWE-Agent sends requests to this proxy thinking it's an LLM API. The proxy:
- Queues requests for VERL to consume
- Blocks until VERL's vLLM engine generates a response
- Returns the response to SWE-Agent

Port assignment: `port=0` (default) lets the OS assign an available port per worker. If a fixed port is set (`port > 0`), ModelProxy auto-increments on conflict.

### SubprocessRunner (`runtime/subprocess_runner.py`)

Manages the `sweagent run` subprocess lifecycle:
- Launches with proper CLI arguments and environment
- Handles timeouts (SIGTERM → SIGKILL escalation)
- Captures logs for debugging
- Uses `PatchExtractor` to extract the generated patch

### Reward Function (`reward/reward.py`)

Computes a 0–1 reward by comparing the agent's generated patch against the expected patch:

| Condition | Score |
|-----------|-------|
| Exact patch match | 1.0 |
| All changed files match | 0.5 |
| Partial file overlap | 0.2 + 0.3 × overlap_ratio |
| Patch generated but no overlap | 0.1 |
| No patch generated | 0.0 |

### Data Preparation (`prepare/prepare_data.py`)

Two modes:
- **`simple`**: Generates synthetic tasks (rename file, create file, fix bug, etc.) with distinct train/val task pools (no overlap)
- **`swebench`**: Loads SWE-bench instances with real GitHub issues and patches; sets per-instance `docker_image` override

Output parquet fields:
- `prompt`: Minimal chat-formatted problem description
- `reward_model.ground_truth`: Expected patch for reward computation
- `extra_info`: Problem statement, repo content, per-instance overrides
- `agent_name`: `"swe_agent"`

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| Training exits at 100% immediately | Old checkpoint matches `total_epochs` | `rm -rf $WORK_BASE/checkpoints/$EXPERIMENT_NAME/` |
| Proxy port conflict (`port > 0`) | Multiple workers on same port | Keep `port: 0` (recommended) or increase `max_port_retries` |
| SWE-Agent TimeoutError | Docker container startup timeout | Pre-pull image: `docker pull swerex-python:3.11` |
| OOM during rollout | Too many concurrent Docker containers | Reduce `train_batch_size` or `docker_memory_limit` |
| No patch found | Agent didn't run `submit` | Increase `max_turns` or improve system prompt |
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

Replace or extend `reward/reward.py`. The function signature is:

```python
def compute_score(solution_str, ground_truth, extra_info=None, **kwargs):
    """Returns a float reward in [0, 1]."""
```

### Custom Templates

Override prompt templates per-instance via `extra_info.agent_overrides.templates` or globally in `swe_agent_config.yaml`.
