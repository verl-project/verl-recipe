# Atropos Integration for verl

GRPO training with [Atropos](https://github.com/NousResearch/atropos) RL environments. Rollout generation and reward scoring are handled by Atropos environments; this recipe pulls the scored batches and trains with verl's FSDP/GRPO pipeline.

Any Atropos environment that pushes `ScoredData` to the trajectory API works out of the box — no code changes needed to switch environments.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Launch script (run_atropos.sh)                                  │
│                                                                  │
│  1. Atropos trajectory API  (run-api, port 8000)                 │
│     └── Buffers scored rollouts from environment(s)              │
│                                                                  │
│  2. verl trainer            (FSDP actor + internal vLLM)         │
│     └── Polls API → computes log probs → GRPO update             │
│     └── Internal vLLM serves inference (HYBRID mode, sleep/wake) │
│                                                                  │
│  3. Generate proxy          (port 9004, CPU-only)                │
│     └── Translates /generate ↔ /v1/completions for environments  │
│                                                                  │
│  4. Atropos environment     (MATH, tool calling, etc.)           │
│     └── Generates rollouts via proxy, scores them, pushes to API │
└──────────────────────────────────────────────────────────────────┘
```

**Training loop** (per step):

1. Poll Atropos API for a batch of `ScoredData` (vLLM is awake, env generates)
2. Convert to verl `DataProto` (left-padded prompts, right-padded responses)
3. Sleep internal vLLM replicas to free GPU memory for FSDP
4. Forward pass: compute log probs (+ ref log probs if KL enabled)
5. Compute GRPO advantages (or use token-level advantages from Atropos)
6. PPO actor update
7. Save checkpoint if at save_freq boundary (for persistence only)
8. Wake replicas, sync weights to internal vLLM via `checkpoint_manager.update_weights()`

## Files

| File | Description |
|------|-------------|
| `main_atropos.py` | Hydra entry point (TaskRunner → RayAtroposTrainer) |
| `atropos_ray_trainer.py` | Core trainer: polls Atropos API, runs GRPO training loop |
| `atropos_client.py` | HTTP client for trajectory API (`/register`, `/batch`) |
| `atropos_data.py` | `ScoredData` → `DataProto` conversion with padding/masking |
| `generate_proxy.py` | CPU-only proxy: translates atropos `/generate` → vLLM `/v1/completions` |
| `config/atropos_trainer.yaml` | Hydra config extending `ppo_trainer` defaults |
| `run_atropos.sh` | Launch script (orchestrates API, trainer, proxy, environment) |
| `tests/` | pytest tests for client, data conversion, and proxy |

## Quick Start

### Prerequisites

```bash
# from verl repo root
pip install -e ".[vllm,atropos]"
git clone https://github.com/NousResearch/atropos.git ../atropos
pip install -e ../atropos
```

### Run training

```bash

# default environment
bash recipe/atropos/run_atropos.sh

# switch environment
ATROPOS_ENV=gsm8k_server bash recipe/atropos/run_atropos.sh

# multi-GPU with a larger model
N_GPUS=4 MODEL=Qwen/Qwen3-8B BATCH_SIZE=16 GROUP_SIZE=8 \
    bash recipe/atropos/run_atropos.sh
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ATROPOS_ENV` | `gsm8k_server` | Atropos environment module name |
| `MODEL` | `Qwen/Qwen3-1.7B` | Model name or HuggingFace path |
| `LR` | `5e-6` | Learning rate |
| `BATCH_SIZE` | `2` | Training batch size (number of prompts) |
| `GROUP_SIZE` | `4` | Responses per prompt (GRPO group size) |
| `TOTAL_STEPS` | `50` | Total training steps |
| `N_GPUS` | `1` | Number of GPUs (data parallel) |
| `MICRO_BATCH_SIZE` | `BATCH_SIZE * GROUP_SIZE` | Micro batch size for actor update (defaults to full batch; reduce to avoid OOM) |
| `MAX_PROMPT_LENGTH` | `512` | Maximum prompt length in tokens |
| `MAX_RESPONSE_LENGTH` | `1536` | Maximum response length in tokens |
| `GPU_MEM_UTIL` | `0.45` | vLLM GPU memory utilization for KV cache |
| `PARAM_OFFLOAD` | `true` | CPU offload model parameters during training |
| `OPTIMIZER_OFFLOAD` | `true` | CPU offload optimizer states during training |
| `GRAD_CLIP` | `1.0` | Gradient clipping norm |
| `ENTROPY_COEFF` | `0` | Entropy bonus coefficient |
| `SAVE_FREQ` | `10` | Checkpoint save frequency (steps) |
| `ATROPOS_DIR` | `../atropos` | Path to Atropos repo clone |
| `ATROPOS_API_PORT` | `8000` | Trajectory API port |
| `PROXY_PORT` | `9004` | Generate proxy port for environments |
| `STEPS_PER_EVAL` | *(env default)* | Environment eval frequency (steps); omit to use env's default |

### Hydra Overrides

All verl training config is accessible via Hydra overrides. Common ones:

| Override | Description |
|----------|-------------|
| `actor_rollout_ref.model.path` | Model name or path |
| `trainer.total_training_steps` | Total training steps (required) |
| `trainer.save_freq` | Checkpoint save frequency |
| `trainer.max_actor_ckpt_to_keep` | Max checkpoints to retain on disk |
| `atropos.api_url` | Trajectory API URL |
| `atropos.poll_timeout` | Max seconds to wait for a batch |
| `algorithm.use_kl_in_reward` | Enable KL penalty (requires ref model) |
| `actor_rollout_ref.actor.optim.lr` | Learning rate |
| `actor_rollout_ref.actor.fsdp_config.param_offload` | CPU offload model parameters |
| `actor_rollout_ref.actor.fsdp_config.optimizer_offload` | CPU offload optimizer states |

## W&B Metrics

The trainer logs the following metrics to Weights & Biases:

| Metric | Description |
|--------|-------------|
| `atropos/batch_groups` | Number of ScoredData groups in the batch |
| `atropos/total_sequences` | Total sequences (groups × group_size) |
| `atropos/mean_score` | Average reward score across all sequences |
| `atropos/identical_score_rate` | Fraction of groups where all scores are the same (zero training signal) |
| `atropos/avg_response_length` | Mean response length in tokens |
| `atropos/empty_response_count` | Number of sequences with empty responses |
| `actor/entropy` | Policy entropy |
| `actor/loss` | Actor loss |
| `training/global_step` | Current training step |

Standard verl metrics (throughput, timing, data statistics) are also logged.

Held-out evaluation is handled by the Atropos environment itself (via `--env.use_wandb true`). Each environment runs its built-in `evaluate()` method at regular intervals and logs accuracy metrics to the same W&B project.

## Switching Environments

Set `ATROPOS_ENV` to any atropos environment module name and adjust `MAX_PROMPT_LENGTH`, `MAX_RESPONSE_LENGTH`, and `GROUP_SIZE` for the task:

```bash
ATROPOS_ENV=gsm8k_server bash recipe/atropos/run_atropos.sh
```

Any environment that pushes `ScoredData` to the trajectory API works — the trainer only consumes batches from the API, it doesn't know or care which environment produced them.

## Tests

```bash
# Client tests (no GPU / verl dependency needed)
pytest recipe/atropos/tests/test_atropos_client.py -v

# Data conversion tests (needs verl installed)
pytest recipe/atropos/tests/test_atropos_data.py -v

# Proxy tests (needs httpx, no GPU needed)
pytest recipe/atropos/tests/test_generate_proxy.py -v

# All tests
pytest recipe/atropos/tests/ -v
```
