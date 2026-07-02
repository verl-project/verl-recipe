# Reward Queue: Decoupled Inference and Reward Computation

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repository, install mode (rolling `main`, pinned release tag, or pinned git commit), and copy-pastable `pip` / `git` instructions where they exist.

## Overview

Reward Queue decouples inference (generation) from reward computation in VERL's fully asynchronous training pipeline. It introduces an intermediate queue between the two stages, enabling concurrent execution and maximizing GPU utilization.

When reward computation involves slow external LLM judges or complex scoring functions, the traditional tightly-coupled pipeline wastes GPU cycles waiting for scores. This recipe solves that bottleneck.

## Architecture

![Reward Queue Architecture](./images/reward_queue_architecture.png)

**Core pipeline:**

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│   Generation    │────▶│ RewardQueue  │────▶│ Reward Compute  │
│   (async)       │     │              │     │ (concurrent)    │
└─────────────────┘     └──────────────┘     └─────────────────┘
```

**Key components:**

| Component           | File              | Role                                                         |
| ------------------- | ----------------- | ------------------------------------------------------------ |
| `RewardQueue`       | `reward_queue.py` | Ray actor-based async queue with producer-consumer semantics |
| `SubRewardDataItem` | `utils.py`        | Data item passed through the queue                           |
| `SampleAggregator`  | `utils.py`        | Accumulates scored sub-items per sample                      |
| `Rollouter`         | `rollouter.py`    | Extended FullyAsyncRollouter with reward queue support       |
| `Trainer`           | `trainer.py`      | Extended FullyAsyncTrainer with timing metadata              |

**Processing flow:**

1. `_processor_worker` launches async generation for each sub-item
2. Generated outputs are immediately buffered into `RewardQueue` (no waiting for scores)
3. `_reward_consumer_worker` pulls from queue and distributes scoring across workers
4. `SampleAggregator` accumulates scored sub-items per sample
5. `_finalize_sample` assembles complete batch and publishes to `MessageQueue` for trainer

## Quick Start

### Enable the Feature

Set `async_training.enable_reward_queue: true` in your config:

```yaml
async_training:
  enable_reward_queue: true
  reward_queue_size: null  # Uses default: max_required_samples * rollout_n
```

Or via command line:

```bash
python -m recipe.reward_queue.main \
    --config-path=config \
    --config-name='fully_async' \
    async_training.enable_reward_queue=true \
    # ... other config
```

### Run Training

```bash
# Single node (8 GPUs)
NNODES=1 NGPUS_PER_NODE=8 \
MODEL_PATH=Qwen3.5-9B \
TRAIN_FILE=./gsm8k/train/gsm8k_tra.jsonl \
VAL_FILE=./gsm8k/eval/gsm8k_ev.jsonl \
bash recipe/reward_queue/train_async.sh
```

### Run with Custom Settings

```bash
NNODES=2 \
NGPUS_PER_NODE=8 \
MODEL_PATH=Qwen3.5-9B \
TRAIN_BATCH_SIZE=8 \
N_SAMPLE=8 \
TOTAL_TRAINING_STEPS=500 \
ASYNC_STALENESS=0.3 \
ASYNC_SYNC_STEP=2 \
ASYNC_REQUIRE_BATCHES=4 \
bash recipe/reward_queue/train_async.sh
```

## Configuration

### Async Training Config

| Parameter                            | Default | Description                                                  |
| ------------------------------------ | ------- | ------------------------------------------------------------ |
| `async_training.enable_reward_queue` | `false` | Enable/disable reward queue decoupling                       |
| `async_training.reward_queue_size`   | `null`  | Max queue size. `null` means `max_required_samples * rollout_n` |

### Environment Variables

| Variable                | Default                         | Description                      |
| ----------------------- | ------------------------------- | -------------------------------- |
| `NNODES`                | `1`                             | Number of nodes                  |
| `NGPUS_PER_NODE`        | `8`                             | GPUs per node                    |
| `MODEL_PATH`            | `Qwen3.5-9B`                    | Model path                       |
| `TRAIN_FILE`            | `./gsm8k/train/gsm8k_tra.jsonl` | Training data                    |
| `VAL_FILE`              | `./gsm8k/eval/gsm8k_ev.jsonl`   | Validation data                  |
| `TRAIN_BATCH_SIZE`      | `8`                             | Training batch size              |
| `N_SAMPLE`              | `8`                             | Responses per prompt (rollout_n) |
| `TOTAL_TRAINING_STEPS`  | `500`                           | Total training steps             |
| `ASYNC_STALENESS`       | `0.3`                           | Staleness threshold              |
| `ASYNC_SYNC_STEP`       | `2`                             | Parameter sync trigger step      |
| `ASYNC_REQUIRE_BATCHES` | `4`                             | Required batches                 |

## Monitoring Metrics

The reward queue exports the following metrics to W&B:

| Metric                            | Description                             |
| --------------------------------- | --------------------------------------- |
| `monitor/queue/reward_queue_size` | Current reward queue size               |
| `reward_queue/total_produced`     | Total items produced to queue           |
| `reward_queue/total_consumed`     | Total items consumed from queue         |
| `reward_queue/dropped_samples`    | Samples dropped due to queue overflow   |
| `static/max_reward_queue_size`    | Maximum configured queue size           |
| `timing_s/reward_compute/mean`    | Mean reward computation time            |
| `timing_s/reward_compute/max`     | Max reward computation time             |
| `timing_s/reward_compute/tp95`    | 95th percentile reward computation time |
| `aggregator/pending_groups_count` | Number of samples awaiting completion   |
| `aggregator/total_pending`        | Total sub-items awaiting scoring        |

## Use Cases

1. **External LLM Judges**: When reward computation calls external LLM APIs (e.g., LLM-as-a-Judge), network latency is overlapped with generation.

2. **Complex Scoring Functions**: Multi-step reward pipelines with multiple model calls benefit from overlapping generation with scoring.

3. **Variable Reward Latency**: When computation time varies significantly across samples, the queue buffers fast results while waiting for slow ones.

4. **Throughput Optimization**: Maximizes GPU utilization by keeping either generation or scoring always active.

## File Layout

```
reward_queue/
├── REQUIRED_VERL.txt
├── README.md
├── reward_queue_architecture.drawio
├── main.py                     # Hydra entry point with TaskRunner
├── rollouter.py                # Extended FullyAsyncRollouter
├── trainer.py                  # Extended FullyAsyncTrainer
├── reward_queue.py             # RewardQueue and RewardQueueClient
├── utils.py                    # SubRewardDataItem, SampleAggregator, etc.
├── train_async.sh              # Launch script
├── agent_loop/
│   └── agent_loop.py           # AgentLoopWorkerForRewardQueue
└── config/
    └── fully_async.yaml        # Base config
```

## Design Notes

- **Backpressure control**: When `MessageQueue` is full, scoring pauses automatically to prevent resource exhaustion.
- **Concurrent scoring**: Multiple reward workers score sub-items in parallel, throttled by `max_concurrent_samples * rollout_n`.
- **Temporal decoupling**: Inference output and reward computation run at their own pace via the queue buffer.