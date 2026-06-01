# Cluster Performance Trace for VeRL

Chrome Trace profiler for VeRL cluster training. Captures wall-clock timelines across all worker roles and DP ranks with **one environment variable**. Output is a JSON file that opens directly in `chrome://tracing` or [Perfetto UI](https://ui.perfetto.dev).

The profiler uses Ray's built-in log collection — trace events are logged to stdout with a `[ClusterProfiler]` marker and automatically collected to the master node.

## Quickstart

### Option A: Using Ray Log Collection

```bash
# 1. Enable tracing
export VERL_CLUSTER_TRACE=1
export RAY_DEDUP_LOGS="0"

# 2. Run training as normal (no changes needed)
python train.py ...

# 3. Parse Ray logs and generate Chrome trace
python -m recipe.async_flow.utils.cluster_trace.log_parser \
    /tmp/ray/session_latest/logs/ \
    -o cluster_trace.json

# 4. Open in browser
#    chrome://tracing  →  Load  →  cluster_trace.json
#    or drag-drop onto https://ui.perfetto.dev
```

### Option B: Using `tee` (No Ray Log Collection Needed)

```bash
# 1. Enable tracing
export VERL_CLUSTER_TRACE=1

# 2. Run training with tee to capture stdout
python3 -m recipe.async_flow.grpo_main \
    2>&1 | tee "logs/${exp_name}_$(date +%Y%m%d_%H%M).log"

# 3. Parse the tee'd log file
python -m recipe.async_flow.utils.cluster_trace.log_parser \
    "logs/${exp_name}_$(date +%Y%m%d_%H%M).log" \
    -o cluster_trace.json

# 4. Open in browser
#    chrome://tracing  →  Load  →  cluster_trace.json
```

**Disable:** `unset VERL_CLUSTER_TRACE` — no other changes needed.

## What You See

### Dimensions

| Dimension | Maps to |
|---|---|
| Process (pid) | Worker role: `rollout`, `actor_fwd`, `ref`, `reward`, `actor_train`, `critic`, `train` |
| Thread (tid) | Worker's global rank within that role group (≈ dp_rank for FSDP tp=1, pp=1) |
| Duration bar | One `marked_timer` / `simple_timer` call (e.g. `generate_sequences`, `update_policy`, `compute_values`) |

Each role is colour-coded. Horizontal alignment across processes uses wall-clock time (`time.time_ns()`), so cross-role idle gaps and imbalance are directly visible.

### Profiled Operations

The following timer points are currently profiled in asyncflow workers:

| Worker | Operation | Timer Name | File Location |
|---|---|---|---|
| **Rollout** | Sequence generation | `rollout` | `agent_loop.py` |
| **Actor Forward** | Log probs computation | `actor_fwd` | `actor_fwd_worker.py:115` |
| | Weight receive | `receive_weight` | `actor_fwd_worker.py:162` |
| **Reference Forward** | Reference log probs | `ref_fwd` | `ref_fwd_worker.py:82` |
| **Reward/Advantage** | Reward computation | `reward` | `reward_adv_worker.py:122` |
| | Advantage computation | `adv` | `reward_adv_worker.py:263` |
| **Actor Train** | Training update | `actor_train` | `actor_train_worker.py:145` |
| **All Workers** | End-to-end step | `e2e_step_time` | `base_async_worker.py:219` |
| | Data fetch wait | `wait_data_s` | `base_async_worker.py:222` |
| | Computation | `compute_s` (wraps process_batch) | `base_async_worker.py:236` |
| | Data write back | `put_data` | `base_async_worker.py:244` |

### Filtering Strategy

The log parser automatically filters out idle polling iterations from workers that use the `AsyncWorkerMixin` structure (actor_fwd, ref, reward, actor_train). These workers have `e2e_step_time` events that indicate actual computation — iterations without `compute_s` or similar timers are excluded.

The rollout worker doesn't follow this pattern, so all its events are included.

Use the `--verbose` flag when running the parser to see detailed filtering information:
```bash
python -m recipe.async_flow.utils.cluster_trace.log_parser /tmp/ray/session_latest/logs/ -o trace.json --verbose
```

## Log Parser CLI

```bash
python -m recipe.async_flow.utils.cluster_trace.log_parser [LOG_PATH] -o OUTPUT_FILE [OPTIONS]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `log_path` | Path to log file or Ray log directory containing worker-*.log files |
| `-o`, `--output` | Output Chrome trace JSON file (required) |
| `--compact` | Output compact JSON (no indentation) |
| `-v`, `--verbose` | Print debug information about event filtering |

### Examples

```bash
# Parse Ray logs directory
python -m recipe.async_flow.utils.cluster_trace.log_parser /tmp/ray/session_latest/logs/ -o trace.json

# Parse a specific log file (e.g., from tee)
python -m recipe.async_flow.utils.cluster_trace.log_parser logs/exp_20260316_1400.log -o trace.json

# Parse with verbose output
python -m recipe.async_flow.utils.cluster_trace.log_parser /tmp/ray/session_latest/logs/ -o trace.json --verbose

# Output compact JSON for large files
python -m recipe.async_flow.utils.cluster_trace.log_parser /tmp/ray/session_latest/logs/ -o trace.json --compact
```

## Integration Details

The cluster trace is automatically installed in asyncflow workers via:

1. **AsyncWorkerMixin** (for most workers: actor_fwd, ref, reward, actor_train)
   - Auto-installs in `init_async_worker()`
   - Workers that inherit from `AsyncWorkerMixin` automatically get tracing

2. **AsyncFlowAgentLoopWorker** (rollout worker)
   - Auto-installs in `__init__()`

All you need to do is set `export VERL_CLUSTER_TRACE=1` before running training.

## Architecture

The stdout-based approach leverages Ray's log collection:

1. **Installation**: When `VERL_CLUSTER_TRACE=1` is set, the worker calls `recipe.async_flow.utils.cluster_trace.install()`
2. **Patch**: The function patches `verl.utils.profiler.performance._timer` to wrap all timer calls
3. **Logging**: Each timer call logs a JSON line to stdout
4. **Collection**: Ray collects all stdout to `/tmp/ray/session_latest/logs/worker-*.log`
5. **Parsing**: The log parser extracts `[ClusterProfiler]` lines via regex
6. **Merging**: Events are merged into Chrome Trace JSON format

## Advantages

- No shared filesystem required
- No periodic flush needed (stdout is naturally flushed)
- No atexit/drain complexity
- Simple implementation
- Minimal overhead

## Troubleshooting

### No trace events found

If you see "Warning: No trace events found", verify:

1. `VERL_CLUSTER_TRACE=1` is set before training starts
2. The log path is correct
3. For Ray collection: `RAY_DEDUP_LOGS="0"` is set (to prevent deduplication)

### Events not showing up in trace

- Ensure you're looking at the correct time range (traces can span long durations)
- Use the search feature in Chrome Tracing to find specific worker names
- Try Perfetto UI which has better search and filtering capabilities

### Too many idle iterations shown

The parser automatically filters idle iterations from `AsyncWorkerMixin` workers. If you still see many, verify:
- Workers are properly inheriting from `AsyncWorkerMixin`
- `e2e_step_time` timer is being used
- Use `--verbose` to see the filtering breakdown