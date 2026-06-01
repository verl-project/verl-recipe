# Cluster Trace Design

## Overview

Cluster Trace is a lightweight performance tracing system for distributed VeRL training. It automatically captures wall-clock timelines across all worker roles and ranks, producing Chrome Trace Event Format output for visualization.

## Design Goals

1. **Zero Setup**: Enable tracing with a single environment variable
2. **Automatic Collection**: Leverage Ray's built-in log infrastructure
3. **Minimal Overhead**: Stdout-based logging without shared filesystem requirements
4. **Clear Visualization**: Role-based grouping with Chrome Tracing/Perfetto support

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     VeRL Training Process                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│  │  Rollout     │    │ Actor Forward│    │  Reference   │     │
│  │   Worker     │    │    Worker    │    │   Worker     │     │
│  │  (rank=0-7)  │    │  (rank=0-7)  │    │  (rank=0-7)  │     │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘     │
│         │                   │                   │             │
│         └───────────────────┼───────────────────┘             │
│                             │                                 │
│                    ┌────────▼────────┐                        │
│                    │  Trace Logger   │                        │
│                    │    (stdout)     │                        │
│                    └────────┬────────┘                        │
└────────────────────────────┼─────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │     Ray         │
                    │  Log Collector  │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │   Log Parser    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ trace.json      │
                    │ (Chrome Format) │
                    └─────────────────┘
```

### Worker Integration

Workers automatically install tracing via two entry points:

1. **AsyncWorkerMixin** (actor_fwd, ref, reward, actor_train)
   - Calls `install()` in `init_async_worker()`
   - Workers inherit trace capability automatically

2. **AsyncFlowAgentLoopWorker** (rollout)
   - Calls `install()` in `__init__()`
   - Uses Ray actor name to derive worker index

## Role Mapping

Worker classes map to trace roles for visualization:

| Worker Class | Trace Role |
|--------------|------------|
| AsyncFlowAgentLoopWorker | rollout |
| ActorForwardWorker | actor_fwd |
| ReferenceForwardWorker | ref |
| RewardAdvWorker | reward |
| ActorTrainWorker | actor_train |

## Event Flow

### Production

1. Worker calls `marked_timer("operation_name")` or `simple_timer("operation_name")`
2. Patched timer wraps execution with `_TraceContext`
3. Context measures duration and logs JSON to stdout
4. Ray collects stdout to `/tmp/ray/session_latest/logs/worker-*.log`

### Collection & Parsing

1. Log parser scans logs for `[ClusterProfiler]` marker
2. Extracts JSON events via regex pattern
3. Groups events by worker and rank
4. Filters idle iterations (see Filtering Strategy)
5. Generates Chrome Trace Event Format

## Filtering Strategy

The log parser removes idle polling iterations to keep traces clean:

**Workers with e2e_step_time** (AsyncWorkerMixin-based):
- Identify iterations via `e2e_step_time` event boundaries
- Keep only iterations containing compute timers
- Discard iterations without `compute_s` or operation-specific timers

**Workers without e2e_step_time** (rollout):
- Keep all events (no idle structure)
- Rollout worker has different async pattern

### Compute Indicators

These timer names indicate actual computation:
- `compute_s`, `rollout`, `generate_sequences`
- `actor_fwd`, `receive_weight`, `ref_fwd`
- `reward`, `adv`, `actor_train`

## Chrome Trace Format

Output maps to Chrome Trace Event Format dimensions:

| Dimension | Maps to |
|-----------|---------|
| Process (pid) | Worker role (unique ID per role) |
| Thread (tid) | Worker rank within role |
| Duration bar | Timer execution |

Events are annotated with process names and thread names for clarity.

## Data Format

### Logged Event Format

```json
{
  "worker": "rollout",
  "rank": 3,
  "ph": "X",
  "name": "generate_sequences",
  "ts": 1234567890000,
  "dur": 12345
}
```

- `ph: "X"` - Complete event (Chrome Trace Event Format)
- `ts` - Timestamp in microseconds (epoch)
- `dur` - Duration in microseconds

### Chrome Trace Output

```json
{
  "traceEvents": [
    {"ph": "M", "name": "process_name", "pid": 0, "args": {"name": "rollout"}},
    {"ph": "M", "name": "thread_name", "pid": 0, "tid": 0, "args": {"name": "rollout[rank=0]"}},
    {"ph": "X", "name": "generate_sequences", "pid": 0, "tid": 0, "ts": 1234567890000, "dur": 12345},
    ...
  ]
}
```

## Timestamp Synchronization

- All workers use `time.time_ns()` for wall-clock synchronization
- Events across workers align by actual execution time
- No master synchronization needed (Ray handles worker startup ordering)

## Advantages of Stdout Approach

1. **No shared filesystem**: Ray collects logs automatically
2. **Automatic flush**: Stdout buffering handled by system
3. **Simple implementation**: No atexit/drain logic
4. **Minimal intrusion**: Single patch point in existing code
5. **Ray-native**: Leverages existing infrastructure