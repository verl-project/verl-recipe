# Context-management example

Runs GRPO with the `naive_summarizer_agent` loop, which compresses the trajectory whenever the model
emits a `<summary>...</summary>` block and continues from `(initial prompt + summary)`.

## Files

- `agent.yaml` — registers `naive_summarizer_agent` and `tool_sliding_window_agent` (forwarded as
  agent-loop `__init__` kwargs). Passed to verl via `agent_loop_config_path`.
- `run_summarizer.sh` — minimal GRPO launch. Set `MODEL_PATH`, `TRAIN_FILES`, `VAL_FILES`.

## Run

```bash
# from a verl checkout that includes this recipe (git submodule update --init --recursive recipe)
export MODEL_PATH=Qwen/Qwen2.5-3B-Instruct
export TRAIN_FILES=$HOME/data/gsm8k/train.parquet
export VAL_FILES=$HOME/data/gsm8k/test.parquet
bash recipe/context_management/example/run_summarizer.sh
```

## Switching strategy

Select a different registered loop without code changes:

```bash
bash run_summarizer.sh actor_rollout_ref.rollout.agent.default_agent_loop=tool_sliding_window_agent
```

## Notes

- Summarization only triggers when a trajectory actually approaches the context window. On tasks whose
  rollouts comfortably fit, the loop is effectively a no-op — to see (and train) compression, use a
  long-horizon task or tighten `data.max_response_length` / `actor_rollout_ref.rollout.max_model_len`.
- The model must already know how to emit a `<summary>` block on demand (e.g. via SFT cold-start);
  otherwise the summarizer never triggers.
