# Context-management agent loops

Plug-in **context management** for verl agent loops: keep multi-turn / long-horizon rollouts within
the model's context window by compressing the trajectory on the fly, instead of truncating or
failing once the window is exceeded.

This recipe provides two ready-to-use agent loops and the `ContextManager` abstraction they share:

| Agent loop (`name`) | Class | Strategy |
|---|---|---|
| `naive_summarizer_agent` | `SummarizerAgentLoop` | When the model emits a `<summary>...</summary>` block, replace the history with `(initial prompt + summary)` and continue. |
| `tool_sliding_window_agent` | `ToolSlidingWindowAgentLoop` | Keep a sliding window over tool-calling turns, dropping the oldest turns when the window is exceeded. |

Both subclass `AgentLoopWithContextManagement`, which drives a generic
`generate → check_and_compress → continue` loop around any `ContextManager`
(`SummarizerContextManager`, `SlidingWindowContextManager`, or your own).

## Background

This code was originally proposed for verl core in
[volcengine/verl#5636](https://github.com/verl-project/verl/pull/5636)
("[algo] feat: supporting agentic rl with context management", see issue
[#5375](https://github.com/verl-project/verl/issues/5375)). At the maintainers' request it now lives
here as a self-contained recipe rather than in `verl/experimental/agent_loop/`, so it can evolve
independently of the core library. The multi-trajectory / session-level GRPO training support that
complements it lands separately in core (see verl#5401, #5969).

## Layout

```
context_management/
  context_manager.py                      # ContextManager + Sliding-window / Summarizer implementations
  agent_loop_with_context_management.py   # AgentLoopWithContextManagement + the two agent loops
  context_manager_plugin.md               # design notes / how to write a custom ContextManager
  test_context_manager.py                 # CPU unit tests
  test_agent_loop_with_context_management.py
  example/                                # runnable GRPO example wiring the summarizer loop
```

## Usage

The loops register themselves under the `name`s above. Point verl at this recipe's agent-loop config
and select a loop:

```bash
actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/context_management/example/agent.yaml
actor_rollout_ref.rollout.agent.default_agent_loop=naive_summarizer_agent
```

See [`example/`](example/) for a full run script, and
[`context_manager_plugin.md`](context_manager_plugin.md) for writing your own `ContextManager`.

## Required verl version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repo and the pinned core-library commit.

## Tests

```bash
pytest recipe/context_management/test_context_manager.py
pytest recipe/context_management/test_agent_loop_with_context_management.py
```
