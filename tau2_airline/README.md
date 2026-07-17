# tau2_airline — multi-turn tool-agent RL on τ²-bench (airline domain)

GRPO over a **multi-turn, tool-calling customer-service agent** in
[τ²-bench](https://github.com/sierra-research/tau2-bench)'s `airline` domain, on a single GPU.

τ²-bench is a *dual-control* benchmark: the agent and a simulated user both act on a shared,
stateful environment (a reservations DB). That makes it a different beast from single-turn RLVR —
reward arrives only at the end of a whole dialogue, the environment mutates as the agent works, and
every rollout needs its **own** private copy of the world.

This recipe supplies the pieces verl does not have for that setting:

| file | what it does |
|---|---|
| `tau2_agent_loop.py` | the `AgentLoop` implementation: policy ↔ tools ↔ simulated-user turn loop, masked tool responses |
| `tau2_bridge.py` | per-trajectory τ²-bench env: seeded messages, tool execution, user simulator, terminal reward |
| `rollout_context.py` | `ContextVar` scoping so N concurrent rollouts never share env state |
| `data_prep_airline.py` | builds the train/test parquet from τ²-bench tasks (deterministic split) |
| `agent_loop_config.yaml` | registers the loop via verl's public `agent_loop_config_path` hook |
| `example/` | the exact launcher used for the reported runs + a local user-simulator server |

It plugs in through verl's **public AgentLoop extension point** — no verl source changes.

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt). Pinned to `ad2e3c2` (2026-07-10), the commit every
number below was produced against. Newer `main` is untested here — not known-broken, just untested.

## Results

Held-out `BINARY mean@4` (20 held-out airline tasks, `VAL_TEMP=0.5`, n=4), `lr=1e-4`, 20 steps:

| seed | val@0 | val@5 | val@10 | val@15 | val@20 |
|---|---|---|---|---|---|
| 42 | 0.275 | 0.525 | 0.5375 | 0.5625 | **0.5625** |
| 123 | 0.375 | 0.4875 | 0.55 | 0.5375 | **0.55** |

**2-seed mean ± std = 0.556 ± 0.01.**

**Measured evaluation noise** (same checkpoint, same eval, 6 repeats of `val@0`):
`0.2375, 0.2875, 0.35, 0.35, 0.2375, 0.3375` → **0.30 ± 0.05**. The endpoint sits ~5σ above that
band, and both seeds land inside 0.0125 of each other.

### ⚠️ Two things you need to know before you try to reproduce this

**1. Those runs start from a teacher-distilled SFT checkpoint, which is not distributed here.**
From raw `Qwen2.5-7B-Instruct`, GRPO on this task is **null**, and that is a property of the task,
not a bug: base success is ~20%, so most groups come back all-fail → zero intra-group variance →
the group-relative advantage is ~0 → no gradient. You need a warm start that already exhibits the
target skills before GRPO has anything to sharpen.

**2. The default learning rate is not the one that works.**
At `lr=4e-6` / `2e-5` the curve is flat and looks like a structural problem. It is not — it is just
too small: `grad_norm ≈ 0.05` against a clip threshold of 1.0 (**20× of headroom**), with
`pg_clipfrac ≈ 0.001`, i.e. clipping never engages. `LR=1e-4` is what produces the table above. If
your curve is flat, check `grad_norm` before you redesign anything.

## Setup

```bash
pip install verl@git+https://github.com/verl-project/verl.git@ad2e3c272ee95fc5627c5007af59b5d25100be1a
pip install tau2-bench        # or install from source: https://github.com/sierra-research/tau2-bench

python data_prep_airline.py   # -> train.parquet / test.parquet
python test_tau2_loop_offline.py   # CPU-only sanity check, see Tests below
```

### User simulator: pick one

The user simulator drives the other half of every dialogue, so it dominates both cost and
determinism.

* **Local (no API key, fully offline):** `bash example/serve_usersim_7b.sh` serves a local 7B on a
  second GPU; point `TAU2_USER_API_BASE` at it. Deterministic at `TAU2_USER_TEMPERATURE=0`.
* **Hosted:** set `USERSIM_BACKEND=openrouter` + `TAU2_USER_LLM`, and put your key in
  `$ROOT/.tau2_env`. The numbers above used `openrouter/meta-llama/llama-3.3-70b-instruct`.
  Frees the second GPU, costs money, adds provider-side nondeterminism.

**The user simulator is part of your experiment.** Changing it changes your numbers; keep it fixed
across any comparison you care about.

## Run

```bash
LR=1e-4 TRAIN_BS=24 ROLLOUT_N=12 EPOCHS=20 TEST_FREQ=5 \
POLICY_MODEL=/path/to/your/sft-checkpoint \
bash example/run_tau2_grpo_7b.sh
```

Everything is env-var driven; see the header of `example/run_tau2_grpo_7b.sh`. Notable knobs:
`LR`, `TRAIN_BS`, `ROLLOUT_N`, `PPO_MINI`, `EPOCHS`, `TEST_FREQ`, `GPU_UTIL`, `MAX_MODEL_LEN`,
`MAX_PROMPT`, `MAX_RESP`, `PARAM_OFFLOAD`, `POLICY_MODEL`, `EXP_NAME`.

## Design notes

**Per-rollout environment isolation.** τ²-bench's env is stateful and its default handles are
process-global, but verl drives many rollouts concurrently in one worker. `rollout_context.py`
scopes each trajectory's env to a `ContextVar`, so trajectory *i* can never observe or mutate
trajectory *j*'s DB. `test_tau2_loop_offline.py::Test C` drives 8 concurrent trajectories and
asserts all 8 DBs stay distinct — this is the failure mode most likely to silently corrupt a
multi-turn agent RL run, and it is silent precisely because nothing crashes.

**Reward is the unmodified τ²-bench verdict.** The evaluator checks both final DB state and what
the agent communicated. A dialogue that hits `MAX_STEPS` scores 0 even if the DB looks right — the
task is not done until the user is done. `extra_fields.outcome_binary` carries the raw, unshaped
success label separately, so group-level filtering and outcome metrics stay correct if you later
add reward shaping.

**Only assistant-generated tokens get gradient.** Tool responses *and* simulated-user turns are
appended with `response_mask = 0` — they are context, not prediction targets. Getting this wrong
trains the policy to imitate the user simulator.

**Memory on one GPU.** LoRA (rank 32, `all-linear`) + colocated vLLM + `use_fused_kernels=True`
(chunked CE, so the `[seq × vocab]` logits are never materialized — this, not
`expandable_segments`, is what fixes long-sequence log-prob OOM; see the launcher header re
pytorch#147851). `use_remove_padding=False` with `attn_implementation=sdpa`, so no flash-attn build
is required.

## Tests

```bash
python test_tau2_loop_offline.py   # CPU only, no GPU, no API key
```

- **A — bridge plumbing:** seeded messages, system prompt, tool schemas, tool execution + id match, user-simulator stop signal.
- **B — reward fidelity:** premature termination → 0.0; replaying the gold actions → 1.0 with the expected DB/COMMUNICATE breakdown.
- **C — concurrency isolation:** 8 concurrent trajectories, all 8 DBs distinct.

GPU is needed only for policy token generation and the local user-simulator server.

## Honest limits

- **20 held-out tasks.** One task is worth 5 percentage points. Treat single-point moves as noise; the ±0.05 band above was measured, not assumed.
- **30 training tasks**, `TRAIN_BS=24` → ~1 optimizer step per epoch. This is a small-data regime.
- **2 seeds**, not 5. Enough to show 0.5625 was not a lucky draw; not enough for tight error bars.
- The reported numbers use a hosted 70B user simulator with provider fallback enabled, so they are not bit-reproducible; the local-usersim path is the deterministic one.
