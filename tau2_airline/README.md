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
| `example/` | training launcher + a local user-simulator server (see reproduction caveat below) |

It plugs in through verl's **public AgentLoop extension point** — no verl source changes.

The warm start and teacher data are public. The original tau2 revision still needs to be
recorded, and the corrected launcher needs a GPU rerun. The reported runs require this warm start:

- **warm start:** [`yuyu0529nya/qwen2.5-7b-tau2-airline-sft-lora`](https://huggingface.co/yuyu0529nya/qwen2.5-7b-tau2-airline-sft-lora) (LoRA, Apache-2.0)
- **teacher data it was distilled from:** [`yuyu0529nya/tau2-airline-deepseek-distill`](https://huggingface.co/datasets/yuyu0529nya/tau2-airline-deepseek-distill) (326 DeepSeek V4 Flash trajectories, Apache-2.0)

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt). Pinned to `ad2e3c2` (2026-07-10), the commit every
number below was produced against. Use this pin: current `main` (checked at `6093e007`) has removed the
`apply_chat_template` helper this loop calls. Supporting it requires a separate port.

## Results

Held-out `BINARY mean@4` (20 held-out airline tasks, `VAL_TEMP=0.5`, n=4), `lr=1e-4`, 20 steps:

| seed | val@0 | val@5 | val@10 | val@15 | val@20 |
|---|---|---|---|---|---|
| 42 | 0.275 | 0.525 | 0.5375 | 0.5625 | **0.5625** |
| 123 | 0.375 | 0.4875 | 0.55 | 0.5375 | **0.55** |

**2-seed mean ± std = 0.556 ± 0.01.**

**Measured evaluation noise** (same checkpoint, same eval, 6 repeats of `val@0`):
`0.2375, 0.2875, 0.35, 0.35, 0.2375, 0.3375` → **0.30 ± 0.05**. These repeats describe baseline evaluation variability; they are not a formal significance
test for the training gain. The evaluation covers only 20 held-out tasks, and the two training
seeds are insufficient to establish broad generalization.

### ⚠️ Two things you need to know before you try to reproduce this

**1. You must start from the distilled SFT warm start, not raw `Qwen2.5-7B-Instruct`.**
In the reported raw-base runs, learning was flat. All-same-reward groups supply no GRPO
outcome contrast, and the observed successful base rollouts lacked write-tool behavior needed
by some held-out tasks. These observations motivated the distilled warm start; they do not
establish that raw-base RL can never learn the task.
Both the warm start and the teacher data it came from are published (links at the top). Merge the
adapter into the base and point `POLICY_MODEL` at the merged weights — that is exactly the
`step 0` of the table above.

**2. Match the reported learning rate.**
The reported `lr=4e-6` / `2e-5` runs were flat; the table above uses `LR=1e-4`, which
is now the launcher default. Low gradient norm or clipping frequency alone does not establish
that an optimizer learning rate is too small. Treat this as an observed hyperparameter result.

## Setup

```bash
pip install verl@git+https://github.com/verl-project/verl.git@ad2e3c272ee95fc5627c5007af59b5d25100be1a
# tau2-bench is the repository name; install its package from a source checkout.
git clone https://github.com/sierra-research/tau2-bench.git ./third_party/tau2-bench
pip install -e ./third_party/tau2-bench
export TAU2_DATA_DIR="$(pwd)/third_party/tau2-bench/data"
# Record `git -C ./third_party/tau2-bench rev-parse HEAD` with each run.
# This PR does not record the original tau2 commit; exact historical data/API
# compatibility therefore remains unverified.

python data_prep_airline.py --n_val 20   # 30 train / 20 test for the 50-task airline dataset
python test_tau2_loop_offline.py   # CPU-only sanity check, see Tests below
```

### Warm start (required — see the note above)

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto")
merged = PeftModel.from_pretrained(
    base, "yuyu0529nya/qwen2.5-7b-tau2-airline-sft-lora"
).merge_and_unload()
merged.save_pretrained("./models/qwen25-7b-sft-airline")   # -> POLICY_MODEL
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct").save_pretrained("./models/qwen25-7b-sft-airline")
```

### User simulator: pick one

The user simulator drives the other half of every dialogue, so it dominates both cost and
determinism.

* **Local (no API key, fully offline):** `bash example/serve_usersim_7b.sh` serves a local 7B on a
  second GPU; use `USERSIM_GPU=1` for the server and `GPU=0` for training, then point
  `TAU2_USER_API_BASE` at it. Temperature 0 reduces sampling noise; it does not guarantee bitwise determinism.
* **Hosted:** set `USERSIM_BACKEND=openrouter` + `TAU2_USER_LLM`, and export `OPENROUTER_API_KEY` in the shell before launching.
  The launcher preserves the hosted model and skips the local `/models` probe. The numbers above used `openrouter/meta-llama/llama-3.3-70b-instruct`.
  Frees the second GPU, costs money, adds provider-side nondeterminism.

**The user simulator is part of your experiment.** Changing it changes your numbers; keep it fixed
across any comparison you care about.

The setup commands above are run from `tau2_airline/`. The model directory must
contain both merged weights and tokenizer files. `SEED` controls data shuffling and
rollout sampling; neither the hosted API nor GPU scheduling is made deterministic by it.
The launcher defaults now match the documented 20-step, 4-trial evaluation protocol;
the corrected entrypoint has not yet been rerun on GPU.

## Run

```bash
GPU=0 LR=1e-4 TRAIN_BS=24 ROLLOUT_N=12 MAX_STEPS=20 TEST_FREQ=5 \
SEED=42 VAL_N=4 VAL_TEMP=0.5 \
POLICY_MODEL=./models/qwen25-7b-sft-airline \
bash example/run_tau2_grpo_7b.sh
```

Everything is env-var driven; see the header of `example/run_tau2_grpo_7b.sh`. Notable knobs:
`LR`, `TRAIN_BS`, `ROLLOUT_N`, `PPO_MINI`, `MAX_STEPS`, `SEED`, `VAL_N`, `VAL_TEMP`, `TEST_FREQ`, `GPU_UTIL`, `MAX_MODEL_LEN`,
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
python -m pytest -q test_launchers_offline.py test_failure_paths_offline.py
```

- **A — bridge plumbing:** seeded messages, system prompt, tool schemas, tool execution + id match, user-simulator stop signal.
- **B — reward fidelity:** premature termination → 0.0; replaying the gold actions → 1.0 with the expected DB/COMMUNICATE breakdown.
- **C — concurrency isolation:** 8 concurrent trajectories, all 8 DBs distinct.

The failure-path suite executes the recipe loop with CPU dependency/service stubs. It checks
that infrastructure failures, empty generations and missing/misaligned log-probs raise instead
of producing synthetic reward-zero examples; normal turn-limit outcomes remain valid. Successful
user/tool turns keep masks and log-probs aligned. These checks do not replace GPU/Ray validation.

Unexpected infrastructure failures abort the current rollout batch with a plain RuntimeError;
the original traceback is logged in the worker. There is no automatic retry in the recipe. This
avoids training on fabricated EOS tokens or labeling a simulator outage as policy failure.
Resolve the underlying service error before restarting the run.

## Honest limits

- **20 held-out tasks.** One task is worth 5 percentage points. Treat single-point moves as noise; the ±0.05 band above was measured, not assumed.
- **30 training tasks**, `TRAIN_BS=24` → ~1 optimizer step per epoch. This is a small-data regime.
- **2 training seeds.** The gains appear in both runs, but these runs do not provide tight uncertainty estimates.
- The reported numbers use a hosted 70B user simulator with provider fallback enabled, so they are not bit-reproducible; the local-usersim path reduces provider variability.
