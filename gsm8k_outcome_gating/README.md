# gsm8k_outcome_gating — phantom advantages under shaped rewards, and the gate that kills them

A small, fully-reproducible study of **when group filtering actually matters in GRPO**,
run on a single RTX 5090 with Qwen2.5-1.5B-Instruct + LoRA on GSM8K (~50 min per arm).

**TL;DR** — under a shaped reward with a mild length term, plain GRPO and a DAPO-style
std filter both collapse to near-zero accuracy within 40 steps; a **binary-outcome gate**
(zero the advantage of groups whose *raw outcome* is all-same) keeps training healthy at
every phantom strength tested:

| held-out EM @ step 40 | plain GRPO | DAPO-style std gate | **outcome gate** |
|---|---|---|---|
| shaped reward, λ=0.30 (n=3) | 0.102 ± 0.126 | 0.037 ± 0.007 | **0.753 ± 0.006** |

All three arms start from the same val@0 ≈ 0.715. Same model, same data, same
hyperparameters — the only difference is which groups' advantage is removed.

## 1. The phantom

Group-relative estimators (GRPO/DAPO/GSPO) normalize per group:
`A = (r − mean(r)) / (std(r) + ε)`. With a **binary** reward, an all-fail group has
`std = 0` and contributes nothing. DAPO's dynamic sampling therefore filters groups by
**reward std**.

Under a **shaped** reward that is exactly the DAPO assumption breaks: an all-fail group
has `std > 0` (different partial credit per failure), so the std filter **keeps** it —
and GRPO then normalizes the tiny shaping differences into full-size advantages. The
group's gradient says nothing about solving the task; it says *"fail the way the shaping
prefers."* We call this a **phantom advantage**.

The shaped reward here makes the phantom concrete and tunable:

```
reward = outcome(0/1) − λ · min(len_chars / 512, 1)        # GATE_PHANTOM=short (default)
```

Among failures the shortest wrong answer scores best — "give up fast". This reproduces a
failure mode we first hit in a search-agent setup where answer length collapsed from ~24
to ~7 characters; here it is dialed in with a single knob λ.

## 2. Design: a confound-free comparison

Three arms, selected by `GATE_MODE`, all inside a wrapper around verl's
`compute_advantage` ([gate_hook.py](gate_hook.py)):

| arm | drops (advantage → 0) | rationale |
|---|---|---|
| `none` | nothing | plain GRPO control |
| `std` | groups with reward std ≈ 0 | DAPO-style dynamic-sampling criterion |
| `outcome` | groups whose **raw outcome** is all-same | the gate under test |

Two design points that matter:

- **Advantage-zeroing, not mask-zeroing.** Zeroing `response_mask` rows shrinks the
  token-mean loss denominator, which multiplies the loss by `1/live_frac` — and Adam's
  scale invariance then largely cancels it. (We measured this separately: under *binary*
  rewards, mask-drop gating is a no-op that merely disguises a learning-rate change.)
  Zeroing advantages leaves the denominator untouched, so the arms differ **only** in
  which groups' gradient is removed.
- **The gate groups on the raw outcome, not the shaped scalar.** The reward function
  returns `{"score": shaped, "outcome_binary": outcome, "acc": outcome}`; the extra keys
  ride through verl's `reward_extra_info` into `non_tensor_batch` (visible in logs as
  `val-aux/openai/gsm8k/outcome_binary/mean@1`), so the outcome gate never confuses
  "barely failed" with "passed".

## 3. Experiment 1 — three arms, three independent replicates (λ = 0.30)

| val EM | @0 | @10 | @20 | @30 | @40 | resp. len @40 (tokens) |
|---|---|---|---|---|---|---|
| none (replicate 1) | 0.725 | 0.733 | 0.613 | 0.487 | **0.247** | 18 |
| std (replicate 1) | 0.717 | 0.701 | 0.531 | 0.267 | **0.044** | 3 |
| outcome (replicate 1) | 0.716 | 0.731 | 0.746 | 0.771 | **0.755** | 244 |

Endpoints across all three replicates:

| val@40 | rep 1 | rep 2 | rep 3 | mean ± std |
|---|---|---|---|---|
| none | 0.247 | 0.038 | 0.022 | 0.102 ± 0.126 |
| std | 0.044 | 0.031 | 0.037 | 0.037 ± 0.007 |
| **outcome** | 0.755 | 0.758 | 0.747 | **0.753 ± 0.006** |

Mechanism, from the gate's own logs: while accuracy is healthy the std gate drops only
~2/32 degenerate groups per batch (the phantom groups all have std>0 and sail through),
whereas the outcome gate drops 16–24/32 (all-pass + all-fail). The std filter only
starts firing *after* the collapse homogenizes outputs (identical 1–3 token answers →
std→0) — it removes the corpse, never the poison.

## 4. Experiment 2 — dose-response over λ

`GATE_LAMBDA` sweeps the phantom strength (0.30 anchor is the n=3 above; the new cells
are single runs):

| val@40 | λ=0.10 | λ=0.30 | λ=0.50 |
|---|---|---|---|
| none | 0.055 | 0.102 ± 0.126 | **0.000** |
| std | 0.033 | 0.037 ± 0.007 | 0.021 |
| outcome | 0.764 | 0.753 ± 0.006 | 0.736 |

- **No safe dose.** Even λ=0.10 fully collapses both baselines within 40 steps; λ only
  sets the collapse speed (mid-run EM orders monotonically with λ).
- **The gate is immune at every dose.** Its mild slope (0.764 → 0.736) is consistent
  with the residual, *legitimate* length penalty that shaping applies to correct answers
  inside live groups.
- At λ=0.50 plain GRPO reaches EM 0.000 with mean response length **1.0 token** — the
  policy learns that shutting up immediately is optimal.
- Plumbing canary: at step 1, `critic/rewards/min` equals exactly −λ (−0.5 / −0.3),
  proving the env knob reaches the Ray reward workers.

## 5. Experiment 3 — reverse phantom (the Goodhart asymmetry)

`GATE_PHANTOM=long` flips the length term so the **longest** wrong answer scores best
(λ=0.30, single runs):

| val@40 | phantom=short | phantom=long |
|---|---|---|
| none | 0.102 ± 0.126 💥 | 0.736 ✅ |
| std | 0.037 ± 0.007 💥 | 0.746 ✅ |
| outcome | 0.753 ± 0.006 ✅ | 0.754 ✅ |

Nobody collapses: mean length drifts up ~15% and then saturates, EM is untouched. The
asymmetry is informative: a phantom is lethal only when its shortcut **destroys task
structure** (truncation deletes the answer itself; padding leaves it intact, and in
mixed groups the +1.0 for a correct answer dwarfs the shaping term, so the phantom
never steers). Footnote: the outcome arm shows the *tightest* length curve of the
three — the gate suppresses even the harmless style drift.

## 6. Honesty & limitations

- **"Independent replicates", not "seeds".** The three Experiment-1 runs used identical
  configs (verl default seed) and differ through rollout/scheduling nondeterminism.
  That is exactly the run-to-run variance that plagues n=1 RL A/Bs — the effect
  (gap ≥ 0.65) dwarfs it (replicate std ≤ 0.13). The launcher now supports true seed
  injection (`REALSEED=…` sets `rollout.seed` + `data.seed`) for stricter runs.
- Experiment 2/3 off-anchor cells are **n=1**; read them as trends anchored by the n=3
  column, not as precise point estimates.
- One model (Qwen2.5-1.5B-Instruct, LoRA r=32), one task (GSM8K), 40 steps. The claim
  is mechanistic (which groups carry the phantom gradient and who drops them), not a
  leaderboard result.
- EM uses a deliberately forgiving extractor (`####`, `\boxed{}`, "answer is", last
  number) so the collapse numbers measure lost ability, not lost formatting.

## 7. Reproduce

```bash
# 0) pinned verl (see REQUIRED_VERL.txt), plus a venv with vllm 0.11.0 / torch 2.8.0
pip install "verl @ git+https://github.com/verl-project/verl.git@e52747a403f55044578d9435069825f949b549bf"

# 1) offline sanity (no GPU needed)
python test_gate_offline.py

# 2) wire the gate into the venv (reversible; inert without GATE_* env vars)
./install_gate.sh /path/to/venv/bin/python

# 3) data + model under $GATE_ROOT
#    - GSM8K parquet via verl's examples/data_preprocess/gsm8k.py -> $GATE_ROOT/data/gsm8k/
#    - Qwen2.5-1.5B-Instruct              -> $GATE_ROOT/models/qwen25-1.5b-instruct

# 4) the three experiments (each arm ~50 min on one RTX 5090)
GATE_ROOT=... VERL_VENV=... ./run_gate_chain.sh      # Exp 1 (repeat with TAG=r2, r3)
GATE_ROOT=... VERL_VENV=... ./run_lambda_chain.sh    # Exp 2
GATE_ROOT=... VERL_VENV=... ./run_phantom_chain.sh   # Exp 3
```

Per-arm curves land in `$GATE_ROOT/gate_results/*.md`; raw logs in `$GATE_ROOT/*.log`
(`grep -aoE "acc[^ ]*mean@1:np.float64\([0-9.]+\)"` extracts the EM curve).

## Files

| file | role |
|---|---|
| [gate_hook.py](gate_hook.py) | advantage-zeroing gate around `compute_advantage` (`GATE_MODE`) |
| [gate_shaped_reward.py](gate_shaped_reward.py) | shaped GSM8K reward (`GATE_LAMBDA`, `GATE_PHANTOM`) |
| [sitecustomize.py](sitecustomize.py) | loads the hook inside every Ray worker |
| [install_gate.sh](install_gate.sh) | one-shot wiring (.pth + built-in reward branch), documented + reversible |
| [run_gate_arm.sh](run_gate_arm.sh) | single-arm launcher (all hydra flags) |
| [run_gate_chain.sh](run_gate_chain.sh) / [run_lambda_chain.sh](run_lambda_chain.sh) / [run_phantom_chain.sh](run_phantom_chain.sh) | Experiments 1 / 2 / 3 |
| [test_gate_offline.py](test_gate_offline.py) | no-GPU tests for partitioning, zeroing, both phantom directions |
| [REQUIRED_VERL.txt](REQUIRED_VERL.txt) | pinned verl commit + stack versions |

### Required `verl` version

See [REQUIRED_VERL.txt](REQUIRED_VERL.txt) — `MODE=pinned_commit` at the exact commit
every number above was produced against.
