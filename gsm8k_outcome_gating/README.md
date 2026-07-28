# gsm8k_outcome_gating — why the group-filter *metric* is safety-critical under shaped rewards

A small, fully-reproducible study of **when group filtering actually matters in GRPO**,
run on a single RTX 5090 with Qwen2.5-1.5B-Instruct + LoRA on GSM8K (~50 min per arm).

**TL;DR** — `filter_groups.metric` decides whether dynamic sampling protects you or not.
Every official DAPO script in this repo sets `filter_groups_metric=acc`, and that choice
is load-bearing: filtering on the **binary outcome** stays safe under reward shaping,
while filtering on the **shaped training reward** (`score` / `seq_reward` — the natural-looking
choice once you have a shaped reward) silently stops dropping all-fail groups and lets a
phantom gradient destroy the policy in 40 steps:

| held-out EM @ step 40 | no filtering (plain GRPO) | filter on **shaped score** | filter on **binary outcome** (= `metric=acc`) |
|---|---|---|---|
| shaped reward, λ=0.30 (n=3) | 0.102 ± 0.126 | 0.037 ± 0.007 | **0.753 ± 0.006** |

All three arms start from the same val@0 ≈ 0.715. Same model, same data, same
hyperparameters — the only difference is which groups' advantage is removed. **~19× on a
config field**, plus the dose-response and direction-asymmetry boundaries below.

## 1. The phantom

Group-relative estimators (GRPO/DAPO/GSPO) normalize per group:
`A = (r − mean(r)) / (std(r) + ε)`. With a **binary** reward, an all-fail group has
`std = 0` and contributes nothing — which is why DAPO filters those groups out.

veRL implements that filter as *"drop groups whose `filter_groups.metric` is all-same"*
([`dapo_ray_trainer.py`](../dapo/dapo_ray_trainer.py)), and the metric is configurable.
The DAPO paper's criterion is accuracy-based, and every official script in
[`recipe/dapo`](../dapo) sets `filter_groups_metric=acc` — under that configuration the
filter is **outcome-based and stays correct even when the reward is shaped**, because
`acc` remains binary no matter what shaping does to `score`.

The failure mode this recipe isolates is what happens when the filter metric is the
**shaped training reward** instead (`score` / `seq_reward` — both are valid values of the
field, and reaching for "the reward I train on" is the natural move once shaping exists).
Then an all-fail group has `std > 0` (different partial credit per failure), the filter
**keeps** it, and GRPO normalizes the tiny shaping differences into full-size advantages.
That group's gradient says nothing about solving the task; it says *"fail the way the
shaping prefers."* We call this a **phantom advantage**.

So the contribution here is not a flaw in DAPO — it is a measurement of how much that one
config field is worth (~19×), why the safe choice is safe, and where its boundaries lie.
**Practical rule: any project that adds reward shaping must expose a binary correctness
signal decoupled from the shaped reward, and filter on that** — otherwise dynamic sampling
degrades silently, with no error and no warning.

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

| arm | drops (advantage → 0) | corresponds to |
|---|---|---|
| `none` | nothing | no dynamic sampling (plain GRPO control) |
| `std` | groups whose **shaped reward** std ≈ 0 | `filter_groups.metric = score` / `seq_reward` |
| `outcome` | groups whose **binary outcome** is all-same | `filter_groups.metric = acc` (every official DAPO script) |

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
  `val-aux/openai/gsm8k/outcome_binary/mean@1`), so outcome-based filtering never confuses
  "barely failed" with "passed".

## 3. Experiment 1 — three arms, three independent replicates (λ = 0.30)

| val EM | @0 | @10 | @20 | @30 | @40 | resp. len @40 (tokens) |
|---|---|---|---|---|---|---|
| no-filter (replicate 1) | 0.725 | 0.733 | 0.613 | 0.487 | **0.247** | 18 |
| shaped-score (replicate 1) | 0.717 | 0.701 | 0.531 | 0.267 | **0.044** | 3 |
| binary-outcome (replicate 1) | 0.716 | 0.731 | 0.746 | 0.771 | **0.755** | 244 |

Endpoints across all three replicates:

| val@40 | rep 1 | rep 2 | rep 3 | mean ± std |
|---|---|---|---|---|
| no-filter | 0.247 | 0.038 | 0.022 | 0.102 ± 0.126 |
| shaped-score | 0.044 | 0.031 | 0.037 | 0.037 ± 0.007 |
| **binary-outcome** | 0.755 | 0.758 | 0.747 | **0.753 ± 0.006** |

Mechanism, from the hook's own logs: while accuracy is healthy the shaped-score filter drops only
~2/32 degenerate groups per batch (the phantom groups all have std>0 and sail through),
whereas the binary-outcome filter drops 16–24/32 (all-pass + all-fail). The shaped-score filter only
starts firing *after* the collapse homogenizes outputs (identical 1–3 token answers →
std→0) — it removes the corpse, never the poison.

## 4. Experiment 2 — dose-response over λ

`GATE_LAMBDA` sweeps the phantom strength (0.30 anchor is the n=3 above; the new cells
are single runs):

| val@40 | λ=0.10 | λ=0.30 | λ=0.50 |
|---|---|---|---|
| no-filter | 0.055 | 0.102 ± 0.126 | **0.000** |
| shaped-score | 0.033 | 0.037 ± 0.007 | 0.021 |
| binary-outcome | 0.764 | 0.753 ± 0.006 | 0.736 |

- **No safe dose.** Even λ=0.10 fully collapses both baselines within 40 steps; λ only
  sets the collapse speed (mid-run EM orders monotonically with λ).
- **The binary-outcome filter is immune at every dose.** Its mild slope (0.764 → 0.736) is consistent
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
| no-filter | 0.102 ± 0.126 💥 | 0.736 ✅ |
| shaped-score | 0.037 ± 0.007 💥 | 0.746 ✅ |
| binary-outcome | 0.753 ± 0.006 ✅ | 0.754 ✅ |

Nobody collapses: mean length drifts up ~15% and then saturates, EM is untouched. The
asymmetry is informative: a phantom is lethal only when its shortcut **destroys task
structure** (truncation deletes the answer itself; padding leaves it intact, and in
mixed groups the +1.0 for a correct answer dwarfs the shaping term, so the phantom
never steers). Footnote: the binary-outcome arm shows the *tightest* length curve of the
three — outcome-based filtering suppresses even the harmless style drift.

## 6. Honesty & limitations

- **This is not a defect report against DAPO.** The DAPO paper's dynamic-sampling
  criterion is accuracy-based, and every official script here sets
  `filter_groups_metric=acc`, which is the safe configuration measured above. What this
  recipe quantifies is the cost of the *other* valid settings of that field once a shaped
  reward is in play, and the reason the safe one is safe.
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
| [gate_hook.py](gate_hook.py) | advantage-zeroing filter around `compute_advantage` (`GATE_MODE` selects the metric) |
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
