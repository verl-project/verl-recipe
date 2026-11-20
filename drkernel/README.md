# DR.Kernel — Triton kernel-generation RL on NPU

This recipe ports **DR.Kernel** ([arXiv:2602.05885](https://arxiv.org/abs/2602.05885)) onto
**verl** and runs it on **Ascend NPU**.

- **Entry point:** `recipe/drkernel/main.py` (`python -m recipe.drkernel.main`)
- **Reward server:** upstream [KernelGYM](https://github.com/hkust-nlp/KernelGYM)
  patched for NPU — the recipe ships only the adaptation (patch + new toolkits) in
  `recipe/drkernel/kernelgym_npu/`. See its own [README](kernelgym_npu/README.md).
- **Advantage estimator:** TRLOO (Turn-level REINFORCE Leave-One-Out) ships with the
  recipe in [`algos.py`](algos.py) and self-registers as `algorithm.adv_estimator=trloo`.

---

## Required `verl` version

Pinned in [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt): upstream
[`verl-project/verl`](https://github.com/verl-project/verl) @ commit
`284d0388`. The recipe builds on
`verl.experimental` (`fully_async_policy`, `reward_loop`, `separation`,
`agent_loop`).

---

## 1. Installation

Follow the instruction in environment setup (base container and framework stack, verl) in this guide: **[`INSTALL.md`](INSTALL.md)**.

---

## 2. Running a training experiment

A training run has **two** pieces that come up in order:

1. the **KernelGYM reward server** (evaluates generated kernels, returns
   correctness + speedup), and
2. the **verl async-PPO trainer** (`recipe.drkernel.main`).

The tracked, canonical launcher
[`scripts/rl/drkernel_kernel_train_native_30b.sh`](scripts/rl/drkernel_kernel_train_native_30b.sh)
wires both together — a **Qwen3-30B-A3B (MoE)** run with MRS on the
2k-difficulty-filtered training set, using the `sandbox_v3` toolkit and `coach`
feedback. Use it as the worked example below.

> An **8B** dense variant is also tracked at
> [`scripts/rl/drkernel_kernel_train_native_8b.sh`](scripts/rl/drkernel_kernel_train_native_8b.sh)
> (same wiring, smaller model / shorter response budget). The knobs below apply
> to both — only the defaults differ.

### 2.1 Prerequisites before launching

- **Model** at `MODEL_PATH` (default a distilled `Qwen3-30B-A3B` checkpoint;
  `/home/model/Qwen3-8B` for the 8B script).
- **Datasets** — train/val parquet files at `TRAIN_FILES` / `VAL_FILES`.
- **KernelGYM server reachable** at `KERNELGYM_SERVER_URL`
  (default `http://127.0.0.1:8002`). Point it at the host/port where
  the patched KernelGYM checkout (`recipe/drkernel/kernelgym_npu/KernelGYM`, created
  by `setup_kernelgym.sh`) is serving — e.g. a remote node's IP for a multi-node setup.
- **NPUs** — the 30B script defaults to `N_GPUS_ROLLOUT=12` rollout +
  `N_GPUS_TRAINING=12` training NPUs, plus the cards the KernelGYM server needs;
  size `NNODES` to your cluster. (The 8B script defaults to 10 rollout + 16 training + 6 KernelGYM.)

### 2.2 Launch procedure (head-node card split: training on 0–7, KernelGYM on 8–15)

The training script auto-starts KernelGYM, so the layout is just: pin Ray (trainer
+ rollout) to the **first 8 cards**, point KernelGYM's `.env` at the **last 8**, and
launch the script from a **new shell that does not carry the trainer's mask** — so
the KernelGYM it spawns lands on cards 8–15 instead of inheriting cards 0–7.

1. **Ray head on the first 8 cards:**
   ```bash
   export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   ray start --head --port=6379          # add `ray start --address=<head-ip>:6379` on other nodes
   ```

2. **Edit `recipe/drkernel/kernelgym_npu/KernelGYM/.env`** (copy from `.env.example`;
   the checkout is created by `setup_kernelgym.sh`, see [INSTALL.md](INSTALL.md)):
   ```ini
   API_HOST=<head-node IP>               # 127.0.0.1 for single-node
   GPU_DEVICES=[8,9,10,11,12,13,14,15]   # the last 8 cards
   ```

3. **Launch the script in a NEW shell** (one *without* `ASCEND_RT_VISIBLE_DEVICES`
   set, so the auto-started KernelGYM can reach cards 8–15):
   ```bash
   export KERNELGYM_SERVER_URL=http://<head-node IP>:8002
   bash recipe/drkernel/scripts/rl/drkernel_kernel_train_native_30b.sh
   ```

### 2.3 Key config knobs (30B defaults)

| Group | Knob (env var) | Default | Meaning |
|---|---|---|---|
| Algorithm | `ALGORITHM` | `trloo` | advantage estimator |
| MRS | `ROLLOUT_RS` / `ROLLOUT_RS_THRESHOLD` | `seq_mean_k1` / `0.997_1.003` | paper geometric-mean rejection |
| MRS | `ROLLOUT_TOKEN_VETO_THRESHOLD` | `1e-4` | per-token catastrophic veto |
| MRS | `ROLLOUT_IS` / `ROLLOUT_IS_THRESHOLD` | `token` / `2.0` | token-level truncated IS |
| Rollout | `ROLLOUT_N` | `15` | samples per prompt |
| Rollout | `MAX_TURN` | `3` | multi-turn turns |
| Lengths | `MAX_PROMPT_LENGTH` / `MAX_RESPONSE_LENGTH` | `4096` / `49152` | token budgets |
| Reward | `REWARD_FUNC_NAME` | `calculate_reward_weighted` | correctness + speedup blend |
| Reward | `INIT_CORRECT_WEIGHT` / `INIT_PERFORMANCE_WEIGHT` | `1.0` / `1.0` | reward weights |
| KernelGYM | `KERNELGYM_TOOLKIT` | `sandbox_v3` | evaluation toolkit (§4) |
| KernelGYM | `KERNELGYM_FEEDBACK_MODE` | `coach` | per-turn feedback rendering (§5) |
| Async | `STALENESS` / `SYNC_STEP` / `REQUIRE_BATCHES` | `0.1` / `4` / `4` | async-PPO pacing |
| Compute | `N_GPUS_ROLLOUT` / `N_GPUS_TRAINING` | `12` / `12` | NPU split |
| Compute | `GEN_TP` / `ULYSSES_SP` | `2` / `4` | rollout TP / training SP |

### 2.4 Outputs

- **Training log:** `logs/${PROJECT_NAME}/run_${EXP_NAME}.log` (teed live).
- **Checkpoints:** `trainer.default_local_dir`
  (default `${REPO_ROOT}/checkpoints/${PROJECT_NAME}/${EXP_NAME}`; override with
  `DEFAULT_LOCAL_DIR`).
- **Rollout dumps:** `./rollout_dump/...` and `./rollout_validation_dump/...`.
- **Metrics:** console + TensorBoard (`trainer.logger=["console","tensorboard"]`).

---

## 3. Running validation only (with Pass@k)

To score a trained checkpoint without launching a full training run, use the
tracked validation launcher
[`scripts/validation/drkernel_validate_native_30b.sh`](scripts/validation/drkernel_validate_native_30b.sh),
which adds the unbiased Pass@k and incremental/resumable knobs described below.
It starts KernelGYM and runs `recipe.drkernel.main_validate` once (rollout-only —
no trainer pool, no parameter sync), reusing the same multi-turn /
kernel_async-reward path as training so the `val-core` / `val-aux` metrics line up
with the curves training logs at `test_freq`.

```bash
MODEL_PATH=/path/to/merged_hf_checkpoint \
VAL_FILES=/path/to/validation.parquet \
    bash recipe/drkernel/scripts/validation/drkernel_validate_native_30b.sh
```

- **`MODEL_PATH` must be a merged HF checkpoint** (e.g. produced from a training
  checkpoint with `merge_to_hf.sh` of verl), not a raw FSDP checkpoint dir.
- **Real, unbiased Pass@k.** Set `VAL_N` stochastic rollouts per prompt
  (default `8`, requires `VAL_DO_SAMPLE=True`) and `VAL_PASS_AT_K` to the
  comma-separated `k` values you want (default `1,8`). Pass@k is the **unbiased
  estimator of Chen et al. 2021** — `pass@k = 1 − C(n−c, k)/C(n, k)` over the
  `n=VAL_N` generations. (`k=1` reduces to
  the mean success rate `c/n`; `k≥VAL_N` coincides with the naive Pass@N.)
- **Incremental & resumable.** `VAL_INCREMENTAL=1` validates in chunks of
  `VAL_CHUNK_SIZE` prompts; `VAL_RESUME=1` skips prompts already present in the
  dump JSONL (so an interrupted run picks up where it left off) instead of
  truncating.
- **Resource pools** are rollout-only: the trainer pool is given
  `n_gpus_per_node=0`, so all NPUs go to rollout (`N_GPUS_ROLLOUT`).
- **Outputs:** metrics JSON (incl. per-prompt Pass@k) + TensorBoard under
  `outputs/${PROJECT_NAME}/${EXP_NAME}`, per-rollout dumps under
  `rollout_validation_dump/...`, and a teed log at
  `logs/${PROJECT_NAME}/val_${EXP_NAME}.log`.

---

## 4. KernelGYM evaluation toolkits

The reward server ships two evaluation toolkits; this recipe documents and
supports the **two** below. The toolkit is selected by setting `KERNELGYM_TOOLKIT` vairable in the training scripts,
which the launcher forwards to **both** the trainer reward
(`reward_model.reward_kwargs.toolkit`) and the per-turn agent-loop feedback
(`config/agent_loop_config.yaml: kernel_eval.toolkit`) — keep them on the same
toolkit so the reward and the feedback the model reads agree. The env var wins
over the YAML defaults.

| Toolkit | `KERNELGYM_TOOLKIT` | Description |
|---|---|---|
| KernelBench (original) | `kernelbench` | The original DR.Kernel paper evaluation path: KernelBench correctness + speedup with operator-level profiling. |
| Sandbox v3 | `sandbox_v3` | NPU-kernel evaluation path: multi-shape correctness, `allclose` precision, **AST-based Triton-implementation (decoy) validation**, and operator-level NPU profiling. |

Switch toolkits per run via the env var, e.g. evaluate with the original
KernelBench toolkit instead of the default `sandbox_v3`:

```bash
KERNELGYM_TOOLKIT=kernelbench bash recipe/drkernel/scripts/rl/drkernel_kernel_train_native_30b.sh
```

The main practical difference: `kernelbench` matches the original paper setup,
while `sandbox_v3` adds AST-level decoy/reward-hacking detection (a kernel that
"passes" without actually Tritonizing the target op is rejected) on the NPU
evaluation path.

> **Reference:** `sandbox_v3` is based on ([AscendOpGenAgent](https://github.com/Just-it/AscendOpGenAgent/tree/br_430/skills/triton/kernel-verifier))
---

## 5. Per-turn feedback modes

Between turns, the KernelGYM result for the model's previous kernel is rendered
into the user message via `{feedback}`. The rendering is selected with
`KERNELGYM_FEEDBACK_MODE` (env var wins over `config/agent_loop_config.yaml:
kernel_eval.feedback_mode`; default `full_json`).

| Mode | `KERNELGYM_FEEDBACK_MODE` | Description |
|---|---|---|
| Original DR.Kernel | `full_json` | JSON-dump of the full result dict, mirroring the original DR.Kernel pipeline. Faithful baseline. |
| Drop keys | `drop_keys` | Same as `full_json`, but strips a configured set of verbose keys (IDs, timestamps, profiling internals) that bloat the prompt without informing the next turn. |
| Coach | `coach` | Natural-language, model-facing coaching: verdict, performance, per-operation NPU device-time broken down Triton-vs-PyTorch, and a next step. Helps smaller models act on the multi-operation signal. |

**Choosing the dropped keys.** In `drop_keys` mode the stripped set comes from
`KERNELGYM_FEEDBACK_DROP_KEYS` (comma-separated env var) → YAML
`kernel_eval.feedback_drop_keys` (list) → the built-in default
`_FEEDBACK_DROP_KEYS` in [`agent/kernel_eval.py`](agent/kernel_eval.py)
(metadata, kernel/total counts, profiling timing fields, task IDs, timestamps).
For example, keep only correctness/speedup-relevant fields:

```bash
KERNELGYM_FEEDBACK_MODE=drop_keys \
KERNELGYM_FEEDBACK_DROP_KEYS=metadata,profiling,task_id,submitted_at,completed_at \
    bash recipe/drkernel/scripts/rl/drkernel_kernel_train_native_30b.sh
```

The config default is `full_json` (the original DR.Kernel rendering). The
tracked launchers override it per run: the 30B scripts use `coach`, the 8B
scripts use `drop_keys`.
