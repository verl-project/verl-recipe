# Recipe: Partial Rollout

English | [简体中文](README_zh.md)

A partial-rollout pipeline for **synchronous RL training**, designed to reclaim GPU bubbles caused by **long-tail response lengths** via sample supplementation and mid-generation interruption with cross-step resume.

> ⚠️ **Don't confuse this with the fully-async framework's partial rollout.** This pipeline still runs the synchronous loop "rollout → wait for batch → one train step"; the only twist is that long-tail samples can be interrupted mid-rollout and resumed in a later step. Trainer and rollout phases remain serial. If you want the trainer/rollout fully decoupled and advancing concurrently, see `verl/experimental/fully_async_policy/` — not here.

> 🔗 **Verl dependency.** Pinned against `verl-project/verl@60546ef2` ([on GitHub](https://github.com/verl-project/verl/commit/60546ef2a7464a158cd170f58f852a62a4e552ba)). Exact `pip install` / `git checkout` recipe in [REQUIRED_VERL.txt](REQUIRED_VERL.txt). Rolling against `main`; bump the pin when refreshing.

## Background

Synchronous PPO/GRPO training in verl waits for every prompt in a batch to finish generating before stepping. RL training datasets are highly **right-skewed in response length**: empirically a small fraction of samples (~3%) emit dramatically longer responses than the median, and these long-tail samples are often the harder, more informative ones — you can't drop them without hurting accuracy. The whole batch stalls on the slowest few; GPUs idle.

![Response Length Distribution across the RL Training Dataset](https://raw.githubusercontent.com/mamazi0131/verl_doc/fca7a6d3acbeca12d69c5de6f85c312c1c9e47b6/Response_Length_Distribution_across_the_RL_Training_Dataset.png)

Partial rollout closes this bubble with two ideas (see the [APRIL paper](https://arxiv.org/pdf/2509.18521) for the academic treatment):

- **Sample supplementation**: when faster workers run out of work, immediately top them up with the next prompt — don't sit idle waiting for the long tail.
- **Mid-generation interruption + cross-step resume**: when the batch's "done" target is met but a few samples are still mid-generation, interrupt them at the step boundary, cache their KV state, and resume them in the next training step. The recipe pays one weight-version drift per resumed sample, corrected by token-level rollout importance sampling (off-policy correction).

![Comparison of GPU Execution Timelines between Standard Synchronous Training and the Proposed Async Partial Rollout](https://raw.githubusercontent.com/mamazi0131/verl_doc/fca7a6d3acbeca12d69c5de6f85c312c1c9e47b6/Comparison_of_GPU_Execution_Timelines_between_Standard_Synchronous_Training_and_the_Proposed_Async_Partial_Rollout.png)

Net effect: convert long-tail GPU bubbles into useful work on the next batch's prompts, at the cost of mild off-policy drift on resumed samples — which token-level IS handles cleanly.

---

## When to use

Use it when:

- The dataset has a **long-tailed response length distribution** (a small fraction of very long samples drags down each step).
- Synchronous PPO/GRPO shows a visible GPU bubble waiting on those long-tail samples.
- Training tolerates **mild off-policy drift** — partial rollout inherently spans multiple weight versions; pair it with IS correction.
- Multi-turn / tool-call workloads — upstream vLLM ≥ 0.12's `pause_generation` + abort is enough; no recipe-side server fork.

Skip it when:

- Response lengths are uniform with no long-tail bubble — the plain synchronous trainer is simpler.
- Strict on-policy is required (every trajectory must come from the current weights).
- You need to mix this pipeline with the upstream sync trainer's batch-shape assumptions — PartialRollout-specific fields (dummy `gen_batch`, continuous-worker semantics) would break them.

---

## Architecture

```
                 trainer (PartialRolloutRayPPOTrainer)
                        │
            push_batch  │  pull_batch
                        ▼
         ┌──────────────────────────────────┐
         │   RolloutPromptManager (Ray)     │
         │                                  │
         │   pending ─pull─► ongoing ─push─► done
         └──────────────────────────────────┘
                        │
            pull_prompts│push_prompts
                        ▼
              PartialRolloutAgentLoopWorker  ×N   (run_continuous loop)
                        │
              llm_client│ generate (FullyLLMServerClient retries aborted)
                        ▼
              PartialRolloutvLLMReplica  ×replicas
                ↳ PartialRolloutvLLMHttpServer  (Python `paused` gate + abort drain)
                        ▲
              cancel/   │
              resume    │
                        │
              PartialRolloutLLMServerManager
```

| Component | File | Role |
|---|---|---|
| `PartialRolloutRayPPOTrainer` | `ray_trainer.py` | Main trainer loop. `_fit_generate` pushes prompts into the manager, awaits one full batch via `async_rollout_manager.generate_sequences`, runs log_prob / advantage / policy update, then `update_weights`. |
| `RolloutPromptManager` | `prompt_manager.py` | Single-threaded Ray actor holding the three queues (pending / ongoing / done). `pull_batch` and `pull_prompts` both block on `asyncio.Event`s — no busy polling, no per-empty-pull RPCs. |
| `PartialRolloutAgentLoopManager` / `PartialRolloutAgentLoopWorker` | `agent_loop/agent_loop.py` | Worker runs a persistent loop (`run_continuous`) sharing one `asyncio.wait` across the pull RPC, the `_run_one` rollout tasks, and a slot-wait sentinel. Instead of overriding upstream `generate_sequences` (left intact for the validation path), the worker adds `generate_for_prompt`, which is upstream's `generate_sequences` with the trailing `outputs = await asyncio.gather(*tasks)` replaced by an `asyncio.wait(FIRST_COMPLETED)` loop that decrements `self.inflight_traj` and signals `self._slot_event` after each trajectory completion. The outer loop pulls the next prompt as soon as `inflight_traj + n <= max_inflight_prompts * n` — so a long-tail trajectory doesn't block other in-flight prompts from making room for the next pull. Manager exposes `cancel()` / `resume()` (delegated to `PartialRolloutLLMServerManager`) for trainer-side bracketing of `update_weights`. |
| `PartialRolloutvLLMHttpServer` / `PartialRolloutvLLMReplica` | `vllm_rollout/vllm_async_server.py` | vLLM HTTP server with a Python-side `_resume_event` gate (vLLM <0.12 lacks `pause_generation`). `cancel()` clears the gate so new `generate()` calls hang at the wrapper layer, then loops `abort_all_requests(reset_prefix_cache=False)` until in-flight drains. `resume()` sets the gate, releasing queued callers. Drop this whole layer once vLLM ≥0.12 is the floor. |
| `PartialRolloutLLMServerManager` | `llm_server.py` | Thin override of upstream `LLMServerManager`: swaps `rollout_replica_class` to `PartialRolloutvLLMReplica`, forces `get_client(fully_async=True)` so callers receive the retry-on-abort `FullyLLMServerClient`, and fans `cancel` / `resume` out to each replica. Installed via a monkey-patch in `PartialRolloutRayPPOTrainer.init_workers` because upstream has no FQN config knob for `LLMServerManager`. |

---

## Key invariants

1. **Prompt ownership during scheduling**: at any instant a prompt lives in exactly one of pending / ongoing / done. `pull_prompts` moves pending→ongoing; `push_prompts` is ongoing→done — terminal only. There is no aborted-back-to-pending re-queue path, because `FullyLLMServerClient.generate()` absorbs the abort/retry cycle inside one logical generate call.
2. **Cross-step abort/resume bracket**: `PartialRolloutAgentLoopManager.generate_sequences` runs `await self.resume()` at entry and `await self.cancel()` after `pull_batch` returns. The naive `checkpoint_engine` backend (PartialRollout's default) short-circuits before its own abort, so the recipe wires this itself; workers' aborted `client.generate(...)` calls wait inside `FullyLLMServerClient`'s retry loop until the next step's `resume()`.
3. **Per-sample weight-version tracking** lives in `gen_batch.meta_info["global_steps"]` (set by the trainer) and `FullyLLMServerClient.generate()` (records the actual versions each retry submitted against). The worker does no per-call version tracking.
4. **Continuous worker loop + trajectory-grained pull pacing**: each `PartialRolloutAgentLoopWorker` runs `run_continuous` for the actor's lifetime. Budget is counted in trajectories (`max_inflight_prompts * n`), not prompts. Inside each in-flight prompt, `generate_for_prompt` awaits the n trajectories via `asyncio.wait(FIRST_COMPLETED)` — every completion decrements `self.inflight_traj` and sets `self._slot_event`, waking the outer loop to pull the next prompt as soon as a prompt's worth of trajectory slots have freed up across all in-flight prompts (no need to wait for any one prompt to fully complete). Pull RPC, `_run_one` tasks, and the slot-wait sentinel all live in the same `asyncio.wait(running)` set; identity checks dispatch the three task kinds.
5. **The dummy `gen_batch` must carry `uid`**: when the dataloader is exhausted at the end of an epoch, the placeholder batch built to drain in-flight prompts still needs `non_tensor_batch["uid"]`, otherwise the manager can't compute the row count.
6. **Stateful dataloader resume**: `PartialRolloutRayPPOTrainer.fit()` resumes via the stateful dataloader automatically — **don't** add manual skip-on-resume logic.
7. **No graceful shutdown**: `run_continuous` runs for the actor's lifetime; Ray terminates workers at process exit (after `fit()` returns). Any rollouts in flight at exit time are abandoned — acceptable because the trainer isn't going to consume them anyway.

---

## Quick start

### Partial-rollout runs

Single-turn:
```bash
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo.sh
```

Multi-turn with tool calls — first generate the tool-agent dataset:
```bash
python3 examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_save_dir $HOME/data/gsm8k_tool
```
then launch:
```bash
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo_tool.sh
```

### Baseline runs (vanilla GRPO, for A/B against partial rollout)

Same model / data / batch / hyperparameters; the only differences from the PR variants are the upstream entry, the upstream agent loop, and no IS correction:
```bash
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo_baseline.sh
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo_tool_baseline.sh
```

### PartialRollout-specific Hydra overrides

| Key | Value | Notes |
|---|---|---|
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `single_turn_agent` / `tool_agent` | Use upstream agent loops directly — `FullyLLMServerClient` handles abort/retry, no PartialRollout wrapper needed. |
| `+async_training.partial_rollout` | `True` | Gates the retry-on-abort path inside `FullyLLMServerClient`. Required for any PartialRollout run; set via Hydra `+` (the key isn't pre-declared in the trainer config). |
| `algorithm.rollout_correction.rollout_is` | `token` (recommended) / `sequence` | Sequence-level ratios easily hit the `exp(±20)` safety clamp once rollouts span several weight versions. |
| `algorithm.rollout_correction.rollout_is_threshold` | `2.0` | TIS upper bound; for IcePop pass `"0.5_5.0"`. |
| `actor_rollout_ref.rollout.multi_turn.enable` | `True` *(tool variant only)* | Enables multi-turn. |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | YAML path | Tool registry; shared across backends. |

---

## File layout

```
partial_rollout/
├── README.md / README_zh.md / REQUIRED_VERL.txt
├── main_ppo.py                # @hydra entry; wraps PartialRolloutTaskRunner
├── ray_trainer.py             # PartialRolloutRayPPOTrainer
├── prompt_manager.py          # RolloutPromptManager (Ray actor)
├── llm_server.py              # PartialRolloutLLMServerManager (force fully_async + cancel/resume)
├── agent_loop/
│   └── agent_loop.py          # PartialRolloutAgentLoopManager / PartialRolloutAgentLoopWorker
├── vllm_rollout/
│   └── vllm_async_server.py   # PartialRolloutvLLMHttpServer / PartialRolloutvLLMReplica (paused gate)
└── run/
    ├── gsm8k_tool_config.yaml                      # tool registry for the tool variants
    ├── run_qwen3-0.6b_gsm8k_grpo.sh                # PR, single-turn
    ├── run_qwen3-0.6b_gsm8k_grpo_tool.sh           # PR, tool-call
    ├── run_qwen3-0.6b_gsm8k_grpo_baseline.sh       # baseline, single-turn
    ├── run_qwen3-0.6b_gsm8k_grpo_tool_baseline.sh  # baseline, tool-call
    └── run_{dapomath,gsm8k}_{nopr,pr}_grpo_4b_*.sh # 4B Qwen3 ports of the recipe PR
```
