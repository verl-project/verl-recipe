# NeMo Gym — browser RL

Step-wise GRPO training on the NeMo Gym `interactive_browser` environment: the
policy drives a live browser through one tool (`observe` / `navigate` / `click` /
`type` / `finish`) and is scored per episode.

The sibling [`nemo_gym`](../README.rst) recipe hands a whole rollout to NeMo Gym
through `RolloutCollectionHelper` and receives a finished result. That works well
for short, deterministic environments. A browser rollout runs for tens of steps
against live sites, and every step can fail for reasons that have nothing to do
with the policy — the session is lost, a page never loads, the judge endpoint is
unreachable. This recipe therefore keeps the loop on the verl side, using verl's
own `ToolAgentLoop`, so the trainer can bound each step, classify a failure while
it happens, and resample instead of learning from it.

Both recipes target the same environments and can be used side by side; they
attach to different verl extension points (`agent_loop_manager_class` versus
`default_agent_loop`) and share no state.

## Requirements

- A NeMo Gym checkout providing `resources_servers/interactive_browser`.
  **This is currently NVIDIA-NeMo/Gym#1865 and is not merged yet** — until it
  lands, point `NEMO_GYM_ROOT` at that branch.
- verl at the commit in `REQUIRED_VERL.txt`, image `verlai/verl:vllm018.latest`.
- A policy endpoint that speaks tool calls as structured `function_call` items.
  For vLLM + Qwen, `--enable-auto-tool-choice --tool-call-parser hermes`.
- For open-ended tasks, an OpenAI-compatible judge endpoint (see below).

## Run

```bash
# 1. Task data. Nothing is committed here; pull it or point at your own JSONL.
python recipe/nemo_gym/browser/prepare_webvoyager_data.py \
    --hf-repo lexmount/webvoyager-clean \
    --output /path/to/webvoyager_train.jsonl

# 2. Environment. Local Chromium needs no account and no network egress beyond
#    the task sites; `interactive_browser/lexmount` drives remote browsers.
gym env start --resources-server interactive_browser --no-agent --no-model

# 3. Training.
cp recipe/nemo_gym/browser/config.env.example config.env   # fill it in
sbatch recipe/nemo_gym/browser/submit_webvoyager.sh
```

## Reward

Two paths, chosen per task:

- **Deterministic** — a task carrying `verifier_metadata` (`final_url`,
  `url_contains`, `dom_contains`, `answer_equals`) is scored by the
  environment's own `/verify`.
- **Open-ended** — everything else is scored by a binary LLM judge in
  `judge.py`, configured through `JUDGE_BASE_URL` / `JUDGE_API_KEY` /
  `JUDGE_MODEL`.

The judge lives here rather than in the environment because the environment's
verifier is deterministic-only today. If NeMo Gym grows a judge-backed verifier,
this path should move there and the recipe should just read `/verify`.

## Environment failures

An infrastructure failure and a policy failure both end with `reward=0`, and the
difference matters: a run where a fifth of the group failed to reach a page is
not a run where the policy got worse. This recipe keeps them apart:

1. `BrowserTool.finalize` classifies the outcome — `environment_seed_session_failed`,
   `environment_verify_failed`, `environment_judge_failed`,
   `generation_aborted_timeout`.
2. A classified rollout is **resampled** once (`NEMO_GYM_BROWSER_ENV_RETRIES`,
   default 1) before it is allowed into the batch.
3. One that stays invalid enters the batch loss-masked, with
   `env_invalid` / `env_invalid_reason` on every sample's `extra_fields`.
4. `group_stats.py` registers a `grpo_env_aware` advantage estimator that keeps
   flagged samples out of the group baseline, and drops a group entirely when
   fewer than two valid samples remain or more than a quarter of it failed.

Steps 1–3 work as-is. **Step 4 needs one upstream change**: registered advantage
estimators receive `non_tensor_batch` only for `gdpo`
(`verl/trainer/ppo/ray_trainer.py`, the `else` branch of `compute_advantage`), so
the flags do not reach the estimator yet and it degrades to stock GRPO behaviour
rather than guessing. The change is two lines and is being proposed separately:

```python
# verl/trainer/ppo/ray_trainer.py, compute_advantage()
adv_kwargs["non_tensor_batch"] = data.non_tensor_batch
adv_kwargs["batch"] = data.batch
```

Until then, leave `algorithm.adv_estimator=grpo`; the loss masking and the
resampling still remove the bulk of the damage. The estimator is unit-tested
offline in the meantime.

Related NeMo Gym issues, which would let the environment report this itself
instead of the recipe inferring it: NVIDIA-NeMo/Gym#2608 (report an
infrastructure failure from `verify`), #2609 (session teardown hook), #2610
(join environment sessions to rollouts).

## Tests

No GPU, no browser, no NeMo Gym server:

```bash
pytest recipe/nemo_gym/browser/tests -q
```

Covers the group-statistics estimator (including the "flags absent must not mean
everything is invalid" case), the dataset conversion, and transcript truncation.

## Files

| File | Purpose |
| --- | --- |
| `browser_agent_loop.py` | `BrowserTool` (NeMo Gym HTTP contract) and `BrowserToolAgentLoop` |
| `judge.py` | Binary LLM judge for open-ended tasks |
| `group_stats.py` | `grpo_env_aware` advantage estimator |
| `dataset.py` | NeMo Gym rollout rows to `tools_kwargs` |
| `prepare_webvoyager_data.py` | Task list to rollout inputs |
| `configs/` | Environment config paths, tool config |
| `submit_webvoyager.sh`, `config.env.example` | Slurm entry point |
