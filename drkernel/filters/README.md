# DR.Kernel batch filters (MRS) — project-side port

Port of `verl_patch/trainer/code/filters/` from
[DR.Kernel](https://github.com/hkust-nlp/KernelGYM) into a project-side
recipe. The filter logic is unchanged from upstream — see the original
README in the DR.Kernel repo for the full design rationale, theory, and
test-coverage notes.

## What's here

```
recipe/drkernel/filters/
├── __init__.py             - public API: PPOBatchFilter, PPOFilterConfig, filter_dataproto
├── unified_filter.py       - PPOBatchFilter pipeline (group/individual/selection stages)
├── two_gate_filter.py      - Two-Gate precision filter (Gate 1 bias, Gate 2 instability)
├── dataproto_adapter.py    - thin DataProto <-> dict bridge so callers can pass a verl DataProto directly
└── README.md               - this file
```

Two changes vs upstream:
1. Dropped an unused `from verl.trainer.ppo.metric_utils import _compute_response_info` import from `unified_filter.py`.
2. Added `dataproto_adapter.py` so the filter can be invoked on a verl `DataProto` without the caller having to repack tensors.

No logic changes anywhere. If DR.Kernel updates the filter implementation,
`cp` the upstream files over `unified_filter.py` and `two_gate_filter.py`,
re-drop the dead import, and the rest stays put.

## Pipeline summary

`PPOBatchFilter.filter_batch()` runs four stages in order:

| Stage | What it does | Knobs |
|---|---|---|
| 0. Two-gate precision filter | (only if oversampling and `enable_two_gate_filter=True`) — Gate 1 detects systematic bias between FSDP/vLLM log probs; Gate 2 detects logit instability. Same idea as verl's `bypass_mode` rollout-correction but with a different signal: Gate 2 uses `top_log_probs` (argmax token logprob) which `bypass_mode` does not have. | `gate1_bias_epsilon`, `gate2_instability_threshold` |
| 1. Group-level filtering | reject prompt-groups with low reward variance (`std < 1e-3`), all-over-length groups, or — if `remove_clip=True` — groups with too few short samples. | `reject_low_variance_groups`, `max_response_length`, `remove_clip`, `min_rollout_n` |
| 2. Individual-sample filtering | reward threshold only. **No** per-sample length filtering — that's intentional; length pressure is handled in stage 3 instead. | `reward_threshold` |
| 3. Group + sample selection | within each surviving group: pick samples by strategy (`uniform` / `efficiency` / `efficiency_stochastic`); across groups: prefer complete groups, accept incomplete groups to maximize utilization. | `sample_selection_strategy`, `target_group_size`, `min_group_size`, `target_num_groups` |

## Quickstart

```python
from recipe.drkernel.filters import PPOBatchFilter, PPOFilterConfig, filter_dataproto

config = PPOFilterConfig(
    sample_selection_strategy="efficiency_stochastic",
    target_group_size=16,                   # = rollout.n
    target_num_groups=32,                   # = train_batch_size in groups
    reject_low_variance_groups=True,
    enable_two_gate_filter=False,           # leave to verl's bypass_mode
)

# Bare API (raw tensors):
indices, metrics = PPOBatchFilter(config).filter_batch(
    batch_data={"rewards": ..., "response_mask": ..., ...},
    uids=["prompt_a"] * 16 + ["prompt_b"] * 16,
    return_indices=True,
)

# DataProto API (drop-in for verl trainers):
filtered_batch, metrics = filter_dataproto(batch, config)
```

## Relationship to verl's `bypass_mode`

There's overlap between MRS's two-gate filter and verl's
`@register_policy_loss("bypass_mode")` + `rollout_corr_helper`:

- **Use `bypass_mode` for log-prob mismatch correction.** verl's IS/RS pipeline
  is a strict superset for the standard rollout/training mismatch case (see
  `drkernel/PORT_PLAN.md` Phase 1.2 audit).
- **Use MRS for the rest** — group-level low-variance / over-length rejection,
  efficiency-based per-group sample selection, optional reward thresholding.
- Default `enable_two_gate_filter=False`. Only flip it on if you specifically
  want Gate 2's `top_log_probs`-based instability check, which verl's
  `bypass_mode` does not provide.

## Proposed integration into `fully_async_policy` (NOT YET WIRED)

Hook point: in `verl/experimental/fully_async_policy/fully_async_trainer.py:fit_step`,
between `_fit_compute_log_prob` (which produces `old_log_probs`) and
`_fit_compute_advantage` (which is where TRLOO needs the filtered batch).

```python
async def fit_step(self, batch_dict: dict = None):
    ...
    with marked_timer("step", self.timing_raw):
        batch = await self._fit_generate(None)
        batch = self._fit_compute_reward(batch)
        batch = self._fit_compute_log_prob(batch)
        batch = self._fit_compute_ref_log_prob(batch)
        batch = self._fit_compute_critic(batch)
        batch = self._fit_filter_batch(batch)              # <-- new hook
        batch = self._fit_compute_advantage(batch)
        ...
```

`_fit_filter_batch` would:
1. Bail out (return `batch` unchanged) if `algorithm.batch_filter.enable` is False.
2. Build a `PPOFilterConfig` from the `algorithm.batch_filter` sub-config the
   first time it's called, and cache the `PPOBatchFilter` instance.
3. Call `filter_dataproto(batch, config, global_step=self.global_steps)`.
4. Merge the returned `metrics` dict into `self.metrics`.
5. If the filter returned an empty batch, log a warning and either skip the
   step or fall back to the unfiltered batch (TBD per user preference).

Why between log_prob and advantage:
- Two-gate filter (if enabled) needs `old_log_probs` (Gate 1) and ideally
  `top_log_probs` from rollout (Gate 2). `_fit_compute_log_prob` produces the
  former; the latter requires upstream rollout config to emit it.
- TRLOO advantage compute should run on the filtered batch so groups that
  fail the filter don't waste advantage compute and don't pollute the LOO
  baselines.
- Critic update happens after advantage today, so the critic also trains on
  filtered data. The trainer's `_fit_update_actor` likewise consumes the
  filtered batch.

Open question for review: where in `algorithm` to put `batch_filter` in
the OmegaConf schema. Likely a new top-level dataclass in
`verl/trainer/config/algorithm.py` mirroring `PPOFilterConfig`, then
omega-conf-converted in `_fit_filter_batch`. Defer until the wiring step.
