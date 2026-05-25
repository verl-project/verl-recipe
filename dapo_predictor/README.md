# DAPO Predictor Reorder

This directory is a portable copy of `recipe/dapo_predictor`. You can copy it into a local `recipe/` tree, for example when adapting the predictor reorder flow around a local verl `0.7.1` environment.

The feature adds predictor-driven prompt reordering to DAPO. Before rollout generation, prompts are scored by a lightweight predictor head and then reordered with serpentine packing so prompts with similar predicted response length are spread across data-parallel ranks. After actor update, the predictor head is trained from the observed rollout response lengths and is used again in the next step.

## What It Changes

- Uses `PredictorAsyncActorRolloutRefWorker` instead of the default actor-rollout worker for FSDP/FSDP2 actor rollout.
- Adds a linear predictor head on the actor worker: `nn.Linear(hidden_size, 1, bias=False)`.
- Scores one sample per prompt group before generation, expands the score to all `rollout.n` samples, and applies `snake_sort_indices`.
- Restores order after DP batch balancing before training the predictor, so labels still correspond to the original prompt groups.
- Trains only the predictor head during `update_predictor`; the actor update still follows the normal DAPO/PPO path.

Prompt-reorder patch examples were removed from this branch; this package documents predictor-driven reorder only.

## Entry Points

- `main_dapo_predictor_reorder.py`
  - Main DAPO entrypoint with predictor score + snake-sort reorder enabled.
- `main_dapo_reorder.py`
  - Backward-compatible alias to the predictor-driven reorder entrypoint.

## Implementation Modules

- `predictor_dapo_trainer.py`
  - Injects predictor scoring before rollout generation.
  - Builds and applies predictor reorder indices.
  - Reverses DP balancing order before predictor training.
  - Calls `actor_rollout_wg.update_predictor(prompt_batch, batch)` after actor update.
- `predictor_worker.py`
  - Adds `PredictorDataParallelPPOActor` with the linear predictor head.
  - Implements `compute_predictor_score` and `update_predictor` worker RPCs.
  - Extracts last-token hidden states from the actor model for scoring and training.
- `predictor_utils.py`
  - Provides `snake_sort_indices` for prompt-level serpentine DP packing.

## Runtime Flow

1. Build `gen_batch` from the training batch and repeat each prompt `rollout.n` times.
2. Hydrate the predictor input with `input_ids`, `attention_mask`, and `position_ids` when needed.
3. Run `compute_predictor_score` on actor workers:
   - sample one item from each prompt group,
   - extract the last-token hidden state,
   - score it with the predictor head,
   - broadcast that score back to all samples from the same prompt.
4. Sort prompt groups by predictor score and apply serpentine DP packing through `snake_sort_indices`.
5. Generate rollouts with the reordered batch.
6. Continue normal reward, KL, advantage, critic, and actor update logic.
7. If DP batch balancing changed row order, restore the pre-balance order.
8. Train the predictor head using the latest prompt hidden states and observed response lengths.

## Predictor Head Training

The predictor head is trained online after each actor update. The training data comes from the same rollout step:

- Inputs: prompt-side last-token hidden states extracted from `prompt_batch`.
- Labels: observed generated response lengths from `response_batch.batch["responses"]`.
- Prompt grouping: response lengths are reshaped by `rollout.n`, and the max response length in each prompt group is used as the label.
- Label scaling: response lengths are bucketed by `max(1, rollout.response_length // 40)` to keep label values in a stable range.
- Loss: ListMLE ranking loss, so the head learns the relative ordering of prompts by response length rather than an exact length regression target.
- Optimizer: AdamW over the linear predictor head only.
- Determinism: the predictor dataloader and ListMLE shuffle use `trainer.predictor_reorder.seed`.

The update path gathers hidden states and labels across distributed ranks. When sequence parallelism is enabled, only SP rank 0 data from each DP group is used to avoid duplicated prompt samples.

Metrics emitted by the predictor update include:

- `predictor/epoch_0_loss`
- `predictor/epoch_0_kendall_tau`
- `predictor/epoch_{last}_loss`
- `predictor/epoch_{last}_kendall_tau`
- `predictor/final_loss`
- `predictor/epochs`
- `predictor/update_time_s`
- `predictor/total_samples`

`scipy` is optional. If it is unavailable, Kendall tau metrics fall back to `0.0` instead of failing the worker.

## Configuration

Enable predictor reorder with Hydra overrides under `trainer.predictor_reorder`. The entrypoint mirrors this config to `actor_rollout_ref.predictor_reorder` so the worker can read it.

Common options:

| Option | Default | Description |
| ------ | ------- | ----------- |
| `enable` | `False` | Enables predictor scoring, reorder, and predictor head training. |
| `epochs` | `10` | Number of predictor-head training epochs per actor update. |
| `batch_size` | `32` | Batch size for predictor-head training. |
| `lr` | `3e-5` | AdamW learning rate for the predictor head. |
| `weight_decay` | `1e-4` | AdamW weight decay for the predictor head. |
| `seed` | `1` | Local seed used by predictor dataloader/ListMLE shuffling. |
| `predictor_keep_actor_loaded` | `False` | Keeps actor parameters on GPU across actor update when predictor training immediately follows. Useful when offload overhead is high. |

## Launch Example

```bash
PYTHONPATH=/workspace/verl python recipe/dapo_predictor/main_dapo_predictor_reorder.py \
  +trainer.predictor_reorder.enable=True \
  +trainer.predictor_reorder.epochs=10 \
  +trainer.predictor_reorder.batch_size=32 \
  +trainer.predictor_reorder.lr=3e-5 \
  +trainer.predictor_reorder.weight_decay=1e-4 \
  +trainer.predictor_reorder.seed=1
```

Use the same DAPO data, model, rollout, critic, and trainer overrides as the normal `recipe.dapo` entrypoint. This package only adds predictor reorder-specific overrides.

## Experimental Setup and Effects

The PR experiment used a long-response DAPO workload where generation time can become unbalanced across DP ranks:

| Parameter | Value |
| --------- | ----- |
| Model | Qwen3-30B-A3B-Instruct-2507 |
| DataLoader seed | 1 |
| Global batch size | 32 |
| Samples per prompt | 8 |
| Max num sequences | 16 |
| Generation TP | 4 |
| Sequence parallel | 4, ulysses |
| Max model length | 22528 |
| Prompt length | about 2k |
| Response length | about 20k |
| NPU count | 32 |
| Training steps | 57 |

### Critic Score

| Metric | Reorder | Baseline |
| ------ | ------- | -------- |
| Average | 0.6179 | 0.6137 |
| First 10 steps avg | 0.4383 | 0.4391 |
| Last 10 steps avg | 0.6680 | 0.6680 |

The critic score is essentially unchanged, so predictor reorder did not degrade training quality in this run.

### Step Time

| Metric | Reorder | Baseline |
| ------ | ------- | -------- |
| Average | 638.98 s/it | 668.40 s/it |
| First 10 steps avg | 616.14 s/it | 621.21 s/it |
| Last 10 steps avg | 616.23 s/it | 711.55 s/it |

The reorder run stayed around 616 s/it, while the baseline degraded from about 621 s/it to 711 s/it. The step-time gap grew from 5.08s to 95.33s.

### Generation Time

| Metric | Reorder | Baseline |
| ------ | ------- | -------- |
| Average | 471.66s | 504.67s |
| First 10 steps avg | 439.39s | 461.45s |
| Last 5 steps avg | 421.14s | 522.81s |
| Trend | -18.25s | +61.37s |

Generation time decreased during the reorder run but increased in the baseline. The generation-time advantage grew from about 22s to 101.67s as training progressed.

### Actor Entropy

| Metric | Reorder | Baseline |
| ------ | ------- | -------- |
| Average | 0.2664 | 0.2626 |
| First 10 steps avg | 0.2571 | 0.2577 |
| Last 5 steps avg | 0.2619 | 0.2611 |
| Trend | +0.0048 | +0.0034 |

Actor entropy stayed comparable between the reorder and baseline runs.

### Summary

- No quality loss was observed: critic score was unchanged.
- Step time stayed stable with predictor reorder, while baseline step time increased late in training.
- Generation became faster and more stable in the reorder run.
- Actor entropy remained similar, suggesting the reorder did not materially change policy entropy.
- The benefit widened over time, especially for generation latency.
