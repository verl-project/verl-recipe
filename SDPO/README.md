# SDPO: Self-Distillation Policy Optimization

This recipe implements the training infrastructure for Self-Distillation Policy Optimization (SDPO) using the verl framework with async 2-GPU training.

**Reference:** [SDPO: Self-Distillation Policy Optimization](https://arxiv.org/abs/2601.20802)

## Overview

SDPO uses the model conditioned on the ground truth answer as a "teacher" to provide dense per-token advantages for policy optimization. The key idea:

- **Student**: The model's response (rollout)
- **Teacher**: EMA of reference + training policy conditioned on `[prompt + ground_truth_answer]`

The SDPO loss computes KL divergence between student and teacher distributions, providing dense per-token credit assignment.

## Architecture

This recipe uses async 2-GPU training with the `one_step_off_policy` infrastructure:

```
GPU 0 (Rollout)          GPU 1 (Training)
     |                        |
     v                        v
  Generate                  Actor
  Rollouts    ----sync--->  Update
     |                        |
     v                        |
  Log Probs   <---weights----+
```

Key features:
- 1 GPU for rollout (vLLM)
- 1 GPU for training (FSDP)
- Importance reweighting for off-policy correction
- GRPO advantage estimation

## Current Status

**Important:** Full SDPO (self-distillation loss with EMA teacher) requires the modified verl code from `/root/SDPO`. The standard verl codebase does not include the `self_distillation` config and `compute_self_distillation_loss` function.

This recipe currently uses:
- Standard GRPO training with vanilla policy loss
- Async 2-GPU training infrastructure
- Importance reweighting for off-policy correction

### Files

| File | Description |
|------|-------------|
| `sdpo_advantage.py` | SDPO loss computation (for use with modified verl) |
| `sdpo_ray_trainer.py` | Extended trainer for SDPO |
| `sdpo_main.py` | Main entry point |
| `config/sdpo_trainer.yaml` | Training configuration |
| `train.sh` | Training script for geo3k |

## Usage

### Data Preparation

Prepare the Geometry3k dataset:

```bash
python examples/data_preprocess/geo3k.py --local_save_dir ~/data/geo3k
```

### Training

Run training with default settings:

```bash
cd recipe/SDPO
bash train.sh
```

### Smoke Test

Run a quick smoke test with minimal iterations:

```bash
SMOKE_TEST=1 bash train.sh
```

### Configuration Options

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL_PATH` | `Qwen/Qwen2.5-VL-3B-Instruct` | Model path |
| `TRAIN_FILE` | `$HOME/data/geo3k/train.parquet` | Training data |
| `TEST_FILE` | `$HOME/data/geo3k/test.parquet` | Test data |
| `TOTAL_TRAIN_STEPS` | 100 | Training steps |
| `SMOKE_TEST` | 0 | Set to 1 for smoke test mode |

## Enabling Full SDPO

To use the full SDPO with self-distillation, you need to:

1. Use the modified verl code from `/root/SDPO` instead of standard verl
2. Set `actor_rollout_ref.actor.policy_loss.loss_mode=sdpo` in the config
3. Configure self-distillation parameters:

```yaml
actor_rollout_ref:
  actor:
    policy_loss:
      loss_mode: sdpo
    self_distillation:
      full_logit_distillation: True
      alpha: 0.0  # 0.0 = forward KL
      ema_update_rate: 0.05  # EMA update rate for teacher
      is_clip: 2.0  # IS clipping threshold
```

## SDPO Loss Details

The SDPO loss (in `sdpo_advantage.py`) computes:

1. **Forward KL (alpha=0.0)**: `KL(teacher || student)` - mode-covering
2. **Reverse KL (alpha=1.0)**: `KL(student || teacher)` - mode-seeking
3. **JSD (0 < alpha < 1)**: Interpolation between forward and reverse KL

With importance sampling clipping:
- Compute IS ratio from old_log_probs
- Clip to prevent extreme updates

The teacher is an EMA of reference and training policy weights:
```
teacher = (1 - ema_rate) * teacher + ema_rate * student
```

This is updated after each actor update step.

## Algorithm Details

From the SDPO paper:

1. Generate rollout from student policy
2. Build teacher context: `[prompt + ground_truth_answer]`
3. Compute teacher log probs for student's tokens
4. SDPO loss: KL divergence between teacher and student
5. Apply IS clipping for stability
6. Update policy with combined GRPO + SDPO gradients

The teacher provides positive advantage for tokens aligned with the correct answer and negative advantage for tokens that diverge.
