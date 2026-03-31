# Recipe: Online Policy Distillation (OPD)

## Overview

Online Policy Distillation trains a smaller student model to match a larger teacher model's token-level distribution, while simultaneously optimizing a task reward (e.g., math accuracy). Unlike offline distillation (SFT on teacher outputs), OPD generates training data from the student's own policy, avoiding distributional mismatch.

**Training objective:**

```
L = L_GRPO + kl_loss_coef * KL(student || teacher)
```

- `L_GRPO`: Standard GRPO policy gradient loss (reward-based)
- `KL(student || teacher)`: Per-token KL divergence between student and teacher distributions

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                 OPD Training Loop                    │
│                                                     │
│  1. Student rollout (vLLM async)                    │
│  2. Compute reward (task-specific)                  │
│  3. Query teacher for per-token log probs (ZMQ)  ◄──┼── Teacher Server
│  4. Compute GRPO advantages                         │   (Qwen3-32B, TP=4)
│  5. Update student: L_GRPO + KL loss                │
│  6. Sync weights to rollout engine                  │
│                                                     │
│  Student: Qwen3-8B (FSDP, 4 GPUs)                 │
└─────────────────────────────────────────────────────┘
```

**Key design:** Zero modification to verl source code. `OPDTrainer` inherits `RayPPOTrainer` and overrides `_compute_ref_log_prob()` to query the external teacher instead of a reference model.

## Quick Start

### 1. Start Teacher Server

Uses the GKD recipe's ZMQ-based teacher server:

```bash
cd recipe/gkd/teacher
export PROXY_FRONTEND_PORT=15555 PROXY_BACKEND_PORT=15556
CUDA_VISIBLE_DEVICES=0,1,2,3 bash start_server.sh
# Edit start_server.sh: set CKPT_PATH and TP size for your teacher model
```

### 2. Run OPD Training

```bash
# Default: kl_loss_coef=0.01
CUDA_VISIBLE_DEVICES=4,5,6,7 bash recipe/opd/run_opd.sh

# Ablation: try different KL coefficients
bash recipe/opd/run_opd.sh 0.001   # weak teacher
bash recipe/opd/run_opd.sh 0.01    # medium (recommended)
bash recipe/opd/run_opd.sh 0.0     # no teacher (GRPO baseline)
```

### 3. Custom Configuration

```bash
STUDENT_MODEL=/path/to/model \
TEACHER_IP=10.0.0.1 \
TEACHER_PORT=15555 \
NGPUS=8 \
TOTAL_STEPS=500 \
bash recipe/opd/run_opd.sh 0.01
```

## Files

| File | Description |
|------|-------------|
| `opd_trainer.py` | `OPDTrainer`: extends `RayPPOTrainer` with teacher KL loss |
| `main_opd.py` | Entry point with `OPDTaskRunner` |
| `reward_gsm8k.py` | GSM8K answer extraction and scoring |
| `run_opd.sh` | Launch script with configurable parameters |

## How It Works

### OPDTrainer (`opd_trainer.py`)

The core change is a single method override:

```python
class OPDTrainer(RayPPOTrainer):
    def _compute_ref_log_prob(self, batch):
        # Instead of querying a reference model, query the teacher
        teacher_log_probs = self.teacher_client.get_teacher_log_probs(
            input_ids, attention_mask, response_length
        )
        return DataProto.from_dict({"ref_log_prob": teacher_log_probs})
```

This plugs into verl's existing `use_kl_loss` mechanism:
- verl computes `kl_penalty(student_log_prob, ref_log_prob)`
- The result is added to the policy loss with coefficient `kl_loss_coef`

### TeacherClient

Communicates with the GKD recipe's ZMQ teacher server:
1. Sends full token sequences (prompt + response) to teacher
2. Teacher returns top-k log probabilities per position
3. Client extracts the log probability of each actual next token
4. Returns `(batch_size, response_length)` tensor of teacher log probs

## KL Loss Coefficient Guide

| kl_loss_coef | Effect | When to Use |
|-------------|--------|-------------|
| 0.0 | No teacher (pure GRPO) | Baseline comparison |
| 0.001 | Weak guidance, stable training | Conservative, long training |
| 0.01 | Moderate guidance | Recommended starting point |
| 0.1 | Strong guidance | Risk of catastrophic forgetting |

## Experimental Results (GSM8K, 500 steps)

Qwen3-8B student + Qwen3-32B teacher, n=8 responses per prompt:

| Config | Avg Score | Max Score | Non-zero Steps |
|--------|-----------|-----------|----------------|
| KL=0.0 (baseline) | 0.0178 | 0.2500 | 169/500 (34%) |
| KL=0.001 (OPD) | 0.0108 | 0.0781 | 213/500 (43%) |

OPD produces more consistent but lower-peak scores. The teacher provides
a smoother learning signal that leads to more frequent near-correct answers,
but the strong reward signal from GRPO alone can achieve higher peaks on
tasks with clear binary rewards like GSM8K.

OPD is expected to show stronger advantages when:
- Reward signals are sparse or noisy
- Teacher and student have large capability gaps
- Tasks require complex reasoning where token-level guidance helps
