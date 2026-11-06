# ObsSpec reproduction experiments

Separate training and evaluation experiments for ObsSpec. See the [recipe README](../README.md) for installation. Run commands from the verl root.

## Data

Prepare both datasets and the evaluation inputs:

```bash
python3 recipe/obsspec/reproduce/prepare_data.py
```

This writes the Endless Terminals and DAPO training/validation parquets to `~/data/terminal_obsspec`.

## Separate training

Train the policy with GRPO, then train the world model on its saved rollouts:

```bash
# Policy
MODEL_PATH=Qwen/Qwen3-8B EXPERIMENT_NAME=obsspec-policy-qwen3-8b \
  bash recipe/obsspec/reproduce/train_policy.sh

# World model
MODEL_PATH=Qwen/Qwen3-0.6B SOURCE_EXPERIMENT=obsspec-policy-qwen3-8b \
EXPERIMENT_NAME=obsspec-wm-qwen3-0.6b \
  bash recipe/obsspec/reproduce/train_offline_wm.sh
```

Each training run uses eight GPUs by default.

## Main evaluation

```bash
# Sequential baseline
POLICY_MODEL_PATH=./outputs/terminal-obsspec/obsspec-policy-qwen3-8b/checkpoints/global_step_500/actor/huggingface \
EXPERIMENT_NAME=obsspec-qwen3-8b-baseline-4k \
ENABLE_SPECULATION=false \
  bash recipe/obsspec/reproduce/eval_4k.sh

# Observation speculation
POLICY_MODEL_PATH=./outputs/terminal-obsspec/obsspec-policy-qwen3-8b/checkpoints/global_step_500/actor/huggingface \
SPECULATOR_MODEL_PATH=./outputs/terminal-obsspec/obsspec-wm-qwen3-0.6b/checkpoints/global_step_500/huggingface \
EXPERIMENT_NAME=obsspec-policy-qwen3-8b-wm-qwen3-0.6b-4k \
  bash recipe/obsspec/reproduce/eval_4k.sh
```

Evaluation uses Endless Terminals by default. Add `DATASET=dapo_tir` to evaluate DAPO instead.

## Batch and async evaluation

Both launchers evaluate 65,536 rollouts using the same terminal task order. Synchronous evaluation runs batches of rollouts; asynchronous evaluation keeps up to 256 rollouts in flight.

```bash
# Synchronous
POLICY_MODEL_PATH=./outputs/terminal-obsspec/obsspec-policy-qwen3-8b/checkpoints/global_step_500/actor/huggingface \
SPECULATOR_MODEL_PATH=./outputs/terminal-obsspec/obsspec-wm-qwen3-0.6b/checkpoints/global_step_500/huggingface \
EXPERIMENT_NAME=obsspec-policy-qwen3-8b-wm-qwen3-0.6b-64k-sync \
  bash recipe/obsspec/reproduce/eval_64k_sync.sh

# Asynchronous
POLICY_MODEL_PATH=./outputs/terminal-obsspec/obsspec-policy-qwen3-8b/checkpoints/global_step_500/actor/huggingface \
SPECULATOR_MODEL_PATH=./outputs/terminal-obsspec/obsspec-wm-qwen3-0.6b/checkpoints/global_step_500/huggingface \
EXPERIMENT_NAME=obsspec-policy-qwen3-8b-wm-qwen3-0.6b-64k-async \
  bash recipe/obsspec/reproduce/eval_64k_async.sh
```
