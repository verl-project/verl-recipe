# Ascend INT8 W8A8 QAT Rollout

## What does this PR do?

This recipe provides an experimental Qwen3-8B GRPO workflow for FSDP actor
quantization-aware training (QAT) with an Ascend INT8 W8A8 rollout backend.
Eligible linear layers fake-quantize weights per output channel and activations
per token during actor training. After each actor update, rollout-ready INT8
weights and scale metadata are synchronized through veRL's existing vLLM weight
update path, reducing the precision mismatch between BF16 optimization and W8A8
sample generation.

## Required `verl` version

This recipe is paired with
[**verl-int8-w8a8**](https://github.com/sunsunsun98/verl-int8-w8a8/tree/int8-w8a8-v0.7.0), which
contains the INT8 W8A8 QAT implementation, worker integration, configuration,
and Ascend rollout support based on official veRL v0.7.0. See
[`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the exact pinned tag and commit as
well as copy-pastable installation commands.

## Scope

- Actor training uses FSDP; Megatron is not supported.
- Rollout uses vLLM with the Ascend ModelSlim quantization backend.
- `actor_rollout_ref.rollout.quantization=ascend` selects the vLLM backend;
  `trainer.device=npu` selects the device.
- `quant_model_description.json` describes the checkpoint's concrete scheme,
  such as `W8A8_DYNAMIC`, and identifies floating-point fallback modules.
- Stochastic rounding is enabled by default with `USE_STOCHASTIC=1`.
- Non-Ascend INT8 rollout kernels have not been validated.

## Environment

Prepare the following before launching:

1. A PyTorch for Ascend environment with compatible vLLM and vLLM Ascend.
2. A BF16 Qwen3-8B actor checkpoint.
3. An Ascend W8A8 rollout checkpoint containing
   `quant_model_description.json` and its required scale/offset metadata.
4. GSM8K or DAPO-Math training and validation parquet files.

The reference environment used vLLM 0.13.0, vLLM Ascend 0.13.0, and 16 Ascend
910C NPUs. Treat these as tested reference versions rather than a universal
compatibility guarantee.

## Installation

From the root of a standalone `verl-recipe` checkout, install the pinned fork:

```bash
./install_verl.sh --recipe int8_w8a8_qat --show
./install_verl.sh --recipe int8_w8a8_qat
```

For an editable checkout instead:

```bash
./install_verl.sh --recipe int8_w8a8_qat --method git --dest ./verl
```

The recipe is pinned to the full commit SHA rather than a moving branch. The
branch and release tag are recorded in `REQUIRED_VERL.txt` for traceability.

## Data Preparation

This recipe supports either GSM8K or DAPO-Math parquet data:

- **GSM8K:** From the pinned `verl-int8-w8a8` checkout, run
  `python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k`.
- **DAPO-Math:** Follow the
  [DAPO data preparation instructions](../dapo/README.md#quickstart), or run
  `bash dapo/prepare_dapo_data.sh` from a standalone `verl-recipe` checkout.

Set `TRAIN_FILE` and `VAL_FILE` to the generated files for the dataset selected
for the experiment.

## Quick Start

The following four path variables are required:

```bash
export MODEL_PATH=/path/to/Qwen3-8B
export ROLLOUT_MODEL_PATH=/path/to/Qwen3-8B-W8A8
export TRAIN_FILE=/path/to/dapo-math-17k.parquet
export VAL_FILE=/path/to/aime-2024.parquet

bash int8_w8a8_qat/run_qwen3_8b_w8a8.sh
```

For GSM8K, use its generated parquet files instead:

```bash
export TRAIN_FILE=~/data/gsm8k/train.parquet
export VAL_FILE=~/data/gsm8k/test.parquet
```

When this repository is checked out as `verl/recipe`, use:

```bash
bash recipe/int8_w8a8_qat/run_qwen3_8b_w8a8.sh
```

For the 16-NPU reference setup:

```bash
N_GPUS_PER_NODE=16 \
CKPT_DIR=/path/to/checkpoints/qwen3-8b-w8a8 \
bash int8_w8a8_qat/run_qwen3_8b_w8a8.sh
```

Additional Hydra overrides can be appended to the command:

```bash
bash int8_w8a8_qat/run_qwen3_8b_w8a8.sh \
    trainer.total_epochs=2 \
    trainer.save_freq=100
```

## Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `MODEL_PATH` | required | BF16 actor checkpoint used for FSDP training. |
| `ROLLOUT_MODEL_PATH` | required | Initial Ascend W8A8 rollout checkpoint and quantization metadata. |
| `TRAIN_FILE` | required | Training parquet path. |
| `VAL_FILE` | required | Validation parquet path. |
| `PROJECT_NAME` | `int8-w8a8-qat` | Logger project name. |
| `EXP_NAME` | `qwen3-8b-int8-w8a8-qat` | Experiment name. |
| `CKPT_DIR` | `./checkpoints/$EXP_NAME` | Checkpoint output directory. |
| `N_GPUS_PER_NODE` | `8` | Number of NPUs per node. |
| `NNODES` | `1` | Number of nodes. |
| `QAT` | `true` | Enable actor QAT. |
| `QAT_W_BIT` | `8` | Actor weight fake-quantization bit width; other values are rejected. |
| `SCALE_SOURCE` | `learned` | Rollout weight-scale source. |
| `USE_STOCHASTIC` | `1` | Enable deterministic value-hash stochastic rounding. |

### Scale Sources

| `SCALE_SOURCE` | Actor behavior | Rollout export behavior |
| --- | --- | --- |
| `learned` or `auto` | Learn per-channel scales with STE and LSQ-style scale gradients. | Export the latest learned scale and offset. |
| `calibrated` | Initialize fixed scales and offsets from checkpoint safetensors. | Reuse calibrated metadata and quantization range. |
| `online` | Do not register learned weight-scale parameters. | Recompute per-channel scales from current actor weights. |

The QAT linear layer also learns a SmoothScale reparameterization. Its forward
path divides activations by the scale and multiplies weights by the same scale,
then updates a slower EMA reference from the smoothed weights. The EMA term uses
stop-gradient, coupling forward statistics while decoupling the gradients of
SmoothScale and WeightScale.

## Validation and Limitations

Before reporting results from a new environment, validate:

- actor loss, reward, KL, and train/rollout log-prob correlation;
- initial and post-update actor-to-rollout weight synchronization;
- sensitive-layer floating-point fallback;
- rollout throughput under the same sampling settings as the BF16 baseline.

Observed rollout speedups depend on model size and workload. The current QAT
forward/backward path is unfused, so this recipe does not claim universal
end-to-end training acceleration. Accuracy and performance must be independently
verified on the target Ascend software stack.
