# Low-Precision RL Recipes

This directory collects RL training recipes that leverage low-precision (FP8) computation for improved efficiency. Recipes are organized by precision strategy:

- **FP8 End-to-End**: Both training and rollout inference run in FP8.
- **FP8 Rollout Only**: Training in BF16, rollout inference quantized to FP8.

> [!TIP]
> For detailed implementation notes, benchmark results, and parameter references, see the [verl FP8 documentation](https://verl.readthedocs.io/en/latest/advance/fp8.html).

## Data Preparation

All scripts in this directory use the same datasets as the DAPO recipe. Please follow the [DAPO data preparation instructions](../dapo/README.md#quickstart) to run `prepare_dapo_data.sh` before launching any training.

## Cluster Setup

The scripts in this directory define training configurations and submit jobs via `ray job submit`. They do **not** handle Ray cluster setup, which depends on your environment (Slurm, manual Ray, SkyPilot, etc.).

Please refer to the [verl multinode training documentation](https://verl.readthedocs.io/en/latest/start/multinode.html) for instructions on setting up a multi-node Ray cluster.

## FP8 End-to-End (Training + Rollout)

### Qwen3-30B-A3B (Megatron)

- **Script**: [run_dapo_qwen3_moe_30b_megatron_fp8e2e.sh](./run_dapo_qwen3_moe_30b_megatron_fp8e2e.sh)
- **Training backend**: Megatron + Megatron-Bridge
- **Min hardware**: 2 nodes × 8 GPUs (H100), CUDA 12.9+
- **Docker image**: `verlai/verl:vllm012.latest`
- **Verified verl commit**: `4849643d`

> [!IMPORTANT]
> **Required environment variables:**
> ```bash
> export NVTE_FP8_BLOCK_SCALING_FP32_SCALES=1  # Use FP32 scales for TE block-wise FP8
> export VLLM_USE_DEEP_GEMM_E8M0=0             # Use FP32 scales for vLLM with DeepGEMM
> ```
> When vLLM uses [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM) as its GEMM backend, it defaults to E8M0 scales. Set `VLLM_USE_DEEP_GEMM_E8M0=0` to explicitly use FP32 scales instead.
>
> For multi-node setups, these variables must be added to the Ray runtime env yaml (default: `verl/trainer/runtime_env.yaml`) so they propagate to all worker nodes.

**FP8 configuration highlights:**

```yaml
# FP8 training via Transformer Engine
actor_rollout_ref.actor.megatron.override_transformer_config:
  fp8: "hybrid"            # FP8 in both forward and backward, support hybrid and e4m3
  fp8_recipe: "blockwise"  # block-wise scaling (requires CUDA 12.9+)

# FP8 optimizer
actor_rollout_ref.actor.optim.override_optimizer_config:
  fp8_recipe: "blockwise"

# FP8 rollout
actor_rollout_ref.rollout:
  quantization: fp8
```

## FP8 Rollout Only

FP8 rollout-only recipes apply FP8 quantization during rollout inference while keeping BF16 for training. This reduces GPU memory usage during generation without modifying the training precision.

### Qwen3-30B-A3B (FSDP)

- **Script**: [run_dapo_qwen3_moe_30b_fsdp_fp8_rollout.sh](./run_dapo_qwen3_moe_30b_fsdp_fp8_rollout.sh)
- **Training backend**: FSDP
- **Min hardware**: 2 nodes × 8 GPUs (H100)
- **Rollout**: vLLM (async, FP8 quantized)

**FP8 configuration highlights:**

```yaml
# Only rollout uses FP8; training remains BF16
actor_rollout_ref.rollout:
  quantization: fp8
```
