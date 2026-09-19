# Fully asynchronous policy training

Launch examples for `verl.experimental.fully_async_policy`, which runs training
and rollout concurrently on separate Ray placement groups.

## Required `verl` version

See [REQUIRED_VERL.txt](REQUIRED_VERL.txt) for the source revision pinned to
[verl @e2ac8f62](https://github.com/verl-project/verl/commit/e2ac8f6222801d5e8ce50447b0c3c2d9237e4770).

## GLM-5.2 on Ascend NPUs

[run_glm5_2_megatron_fully_async.sh](examples/ascend/run_glm5_2_megatron_fully_async.sh)
uses Megatron for training and vLLM-Ascend for rollout on DAPO-Math-17k, with
GRPO advantages and GSPO policy loss by default. The script preserves the settings
and dependency commit table from the source PR.

1. Install the pinned verl revision and the Ascend dependencies listed at the top
   of the script on every node.
2. Start a Ray cluster with 32 nodes, each with 16 NPUs. The default configuration
   allocates 16 nodes to training and 16 separate nodes to rollout.
3. Make the model weights and parquet datasets available at the configured paths
   on all nodes, and run from the **verl repository root**, with this repository
   checked out as `recipe/`:

```bash
MODEL_PATH=/shared/glm52_weights \
TRAIN_FILE=/shared/datasets/dapo-math-17k.parquet \
TEST_FILE=/shared/datasets/dapo-math-17k.parquet \
bash recipe/fully_async_policy/examples/ascend/run_glm5_2_megatron_fully_async.sh
```

This example uses upstream's fully asynchronous trainer. The separate
[`async_flow`](../async_flow/README.md) recipe implements a four-worker pipeline,
and [`partial_rollout`](../partial_rollout/README.md) implements synchronous
training with interrupted and resumed rollouts.
