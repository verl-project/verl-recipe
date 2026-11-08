# Tree Prefix Reuse (TPR)

TPR reuses shared token prefixes across trajectories during VERL V1 Megatron Actor training. See [VERL RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the compatible VERL and Uni-Agent revisions.

## Requirements

- A working VERL V1 Megatron Actor environment with Megatron-Core and an Ascend NPU training stack.
- For the SWE example, follow the [Uni-Agent project](https://github.com/verl-project/uni-agent), [installation guide](https://uni-agent.readthedocs.io/en/latest/quickstart/installation.html), and [RL training guide](https://uni-agent.readthedocs.io/en/latest/quickstart/rl-training.html). Prepare the model, data, sandbox, and Ray environment as in Uni-Agent.
- Install the TPR package and patched VERL code in the Megatron Actor worker environment.

## Quick Start

Apply the patch to the compatible VERL checkout and install TPR:

```bash
export VERL_ROOT=/path/to/verl
export RECIPE_ROOT=/path/to/verl-recipe

git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
python -m pip install --no-deps --no-build-isolation -e "$RECIPE_ROOT/tpr"
```

Enable TPR in the existing Megatron Actor launcher:

```bash
+actor_rollout_ref.actor.megatron.tpr_enabled=True \
```

Set to `False` for native Full training. To run the included Uni-Agent SWE example with the prepared Uni-Agent environment:

```bash
bash "$RECIPE_ROOT/tpr/run_train.sh"
```

## Support

**Supported:** Dense causal full-attention (softmax attention) models using the compatible Megatron-Core attention path; TP, DP, CP, and SP; full recomputation (`uniform/full/1`).

**Not supported:** Pipeline parallelism (PP > 1), expert parallelism (EP > 1), MoE models, linear attention models, and hybrid linear-attention architectures.
