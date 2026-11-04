# Tree Prefix Reuse (TPR)

TPR reuses shared prefixes across agent trajectories during VERL Megatron Actor training. See [RFC #7561](https://github.com/verl-project/verl/issues/7561).

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt).

## Requirements

- An Ascend NPU environment with Megatron-Core, MindSpeed, Ray, vLLM-Ascend and Megatron-Bridge. The example uses eight NPUs.
- Uni-Agent: [Install Uni-Agent](https://uni-agent.readthedocs.io/en/latest/quickstart/installation.html). Use the revision specified in [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt).
- Docker CLI for the included SWE-ReBench sandbox configuration.

The SWE-ReBench sandbox image uses `linux/amd64`. Uni-Agent installs Claude Code inside the sandbox automatically when network access is available.

## Quick Start

This example assumes you already have a VERL checkout. Install Uni-Agent using the instructions above, and prepare a local Qwen3-4B checkpoint.

Set the paths to your existing checkouts and model:

```bash
export VERL_ROOT=/path/to/verl
export RECIPE_ROOT=/path/to/verl-recipe
export UNIAGENT_ROOT=/path/to/uni-agent
export MODEL_PATH=/path/to/Qwen3-4B
```

Apply the TPR patch to a clean VERL checkout:

```bash
git -C "$VERL_ROOT" apply --check "$RECIPE_ROOT/tpr/verl-tpr.patch"
git -C "$VERL_ROOT" apply "$RECIPE_ROOT/tpr/verl-tpr.patch"
```

Apply the patch only once.

Prepare the SWE-ReBench training data:

```bash
export TRAIN_FILE="$HOME/data/tpr/swe_rebench_pelita875.parquet"
mkdir -p "$(dirname "$TRAIN_FILE")"
python "$RECIPE_ROOT/tpr/prepare_swe_rebench.py" --output "$TRAIN_FILE"
```

Start training:

```bash
export TPR_ENABLED=True
bash "$RECIPE_ROOT/tpr/run_uniagent_rebench_ppo.sh"
```

The example runs 10 training steps by default. Set `TPR_ENABLED=False` to use the Base training path. Logs are saved under `$VERL_ROOT/outputs/tpr/`.
