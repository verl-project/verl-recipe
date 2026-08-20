# TMax

This recipe reproduces [TMax](https://arxiv.org/abs/2606.23321), an RL recipe for terminal agents.

## Required `verl` version

This recipe tracks upstream `main` and was most recently tested at [`8bda42207c`](https://github.com/verl-project/verl/commit/8bda42207cc08a947a49587d38315647740b9e14); see [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for installation details.

## Data

Prepare the [TMax-15K-Harbor](https://hub.harborframework.com/datasets/tmax/TMax-15K-Harbor/latest) dataset:

```bash
pip install harbor
python3 recipe/tmax/prepare_data.py
```

The script downloads the Harbor tasks and writes `train.parquet` and `val.parquet` to `~/data/tmax`.

## Training

This recipe requires Python >= 3.11, vLLM >= 0.26, and Modal. Install the dependencies and set Modal credentials as follows:

```bash
pip install modal "vllm>=0.26"
export MODAL_TOKEN_ID="<token-id>"
export MODAL_TOKEN_SECRET="<token-secret>"
```

Run the paper-scale Qwen3.5-9B DPPO-TV reproduction on 16 learner and 48 rollout GPUs:

```bash
bash recipe/tmax/run_tmax_9b.sh
```
