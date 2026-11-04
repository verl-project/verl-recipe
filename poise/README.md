# POISE

**Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States** — accepted at **NeurIPS 2026**.

[Paper](https://arxiv.org/abs/2605.07579)  · [Project](https://holi-lab.github.io/POISE/)

POISE estimates values from the actor's hidden states and token entropies using PCA and ridge regression. With two independent rollouts per prompt, each rollout predicts its sibling's reward and uses its sibling's prediction as the policy baseline:

$$y_1=r_2,\quad y_2=r_1,\qquad A_1=r_1-v(h_2),\quad A_2=r_2-v(h_1).$$

Implemented on **veRL's V1 synchronous trainer and FSDP engine**. Hidden states come from the log-probability forward pass. Domain probes are refitted after actor updates; checkpoints include probes and their buffers.

## Setup

Place this recipe at `verl/recipe/poise`. From the veRL root, install the version pinned in [REQUIRED_VERL.txt](REQUIRED_VERL.txt):

```bash
git checkout 8718ca30a3f002f93b7c4fd99b9b2506718681bc
uv sync --locked --extra fsdp --extra vllm
source .venv/bin/activate
uv pip install -r recipe/poise/requirements.txt
```

The pinned environment requires CUDA 13 driver support, either natively or through NVIDIA's [forward compatibility package](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html).

## Training

Example using veRL's GSM8K data preparation and reward:

```bash
python examples/data_preprocess/gsm8k.py --local_save_dir data/gsm8k
bash recipe/poise/run_qwen3_4b.sh \
  data.train_files=data/gsm8k/train.parquet data.val_files=data/gsm8k/test.parquet \
  '~poise.buffer_rows.code' '~poise.buffer_rows.other'
```

For OLMo3, use `run_olmo3_7b.sh` with the same arguments. For other datasets, supply veRL-format parquet files and set `reward.custom_reward_function.path` to a scorer returning correctness in `[0, 1]`.

`data_source` selects independent math/code/other probe buffers. Remove unused domains as shown above; every configured domain needs bootstrap examples. Defaults: 8 GPUs, 200 steps, 2 rollouts per prompt. Supported: FSDP/FSDP2, single-turn generation, sequence-parallel size 1 and synchronous checkpoints on local/shared storage. Inspect resolved settings with `--cfg job --resolve`.

```bibtex
@article{choi2026poise,
  title={Your Language Model is Its Own Critic: Reinforcement Learning with Value Estimation from Actor's Internal States},
  author={Choi, Yunho and Lim, Jongwon and Ahn, Woojin and Oh, Minjae and Shim, Jeonghoon and Jo, Yohan},
  journal={arXiv preprint arXiv:2605.07579},
  year={2026}
}
```
