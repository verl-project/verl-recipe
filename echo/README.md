# ECHO

This recipe ports [ECHO](https://arxiv.org/abs/2605.24517) to verl, following the [official implementation](https://github.com/microsoft/echo-rl). ECHO combines GRPO with an on-policy cross-entropy loss over environment-observation tokens.

## Required `verl` version

This recipe is pinned to upstream `verl` commit [`8bda42207c`](https://github.com/verl-project/verl/commit/8bda42207cc08a947a49587d38315647740b9e14); see [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for installation details. The ECHO-specific rollout field and auxiliary-loss integration are vendored in this recipe, so no changes to `verl` source are required.

## Data

The paper's training data was not released. This recipe uses [obiwan96/endless-terminals](https://huggingface.co/datasets/obiwan96/endless-terminals), with 100 validation tasks sampled using seed 42 and the remaining 2,392 tasks used for training.

From the verl root:

```bash
python3 recipe/echo/prepare_data.py
```

## Training

Install Modal and set the required credentials:

```bash
pip install modal
export MODAL_TOKEN_ID="<token-id>"
export MODAL_TOKEN_SECRET="<token-secret>"
```

```bash
bash recipe/echo/run_echo.sh
```

The launcher defaults to Qwen3-8B with 16 rollouts per training prompt and an ECHO coefficient of `0.05`.

## Reproduction notes

When the paper and released config disagree, this recipe follows the released config:

- AdamW betas: paper `[0.9, 0.95]`; config `[0.9, 0.999]`.
- PPO clipping: paper `[0.2, 0.28]`; config `[0.2, 0.2]`.

The port makes two simplifications relative to the reference:

- It uses verl's XML tool parser, which provides less granular parsing warnings.
- It normalizes the ECHO loss by selected environment tokens rather than the full observation-token count.
