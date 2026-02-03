# SDPO: Self-Distillation Policy Optimization

This recipe implements the training infrastructure for Self-Distillation Policy Optimization (SDPO) using the verl framework with 1 GPU.

**References:** 
- [SDPO: Self-Distillation Policy Optimization](https://arxiv.org/abs/2601.20802)
- [Original veRL Implementation from Authors](https://github.com/lasgroup/SDPO)

## Setup

First, download the Qwen3 VL 4b model:
```
huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct \
  --local-dir "$HOME/models/Qwen2.5-VL-7B-Instruct" \
  --local-dir-use-symlinks False
```

Next, setup the dataset:
```
python examples/data_preprocess/geo3k.py
```