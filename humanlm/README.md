# Recipe: HumanLM

Train user simulators by aligning on psychological state dimensions (belief, emotion, stance, value, goal, communication) instead of imitating response text.

**Paper:** [HUMANLM: Simulating Users with State Alignment Beats Response Imitation](https://humanlm.stanford.edu/HumanLM_paper.pdf)

**Project Page:** [https://humanlm.stanford.edu/](https://humanlm.stanford.edu/)

## Environment Setup

### Install Dependencies

```bash
# Install verl with trainer support
pip install -e /path/to/verl  # verl repo with trainer module

# Additional dependencies
pip install litellm datasets polars
```

### Configure API Keys
```bash
# Required for LLM-as-judge rewards (RL training)
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

---

## Datasets

Official HuggingFace datasets:

| Dataset | HuggingFace Repo | `--dataset` arg |
|---------|------------------|-----------------|
| Humanual-Books | `snap-stanford/humanual-book` | `amazon` |
| Humanual-Opinion | `snap-stanford/humanual-opinion` | `reddit` |
| Humanual-Politics | `snap-stanford/humanual-politics` | `medium` |
| Humanual-News | `snap-stanford/humanual-news` | `youtube` |
| Humanual-Chat | `snap-stanford/humanual-chat` | `wildchat_english` |
| Humanual-Email | `snap-stanford/humanual-email` | `enron` |

---

## SFT Training

### Step 1: Process Dataset

Convert HuggingFace dataset to SFT format:

```bash
# No-thinking mode (response only)
python -m humanlm.process_dataset \
    --dataset amazon \
    --raw_dataset_repo snap-stanford/humanual-book \
    --save_data_dir ./data/humanual-book \
    --sft \
    --no_tag

# With thinking traces (requires API key for trace generation)
python -m humanlm.process_dataset \
    --dataset amazon \
    --raw_dataset_repo snap-stanford/humanual-book \
    --save_data_dir ./data/humanual-book \
    --sft \
    --thinking_sft \
    --thinking_model gpt-4o-mini
```

This creates:
```
./data/humanual-book/
└── sft/
    └── r_no_tag/
        ├── train.parquet
        ├── val.parquet
        └── test.parquet
```

### Step 2: Run SFT Training

```bash
DATASET_DIR=./data/humanual-book bash humanlm/train_sft_humanlm.sh \
    "0,1,2,3,4,5,6,7" \
    amazon \
    no_thinking
```

**Arguments:**
| Position | Name | Example | Description |
|----------|------|---------|-------------|
| 1 | GPU_LIST | `"0,1,2,3,4,5,6,7"` | Comma-separated GPU IDs |
| 2 | DATASET_NAME | `amazon` | Dataset identifier |
| 3 | THINKING_MODE | `no_thinking` or `thinking` | Whether to use thinking traces |

**Environment Variables:**
| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET_DIR` | (required) | Path to processed data |
| `OUTPUT_ROOT` | `./outputs` | Where to save checkpoints |
| `HF_CACHE_DIR` | system default | HuggingFace cache location |

**Output:**
- Model checkpoints: `./outputs/sft_amazon_no_thinking_r_no_tag/`
- WandB project: `humanlm`

---

## RL Training (GRPO)

Before training, update ```cluster_config.sh``` with your custom project paths and your .env file.  

### Train HumanLM
```bash
bash humanlm/train_rl_humanlm.sh \
    "0,1,2,3,4,5,6,7" \
    amazon \
    train_humanlm \
    "" \
    base
```

### Evaluation
```bash
bash humanlm/train_rl_humanlm.sh \
    "0,1,2,3,4,5,6,7" \
    amazon \
    eval_only \
    "/path/to/checkpoint" \
    humanlm
```

---

## Citation

```bibtex
@article{wu2026humanlm,
  title={HUMANLM: Simulating Users with State Alignment Beats Response Imitation},
  url={https://humanlm.stanford.edu/},
  author={Wu, Shirley and Choi, Evelyn and Khatua, Arpandeep and
          Wang, Zhanghan and He-Yueya, Joy and Weerasooriya, Tharindu Cyril and
          Wei, Wei and Yang, Diyi and Leskovec, Jure and Zou, James},
  year={2026}
}
```
