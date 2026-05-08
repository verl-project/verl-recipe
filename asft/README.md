# Anchored Supervised Fine-Tuning (ASFT) Recipe

This recipe adds **Anchored Supervised Fine-Tuning (ASFT)** to verl as a
self-contained `recipe/` (no changes to core verl modules). ASFT extends
*Dynamic Fine-Tuning (DFT)* with a KL anchor against a frozen reference
model, recovering DFT's tighter RL lower bound while removing its
distributional drift.

> Reference: He Zhu et al., *Anchored Supervised Fine-Tuning*, ICLR 2026
> ([arXiv:2509.23753](https://arxiv.org/abs/2509.23753)).
> Project page: https://github.com/zhuchichi56/ASFT

## What's included

```
recipe/asft/
├── README.md                 # this file
├── fsdp_sft_trainer_asft.py  # FSDP trainer with sft / dft / asft loss modes
├── sft_trainer.yaml          # default config (loss_mode, asft_kl_coef, ...)
├── prepare_data.py           # download/prep med + math parquet from HF chichi56/ASFT
└── run_asft.sh               # 8-GPU launcher: sweeps sft / dft / asft
```

The trainer is a fork of `verl/trainer/fsdp_sft_trainer.py` from verl
0.6.x, with the following additions:

1. `trainer.loss_mode ∈ {sft, dft, asft}`
2. `trainer.asft_kl_coef` (default `0.1`; we recommend `0.03` under bf16/fp16)
3. A frozen FSDP reference model built once for the `asft` mode
4. The loss
   ```
   L_ASFT = -E[ p_θ(y|x).detach() · log p_θ(y|x) ]   # DFT term
            + β · KL( p_θ(·|x) ‖ p_ref(·|x) )         # anchor term
   ```
5. An MCQ benchmark eval hook (medqa / mmlu / medmcqa) for in-training tracking

The ASFT trainer is intentionally placed under `recipe/` and does **not**
modify the core `verl/trainer/sft_trainer.py`; users opt in by launching
`recipe/asft/fsdp_sft_trainer_asft.py` instead of the standard SFT trainer.

## Quick start

```bash
# 1. Prepare data (downloads chichi56/ASFT med dataset to ./data/asft_med)
python recipe/asft/prepare_data.py --dataset med --output_dir ./data/asft_med

# 2. Train with ASFT on 8 GPUs (default sweeps sft/dft/asft)
LOSS_MODES=asft bash recipe/asft/run_asft.sh
```

Key knobs (env vars in `run_asft.sh`):

| Var               | Default  | Notes                              |
|-------------------|----------|------------------------------------|
| `LOSS_MODES`      | `"sft dft asft"` | space-separated subset       |
| `ASFT_KL_COEF`    | `0.03`   | recommended for bf16               |
| `EPOCHS`          | `3`      |                                    |
| `GLOBAL_BSZ`      | `128`    |                                    |
| `LR`              | `2e-5`   | full FT; for LoRA use `5e-4`       |
| `SAVE_EVERY`      | `200`    |                                    |

## Why a separate trainer (and not just a flag)?

ASFT requires a frozen reference model held in FSDP throughout training —
this changes memory accounting and the inner loop enough that bolting it
onto the canonical SFT trainer would muddy the common path. Keeping it as
an opt-in recipe matches the existing convention (cf. `dapo/`, `spin/`,
`sppo/`, `gvpo/`): one self-contained folder, no patches to core verl —
and lets users sweep `sft / dft / asft` with the same launcher for fair
comparison.

## Citation

```bibtex
@misc{zhu2025anchoredsupervisedfinetuning,
  title  = {Anchored Supervised Fine-Tuning},
  author = {He Zhu and Junyou Su and Peng Lai and Ren Ma and Wenjia Zhang and Linyi Yang and Guanhua Chen},
  year   = {2025},
  eprint = {2509.23753},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url    = {https://arxiv.org/abs/2509.23753}
}
```
