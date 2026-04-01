# Confidence-Aligned On-Policy Distillation (TCAD)

Official research implementation of **"Learning from What the Teacher Trusts: Confidence-Aligned On-Policy Distillation"**.

## Method Overview
- **TCAD** (teacher-confidence reweighting) 
- **Support-Symmetric Truncation**
- **Coverage-Aware Regularization**

## TL;DR
On-policy distillation can overfit to low-quality student rollouts (hallucinations/flawed reasoning).  
**TCAD** reweights on-policy trajectories by the **teacher's sequence-level confidence** (length-normalized log-likelihood), acting as a lightweight "soft rejection sampling" without reward models.

## Key Features
- **Teacher-confidence reweighting**: Weight student rollouts by teacher's confidence scores
- **Support-Symmetric Truncation**: Build Top-K support using both teacher + student high-probability tokens
- **Coverage-Aware Regularization**: Prevent probability mass leakage outside selected support

## Project Structure
```
gkd/fsdp/
├── config/ - Configuration files for different components
├── dp_actor.py - Distributed actor implementation
├── fsdp_workers.py - FSDP worker processes
├── main_opkd.py - Main training entry point
├── ray_trainer.py - Ray-based distributed trainer
└── run_opkd.sh - Launch script
```

## Quick Start
```bash
bash run_opkd.sh
```

## Reference

```bibtex
@misc{chen2026towards,
  title        = {Learning from What the Teacher Trusts: Confidence-Aligned On-Policy Distillation},
  author       = {Chen, Xinran and Chen, Jiamin and Kong, Rui and Li, Yuchen and Wang, Yu and Li, Lei and Wu, Hui and Xu, Han and Cai, Hengyi and Wang, Shuaiqiang and Yin, Dawei},
  journal      ={arXiv preprint},
  year         = {2026},
}
```