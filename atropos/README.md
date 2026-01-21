# Atropos-VERL Integration (GRPO)

![GRPO + Atropos training metrics](wandb_grpo_step301.png)

- `val-aux/openai/gsm8k/reward/mean@1` and `val-core/openai/gsm8k/acc/mean@1`
  show a steady rise across training steps.
- `critic/rewards/mean` trends upward with expected noise for GSM8K.
- Stability metrics (KL, entropy, grad norm) remain bounded.

Focused **GRPO integration** between Atropos environments and VERL:
- **GRPO training** with optional **token-level advantage overrides** from Atropos.
- **VERL-managed inference servers** (vLLM) with endpoint registration.
- A **working GSM8K example** that improves task metrics.

Note: GRPO here is implemented via VERL’s PPO trainer scaffold with
`adv_estimator: grpo` and critic disabled. This is the canonical VERL GRPO path.

## Key Components

- `atropos_integration.py`: Atropos API client + advantage override logic
- `grpo_atropos_trainer.py`: GRPO trainer with Atropos token-level advantages
- `launch_atropos_verl_services.py`: Orchestrates Atropos API, vLLM, and training
- `config/atropos_grpo_small.yaml`: Minimal GRPO config used by the launcher

## Run Commands

### Launch services + training (recommended)

```bash
cd verl
python recipe/atropos/launch_atropos_verl_services.py \
  --config recipe/atropos/config/atropos_grpo_small.yaml

# You can pass standard VeRL overrides after `--`:
python recipe/atropos/launch_atropos_verl_services.py \
  --config recipe/atropos/config/atropos_grpo_small.yaml -- \
  trainer.total_epochs=1 data.train_batch_size=2 actor_rollout_ref.rollout.n=2
```

This starts:
- Atropos API server
- vLLM inference server
- GRPO training via `RayGRPOAtroposTrainer`


