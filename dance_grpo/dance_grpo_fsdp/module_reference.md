# dance_grpo_fsdp Module Reference

| Module | File | Description |
|--------|------|-------------|
| **Entry** | `main_dance.py` | Hydra entry point. Initializes Ray, wires up workers/datasets/trainer, launches training. |
| **Algorithm** | `algorithms/grpo_diffusion.py` | Core DANCE-GRPO math: GRPO advantage normalization, SDE log-prob, PPO-clip loss, KL regularizer. |
| **Config** | `config/actor.py`, `config/optimizer.py`, `config/dance_ppo_trainer.yaml` | Hydra config. Extends base FSDP configs with diffusion-specific fields (timestep_fraction, sampling_steps, etc.). |
| **Dataset** | `dataset/pickapic_dataset.py` | Lightweight JSON dataset loading only the `prompt` field. Provides collate/sampler factories. |
| **Reward** | `reward/hpsv3_reward_manager.py` | Scores images with HPSv3 model. Keeps reward model on CPU, moves to accelerator only during scoring. |
| **Trainer** | `trainer/dance_trainer.py` | Diffusion-specific training loop (no critic, no token-level KL). Handles checkpointing, logging, fixed-eval. |
| **Actor** | `workers/diffusion_actor.py` | PPO policy for DiT. Runs GRPO advantage, SDE log-prob recomputation, PPO-clip update across sampled timesteps. |
| **Rollout** | `workers/diffusion_rollout.py` | `GRPOMockScheduler` injects SDE math and records trajectories. `DiffusionRollout` runs MLLM→DiT→VAE pipeline. |
| **Worker** | `workers/diffusion_worker.py` | Ray remote worker. Owns all models (frozen MLLM/VAE, trainable DiT). Orchestrates rollout→reward→update. Manages FSDP2 and checkpoints. |
| **Utils** | `utils/hpsv3_client.py`, `utils/rl_latent_dataset*.py` | HTTP client for remote HPSv3 scoring; general RLHF dataset classes from verl framework. |

## Data Flow

```
main_dance.py → TaskRunner.run()
  ├─ PickAPicDataset (prompts)
  └─ RayDANCETrainer.fit() [loop]
       ├─ generate_sequences()
       │    ├─ DiffusionRollout (GRPOMockScheduler + MLLM→DiT→VAE)
       │    └─ HPSv3RewardManager (score images)
       └─ update_actor()
            └─ DiffusionActor.update_policy()
                 ├─ compute_grpo_advantages()
                 ├─ compute_sde_log_prob()
                 ├─ compute_diffusion_ppo_loss()
                 └─ compute_mean_prediction_kl() [optional]
```
