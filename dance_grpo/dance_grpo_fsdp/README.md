<p align="center">
<h1 align="center"> Recipe: Reinforcement learning for generative models using FSDP as the backend (DanceGRPO) </h1>

## 1. Environment installation ##

\[You are advised to use the matching environment version during model development.\]

Install CANN 8.5.0 from https://www.hiascend.com/cann/download
```shell
# Importing CANN Environment Variables
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh

# Creating the python3.11 conda environment
conda create -n test python=3.11
conda activate test

# Download and Install Verl
git clone https://github.com/volcengine/verl.git
cd verl
git checkout v0.7.0
pip install -r requirements-npu.txt
pip install -r requirements.txt
pip install -v -e .
cd ..

# Update the recipe directory.
git clone https://github.com/verl-project/verl-recipe.git
cp -rf verl-recipe/dance_grpo/dance_grpo_fsdp verl/recipe/dance_grpo/

# Installing mammothmoda Model
git clone https://github.com/bytedance/mammothmoda.git
cp verl/recipe/dance_grpo/mm2.patch
cd mammothmoda
git apply mm2.patch
cd ..
cp -r mammothmoda/mammothmoda2 verl/recipe/dance_grpo/

# Installing the HPSv3 Scoring Model
git clone https://github.com/MizzenAI/HPSv3.git
cd HPSv3
git checkout upgrade_transformers_version
pip install -e .
cd ..

# Installing Other Packages
pip install diffusers==0.35.1 peft==0.17.1 torch_npu==2.7.1 loguru==0.7.3 opencv-python-headless==4.10.0.84 tf-keras matplotlib==3.8.4

cd verl

# The directory structure after the preparation is as follows:
# HPSv3
# verl-recipe
# verl
# mammothmoda
# ├── recipe
#     ├── dance_grpo
#         ├── dance_grpo_fsdp
```

## 2. Dataset preparation ##

This repository include a demo dataset ` verl/recipe/dance_grpo/dance_grpo_fsdp/data/pickapic_single.json`. It is a subset of the 
HPDv3 dataset, available at https://huggingface.co/datasets/MizzenAI/HPDv3/blob/main/pickapic.json

## 3. Training model preparation ##

Mammothmoda2 model download address:

https://huggingface.co/bytedance-research/MammothModa2-Dev

## 4. Scoring Model preparation ##

1. Download the HPSv3 model: https://huggingface.co/MizzenAI/HPSv3/tree/main
In the training script, change the value of +actor_rollout_ref.model.reward_model_path to the HPSv3 weight path.
```shell
+actor_rollout_ref.model.reward_model_path=/home/CKPT/HPSv3/HPSv3.safetensors
```

2. Download the Qwen2.5VL 7B model: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct/tree/main
Modify the HPSv3/hpsv3/config/HPSv3_7B.yaml file to configure the directory for the Qwen2-VL-7B model.
```shell
model_name_or_path: "/home/CKPT/Qwen2-VL-7B-Instruct"
```

## 5. Model RL training ##

Modifying Parameters in the verl-recipe/dance_grpo/dance_grpo_fsdp/run_verl_dance.sh RL Training Script

### 1. Environment Variables

| Parameter                     | Default                    | Description                                                                               |
| ----------------------------- | -------------------------- | ----------------------------------------------------------------------------------------- |
| `CUSTOM_TIMESTAMP`            | `$(date +"%Y%m%d_%H%M%S")` | Timestamp used to generate unique log file names for each run                             |
| `PYTORCH_NPU_ALLOC_CONF`      | `expandable_segments:True` | NPU memory allocation strategy; enables expandable segments for better memory utilization |
| `HYDRA_FULL_ERROR`            | `1`                        | Hydra full error output; prints complete stack traces for debugging                       |
| `HCCL_HOST_SOCKET_PORT_RANGE` | `auto`                     | HCCL (Huawei Collective Communication Library) host-side socket port range                |
| `HCCL_NPU_SOCKET_PORT_RANGE`  | `auto`                     | HCCL NPU-side socket port range                                                           |
| `WORKING_DIR`                 | `${PWD}`                   | Working directory; affects checkpoint and log storage paths                               |

---

### 2. Data & Model Paths

| Parameter           | Default                    | Description                                                             |
| ------------------- | -------------------------- | ----------------------------------------------------------------------- |
| `TRAIN_DATA`        | `.../pickapic_single.json` | Path to training data file (PickAPic image preference dataset)          |
| `VAL_DATA`          | Same as `TRAIN_DATA`       | Path to validation data file; defaults to the same as training data     |
| `MODEL_PATH`        | `.../MammothModa2-Dev`     | Path to the base diffusion model (shared by Actor and Reference policy) |
| `REWARD_MODEL_PATH` | `.../HPSv3.safetensors`    | Path to the reward model weights (HPSv3 human preference scoring model) |

---

### 3. Training Batch & Steps

| Parameter              | Default | Description                                                                               |
| ---------------------- | ------- | ----------------------------------------------------------------------------------------- |
| `TRAIN_BATCH_SIZE`     | `8`     | Number of prompts sampled per training iteration                                          |
| `PPO_BATCH_SIZE`       | `8`     | PPO mini-batch size (also equals micro-batch size per GPU)                                |
| `LOG_PROB_BATCH_SIZE`  | `8`     | Micro-batch size per GPU when computing log probabilities                                 |
| `ROLLOUT_N`            | `8`     | Number of images generated per prompt (GRPO needs multiple samples to estimate advantage) |
| `TOTAL_TRAINING_STEPS` | `500`   | Maximum total training steps                                                              |
| `TOTAL_EPOCHS`         | `500`   | Maximum total training epochs (training stops when either limit is reached)               |
| `MAX_PROMPT_LENGTH`    | `512`   | Maximum token length for prompts; overlong prompts are filtered/truncated                 |

---

### 4. Checkpoint Controls

| Parameter                | Default                          | Description                                                                                                 |
| ------------------------ | -------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `CHECKPOINT_ROOT_DIR`    | `${WORKING_DIR}/checkpoints/...` | Root directory for checkpoints; saved as `global_step_<step>/actor/actor_model.pt`                          |
| `SAVE_FREQ`              | `50`                             | Save a checkpoint every N steps (`-1` means no saving)                                                      |
| `MAX_ACTOR_CKPT_TO_KEEP` | `3`                              | Maximum number of Actor checkpoints to keep; oldest is deleted when exceeded                                |
| `RESUME_MODE`            | `disable`                        | Resume mode: `disable` (no resume), `auto` (resume from latest), `resume_path` (resume from specified path) |
| `RESUME_FROM_PATH`       | `""`                             | Checkpoint path to resume from when `RESUME_MODE=resume_path`                                               |

---

### 5. Rollout (Image Generation / Sampling) Parameters

| Parameter                  | Default               | Description                                                                                         |
| -------------------------- | --------------------- | --------------------------------------------------------------------------------------------------- |
| `ROLLOUT_HEIGHT`           | `512`                 | Height of generated images in pixels                                                                |
| `ROLLOUT_WIDTH`            | `512`                 | Width of generated images in pixels                                                                 |
| `ROLLOUT_STEPS`            | `40`                  | Number of denoising steps in diffusion inference (DDIM steps)                                       |
| `ROLLOUT_ETA`              | `0.3`                 | DDIM eta parameter controlling stochasticity; 0 = deterministic, 1 = fully stochastic               |
| `ROLLOUT_INIT_SAME_NOISE`  | `true`                | Whether rollouts from the same prompt use the same initial noise (true in GRPO for fair comparison) |
| `ROLLOUT_CFG_SCALE`        | `3.0`                 | Classifier-Free Guidance scale for image conditioning                                               |
| `ROLLOUT_TEXT_CFG_SCALE`   | `3.0`                 | Classifier-Free Guidance scale for text conditioning                                                |
| `ROLLOUT_VAE_SCALE_FACTOR` | `16`                  | VAE latent space scale factor (ratio between latent and pixel space)                                |
| `ROLLOUT_OUTPUT_DIR`       | `${WORKING_DIR}/logs` | Output directory for rollout-generated images                                                       |

---

### 6. Actor (Policy Network) Parameters

| Parameter                   | Default                      | Description                                                                                                                                                           |
| --------------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `ACTOR_TIMESTEP_FRACTION`   | `0.6`                        | DANCE core parameter — fraction of timesteps to optimize during training; only the first 60% of timesteps are updated (early timesteps matter more for image quality) |
| `ACTOR_SAMPLING_STEPS`      | Same as `ROLLOUT_STEPS` (40) | Number of sampling steps during Actor training; defaults to match rollout                                                                                             |
| `ACTOR_CLIP_RANGE`          | `0.0001`                     | PPO clip range ε; limits policy update magnitude. Extremely small value indicates very conservative updates                                                           |
| `ACTOR_ADV_CLIP_MAX`        | `5.0`                        | Advantage function clipping upper bound; prevents extreme advantage values from destabilizing training                                                                |
| `ACTOR_KL_COEFF`            | `0.0`                        | KL divergence penalty coefficient in PPO loss; 0 means no KL constraint                                                                                               |
| `ACTOR_MAX_GRAD_NORM`       | `1.0`                        | Maximum gradient norm for gradient clipping; prevents gradient explosion                                                                                              |
| `ACTOR_LR`                  | `5e-6`                       | Actor optimizer learning rate                                                                                                                                         |
| `ACTOR_WEIGHT_DECAY`        | `0.01`                       | Weight decay (L2 regularization) coefficient                                                                                                                          |
| `ACTOR_FREEZE_NON_ATTN_FFN` | `False`                      | Whether to freeze all parameters except attention and FFN layers (saves memory)                                                                                       |

---

### 7. Hydra Command-Line Override Parameters

#### Data

| Parameter                      | Value     | Description                                                                          |
| ------------------------------ | --------- | ------------------------------------------------------------------------------------ |
| `data.max_response_length`     | `128`     | Maximum response length (kept for compatibility; less relevant for diffusion models) |
| `data.filter_overlong_prompts` | `True`    | Filter out prompts exceeding `max_prompt_length`                                     |
| `data.truncation`              | `'error'` | Truncation strategy; `error` raises an exception instead of truncating               |

#### Optimizer

| Parameter                                                       | Value        | Description                                                               |
| --------------------------------------------------------------- | ------------ | ------------------------------------------------------------------------- |
| `actor_rollout_ref.actor.optim.lr_scheduler.name`               | `'constant'` | Learning rate scheduler type; constant means fixed learning rate          |
| `actor_rollout_ref.actor.optim.lr_scheduler.num_warmup_steps`   | `0`          | Number of warmup steps; 0 means no warmup                                 |
| `actor_rollout_ref.actor.optim.lr_scheduler.num_training_steps` | `10000`      | Total steps for the scheduler (no effect with constant scheduler)         |
| `actor_rollout_ref.actor.optim.lr_scheduler.num_cycles`         | `1`          | Number of cycles for cosine scheduler (no effect with constant scheduler) |

#### Strategy & Training

| Parameter                                               | Value   | Description                                                                                      |
| ------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------ |
| `actor_rollout_ref.actor.strategy`                      | `fsdp`  | Use FSDP (Fully Sharded Data Parallel) strategy for distributed training                         |
| `actor_rollout_ref.actor.fsdp_config.offload_policy`    | `False` | Do not offload FSDP parameters to CPU                                                            |
| `actor_rollout_ref.model.enable_gradient_checkpointing` | `True`  | Enable gradient checkpointing; trades computation for memory savings                             |
| `algorithm.adv_estimator`                               | `grpo`  | Use GRPO (Group Relative Policy Optimization) for advantage estimation; no Critic network needed |
| `algorithm.use_kl_in_reward`                            | `False` | Do not add KL penalty to the reward                                                              |

#### Rollout

| Parameter                        | Value        | Description                                                  |
| -------------------------------- | ------------ | ------------------------------------------------------------ |
| `actor_rollout_ref.rollout.mode` | `"sync"`     | Synchronous rollout mode; all GPUs generate then synchronize |
| `actor_rollout_ref.rollout.n`    | `$ROLLOUT_N` | Number of rollouts per prompt                                |

#### Trainer

| Parameter                  | Value                       | Description                                                                      |
| -------------------------- | --------------------------- | -------------------------------------------------------------------------------- |
| `trainer.critic_warmup`    | `0`                         | Critic warmup steps (0 because GRPO does not use a Critic)                       |
| `trainer.logger`           | `console`                   | Log output to console                                                            |
| `trainer.val_before_train` | `False`                     | Do not run validation before training starts                                     |
| `trainer.project_name`     | `'mammothmoda2_project'`    | Project name for logging/experiment management                                   |
| `trainer.experiment_name`  | `'mammothmoda2_experiment'` | Experiment name for logging/experiment management                                |
| `trainer.n_gpus_per_node`  | `8`                         | Number of GPUs per node                                                          |
| `trainer.nnodes`           | `1`                         | Number of nodes (single-machine training)                                        |
| `trainer.test_freq`        | `1`                         | Run test/validation every 1 step                                                 |
| `trainer.device`           | `npu`                       | Use Huawei NPU device for training                                               |
| `trainer.fixed_eval`       | `False`                     | Do not use a fixed evaluation set (`++` prefix means appending a new config key) |

## 6.Start RL training.

```shell
# cd verl

bash recipe/dance_grpo/dance_grpo/run_verl_dance.sh
```

## Result

![Alt text](https://github.com/user-attachments/assets/60037777-5644-49ac-a5de-fbdc5c9d7e17)

## Design
### Module Reference

| Module        | File                                                                      | Description                                                                                                                             |
| ------------- | ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **Entry**     | `main_dance.py`                                                           | Hydra entry point. Initializes Ray, wires up workers/datasets/trainer, launches training.                                               |
| **Algorithm** | `algorithms/grpo_diffusion.py`                                            | Core DANCE-GRPO math: GRPO advantage normalization, SDE log-prob, PPO-clip loss, KL regularizer.                                        |
| **Config**    | `config/actor.py`, `config/optimizer.py`, `config/dance_ppo_trainer.yaml` | Hydra config. Extends base FSDP configs with diffusion-specific fields (timestep_fraction, sampling_steps, etc.).                       |
| **Dataset**   | `dataset/pickapic_dataset.py`                                             | Lightweight JSON dataset loading only the `prompt` field. Provides collate/sampler factories.                                           |
| **Reward**    | `reward/hpsv3_reward_manager.py`                                          | Scores images with HPSv3 model. Keeps reward model on CPU, moves to accelerator only during scoring.                                    |
| **Trainer**   | `trainer/dance_trainer.py`                                                | Diffusion-specific training loop (no critic, no token-level KL). Handles checkpointing, logging, fixed-eval.                            |
| **Actor**     | `workers/diffusion_actor.py`                                              | PPO policy for DiT. Runs GRPO advantage, SDE log-prob recomputation, PPO-clip update across sampled timesteps.                          |
| **Rollout**   | `workers/diffusion_rollout.py`                                            | `GRPOMockScheduler` injects SDE math and records trajectories. `DiffusionRollout` runs MLLM→DiT→VAE pipeline.                           |
| **Worker**    | `workers/diffusion_worker.py`                                             | Ray remote worker. Owns all models (frozen MLLM/VAE, trainable DiT). Orchestrates rollout→reward→update. Manages FSDP2 and checkpoints. |
| **Utils**     | `utils/hpsv3_client.py`, `utils/rl_latent_dataset*.py`                    | HTTP client for remote HPSv3 scoring; general RLHF dataset classes from verl framework.                                                 |

### Data Flow

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
