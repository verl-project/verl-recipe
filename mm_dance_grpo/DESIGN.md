# MMDanceGRPO Design Document

## 1. Overview

MMDanceGRPO is a reinforcement learning training scheme for diffusion models implemented based on the verl framework, integrating the MindSpeed-MM distributed training framework and the GRPO (Group Relative Policy Optimization) algorithm.

### Core Features

- **Diffusion Model RL Training**: Supports reinforcement learning for video generation models such as Wan2.2.
- **GRPO Algorithm**: Optimization strategy based on Group Relative Policy Optimization.
- **Distributed Training**: Supports FSDP/FSDP2, Ulysses sequence parallelism.
- **Multimodal Reward**: Integrates HPSv3 scoring model.
- **Ray Distribution**: Distributed training orchestration based on Ray.

## 2. System Architecture

### 2.1 Overall Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Driver Process                                │
│                         (RayDANCETrainer)                               │
└───────────────────────────────────┬─────────────────────────────────────┘
                                    │
                                    │ Ray RPC
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        DiffusionActorRolloutWorker                      │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                          Core Components                          │  │
│  │  ┌──────────────┐     ┌──────────────┐                            │  │
│  │  │  HFRollout   │     │   HPSv3 RM   │                            │  │
│  │  │ (Generation) │     │ (Reward Calc)│                            │  │
│  │  └──────┬───────┘     └──────┬───────┘                            │  │
│  │         │                    │                                    │  │
│  │         ▼                    ▼                                    │  │
│  │  ┌───────────────────────────────────────────────────────────┐    │  │
│  │  │                  DataParallelPPOActor                     │    │  │
│  │  │                   (GRPO Policy Update)                    │    │  │
│  │  └───────────────────────────────────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                        MindSpeed-MM Backend                       │  │
│  │  ┌──────────────┐               ┌──────────────┐                  │  │
│  │  │  FSDP Module │               │   Optimizer  │                  │  │
│  │  │(Diffusion Model)             │  (Optimizer) │                  │  │
│  │  └──────────────┘               └──────────────┘                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Training Flow Diagram

```
┌─────────────────┐
│    Dataset      │
│   (prompts)     │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Step 1: Rollout Generation                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  1. Encode prompts to embeddings                              │  │
│  │  2. Sample initial noise latents                              │  │
│  │  3. Diffusion sampling (n samples)                            │  │
│  │  4. Store log_probs and latents                               │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Step 2: Reward Computation                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  1. Decode latents to images/videos                           │  │
│  │  2. Extract first frame                                       │  │
│  │  3. HPSv3 scoring                                             │  │
│  │  4. Apply reward coefficient (0.1)                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         Step 3: GRPO Update                         │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │  1. Compute group advantages                                  │  │
│  │     - Group normalization: (r - mean) / std                   │  │
│  │  2. Sample timesteps for training                             │  │
│  │  3. For each timestep:                                        │  │
│  │     a. Forward pass (compute new log_probs)                   │  │
│  │     b. Compute ratio = exp(new_log - old_log)                 │  │
│  │     c. PPO clipping on ratio                                  │  │
│  │     d. Compute policy loss                                    │  │
│  │     e. Backward pass                                          │  │
│  │  4. Gradient clipping                                         │  │
│  │  5. Optimizer step                                            │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     Step 4: Checkpoint & Logging                    │
│  - Save model checkpoints (DCP format)                              │
│  - Log metrics (loss, reward, grad_norm)                            │
│  - Online testing (optional)                                        │
└─────────────────────────────────────────────────────────────────────┘
```

## 3. Core Components Detail

### 3.1 RayDANCETrainer

**Location**: `recipe/mm_dance_grpo/dance_ray_trainer.py`

**Responsibilities**:
- Inherits from `RayPPOTrainer`, reuses verl's distributed training infrastructure.
- Manages Worker lifecycle, coordinates the training loop.

**Key Methods**:
```python
class RayDANCETrainer(RayPPOTrainer):
    def __init__(self, config, tokenizer, ...):
        # Initialize configuration and resource pool
        
    def init_workers(self):
        # Create DiffusionActorRolloutWorker
        
    def fit(self):
        # Main training loop
        for batch in train_dataloader:
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            batch = self.actor_rollout_wg.compute_rm_score(batch)
            actor_output = self.actor_rollout_wg.update_actor(batch)
            if should_save:
                self._save_checkpoint()
```

### 3.2 DiffusionActorRolloutWorker

**Location**: `recipe/mm_dance_grpo/diffusion_workers.py`

**Responsibilities**:
- Encapsulates the complete logic of Rollout, reward computation, and GRPO update.
- Manages MindSpeed-MM FSDP models and optimizers, handles distributed communication.

**Core Components**:
```python
class DiffusionActorRolloutWorker(Worker):
    def __init__(self, config, role):
        self.actor_module_fsdp = get_model(mm_model_provider)
        self.actor_optimizer = get_megatron_optimizer(...)
        self.rollout = HFRollout(self.actor_module_fsdp, ...)
        self.actor = DataParallelPPOActor(self.actor_module_fsdp, ...)
        self.reward_module = HPSv3RewardInferencer(...)
```

**Key Methods**:

#### 3.2.1 Generate Sequences (generate_sequences)

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
def generate_sequences(self, data: DataProto):
    """
    Generate video sequences using the diffusion model.
    
    Process:
    1. Encode prompts to text embeddings.
    2. Sample initial noise latents.
    3. Perform diffusion sampling (DDPM/DDIM).
    4. Record log_probs at each timestep.
    """
    output = self.rollout.generate_sequences(data)
    data = data.repeat(repeat_times=self.config.rollout.n)
    data = data.union(output)
    return data
```

#### 3.2.2 Compute Reward (compute_rm_score)

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
def compute_rm_score(self, data: DataProto):
    """
    Compute reward scores for generated videos using HPSv3.
    
    Process:
    1. Decode latents to videos.
    2. Extract the first frame.
    3. Call HPSv3 scoring.
    4. Apply reward coefficient (0.1).
    """
    for i in range(batch_size):
        image = video_first_frame_to_pil(images_path)
        hps_score = self.reward_module.reward([image], [prompt])
        hps_score = reward_coeff * hps_score
        all_rewards.append(hps_score)
    
    data.batch["rewards"] = torch.cat(all_rewards)
    return data
```

#### 3.2.3 GRPO Advantage Computation (_compute_grpo_advantages)

```python
def _compute_grpo_advantages(self, rewards):
    """
    Compute GRPO advantages.
    
    Two normalization methods:
    1. Group normalization: (r - group_mean) / group_std
    2. Global normalization: (r - mean) / std
    
    Supports reward threshold filtering.
    """
    if use_group:
        group_mean = rewards.mean()
        group_std = rewards.std() + 1e-8
        if group_mean < reward_threshold:
            advantages[:] = 0  # Filter low-quality samples
        else:
            advantages[:] = (rewards - group_mean) / group_std
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return advantages
```

#### 3.2.4 Policy Update (update_actor)

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
def update_actor(self, data: DataProto):
    """
    GRPO policy update.
    
    Process:
    1. Get latents, old_log_probs, rewards.
    2. Compute advantages.
    3. Sample training timesteps.
    4. For each timestep:
       a. Forward pass to compute new_log_probs.
       b. Compute probability ratio ratio = exp(new_log - old_log).
       c. PPO clipping: clipped_ratio = clamp(ratio, 1-ε, 1+ε).
       d. Compute loss: loss = -adv * max(ratio, clipped_ratio).
       e. Backward pass.
    5. Gradient clipping.
    6. Optimizer step.
    """
    advantages = self._compute_grpo_advantages(rewards)
    train_timesteps = random.sample(timesteps, int(len(timesteps) * fraction))
    
    for timestep_idx in train_timesteps:
        for micro_batch in micro_batches:
            new_log_probs = self.actor.forward_micro_batch(...)
            ratio = torch.exp(new_log_probs - old_log_probs[:, timestep_idx])
            clamped_advantages = torch.clamp(advantages, -clip_max, clip_max)
            clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            unclipped_loss = -clamped_advantages * ratio
            clipped_loss = -clamped_advantages * clipped_ratio
            loss = torch.mean(torch.max(clipped_loss, unclipped_loss))
            loss.backward()
    
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm)
    self.actor_optimizer.step()
    self.actor_lr_scheduler.step(batch_size)
```

### 3.3 DataParallelPPOActor

**Location**: `recipe/mm_dance_grpo/actor.py`

**Responsibilities**:
- Encapsulates the forward pass logic of the diffusion model.
- Implements single-step GRPO training.

```python
class DataParallelPPOActor(BasePPOActor):
    def __init__(self, actor_module, config, scheduler, tokenizer):
        self.actor_train = ModelingSoraModelTrain(actor_module, ...)
        
    def forward_micro_batch(self, latents, pre_latents, i, 
                       text_hidden_states, negative_text_hidden_states):
        """
        GRPO single-step forward pass.
        """
        log_probs = self.actor_train.train(
            latents, pre_latents, i,
            text_hidden_states, negative_text_hidden_states
        )
        return log_probs
```

### 3.4 HFRollout

**Location**: `recipe/mm_dance_grpo/rollout.py`

**Responsibilities**:
- Performs the sampling process of the diffusion model, generates video sequences and records log_probs.

```python
class HFRollout:
    def __init__(self, module, config, scheduler, tokenizer):
        self.sora_rollout = ModelingSoraModelInference(module, ...)
        
    def generate_sequences(self, prompts: DataProto):
        """
        Generate video sequences.
        """
        prompt_embeds, negative_prompt_embeds = self.sora_rollout.encode_texts(...)
        src_latents = self.sora_rollout.get_noise_latents(...)
        imgs, all_latents, all_log_probs = self.sora_rollout.generate(
            prompt_embeds, negative_prompt_embeds, src_latents
        )
        return DataProto.from_dict({
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "all_latents": all_latents,
            "all_log_probs": all_log_probs,
        })
```

## 4. Data Flow

### 4.1 DataProto Structure

```python
# Input data
DataProto {
    batch: {"input_ids": torch.Tensor},
    non_tensor_batch: {"raw_prompt": str, "index": int}
}

# Rollout output
DataProto {
    batch: {
        "prompt_embeds": torch.Tensor,
        "negative_prompt_embeds": torch.Tensor,
        "all_latents": torch.Tensor,      # [B, T, C, H, W]
        "all_log_probs": torch.Tensor,    # [B, T]
    },
    non_tensor_batch: {"all_imgs": List[str]}
}

# After reward computation
DataProto {
    batch: {... , "rewards": torch.Tensor},
    non_tensor_batch: {... , "global_steps": int}
}
```

### 4.2 Training Loop Data Flow

```
1. DataLoader → DataProto {input_ids, raw_prompt}
2. generate_sequences → DataProto {prompt_embeds, ..., all_log_probs, all_imgs}
3. compute_rm_score → DataProto {..., rewards, global_steps}
4. update_actor → DataProto {metrics: loss, reward_mean, grad_norm}
```

## 5. GRPO Algorithm Details

### 5.1 GRPO vs PPO

| Feature        | PPO                          | GRPO                               |
|----------------|------------------------------|------------------------------------|
| Advantage Est. | Value function              | Group normalization               |
| Baseline       | Requires separate model     | Not required                      |
| Advantage Calc | A(s) = Q(s,a) - V(s)        | A(s) = (r - mean) / std           |
| Applicability  | General RL                  | Diffusion model RL                |

### 5.2 GRPO Advantage Computation

```python
def compute_grpo_advantages(rewards, use_group=True):
    """
    Key ideas:
    1. Use relative reward values rather than absolute.
    2. Stabilize training via normalization.
    3. Support group-wise normalization (use_group=True).
    """
    if use_group:
        group_mean = rewards.mean()
        group_std = rewards.std() + 1e-8
        advantages = (rewards - group_mean) / group_std
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return advantages
```

### 5.3 GRPO Loss Function

```python
def grpo_loss(old_log_probs, new_log_probs, advantages, 
            clip_range=0.2, adv_clip_max=10.0):
    """
    Args:
        old_log_probs: Log probabilities of the old policy.
        new_log_probs: Log probabilities of the new policy.
        advantages: GRPO advantages.
        clip_range: PPO clipping range.
        adv_clip_max: Maximum advantage clipping value.
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    clamped_advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    unclipped_loss = -clamped_advantages * ratio
    clipped_loss = -clamped_advantages * clipped_ratio
    loss = torch.mean(torch.max(clipped_loss, unclipped_loss))
    return loss
```

## 6. Configuration Description

### 6.1 Main Configuration Items

```yaml
actor_rollout_ref:
  model:
    reward_model: "hpsv3"
    reward_model_path: "/path/to/HPSv3.safetensors"
  
  actor:
    strategy: "fsdp"
    fsdp_config:
      fsdp_size: 8
      param_offload: True
      optimizer_offload: True
    optim:
      lr: 5e-8
      weight_decay: 0.01
      lr_scheduler_name: "cosine"
      lr_scheduler_num_warmup_steps: 1000
      lr_scheduler_num_training_steps: 10000
    ppo_adv_clip_max: 10.0
    ppo_kl_coeff: 1.0
    ppo_max_grad_norm: 1.0
    clip_range: 1e-4
    shift: 1.0
    timestep_fraction: 1
    sampling_steps: 10
    micro_batch_size: 2
  
  rollout:
    n: 8  # Generate n samples per prompt
    micro_batch_size: 2
    latent_w: 128
    latent_h: 128
    init_same_noise: True
    online:
      test: True
      step:
        interval: 10
      save:
        path: "/path/to/save"
      prompt: "test prompt"
    only: False
    result:
      save:
        path: "/path/to/save"

data:
  train_files: "/path/to/train.parquet"
  val_files: "/path/to/test.parquet"
  train_batch_size: 8
  max_prompt_length: 1024
  max_response_length: 128

algorithm:
  adv_estimator: "grpo"
  use_kl_in_reward: False

trainer:
  n_gpus_per_node: 8
  nnodes: 1
  total_epochs: 2
  total_training_steps: 200
  save_freq: -1
  device: "npu"
  logger: "console"
```

### 6.2 MindSpeed-MM Configuration

MindSpeed-MM configuration is passed via pickle files:

```python
# mindspeed_args.pkl
{
    "micro_batch_size": 2,
    "rampup_batch_size": None,
    "global_batch_size": 8,
    "data_parallel_size": 1,
    ...
}

# mm_args.pkl
{
    "mm": {
        "model": {
            "diffusion": {...},
            "tokenizer": {...},
            ...
        }
    },
    ...
}
```

## 7. Integration with verl

### 7.1 Inheritance Relationship

```
verl.trainer.ppo.ray_trainer.RayPPOTrainer
    └─> recipe.mm_dance_grpo.dance_ray_trainer.RayDANCETrainer

verl.workers.actor.base.BasePPOActor
    └─> recipe.mm_dance_grpo.actor.DataParallelPPOActor

verl.single_controller.base.Worker
    └─> recipe.mm_dance_grpo.diffusion_workers.DiffusionActorRolloutWorker
```

### 7.2 verl Components Used

| verl Component                   | Purpose                         |
|----------------------------------|---------------------------------|
| `DataProto`                      | Data transfer protocol          |
| `RayWorkerGroup`                 | Ray Worker group management     |
| `ResourcePoolManager`            | GPU resource pool management    |
| `Dispatch` decorator             | Distributed communication       |
| `FSDPUlyssesShardingManager`     | Sequence parallel sharding mgmt |
| `CheckpointEngineManager`        | Checkpoint management           |

### 7.3 Custom Components

| Component                         | Description                           |
|-----------------------------------|---------------------------------------|
| `DiffusionActorRolloutWorker`     | Merges Actor, Rollout, Reward         |
| `DataParallelPPOActor`            | Diffusion model Actor implementation  |
| `HFRollout`                       | Diffusion model Rollout implementation|
| `RayDANCETrainer`                 | GRPO trainer                          |

## 8. Extension Guide

### 8.1 Adding a New Reward Model

```python
class DiffusionActorRolloutWorker(Worker):
    def _init_reward_module(self):
        from your_reward_module import YourRewardModel
        self.reward_module = YourRewardModel(...)
    
    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    def compute_rm_score(self, data: DataProto):
        rewards = self.reward_module.compute(data)
        data.batch["rewards"] = rewards
        return data
```

### 8.2 Modifying GRPO Advantage Computation

```python
def _compute_grpo_advantages(self, rewards):
    advantages = your_custom_advantage_computation(rewards)
    return advantages
```

### 8.3 Supporting a New Diffusion Model

```python
class HFRollout:
    def __init__(self, module, config, scheduler, tokenizer):
        from your_model import YourModelInference
        self.your_rollout = YourModelInference(module, ...)
```

## 9. Troubleshooting

### 9.1 Common Issues

| Issue                | Possible Cause         | Solution                              |
|----------------------|------------------------|---------------------------------------|
| OOM                  | Batch too large        | Reduce `micro_batch_size`             |
| Unstable training    | Learning rate too high | Lower `lr`, increase `warmup_steps`   |
| Abnormal rewards     | HPSv3 config error     | Check `reward_model_path`             |
| Checkpoint load fail | DCP format mismatch    | Verify weight conversion steps        |
| Distributed comm fail| NCCL config error      | Check `HCCL_CONNECT_TIMEOUT`          |

### 9.2 Debugging Tips

```bash
export VERL_LOGGING_LEVEL=DEBUG
```

```python
log_gpu_memory_usage("checkpoint", logger=logger)
print(f"latents shape: {latents.shape}")
print(f"rewards shape: {rewards.shape}")
```

## 10. References

- [verl Documentation](https://verl.readthedocs.io/)
- [MindSpeed-MM Documentation](https://gitcode.com/Ascend/MindSpeed-MM)
- [GRPO Paper](https://arxiv.org/abs/2409.19256)
- [Wan2.2 Model](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
- [HPSv3 Model](https://github.com/MizzenAI/HPSv3)