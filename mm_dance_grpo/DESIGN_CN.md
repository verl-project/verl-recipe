# MMDanceGRPO 设计文档

## 1. 概述

MMDanceGRPO 是基于 verl 框架实现的扩散模型强化学习训练方案，结合了 MindSpeed-MM 分布式训练框架和 GRPO（Group Relative Policy Optimization）算法。

### 核心特性

- **扩散模型RL训练**：支持 Wan2.2 等视频生成模型的强化学习。
- **GRPO算法**：基于 Group Relative Policy Optimization 的优化策略。
- **分布式训练**：支持 FSDP/FSDP2、Ulysses 序列并行。
- **多模态奖励**：集成 HPSv3 评分模型。
- **Ray分布式**：基于 Ray 的分布式训练编排。

## 2. 系统架构

### 2.1 整体架构图

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
│  │                          Core Component                           │  │
│  │  ┌──────────────┐     ┌──────────────┐                            │  │
│  │  │  HFRollout   │     │   HPSv3 RM   │                            │  │
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
│  │  └──────────────┘               └──────────────┘                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 训练流程图

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

## 3. 核心组件详解

### 3.1 RayDANCETrainer

**位置**: `recipe/mm_dance_grpo/dance_ray_trainer.py`

**职责**:
- 继承自 `RayPPOTrainer`，复用 verl 的分布式训练基础设施。
- 管理 Worker 生命周期，协调训练循环。

**关键方法**:
```python
class RayDANCETrainer(RayPPOTrainer):
    def __init__(self, config, tokenizer, ...):
        # 初始化配置和资源池
        
    def init_workers(self):
        # 创建 DiffusionActorRolloutWorker
        
    def fit(self):
        # 主训练循环
        for batch in train_dataloader:
            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
            batch = self.actor_rollout_wg.compute_rm_score(batch)
            actor_output = self.actor_rollout_wg.update_actor(batch)
            if should_save:
                self._save_checkpoint()
```

### 3.2 DiffusionActorRolloutWorker

**位置**: `recipe/mm_dance_grpo/diffusion_workers.py`

**职责**:
- 封装 Rollout、奖励计算和 GRPO 更新的完整逻辑。
- 管理 MindSpeed-MM FSDP 模型和优化器，处理分布式通信。

**核心组件**:
```python
class DiffusionActorRolloutWorker(Worker):
    def __init__(self, config, role):
        self.actor_module_fsdp = get_model(mm_model_provider)
        self.actor_optimizer = get_megatron_optimizer(...)
        self.rollout = HFRollout(self.actor_module_fsdp, ...)
        self.actor = DataParallelPPOActor(self.actor_module_fsdp, ...)
        self.reward_module = HPSv3RewardInferencer(...)
```

**关键方法**:

#### 3.2.1 生成序列 (generate_sequences)

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
def generate_sequences(self, data: DataProto):
    """
    使用扩散模型生成视频序列
    
    流程:
    1. 编码 prompt 为 text embeddings
    2. 采样初始噪声 latents
    3. 执行扩散采样 (DDPM/DDIM)
    4. 记录每个时间步的 log_probs
    """
    output = self.rollout.generate_sequences(data)
    data = data.repeat(repeat_times=self.config.rollout.n)
    data = data.union(output)
    return data
```

#### 3.2.2 计算奖励 (compute_rm_score)

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
def compute_rm_score(self, data: DataProto):
    """
    使用 HPSv3 计算生成视频的奖励分数
    
    流程:
    1. 从 latents 解码为视频
    2. 提取第一帧
    3. 调用 HPSv3 评分
    4. 应用奖励系数 (0.1)
    """
    for i in range(batch_size):
        image = video_first_frame_to_pil(images_path)
        hps_score = self.reward_module.reward([image], [prompt])
        hps_score = reward_coeff * hps_score
        all_rewards.append(hps_score)
    
    data.batch["rewards"] = torch.cat(all_rewards)
    return data
```

#### 3.2.3 GRPO 优势计算 (_compute_grpo_advantages)

```python
def _compute_grpo_advantages(self, rewards):
    """
    计算 GRPO 优势
    
    两种归一化方式:
    1. Group normalization: (r - group_mean) / group_std
    2. Global normalization: (r - mean) / std
    
    支持奖励阈值过滤
    """
    if use_group:
        group_mean = rewards.mean()
        group_std = rewards.std() + 1e-8
        if group_mean < reward_threshold:
            advantages[:] = 0  # 过滤低质量样本
        else:
            advantages[:] = (rewards - group_mean) / group_std
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return advantages
```

#### 3.2.4 策略更新 (update_actor)

```python
@register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
def update_actor(self, data: DataProto):
    """
    GRPO 策略更新
    
    流程:
    1. 获取 latents, old_log_probs, rewards
    2. 计算优势
    3. 采样训练时间步
    4. 对每个时间步:
       a. 前向传播计算 new_log_probs
       b. 计算概率比率 ratio = exp(new_log - old_log)
       c. PPO clipping: clipped_ratio = clamp(ratio, 1-ε, 1+ε)
       d. 计算损失: loss = -adv * max(ratio, clipped_ratio)
       e. 反向传播
    5. 梯度裁剪
    6. 优化器更新
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

**位置**: `recipe/mm_dance_grpo/actor.py`

**职责**:
- 封装扩散模型的前向传播逻辑。
- 实现 GRPO 的单步训练。

```python
class DataParallelPPOActor(BasePPOActor):
    def __init__(self, actor_module, config, scheduler, tokenizer):
        self.actor_train = ModelingSoraModelTrain(actor_module, ...)
        
    def forward_micro_batch(self, latents, pre_latents, i, 
                       text_hidden_states, negative_text_hidden_states):
        """
        GRPO 单步前向传播
        """
        log_probs = self.actor_train.train(
            latents, pre_latents, i,
            text_hidden_states, negative_text_hidden_states
        )
        return log_probs
```

### 3.4 HFRollout

**位置**: `recipe/mm_dance_grpo/rollout.py`

**职责**:
- 执行扩散模型的采样过程，生成视频序列并记录 log_probs。

```python
class HFRollout:
    def __init__(self, module, config, scheduler, tokenizer):
        self.sora_rollout = ModelingSoraModelInference(module, ...)
        
    def generate_sequences(self, prompts: DataProto):
        """
        生成视频序列
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

## 4. 数据流

### 4.1 DataProto 结构

```python
# 输入数据
DataProto {
    batch: {"input_ids": torch.Tensor},
    non_tensor_batch: {"raw_prompt": str, "index": int}
}

# Rollout 输出
DataProto {
    batch: {
        "prompt_embeds": torch.Tensor,
        "negative_prompt_embeds": torch.Tensor,
        "all_latents": torch.Tensor,      # [B, T, C, H, W]
        "all_log_probs": torch.Tensor,    # [B, T]
    },
    non_tensor_batch: {"all_imgs": List[str]}
}

# 奖励计算后
DataProto {
    batch: {... , "rewards": torch.Tensor},
    non_tensor_batch: {... , "global_steps": int}
}
```

### 4.2 训练循环数据流

```
1. DataLoader → DataProto {input_ids, raw_prompt}
2. generate_sequences → DataProto {prompt_embeds, ..., all_log_probs, all_imgs}
3. compute_rm_score → DataProto {..., rewards, global_steps}
4. update_actor → DataProto {metrics: loss, reward_mean, grad_norm}
```

## 5. GRPO 算法详解

### 5.1 GRPO vs PPO

| 特性       | PPO                          | GRPO                               |
|------------|------------------------------|------------------------------------|
| 优势估计   | Value function              | Group normalization               |
| 基准策略   | 需要单独模型                | 不需要                            |
| 优势计算   | A(s) = Q(s,a) - V(s)        | A(s) = (r - mean) / std           |
| 适用场景   | 通用 RL                     | 扩散模型 RL                       |

### 5.2 GRPO 优势计算

```python
def compute_grpo_advantages(rewards, use_group=True):
    """
    关键思想：
    1. 使用奖励的相对值而非绝对值
    2. 通过归一化稳定训练
    3. 支持组内归一化 (use_group=True)
    """
    if use_group:
        group_mean = rewards.mean()
        group_std = rewards.std() + 1e-8
        advantages = (rewards - group_mean) / group_std
    else:
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
    return advantages
```

### 5.3 GRPO 损失函数

```python
def grpo_loss(old_log_probs, new_log_probs, advantages, 
            clip_range=0.2, adv_clip_max=10.0):
    """
    Args:
        old_log_probs: 旧策略的对数概率
        new_log_probs: 新策略的对数概率
        advantages: GRPO 优势
        clip_range: PPO clipping 范围
        adv_clip_max: 优势裁剪最大值
    """
    ratio = torch.exp(new_log_probs - old_log_probs)
    clamped_advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
    unclipped_loss = -clamped_advantages * ratio
    clipped_loss = -clamped_advantages * clipped_ratio
    loss = torch.mean(torch.max(clipped_loss, unclipped_loss))
    return loss
```

## 6. 配置说明

### 6.1 主要配置项

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
    n: 8  # 每个 prompt 生成 n 个样本
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

### 6.2 MindSpeed-MM 配置

MindSpeed-MM 的配置通过 pickle 文件传递：

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

## 7. 与 verl 的集成点

### 7.1 继承关系

```
verl.trainer.ppo.ray_trainer.RayPPOTrainer
    └─> recipe.mm_dance_grpo.dance_ray_trainer.RayDANCETrainer

verl.workers.actor.base.BasePPOActor
    └─> recipe.mm_dance_grpo.actor.DataParallelPPOActor

verl.single_controller.base.Worker
    └─> recipe.mm_dance_grpo.diffusion_workers.DiffusionActorRolloutWorker
```

### 7.2 使用 verl 组件

| verl 组件                      | 用途                       |
|------------------------------|----------------------------|
| `DataProto`                  | 数据传输协议               |
| `RayWorkerGroup`             | Ray Worker 组管理          |
| `ResourcePoolManager`        | GPU 资源池管理             |
| `Dispatch`                   | 分布式通信                 |
| `FSDPUlyssesShardingManager` | 序列并行分片管理           |
| `CheckpointEngineManager`    | 检查点管理                 |

### 7.3 自定义组件

| 组件                                | 说明                           |
|-------------------------------------|--------------------------------|
| `DiffusionActorRolloutWorker`       | 融合 Actor、Rollout、Reward    |
| `DataParallelPPOActor`              | 扩散模型 Actor 实现            |
| `HFRollout`                         | 扩散模型 Rollout 实现          |
| `RayDANCETrainer`                   | GRPO 训练器                    |

## 8. 扩展指南

### 8.1 添加新的奖励模型

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

### 8.2 修改 GRPO 优势计算

```python
def _compute_grpo_advantages(self, rewards):
    advantages = your_custom_advantage_computation(rewards)
    return advantages
```

### 8.3 支持新的扩散模型

```python
class HFRollout:
    def __init__(self, module, config, scheduler, tokenizer):
        from your_model import YourModelInference
        self.your_rollout = YourModelInference(module, ...)
```

## 9. 故障排查

### 9.1 常见问题

| 问题           | 可能原因           | 解决方案                             |
|----------------|--------------------|--------------------------------------|
| OOM            | 批次过大           | 减小 `micro_batch_size`              |
| 训练不稳定     | 学习率过大         | 降低 `lr`，增加 `warmup_steps`       |
| 奖励异常       | HPSv3 配置错误     | 检查 `reward_model_path`             |
| 检查点加载失败 | DCP 格式不匹配     | 检查权重转换步骤                     |
| 分布式通信失败 | NCCL 配置错误      | 检查 `HCCL_CONNECT_TIMEOUT`          |


## 10. 参考资料

- [verl 文档](https://verl.readthedocs.io/)
- [MindSpeed-MM 文档](https://gitcode.com/Ascend/MindSpeed-MM)
- [GRPO 论文](https://arxiv.org/abs/2409.19256)
- [Wan2.2 模型](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B-Diffusers)
- [HPSv3 模型](https://github.com/MizzenAI/HPSv3)