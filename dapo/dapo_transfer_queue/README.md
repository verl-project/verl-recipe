# DAPO + TransferQueue

本 recipe 在 [DAPO（Data-Assisted Policy Optimization）](https://github.com/volcengine/verl/tree/master/recipe/dapo) 上接入 [TransferQueue](https://github.com/TransferQueue/TransferQueue)，用 BatchMeta + tq_client 替代单控制器下整批 DataProto 的中转，减轻数据经 Driver 的瓶颈，同时保留 DAPO 的动态采样、filter_groups 等逻辑。

## 目录与文件

| 路径 | 说明 |
|------|------|
| `main_dapo.py` | 入口：Hydra 加载 `dapo_transfer_queue_trainer`，设置 `TRANSFER_QUEUE_ENABLE=1`，创建 `RayDAPOTrainerTQ` 并执行 `fit()` |
| `dapo_ray_trainer.py` | `RayDAPOTrainerTQ` 实现：继承 `recipe.transfer_queue.ray_trainer.RayPPOTrainer`，重写 `compute_kl_related_metrics` 与 `fit()` |
| `config/dapo_transfer_queue_trainer.yaml` | 主配置：继承 `ppo_trainer`，DAPO 数据/奖励/algorithm 配置 + TransferQueue 开关与存储参数 |
| `config/dapo_transfer_queue_quickstart.yaml` | 快速开始配置：继承 `dapo_transfer_queue_trainer`，少量 step、仅 console 日志 |
| `30B_megatron_dapo_npu.sh` | 示例脚本：30B Megatron + NPU 上以 `ray job submit` 提交 DAPO+TQ 任务（含 filter_groups、GRPO 等） |

## 依赖与启用

- **TransferQueue**：需安装，例如 `pip install TransferQueue`（或见项目 `setup.py` 中 `transferqueue` 可选依赖）。
- **启用方式**：使用本 recipe 的 config（`--config-name=dapo_transfer_queue_trainer`）会自动设置 `transfer_queue.enable: True` 并在 Ray 的 `runtime_env` 中设置 `TRANSFER_QUEUE_ENABLE=1`。

## 运行示例

**主配置启动：**

```bash
python3 -m recipe.dapo_transfer_queue.main_dapo \
  --config-name=dapo_transfer_queue_trainer \
  data.train_files=... \
  data.val_files=... \
  actor_rollout_ref.model.path=... \
  ...
```

**快速开始（少量 step、仅 console）：**

```bash
python3 -m recipe.dapo_transfer_queue.main_dapo \
  --config-name=dapo_transfer_queue_quickstart \
  data.train_files=... \
  data.val_files=... \
  actor_rollout_ref.model.path=...
```

可参考同目录下的 `30B_megatron_dapo_npu.sh` 或 `recipe/dapo/run_dapo_*.sh` 的参数；DAPO 使用 async rollout，需保证 rollout 相关配置一致。

## 与 DAPO / TransferQueue 的关系

- **基类**：`RayDAPOTrainerTQ` 继承自 `recipe.transfer_queue.ray_trainer.RayPPOTrainer`（带 TransferQueue 的 PPO Trainer），具备 TQ 的初始化（Controller、Storage、tq_client）、BatchMeta 流转和 tqbridge 等能力。
- **覆盖逻辑**：
  - **compute_kl_related_metrics(batch_meta, metrics, timing_raw)**：基于 BatchMeta 和 tq_client 计算 response_mask、old_log_prob、ref_log_prob，返回更新后的 BatchMeta。
  - **fit()**：实现 DAPO 训练循环；每个 dataloader batch 先 put 到 TQ → generate → reward，按需做 KL 相关；若启用 filter_groups 则在 Driver 侧 get_data 后做过滤与累积，满足条件后将合并 batch 再 put 回 TQ 作为“更新用” batch_meta，再做 balance、values、advantage、update_critic、update_actor；步骤结束时对用过的 gen/update batch_meta 做 clear_samples。

## 配置

- 默认继承 **ppo_trainer**（见 `config/dapo_transfer_queue_trainer.yaml` 的 `defaults`），在此基础上增加/覆盖：
  - DAPO 相关：`data.gen_batch_size`、`reward_model.reward_manager: dapo`、`algorithm.filter_groups` 等。
  - TransferQueue：`transfer_queue.enable: True`，以及 `num_global_batch`、`storage_backend`、`num_data_storage_units` 等（详见该 yaml）。

## 文档

- TransferQueue 在 verl 中的整体说明与 BatchMeta、tqbridge 用法：[docs/data/transfer_queue.md](../../docs/data/transfer_queue.md)。
- DAPO 原版说明：[recipe/dapo/README.md](../dapo/README.md)。
