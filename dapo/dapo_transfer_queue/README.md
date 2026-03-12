# DAPO + TransferQueue

**中文**  
本 recipe 在 [DAPO（Data-Assisted Policy Optimization）](https://github.com/volcengine/verl/tree/master/recipe/dapo) 上接入 [TransferQueue](https://github.com/TransferQueue/TransferQueue)，用 BatchMeta + tq_client 替代单控制器下整批 DataProto 的中转，减轻数据经 Driver 的瓶颈，同时保留 DAPO 的动态采样、filter_groups、KL 惩罚、overlong reward、GRPO/GAE 等逻辑。

**English**  
This recipe integrates [TransferQueue](https://github.com/TransferQueue/TransferQueue) with [DAPO (Data-Assisted Policy Optimization)](https://github.com/volcengine/verl/tree/master/recipe/dapo). It uses **BatchMeta** and **tq_client** instead of passing full **DataProto** batches through a single controller, reducing the Driver bottleneck while keeping DAPO behavior such as dynamic sampling, `filter_groups`, KL penalty, overlong reward, and GRPO/GAE.

> **兼容性 / Compatibility:**  
> 此特性目前仅支持 **verl 在 commit `706a807` 及更早版本**。更新版本的 verl 不保证可用。  
> This feature currently only supports **verl at commit `706a807` and earlier**. Newer verl commits are not guaranteed to work with this recipe.

---

## 目录与文件 / Directory and Files

| 路径 Path | 说明 Description |
|-----------|------------------|
| `main_dapo.py` | 入口：Hydra 加载 `dapo_transfer_queue_trainer`，设置 `TRANSFER_QUEUE_ENABLE=1`，创建 `RayDAPOTrainer` 并执行 `fit()` / Entry: Hydra loads `dapo_transfer_queue_trainer`, sets `TRANSFER_QUEUE_ENABLE=1`, creates `RayDAPOTrainer` and runs `fit()` |
| `dapo_ray_trainer.py` | `RayDAPOTrainer` 实现：继承 `RayPPOTrainer`，重写 `compute_kl_related_metrics` 与 `fit()` / Implements `RayDAPOTrainer`: extends `RayPPOTrainer`, overrides `compute_kl_related_metrics` and `fit()` |
| `config/dapo_transfer_queue_trainer.yaml` | 主配置：继承 `ppo_trainer`，DAPO + TransferQueue 配置 / Main config: extends `ppo_trainer` with DAPO and TransferQueue options |
| `config/dapo_transfer_queue_quickstart.yaml` | 快速开始：少量 step、仅 console 日志 / Quickstart: fewer steps, console-only logging |
| `30B_megatron_dapo_npu.sh` | 示例脚本：30B Megatron + NPU 上 `ray job submit` 提交 DAPO+TQ / Example: submit DAPO+TQ via `ray job submit` (Megatron, NPU) |
| `DESIGN.md` | 设计文档（架构、数据流、配置）/ Design document (architecture, data flow, config) |

---

## 依赖与启用 / Dependencies and Enabling

**中文**
- **TransferQueue**：需安装，例如 `pip install TransferQueue==0.1.5`（或见项目 `setup.py` 中 `transferqueue` 可选依赖）。
- **启用**：使用本 recipe 的 config（`--config-name=dapo_transfer_queue_trainer`）会自动设置 `transfer_queue.enable: True` 并在 Ray 的 `runtime_env` 中设置 `TRANSFER_QUEUE_ENABLE=1`。

**English**
- **TransferQueue:** must be installed, e.g. `pip install TransferQueue==0.1.5` (or use the project’s `transferqueue` optional dependency in `setup.py`).
- **Enabling:** Using this recipe’s config (`--config-name=dapo_transfer_queue_trainer`) sets `transfer_queue.enable: True` and adds `TRANSFER_QUEUE_ENABLE=1` to Ray’s `runtime_env`.

---

## 运行示例 / Run Examples

**主配置 / Main config:**

```bash
python3 -m recipe.dapo_transfer_queue.main_dapo \
  --config-name=dapo_transfer_queue_trainer \
  data.train_files=... \
  data.val_files=... \
  actor_rollout_ref.model.path=... \
  ...
```

**快速开始 / Quickstart (少量 step、仅 console / few steps, console only):**

```bash
python3 -m recipe.dapo_transfer_queue.main_dapo \
  --config-name=dapo_transfer_queue_quickstart \
  data.train_files=... \
  data.val_files=... \
  actor_rollout_ref.model.path=...
```

**中文**  
更多参数可参考同目录下的 `30B_megatron_dapo_npu.sh` 或 `recipe/dapo/run_dapo_*.sh`；DAPO 使用 async rollout，需保证 rollout 相关配置一致。

> **注意（bsz 与 n_gpus）：** 脚本 `30B_megatron_dapo_npu.sh` 内备注要求 `train_prompt_bsz`（bsz）> `n_gpus`，但脚本中当前写的是 `train_prompt_bsz=8`、`trainer.n_gpus_per_node=16`，即 8 > 16 不成立，二者冲突。实际使用时请按需求二选一调整：要么增大 bsz 使其大于总 GPU 数（如 `nnodes * n_gpus_per_node`），要么减小 `n_gpus_per_node`/节点数以使 n_gpus < bsz，否则可能不符合框架对 batch 与 GPU 数的约束。

**English**  
For more arguments, see `30B_megatron_dapo_npu.sh` or `recipe/dapo/run_dapo_*.sh`. DAPO uses async rollout; keep rollout-related config consistent.

> **Note (bsz vs n_gpus):** The script `30B_megatron_dapo_npu.sh` comments that `train_prompt_bsz` (bsz) must be > `n_gpus`, but the script currently sets `train_prompt_bsz=8` and `trainer.n_gpus_per_node=16`, so 8 > 16 does not hold—they conflict. When running, adjust one or the other: either increase bsz so it is greater than total GPUs (e.g. `nnodes * n_gpus_per_node`), or reduce `n_gpus_per_node`/node count so that n_gpus < bsz, otherwise the batch vs GPU constraint may not be satisfied.

---

## 与 DAPO / TransferQueue 的关系 / Relation to DAPO and TransferQueue

**中文**
- **基类**：`RayDAPOTrainer` 继承自 `verl.trainer.ppo.ray_trainer.RayPPOTrainer`（带 TransferQueue 的 PPO Trainer），具备 TQ 的初始化（Controller、Storage、tq_client）、BatchMeta 流转和 tqbridge 等能力。
- **覆盖逻辑**：
  - **compute_kl_related_metrics(batch_meta, metrics, timing_raw)**：基于 BatchMeta 和 tq_client 计算 response_mask、old_log_prob、ref_log_prob，返回更新后的 BatchMeta。
  - **fit()**：实现 DAPO 训练循环；每个 dataloader batch 先 put 到 TQ → generate → reward，按需做 KL 相关；若启用 filter_groups 则在 Driver 侧 get_data 后做过滤与累积，满足条件后将合并 batch 再 put 回 TQ 作为“更新用” batch_meta，再做 balance、values、advantage、update_critic、update_actor；步骤结束时对用过的 gen/update batch_meta 做 clear_samples。

**English**
- **Base class:** `RayDAPOTrainer` extends `verl.trainer.ppo.ray_trainer.RayPPOTrainer` (PPO Trainer with TransferQueue), so it gets TQ setup (Controller, Storage, tq_client), BatchMeta flow, and tqbridge.
- **Overrides:**
  - **compute_kl_related_metrics(batch_meta, metrics, timing_raw):** computes response_mask, old_log_prob, ref_log_prob from BatchMeta and tq_client; returns updated BatchMeta.
  - **fit():** DAPO training loop: each dataloader batch is put to TQ → generate → reward; KL-related steps when needed; if `filter_groups` is on, Driver does get_data → filter and accumulate → when enough, merge batch and put back to TQ as the “update” batch_meta → balance, values, advantage, update_critic, update_actor; at step end, clear_samples for used gen/update batch_metas.

---

## 配置 / Configuration

**中文**  
默认继承 **ppo_trainer**（见 `config/dapo_transfer_queue_trainer.yaml` 的 `defaults`），在此基础上增加/覆盖：DAPO 相关（如 `data.gen_batch_size`、`reward_model.reward_manager: dapo`、`algorithm.filter_groups`）；TransferQueue（`transfer_queue.enable: True`，以及 `num_global_batch`、`storage_backend`、`num_data_storage_units` 等，详见该 yaml）。

**English**  
Config inherits from **ppo_trainer** (see `defaults` in `config/dapo_transfer_queue_trainer.yaml`), then adds/overrides: **DAPO** (e.g. `data.gen_batch_size`, `reward_model.reward_manager: dapo`, `algorithm.filter_groups`); **TransferQueue** (`transfer_queue.enable: True`, and optionally `num_global_batch`, `storage_backend`, `num_data_storage_units` — see that yaml).

---

## 文档 / Documentation

**中文**
- TransferQueue 在 verl 中的说明（BatchMeta、tqbridge）：`docs/data/transfer_queue.md`（若仓库中存在）。
- DAPO 原版说明：[recipe/dapo/README.md](../dapo/README.md)。
- 本特性设计文档：[DESIGN.md](./DESIGN.md)。

**English**
- TransferQueue in verl (BatchMeta, tqbridge): `docs/data/transfer_queue.md` (if present in your repo).
- DAPO recipe: [recipe/dapo/README.md](../dapo/README.md).
- This feature design: [DESIGN.md](./DESIGN.md).
