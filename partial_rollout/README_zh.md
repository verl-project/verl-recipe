# Recipe: Partial Rollout

[English](README.md) | 简体中文

**同步 RL 训练模式**下的 partial-rollout 流水线，针对**长尾响应长度**带来的 GPU 闲置问题做样本补充与中断续跑。

> ⚠️ **不要跟全异步框架的 partial rollout 混淆**：本流水线仍走「rollout → 等批次齐 → 一次性 train step」的同步循环，只是允许在 rollout 阶段中断长尾样本、跨 step 续跑；trainer 与 rollout 之间是顺序串行的。如果你需要的是 trainer / rollout 完全解耦异步推进的版本，看 `verl/experimental/fully_async_policy/`，不是这里。

> 🔗 **Verl 依赖**：pin 在 `verl-project/verl@60546ef2`（[GitHub](https://github.com/verl-project/verl/commit/60546ef2a7464a158cd170f58f852a62a4e552ba)）。完整 `pip install` / `git checkout` 命令见 [REQUIRED_VERL.txt](REQUIRED_VERL.txt)。rolling 在 `main` 上跟，刷新时同步 bump pin。

## 背景

verl 的同步 PPO/GRPO 训练要等整批 prompt 全部 generate 完才能 step。但 RL 数据集的 response 长度分布是**严重右偏**的：经验上约 3% 的样本输出长度比中位数长得多，而这些长尾样本往往恰好是更难、信息量更高的样本 —— 不能直接丢弃，否则训练效果下降。整批因此被最慢的几条卡住，GPU 空转。

![Response Length Distribution across the RL Training Dataset](https://raw.githubusercontent.com/mamazi0131/verl_doc/fca7a6d3acbeca12d69c5de6f85c312c1c9e47b6/Response_Length_Distribution_across_the_RL_Training_Dataset.png)

Partial rollout 用两个手段填这个 bubble（学术对照见 [APRIL 论文](https://arxiv.org/pdf/2509.18521)）：

- **样本补充 (sample supplementation)**：快的 worker 把当前任务做完就立刻补一条新的 prompt 进来，不要空等长尾。
- **跨 step 的中断 + 续跑 (mid-generation interruption + cross-step resume)**：当前 batch 拿到足够的 "done" 样本就在 step 边界处中断剩下的长尾样本，缓存它们的 KV 状态，下个训练 step 继续从断点跑。每个被续跑的样本付出 1 个 weight 版本的 drift，靠 token-level rollout importance sampling（off-policy 修正）补回。

![Comparison of GPU Execution Timelines between Standard Synchronous Training and the Proposed Async Partial Rollout](https://raw.githubusercontent.com/mamazi0131/verl_doc/fca7a6d3acbeca12d69c5de6f85c312c1c9e47b6/Comparison_of_GPU_Execution_Timelines_between_Standard_Synchronous_Training_and_the_Proposed_Async_Partial_Rollout.png)

净效果：把长尾的 GPU 空闲转换成下一批 prompt 的有效计算，代价是续跑样本的轻度 off-policy drift —— 这点 token-level IS 能干净 handle。

---

## 使用场景

适用：

- 数据集**响应长度分布长尾**（少量超长样本拖慢整批 step）
- 同步 PPO/GRPO 因等待长尾样本导致 GPU bubble 明显
- 训练对**轻微 off-policy** 容忍（partial rollout 必然引入 weight-version 跨越，需配合 IS 修正）
- 多轮 / tool-call 场景 —— 上游 vLLM ≥ 0.12 的 `pause_generation` + abort 已经够用，本 recipe 不 fork server

不适用：

- 响应长度均匀、没有 long-tail bubble — 同步 trainer 更简单
- 严格 on-policy 必须保证（每个 trajectory 只能由当前权重产出）
- 同时使用本流水线 + 上游 sync trainer 的 batch shape 假设（dummy gen_batch、continuous-worker 语义会破坏）

---

## 整体架构

```
                 trainer (PartialRolloutRayPPOTrainer)
                        │
            push_batch  │  pull_batch
                        ▼
         ┌──────────────────────────────────┐
         │   RolloutPromptManager (Ray)     │
         │                                  │
         │   pending ─pull─► ongoing ─push─► done
         └──────────────────────────────────┘
                        │
            pull_prompts│push_prompts
                        ▼
              PartialRolloutAgentLoopWorker  ×N   (run_continuous 持续循环)
                        │
              llm_client│ generate (FullyLLMServerClient retry aborted)
                        ▼
              PartialRolloutvLLMReplica  ×replicas
                ↳ PartialRolloutvLLMHttpServer  (Python `paused` gate + abort drain)
                        ▲
              cancel/   │
              resume    │
                        │
              PartialRolloutLLMServerManager
```

- **`PartialRolloutRayPPOTrainer`** (`ray_trainer.py`)：trainer 主循环。`_fit_generate` 把 prompt push 进 manager，调 `async_rollout_manager.generate_sequences` 等一个完整 batch 回来，再走 log_prob / advantage / policy update / `update_weights`。
- **`RolloutPromptManager`** (`prompt_manager.py`)：单线程 Ray actor，维护三个队列（pending / ongoing / done）。`pull_batch` 和 `pull_prompts` 都用 `asyncio.Event` 阻塞 —— 无 busy poll，无空 pull 的 RPC。
- **`PartialRolloutAgentLoopManager` / `PartialRolloutAgentLoopWorker`** (`agent_loop/agent_loop.py`)：worker 跑常驻 `run_continuous` 循环，pull RPC、`_run_one` rollout task、slot-wait sentinel 共用一个 `asyncio.wait`。worker 不 override 上游 `generate_sequences`（留给 validation 用），而是新增一个 `generate_for_prompt` —— 把上游 `generate_sequences` 末尾的 `outputs = await asyncio.gather(*tasks)` 换成 `asyncio.wait(FIRST_COMPLETED)` 循环，每个 trajectory 完成就 decrement `self.inflight_traj` 并 set `self._slot_event` 唤醒外 loop。外 loop 用 `inflight_traj + n <= max_inflight_prompts * n` 判断 pull —— 一个 prompt 的 n trajectory 已经完成了 k 个就空出 k 格预算，长尾 prompt 的慢 traj 不卡住后面新 prompt 进入。Manager 暴露 `cancel()` / `resume()`（委托给 `PartialRolloutLLMServerManager`）让 trainer 包住 `update_weights`。
- **`PartialRolloutvLLMHttpServer` / `PartialRolloutvLLMReplica`** (`vllm_rollout/vllm_async_server.py`)：上游 vLLM HTTP server 的 Python 层 `_resume_event` 闸门（vLLM <0.12 没有 `pause_generation`）。`cancel()` 关闸门，新 `generate()` 请求挂在 wrapper 层而不进 engine；同时循环调 `abort_all_requests(reset_prefix_cache=False)` 直到 inflight 排空。`resume()` 打开闸门，挂着的 caller 一起放行。verl 升到 vLLM ≥0.12 之后这一层可以整个删掉。
- **`PartialRolloutLLMServerManager`** (`llm_server.py`)：上游 `LLMServerManager` 的薄壳子类。swap `rollout_replica_class` 到 `PartialRolloutvLLMReplica`、强制 `get_client(fully_async=True)`（让所有 caller 拿到带 retry-on-abort 的 `FullyLLMServerClient`）、`cancel` / `resume` fan out 到每个 replica。因为上游没有 `LLMServerManager` 的 FQN 配置开关，所以在 `PartialRolloutRayPPOTrainer.init_workers` 里 monkey-patch 替换。

---

## 关键不变量

1. **prompt 流转的所有权**：每个 prompt 同一时刻只在 pending / ongoing / done 之一。`pull_prompts` 移 pending→ongoing；`push_prompts` 只走 ongoing→done。没有 aborted-回 pending 的二次入队，因为 `FullyLLMServerClient.generate()` 在单次 generate 调用内部就把 abort/retry 吸收了。
2. **跨 step 的 abort/resume 包装**：`PartialRolloutAgentLoopManager.generate_sequences` 开头 `await self.resume()`，`pull_batch` 拿到完整 batch 后 `await self.cancel()`。naive `checkpoint_engine` backend（PartialRollout 默认）会跳过自己的 abort，所以这一步必须由 recipe 自己来；worker 的 `client.generate(...)` 被 abort 后在 `FullyLLMServerClient` retry 循环里等下一个 step 的 `resume()`。
3. **per-sample weight-version 跟踪**：放在 `gen_batch.meta_info["global_steps"]`（trainer 写）和 `FullyLLMServerClient.generate()`（记录每次 retry 真正提交时的版本）里。worker 自己不做版本跟踪。
4. **持续 worker 循环 + trajectory-grained pull pacing**：每个 `PartialRolloutAgentLoopWorker` 跑 `run_continuous` 直到 actor 销毁。预算按 trajectory 算（`max_inflight_prompts * n`），不是按 prompt。每个 in-flight prompt 的 n 个 trajectory 在 `generate_for_prompt` 内用 `asyncio.wait(FIRST_COMPLETED)` 等，每 traj 完成就 decrement `self.inflight_traj`、set `self._slot_event` 唤醒外 loop；外 loop 看 inflight 还差 n 就 pull 下一个 prompt，不必等 prompt 全部 n traj 跑完。Pull RPC、`_run_one` task、slot-wait sentinel 三类对象都在同一个 `asyncio.wait(running)` 里，识别时按对象身份分支。
5. **dummy gen_batch 也要带 uid**：epoch 末 dataloader 耗尽时构造的占位 batch 必须填 `non_tensor_batch["uid"]`，否则 manager 端取不到行数。
6. **stateful dataloader 续训**：`PartialRolloutRayPPOTrainer.fit()` 走 stateful loader 自动恢复进度，**不要**手工加 skip-on-resume 逻辑。
7. **不做 graceful shutdown**：`run_continuous` 跑到 actor 销毁为止；fit() return 后 Ray 进程退出会把 worker 杀掉。退出时还在跑的 rollout 直接丢弃 —— trainer 反正也不会再消费它们，无所谓。
8. **vLLM 0.11 需要 Python 层 paused 闸门**：`PartialRolloutvLLMHttpServer.generate` 开头 `await self._resume_event.wait()`，`cancel()` 关闸门并循环 abort + drain。等 verl 升到 vLLM ≥0.12，这一层（含 `PartialRolloutvLLMReplica`）整个可以删，直接用上游的 `pause_generation` 就够。

---

## 快速上手

### 单轮 (single-turn)
```bash
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo.sh
```

### 多轮工具调用 (tool agent)
先生成 tool-agent 数据集：
```bash
python3 examples/data_preprocess/gsm8k_multiturn_w_tool.py \
    --local_save_dir $HOME/data/gsm8k_tool
```
再启动：
```bash
bash recipe/partial_rollout/run/run_qwen3-0.6b_gsm8k_grpo_tool.sh
```

### 关键 Hydra override

| 项 | 值 | 说明 |
|---|---|---|
| `actor_rollout_ref.rollout.agent.default_agent_loop` | `single_turn_agent` / `tool_agent` | 用上游 agent loop（不再需要 PartialRollout wrapper —— abort/retry 由 `FullyLLMServerClient` 接管） |
| `+async_training.partial_rollout` | `True` | 打开 `FullyLLMServerClient` 里的 retry-on-abort。**必填**；前面 `+` 因为 trainer config 里默认没声明这个 key |
| `algorithm.rollout_correction.rollout_is` | `token` 或 `sequence` | 推荐 `token`（partial rollout 跨权重版本，sequence 级 ratio 易被 clamp） |
| `algorithm.rollout_correction.rollout_is_threshold` | `2.0` | TIS 上限；想用 IcePop 写 `"0.5_5.0"` |
| `actor_rollout_ref.rollout.multi_turn.enable` | `True`（仅 tool 版） | 启用多轮 |
| `actor_rollout_ref.rollout.multi_turn.tool_config_path` | YAML 路径 | tool registry，跨 backend 共用 |

