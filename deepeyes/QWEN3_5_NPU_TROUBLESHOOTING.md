# Qwen3.5 DeepEyes NPU 问题归档

记录 Qwen3.5-9B DeepEyes GRPO 在 Ascend A2 上的部署与运行问题、根因和处理方式。

## 环境

| 项 | 版本 / 说明 |
| --- | --- |
| 节点 | 71.10.29.142（HDK 25.2.0）、71.10.29.138（HDK 25.5.2） |
| 镜像 | `quay.io/ascend/vllm-ascend:v0.27.1rc1`（CANN 9.1.0、vLLM 0.27.1、transformers 5.14.1） |
| 运行环境 | 镜像内 micromamba conda env（Python 3.12.16），通过 `image.pth` 引入镜像 site-packages |
| 训练 | verl main（本 recipe），FSDP2，Ray 2.48，2 节点 |
| 推理 | vLLM-Ascend 0.27.1rc1，每节点 4 个 TP=2 副本 |

## 问题列表

### 1. Ray 集群与驱动的 Python 版本不一致

- 现象：Ray worker 报 Python 版本不匹配，连接失败。
- 根因：Ray 集群使用镜像 Python 3.12.13 启动，驱动使用 conda Python 3.12.16。
- 处理：统一用 conda 解释器启动 Ray：`/workspace/conda/verl/bin/python /usr/local/python3.12.13/bin/ray ...`。

### 2. vLLM WorkerProc 找不到 acl

- 现象：`ModuleNotFoundError: No module named 'acl'`。
- 根因：conda env 未包含 CANN Python 包路径。
- 处理：`image.pth` 追加 `/usr/local/Ascend/ascend-toolkit/latest/python/site-packages`。

### 3. HCCL 选错网卡导致 FSDP2 建链失败

- 现象：`hcclCommInitRootInfoConfig error code 1`，部分 rank 未连接。
- 根因：节点存在 tun/docker/br 等多张网卡，HCCL 自动选择错误。
- 处理：设置 `HCCL_SOCKET_IFNAME=enp67s0f0np0`（71.10.29.x 网卡），并写入 Ray `runtime_env`。

### 4. 单卡多进程 NPU socket 端口冲突

- 现象：`EI0020 ... IP address ... port 16666 have already been bound`。
- 根因：每节点 4 个 vLLM 副本共用默认 NPU socket 端口；端口范围变量只设置在驱动 shell，未进入 Ray actor。
- 处理：`HCCL_HOST_SOCKET_PORT_RANGE=60000-60050`、`HCCL_NPU_SOCKET_PORT_RANGE=61000-61050` 写入 Ray `runtime_env`。

### 5. 工具配置相对路径在 Ray actor 中失效

- 现象：`Failed to initialize tools ...`。
- 根因：Ray actor 不继承启动目录。
- 处理：`tool_config_path` 使用 `${SCRIPT_DIR}` 绝对路径。

### 6. 数据过滤缺少 qwen_vl_utils

- 现象：`ModuleNotFoundError: No module named 'qwen_vl_utils'`。
- 处理：conda env 安装 `qwen-vl-utils==0.0.14 --no-deps`。

### 7. 解释器退出时 glibc abort（corrupted size vs. prev_size）

- 现象：进程退出码 134，日志末尾 `corrupted size vs. prev_size`。
- 根因：conda 解释器与镜像预编译扩展组合在退出阶段触发堆破坏。
- 处理：preflight 使用 `import verl; import os; os._exit(0)`；训练正常完成后的退出 abort 不影响结果。

### 8. HDK 25.2.0 无法进行 ACL graph capture

- 现象：`capture_end ... error code 507903`，CANN plog 报 `207005 driver error:resource alloc fail`；同一节点部分或全部 vLLM 副本失败。
- 根因：142 节点 HDK 25.2.0 低于 CANN 9.1.0 镜像要求（138 的 25.5.2 正常）。
- 处理：`actor_rollout_ref.rollout.enforce_eager=True`。彻底修复需将 142 驱动升级到 ≥25.5。

### 9. FRACTAL_NZ 导致权重同步 wake_up 失败

- 现象：`ValueError: FRACTAL_NZ mode is enabled ... set weight_nz_mode=0`。
- 根因：vllm-ascend 0.27.1 使用 `additional_config.weight_nz_mode`，旧的 `VLLM_ASCEND_ENABLE_NZ` 已失效。
- 处理：`+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0`。

### 10. transformers 版本超出 verl 支持范围

- 现象：`AttributeError: 'FSDPQwen3_5DecoderLayer' object has no attribute 'layer_type'`。
- 根因：镜像 transformers 5.14.1 将 Qwen3.5 decoder layer 的 `layer_type` 改名为 `block_type`；verl 要求 `transformers<5.13`。
- 处理：conda env 安装 `transformers==5.12.1`（满足 vLLM `>=5.5.3` 与 verl `<5.13`），两节点保持一致。

### 11. 3D jagged nested position_ids unbind 失败

- 现象：actor 更新时 `split_with_sizes expects split_sizes to sum exactly to <N> ... got split_sizes=[4]`。
- 根因：TransferQueue 重组 batch 时用 `as_nested_tensor` 打包多分量 position_ids，单样本/等长时把分量轴（4）误判为 ragged 轴；PyTorch 3D jagged `unbind` 本身也有已知问题（pytorch#153238）。
- 处理：应用上游 #8413（`nested_tensor_from_jagged(..., jagged_dim=token)`），并在 `index_select_tensor_dict` 增加 unbind 失败时按 `values`/`offsets` 还原样本的兜底。

### 12. judge 长尾生成导致客户端 60s 超时

- 现象：`Chat completion request failed: ... Read timed out (read timeout=60.0)`；14 小时内出现 184 次。
- 影响：超时请求重试 3 次后 acc=0，污染奖励（约 3.6%），并拖慢 rollout 尾部。
- 根因：judge 模型默认 `max_tokens=2048`，个别回复不按格式输出（`Judgement format error` 37 次），长时间占用推理引擎，并发请求排队超过 60s。judge 端观测 `Running: 1–3, Waiting: 0`，算力不是瓶颈；瓶颈在客户端并发（4 个 RewardLoopWorker 串行请求）与长尾生成。关闭 thinking 后正常请求实测 0.11s（3 tokens）。
- 处理：judge 侧限制生成长度：`--override-generation-config '{"max_new_tokens": 64}'`。vLLM 的 override 只识别 `max_new_tokens`（会映射为全局默认 `max_tokens`）；或客户端设置 `max_tokens`、提高超时。

### 13. 节点被其他容器占用

- 现象：vLLM 启动报 `Free memory on device (4.29/60.96 GiB) is less than desired`。
- 根因：138 上其他容器占满 8 卡（约 50GB/卡），Ray 不感知容器外占用。
- 处理：启动前用宿主机 `docker ps` 与 `npu-smi` 检查；由资源方停止占用容器。

### 14. vLLM 引擎失败后 worker 残留占显存

- 现象：后续启动显存不足，宿主机 `npu-smi` 仍显示 `VLLMWorker_TP`。
- 处理：重启前在容器内按 PID 清理残留 worker。

### 15. 权重同步慢（HDK 缺少 IPC 支持）

- 现象：每步 rollout 结束后有约 27 分钟仅有 host 侧工作，NPU 空闲。
- 根因：HDK 25.2.0 < 25.3.rc1，verl 回退到共享内存做权重同步（`IPC is not supported ... Falling back to shared memory`）。
- 建议：升级 142 HDK（≥25.5，与 138 一致）以启用 IPC 与 ACL graph，或将训练迁移到 138。

## 单步耗时分析

配置：8 卡（142），`train_batch_size=64`、`rollout.n=8`、`max_response_length=16384`，eager 推理。

- 单步 79–92 分钟（进度条累计平均 4850s/步）；517 步约 28 天。
- 构成（按日志中 pending/running 时间戳实测）：
  - rollout（生成 + 逐条判分，判分与生成交错）：40–50 分钟；
  - actor 更新（NPU）：约 5–10 分钟；
  - 权重同步（共享内存 + vLLM 装载）：约 25–30 分钟；
  - 两者合计（rollout 结束到下一步 rollout 开始）：29–44 分钟。
- 主要瓶颈：
  1. 权重同步依赖共享内存（HDK 25.2.0 无 IPC）；
  2. rollout 使用 eager 模式（HDK 25.2.0 无 ACL graph）；
  3. judge 超时拖慢 rollout 尾部。
- 提速路径：升级 142 驱动后预计每步 30–40 分钟；或使用 16 卡训练（rollout 与更新近似减半）。

## 复现所需的本地改动与启动开关

- `verl/workers/utils/padding.py`：采用上游 #8413 的 `nested_tensor_from_jagged(..., jagged_dim=token)`。
- `verl/utils/tensordict_utils.py::index_select_tensor_dict`：nested `unbind` 失败时按 `values`/`offsets` 还原样本。
- conda env：`transformers==5.12.1`、`qwen-vl-utils==0.0.14 --no-deps`。
- 训练开关：`actor_rollout_ref.rollout.enforce_eager=True`、`+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0`。
- judge 启动：`--override-generation-config '{"max_new_tokens": 64}'`。

## 参考

- verl PR #8413：position_ids packing 修复。
- pytorch#153238：3D jagged NestedTensor unbind 问题。
- vllm-ascend PR #6033 / #5725：EI0020 端口哨兵值（65536/16666）修复。
