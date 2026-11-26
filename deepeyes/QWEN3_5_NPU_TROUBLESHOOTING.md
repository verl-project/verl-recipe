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

- 现象：启动日志出现 2 条 `Failed to initialize tools (tool_config_path=recipe/deepeyes/configs/image_zoom_in_tool_config.yaml, ...): [Errno 2] No such file or directory: '/workspace/recipe/...'`（训练集与验证集各一次）。
- 根因：`data.tool_config_path` 继承 `actor_rollout_ref.rollout.multi_turn.tool_config_path` 的相对路径；Ray actor 的 cwd 是集群启动目录 `/workspace`（主驱动是 `/workspace/verl`），相对路径解析失败。agent loop 加载工具时会走 `resolve_config_path` 回退到 verl 项目根目录，因此 rollout 的工具调用不受影响；只有 `RLHFDataset` 直接使用原始路径，导致过滤时 `tool_schemas=None`（长度估计未计入 tool schema token，少量边界 prompt 会略微超长；rollout `max_model_len=24576`，不会因此报错）。
- 处理（配置解决，不改代码）：启动脚本显式设置 `data.tool_config_path=/workspace/verl/recipe/deepeyes/configs/image_zoom_in_tool_config.yaml`（绝对路径），同时保留 `actor_rollout_ref.rollout.multi_turn.tool_config_path="${SCRIPT_DIR}/configs/image_zoom_in_tool_config.yaml"`。已用 `--cfg job` 验证组合后的 `data.tool_config_path` 与 rollout 侧均为绝对路径、工具可正常加载。上游代码修复方案见 issue #8554。

### 6. 数据过滤缺少 qwen_vl_utils

- 现象：`ModuleNotFoundError: No module named 'qwen_vl_utils'`。
- 处理：conda env 安装 `qwen-vl-utils==0.0.14 --no-deps`。

### 7. 解释器退出时 glibc abort（corrupted size vs. prev_size）

- 现象：进程退出码 134，日志末尾 `corrupted size vs. prev_size`。
- 根因：conda 解释器与镜像预编译扩展组合在退出阶段触发堆破坏。
- 处理：preflight 使用 `import verl; import os; os._exit(0)`；训练正常完成后的退出 abort 不影响结果。

### 8. HDK 25.2.0 无法进行 ACL graph capture

- 现象：`capture_end ... error code 507903`，CANN plog 报 `rtStreamWaitEvent: ErrCode=207005, desc=[driver error:resource alloc fail]`；单实例同样失败。
- 概念：HDK 是宿主侧昇腾驱动包（驱动 + 固件 + npu-smi/hccn_tool），挂载在容器 `/usr/local/Ascend/driver`；CANN 是镜像内软件栈（runtime/acl/HCCL/算子，版本 9.1.0）。两者独立版本化；CANN 的图捕获 API 最终调用驱动的流/事件捕获与资源分配。
- 根因：142 的 HDK 25.2.0（固件 7.7.0.3）低于 CANN 9.1.0 镜像配套的驱动版本。同一镜像在 138（HDK 25.5.2、固件 7.8.0.7）可正常 capture；plog 中 207005/507903 均为驱动/运行时错误，并非 CANN 不支持该 API。
- 处理：`actor_rollout_ref.rollout.enforce_eager=True`。彻底修复需将 142 的驱动与固件升级到 ≥25.5（与 138 一致）。

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
- 根因：OpenAI 请求的默认生成长度取剩余上下文（`max_model_len - prompt_len`），个别回复不按格式输出（`Judgement format error` 37 次）可持续生成数千 token（实测单请求 2408 tokens / 29s），占用推理引擎，并发请求排队超过 60s。judge 端观测 `Running: 1–3, Waiting: 0`，算力不是瓶颈；瓶颈在客户端并发（4 个 RewardLoopWorker 串行请求）与长尾生成。关闭 thinking 后正常请求实测 0.11s（3 tokens）。
- 处理：已验证 judge 侧 `--override-generation-config '{"max_new_tokens": 64}'` 生效（thinking-on 生成被截断到 64 tokens，单请求 1.08s，`finish_reason=length`）；正式 judge 已用该参数重启。客户端在请求中设置 `max_tokens` 或提高超时同样有效，但需要重启训练进程。
- 已应用：2026-11-04 02:07 在 actor 更新窗口内重启 judge（未造成奖励失败）。验证：长回复被截断为 64 tokens（0.87s），正常请求 0.10s；重启后未再出现 60s 超时。
- 副作用：显式 `max_new_tokens=64` 后，prompt 超过 8128 token 的 judge 请求会被 vLLM 以 400 拒绝（实测报错 `maximum context length is 8192 ... prompt contains at least 8129 input tokens`）。模型回答很长（长尾）时 judge prompt 超限，重试 3 次后该样本奖励记 0；基线 run 出现 19 次（5 步），关闭长尾生成后约 8 次/步。待处理：在 `_judge_accuracy` 中截断 `answer_text`（保留尾部最终答案），或将 judge `max_model_len` 提高到 16384 以上。

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

### 16. 每次启动重复过滤超长 prompt（filter 缓存不命中）

- 现象：每次重启都重新过滤 33137 条训练与 256 条验证 prompt（96 进程约 13–16 分钟），期间 NPU 空闲。
- 根因：过滤由数据集构造时的 `filter_overlong_prompts` 执行；HF `datasets.filter` 的结果缓存按函数 fingerprint 命中，而 verl 传入的闭包捕获了 tokenizer/processor 和数据集对象，fingerprint 跨进程不稳定，导致每次启动都全量重算并另写一份分片缓存。实测同一 `train.parquet` 一天内三次启动（03:49 / 04:38 / 08:02）产生三套 96 分片过滤缓存（约 2.6 GB）。
- 处理（配置解决，不改代码）：用数据集自身的过滤逻辑一次性生成 `train.filtered_8192.parquet`、`val.filtered_8192.parquet`（`recipe/deepeyes/scripts/prefilter_deepeyes.py`，64 进程），启动时 `data.train_files`/`data.val_files` 指向过滤后文件并设 `data.filter_overlong_prompts=False`。实测该数据在 8192 限制下过滤前后均为 33137 / 256 行（一条未丢）；已用数据集类加载校验（含 image_zoom_in_tool、system/user 消息正常）。注意：tokenizer、tool schema 或 `max_prompt_length` 变化时需重新生成过滤文件。上游代码修复方案见 issue #8552。

### 17. CPU-only Ray actor 报 PlatformNPU 未可用

- 现象：日志出现 `Platform 'huawei' (PlatformNPU) is registered but not available. This may be due to this ray actor being a CPU-only actor.`，一次启动 5–6 条。
- 根因：发出警告的是驱动进程和 `AgentLoopWorkerTQ`（无 NPU 的 CPU actor）。它们 import vLLM 时 vllm-ascend 注册了 Huawei 平台插件，但 actor 内不可见 NPU，平台探测失败；实际推理在持有 NPU 的 VLLMWorker/WorkerDict 中运行。
- 影响：无。基线 run 同样存在（5 条），step 正常推进；属日志噪声，无需处理。若该警告出现在 VLLMWorker/WorkerDict 中才需要排查设备映射。

### 18. vLLM 启动时 torch.compile 缓存加载失败并重新编译

- 现象：每个 rollout worker 启动时打一条 `WARNING ... [decorators.py:321] Compiling model again due to a load failure from /root/.cache/vllm/torch_compile_cache/torch_aot_compile/<hash>/rank_X_0/model, reason: While executing %triton_kernel_wrapper_mutation ...`，本 run 3 条、基线 4 条，均出现在 vLLM 引擎初始化阶段（08:20:54–55）。
- 根因：vLLM 的 AOT compile 缓存产物（含 inductor 生成的 Triton kernel wrapper，如 RMSNorm）在 Ascend 环境无法加载，vLLM 捕获 `load failure` 后回退为现场重新编译。138 未设 `enforce_eager`（使用 compile + ACL graph），因此每个 TP worker 各出现一次。
- 影响：仅启动阶段每个 worker 多花约 1–2 分钟；重新编译成功后正常服务，稳态 rollout 不受影响。属预期回退，不是故障。
- 处理：无需处理；如需消除该日志，可删除 `/root/.cache/vllm/torch_compile_cache/torch_aot_compile` 或设置 `VLLM_DISABLE_COMPILE_CACHE=1`（效果相同，每次启动仍会编译）。

### 19. 工具调用参数为 `...` 导致 JSON 序列化失败

- 现象：`ERROR ... Error in extracting tool call from response: Object of type ellipsis is not JSON serializable`，本 run 出现 2 次/38 步，基线 5 步内 0 次。
- 根因：`tool_parser.py::_parse_xml_function_call` 用 `ast.literal_eval` 解析参数值，模型输出 `...` 会被解析成 Python `Ellipsis`，随后 `json.dumps(param_dict)` 失败，整个 tool call 提取抛异常，该轮工具调用丢失。
- 处理：`literal_eval` 结果为 `Ellipsis` 时保留原始字符串（`"..."`），再交给 `json.dumps`；已加冒烟验证 `{"bbox_2d": "..."}` 可正常序列化。

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

## 提速实验（138，2026-11-04）：bypass old_log_prob + actor torch.compile

开关（`run_full_138.sh` 末尾追加）：

```bash
algorithm.rollout_correction.bypass_mode=True
actor_rollout_ref.actor.use_torch_compile=True
```

环境：138 单机 8 卡（ACL graph + IPC）。基线为同机 step 11–15；实验从 step 15 checkpoint 恢复（step 16 起）。计时取自日志中的 `timing_s/*`。

| 阶段 (s) | 基线 step 11–15 均值 | 实验 step 17–53 均值（37 步） |
| --- | --- | --- |
| gen | 386.8 | 300.3 |
| old_log_prob | 222.1 | **0.3** |
| adv | 1.2 | 0.6 |
| update_actor | 1368.1 | 1262.8 |
| update_weights | 7.9 | 7.3 |
| step | 1997.7 | 1581.8 |

结论：

- **bypass_mode 生效且收益确定**：`_compute_old_log_prob` 直接复用 TQ 中的 `rollout_log_probs`，old_log_prob 阶段从约 222 s 降到 0.3 s，按基线 workload 单步固定节省约 220 s（约 11%）。
- **torch.compile 无可测收益**：当前 FSDP 引擎里 `use_torch_compile` 只包装 `entropy_from_logits`（`verl/workers/engine/fsdp/transformer_impl.py`），且仅在 `calculate_entropy=True` 时调用；本任务 `calculate_entropy=False`、`entropy_coeff=0`，该函数不会被执行。它唯一的实际作用是通过 `oc.select` 把 ref 的开关也置 True，而本任务不用 ref。同 workload 下 update_actor 仍为 1391 s（基线 1330–1420 s 区间内）。
- 实验期 step 均值 1998→1582 s（−21%）中，除 bypass 的约 222 s 外，其余来自 workload 变化（`response_length/mean` 由 1682 降至 1385），不是开关收益；steps 51–53 回复长度回到约 2100 时单步回到 1813–1850 s，可作对照。
- **正确性**：bypass 后 `actor/ppo_kl` 0.0007–0.0021、`actor/pg_clipfrac` 0.001–0.003（基线因 old=当前权重恒为 0），说明 rollout 与训练策略数值差异很小；`critic/rewards/mean` 从 1.11 升到 1.37–1.46，训练正常。`actor/entropy` 指标不再产生（bypass 不计算 entropy）。
- 运行至 2026-11-05 01:30 已完成 step 53，进度正常。

## 复现所需的本地改动与启动开关

- `verl/workers/utils/padding.py`：采用上游 #8413 的 `nested_tensor_from_jagged(..., jagged_dim=token)`。
- `verl/utils/tensordict_utils.py::index_select_tensor_dict`：nested `unbind` 失败时按 `values`/`offsets` 还原样本。
- `verl/experimental/agent_loop/tool_parser.py`：参数值 `...` 被 `ast.literal_eval` 解析为 `Ellipsis` 时保留原字符串（配置层无等效开关）。
- 数据与配置（不改代码）：先用 `recipe/deepeyes/scripts/prefilter_deepeyes.py` 生成 `train/val.filtered_8192.parquet`，启动时 `data.train_files`/`data.val_files` 指向过滤后文件、`data.filter_overlong_prompts=False`，并设置绝对路径 `data.tool_config_path`。
- conda env：`transformers==5.12.1`、`qwen-vl-utils==0.0.14 --no-deps`。
- 训练开关：`actor_rollout_ref.rollout.enforce_eager=True`、`+actor_rollout_ref.rollout.engine_kwargs.vllm.additional_config.weight_nz_mode=0`。
- judge 启动：`--override-generation-config '{"max_new_tokens": 64}'`。

## 参考

- verl PR #8413：position_ids packing 修复。
- pytorch#153238：3D jagged NestedTensor unbind 问题。
- vllm-ascend PR #6033 / #5725：EI0020 端口哨兵值（65536/16666）修复。
