# Step 4 复杂度与可行性分析：把 Dynamo 真接进 `DynamoHttpServer.launch_server`

**Last updated:** 2026-04-27
**Source code reference:** `/lustre/fsw/general_sa/sopyang/dynamo` (commit at exploration time)
**Scope:** 评估 design doc §3、§5.2、§9.3 在 P1 阶段是否能落地，给出三种实施路线与各自的工作量/风险。

---

## 0. TL;DR

可行，但**比 design doc 当初设想的更脏**。三个关键事实：

1. **Dynamo Frontend 没有 in-process Python API**——必须 `python -m dynamo.frontend` 拉子进程。
2. **Dynamo worker 是普通 Python 进程**，里面的 `engine_client` 就是原生 vLLM `AsyncLLM`（`AsyncLLM.from_vllm_config`，`dynamo/components/src/dynamo/vllm/main.py:574`）。也就是说**控制面 RPC（`collective_rpc`）依然走 vLLM 自己的 API**——Dynamo 没有屏蔽它。
3. **`WorkerFactory` 接受 `setup_vllm_engine_fn` 作为回调**（`dynamo/components/src/dynamo/vllm/main.py:170-176`），所以可以注入一个 wrapper，在 `AsyncLLM.from_vllm_config` 之前把 `engine_args.worker_extension_cls` 设为 verl 已有的 extension。

最大的设计抉择是：**DynamoHttpServer Ray actor 自己当 dynamo worker（同进程）** vs **拉一组独立 dynamo worker 子进程，actor 只当 RPC 桥**。前者实现量小，但放弃了 Dynamo "跨节点 router" 的全部价值；后者保留 Dynamo 价值，但要补一条独立的控制面 RPC 通道。

| 路线 | P1 工作量 | 保留 Dynamo 哪些价值 | 主要风险 |
|---|---|---|---|
| **A) Actor-as-worker**（同进程） | ~1 周 | 仅 Frontend HTTP + KV-aware routing（如果跑 ≥2 actors） | "为啥不直接用 vllm backend"——可能被合并请求 |
| **B) Subprocess-per-worker + bridge** | ~3 周 | 全部（disagg、跨节点、KVBM） | 控制面 RPC 实现复杂、调试难、与 dynamo 内部 RPC 解耦容易裂 |
| **C) Headless mode 兜底** | ~3 天 | 几乎没有 | 用 dynamo 跑一个不是 dynamo 的部署，浪费了路 |

**P1 推荐 A**，把 disagg 和跨节点推到 P3+。理由见 §6。

---

## 1. Dynamo 的实际 Python API 表面

### 1.1 Frontend：CLI-only

- 入口：`dynamo/components/src/dynamo/frontend/__main__.py`，做的事就是 `from .main import main; main()`。
- 主体在 Rust（`lib/`），Python 侧没有可被 `await` 的 Frontend 类。
- 启动命令的最小形态（取自 `dynamo/examples/backends/vllm/launch/agg.sh:40-55`）：

  ```bash
  python -m dynamo.frontend [--http-port 8000] \
                            [--router-mode kv|round-robin|random]
  ```
- 进程发现 worker 通过 discovery backend（见 §1.4），Frontend 只读 discovery，不主动启动 worker。

**结论**：Frontend 必须用 `subprocess.Popen` 起，actor 负责 wait/kill。

### 1.2 Worker：可以编程化

入口：`dynamo/components/src/dynamo/vllm/main.py:114` 的 `worker()` 协程。流程：

```python
config = parse_args()                          # 解析 CLI
runtime, loop = create_runtime(...)            # 起 Rust runtime
factory = WorkerFactory(
    setup_vllm_engine_fn=setup_vllm_engine,    # 可注入！
    setup_kv_event_publisher_fn=setup_kv_event_publisher,
    register_vllm_model_fn=register_vllm_model,
    ...
)
await factory.create(runtime, config, ...)     # 内部建 endpoint + AsyncLLM + serve
```

`factory.create` → `_create_decode_worker`（`worker_factory.py:187`）里：

- 创建 `runtime.endpoint(...)` 把自己注册进 dynamo 的 endpoint 表。
- 调 `setup_vllm_engine(config, ...)` 拿到 `engine_client: AsyncLLM`（`worker_factory.py:260`）。
- 构造 `DecodeWorkerHandler(runtime, config, engine_client, ...)` 并 `serve_endpoint`。

**关键事实**：`engine_client` 就是 vLLM 的 `AsyncLLM` 对象（`main.py:574`），它原生有 `collective_rpc(method, args, kwargs)` 方法。Dynamo 只是把它当推理引擎用，**没**对这个方法做任何屏蔽或代理。

### 1.3 WorkerExtension 注入

verl 现在用 `engine_args.worker_extension_cls = "verl....OurExtension"`（`vllm_async_server.py:247`）来给 vLLM worker mixin 进自定义方法。

Dynamo 的 `setup_vllm_engine`（`main.py:416-598`）里：

- `engine_args = config.engine_args`（line 459）——这是 vLLM 自己的 `AsyncEngineArgs`，**完整暴露**。
- 在 `engine_args.create_engine_config(...)` 之前可以任意改 `engine_args.worker_extension_cls`。

**因此**：只要我们能让自己的 `setup_vllm_engine_fn` 替代 dynamo 默认的，就能注入 verl 的 WorkerExtension。`WorkerFactory.__init__` 直接接 callback，所以替换的成本是零。

### 1.4 Service discovery

`dynamo/components/src/dynamo/vllm/main.py:158-163`：

```python
runtime, loop = create_runtime(
    discovery_backend=config.discovery_backend,  # "file" | "etcd" | "kubernetes" | "mem"
    request_plane=config.request_plane,           # "tcp" | "http" | "nats"
    event_plane=config.event_plane,               # "nats" | "zmq"
    use_kv_events=config.use_kv_events,
)
```

- **`file`**：单机用，写到共享路径（默认 `/tmp/dynamo`）。**actor 进程嵌入的 worker 与 Frontend 子进程靠这个互相发现**——这是路线 A 的正解。
- **`mem`**：仅同进程多组件可见，无法跨进程；Frontend 子进程看不到 actor 进程的 `mem` 表，因此**不适用于路线 A**。
- **`etcd` / `nats`**：跨节点必需，需要外部基础设施。

P1 用 `file` 最干净（无外部依赖）；P4（跨节点）再切 etcd。

### 1.5 Headless mode

`main.py:151-155`：

```python
if config.headless:
    run_dynamo_headless(config)   # bypass DistributedRuntime entirely
    return
```

意思是：连 dynamo runtime 都不起，纯 vLLM 跑。这条路存在的意义是 dynamo 做 checkpoint/snapshot 测试用的，**不是给生产用的**。

---

## 2. 三种实施路线

### 路线 A — Actor-as-worker（推荐 P1）

**思路**：`DynamoHttpServer` Ray actor 自己 = dynamo worker。同进程持有 `AsyncLLM`，`collective_rpc` 直接调本地对象。Frontend 单独子进程。

**架构**：
```
Trainer process(s)
   │
   │ Ray actor handle
   ▼
┌────────────────────────────────────────┐
│ DynamoHttpServer (Ray actor, GPU proc) │
│   - hosts AsyncLLM (= vllm engine)     │
│   - registers endpoint into dynamo     │
│   - collective_rpc → AsyncLLM directly │
│                                        │
│   subprocess(`python -m dynamo.frontend`)
│   ↑ talks via mem/file discovery       │
└──────────┬─────────────────────────────┘
           │ HTTP /v1/chat/completions
           ▼
       (Trainer agent loop sends generate requests)
```

**改动清单**：

| 文件 | 改动 | 难度 |
|---|---|---|
| `recipe/dynamo/dynamo_async_server.py` `DynamoHttpServer.launch_server` | 复制 `dynamo/components/src/dynamo/vllm/main.py:114-183` 的 `worker()` 流程到 actor 内；以 `setup_vllm_engine_fn=our_setup` 注入 worker_extension_cls；`subprocess.Popen("python -m dynamo.frontend ...")` 拉 Frontend；从 `WorkerFactory._create_decode_worker` 里把 `engine_client` 抓出来存到 `self.engine` | 中（机械搬运 + 一处需 fork WorkerFactory 让它返回 engine_client）|
| `recipe/dynamo/dynamo_worker_extension.py` | 直接 import 复用 `verl/workers/rollout/vllm_rollout/utils.py` 已有的 worker extension（如 `BucketedWeightReceiver` 那一侧）| 低（几乎零代码） |
| `recipe/dynamo/dynamo_async_server.py` `DynamoHttpServer.collective_rpc` | `await self.engine.collective_rpc(method, timeout, args, kwargs)` | trivial |
| `recipe/dynamo/dynamo_async_server.py` `wake_up/sleep/clear_kv_cache` | 沿用 vllm `vllm_async_server.py:555-602` 的实现（vLLM 原生方法） | trivial |
| `recipe/dynamo/dynamo_async_server.py` `DynamoReplica.launch_servers` | 抄 `vllm_async_server.py:885-972`，prefix 换成 `dynamo_` | 低 |
| `recipe/dynamo/dynamo_frontend_launcher.py`（新文件）| `subprocess.Popen` + 端口分配 + `__del__` 里 `terminate()` | 低 |
| `recipe/dynamo/main_dynamo.py` + minimal config | 入口 + hydra group | 低 |

**估时**：1 名熟悉 verl 的工程师 5–7 个工作日。

**保留的 Dynamo 价值**：
- ✅ Frontend 的 OpenAI-compatible HTTP 入口（与 vllm/trtllm 行为一致，verl agent loop 不用改）。
- ✅ KV-aware routing：当 ≥2 个 replica 时，Frontend 把同 prefix 的请求路由到命中的 actor。
- ❌ Disaggregated prefill/decode（actor 只跑一个 AsyncLLM）。
- ❌ 跨节点 worker（一个 actor 一个 GPU 集合）。
- ❌ KVBM 跨 GPU/CPU/SSD KV 管理。

**风险**：
- 中：dynamo `WorkerFactory._create_decode_worker` 把 `engine_client` 持有在 `DecodeWorkerHandler` 里，没有暴露 getter。要么 monkey-patch handler，要么 fork _create_decode_worker。**风险中等**——逻辑稳定，但 dynamo 升级时这条 patch 可能失效。
- 低：Frontend 子进程的端口分配 + 健康检查需要写死等待逻辑。
- 低："为啥不直接用 verl 的 vllm backend？"——P1 的答案是"为 P3+ disagg 做铺垫"。需要在 PR 描述里明说。

### 路线 B — Subprocess-per-worker + bridge

**思路**：actor 不持有 GPU，只起 N 个 dynamo worker 子进程（`python -m dynamo.vllm`）+ 1 个 Frontend 子进程。控制面 RPC 通过额外通道。

**架构**：
```
Ray actor (CPU only)
   ├── Frontend subprocess (HTTP)
   ├── Worker subprocess 0 (GPU, hosts AsyncLLM_0)
   │     └── side-channel ZMQ for collective_rpc  ◄── 这个要新写
   ├── Worker subprocess 1 (GPU, hosts AsyncLLM_1)
   │     └── side-channel ZMQ
   └── ...
```

**核心难点**：dynamo 子进程里持有 `AsyncLLM`，actor 拿不到这个引用。要把 `collective_rpc` 跨进程送过去，必须加一个独立的控制面通道。可选：

1. **ZMQ 控制 socket**：每个 worker 子进程 fork 时在 `our_setup_vllm_engine` 里启一个 thread 监听 ZMQ，收到方法名后调本地 `engine.collective_rpc`。verl 已有 ZMQ 周边代码（`bucketed_weight_transfer.py`）可以借鉴。
2. **复用 dynamo 的 NATS event plane**：把 `collective_rpc` 当作一种 dynamo event 发。但 dynamo events 是 inference 相关的，往里塞控制面会污染语义，不推荐。
3. **dynamo 自定义 endpoint**：worker 注册一个 `f"{ns}.{cmp}.collective_rpc"` endpoint。**最干净**，但需要写 dynamo Rust binding 还是 Python？看 `_create_decode_worker` 是 Python 起的 endpoint，应该 Python 就够。需验证 endpoint 能不能 dispatch 任意 Python 函数。

**改动清单**（相对 A 增量）：

| 项 | 增量改动 | 难度 |
|---|---|---|
| 控制面 RPC 通道（ZMQ 或 dynamo endpoint）| 新写 ~300-500 行 + 序列化协议 | **高** |
| Worker 子进程入口 `recipe/dynamo/dynamo_vllm_worker_main.py` | 复制 `dynamo.vllm.main` + 注入 worker_extension_cls + 启 RPC 监听 | 中 |
| Actor 侧 RPC client | 维护 N 个 ZMQ socket / endpoint client，`collective_rpc` 做 fan-out | 中 |
| 进程生命周期管理 | actor 死亡时清理 N+1 个子进程；worker crash 时 actor 怎么处理 | 中 |
| 多机版 | 走 etcd discovery，跨节点 ZMQ 还是用 dynamo endpoint | 高 |

**估时**：3 周（单机），多机再加 1–2 周。

**保留的 Dynamo 价值**：
- ✅ 全部：disagg、跨节点、KVBM、KV-aware routing。

**风险**：
- 高：控制面 RPC 自己实现，调试成本高。
- 高：与 dynamo 内部进程模型耦合。dynamo 升级（worker_factory 重构、endpoint API 改名）容易裂。
- 中：weight update 的延迟比路线 A 高一跳（actor → 子进程 ZMQ → AsyncLLM → vLLM workers）。

### 路线 C — Headless mode 兜底

**思路**：用 dynamo headless（`main.py:151-155`），相当于 actor 进程内跑 vllm，dynamo 啥也不做。

**改动清单**：基本就是把 `recipe/dynamo` 改写成 `recipe/vllm-clone`。

**估时**：3 天。

**保留的 Dynamo 价值**：几乎没有。可能保留的：
- ⚠️ `setup_vllm_engine` 里关于 prometheus / model loaders 的初始化代码（但这些 verl 自己也有）。
- ❌ 别的全没有。

**何时考虑**：作为 A/B 失败的 fallback，确保至少可以跑通流程。**P1 不做**。

---

## 3. 关键 design doc 开放问题（§9）的答复

### §9.3 "Dynamo Python API 是否暴露 worker handles"

**答**：不暴露。但**用不到**——因为 dynamo worker 内部直接持有 vLLM `AsyncLLM`，`AsyncLLM.collective_rpc` 是 vLLM 原生的，未被 dynamo 屏蔽。所以问题等价于"我们怎么从 actor 跨进程调到那个 AsyncLLM"，而不是"dynamo 让不让我们调它的 worker"。

→ **路线 A 把这个问题消解**（同进程，无跨进程）。
→ **路线 B 把它转化为"自己写控制面 RPC"**（dynamo 不干预，但也不帮忙）。

### §9.4 "FP8 quant patch 在 dynamo worker 起来时是否生效"

由于 `setup_vllm_engine_fn` 是回调，我们在自己的 wrapper 里第一行调 `apply_vllm_fp8_patches()` 即可。**无问题**。

### §9.5 "supports_partial_loading 是否暴露"

`AsyncLLM.collective_rpc` 直通 vLLM worker，在 worker extension 里照样能 `getattr(self, "supports_partial_loading", False)`。**无问题**。

---

## 4. 最小可验证里程碑（路线 A）

按"最小代码路径先跑通端到端"的顺序：

1. **m1（1 天）**：`recipe/dynamo/main_dynamo.py` + minimal hydra config，跑 `import recipe.dynamo` 触发注册。`get_rollout_replica_class("dynamo")` 解析成功。
2. **m2（2 天）**：`DynamoReplica.launch_servers` 抄 vllm 版本；`DynamoHttpServer.launch_server` 仅起 vLLM AsyncLLM（**不**起 dynamo runtime），`collective_rpc` 转发到 AsyncLLM。这一刻其实就是 vllm 的复刻——验证 verl 的 ServerAdapter ↔ HttpServer 链路在 recipe 侧没问题。
3. **m3（2 天）**：在 m2 的基础上加 `create_runtime(discovery_backend="file", ...)` + `factory.create(...)`，让 dynamo runtime 跑起来；endpoint 注册写入 `/tmp/dynamo`。这一步**还没**起 Frontend，所以 verl 那侧 generate 还走原来的 vllm 路径（不通过 Frontend）。验证 dynamo runtime 同进程嵌入是否成立。
4. **m4（1 天）**：`subprocess.Popen` 起 Frontend；让 verl agent loop 把 `server_address` 指向 Frontend 的 HTTP 端口，跑通一次 generate。
5. **m5（1 天）**：`update_weights` 端到端：actor `engine.collective_rpc("update_weights_from_ipc", ...)` → vLLM worker 的 verl WorkerExtension。
6. **m6（1 天）**：`resume/release` 端到端。

如果 m3 跑不通——即 dynamo runtime 不能在 Ray actor 进程内嵌——退路线 C。如果 m5 跑不通——即 dynamo 在 vLLM 起来后塞了什么 hook 阻碍 collective_rpc——回到 m2 那个状态作为最小可发布版本。

**总计：8 个工作日**，包含两个明确的 fall-back 决策点。

---

## 5. 与 design doc §3-§5 的差异

`docs/design/dynamo_rollout_design.md` 当时假设：
- Dynamo 暴露完整 Python API（包括 Frontend 类）。
- 控制面 RPC 走 dynamo "collective_rpc 桥接"（§5.2）。

实际：
- Frontend Python API 没有，必须 subprocess。
- "collective_rpc 桥接" 这词在 dynamo 里**根本不存在**——controle 面 RPC 走 vLLM 自己的 `AsyncLLM.collective_rpc`，dynamo 只管推理路由。

**design doc 需要更新的章节**（建议另起 PR）：
- §3.1 架构图：去掉"DynamoHttpServer 直接跨 worker collective_rpc"那条线，改成"actor 进程内直接持有 AsyncLLM"（路线 A）。
- §5.2 重写："collective_rpc 桥接"应改为"vLLM-native collective_rpc，actor 必须与 vLLM AsyncLLM 同进程"。
- §9.3 关掉这个开放问题，结论：靠路线 A 绕过；路线 B 需要自建控制面通道。

---

## 6. 推荐

**P1：路线 A**，理由：

1. **可在 8 天内端到端**，里程碑明确，每步都有 fall-back。
2. **价值兑现**：拿到 Frontend HTTP + KV-aware routing（多 replica 场景），这是 P1 验证 dynamo "至少不比 vllm 差" 的最小条件。
3. **不预付 disagg/跨节点的复杂度**。disagg 是 design doc §3.2 的核心卖点，但只有在长上下文 / 长 response 场景才显著——这是 P3 的事，等 P1 数据确认 dynamo 路径不偏 OOM/精度问题再做。
4. **fall-back 路径短**：路线 A 失败回 C；路线 B 失败可能要重写整套控制面。

**P3：评估升级到路线 B**。前提是：
- P1/P2 验证了 vLLM-as-dynamo-worker 在 RL 训练里精度/吞吐持平。
- 长上下文 benchmark 显示 disagg 有显著 TTFT 收益。

**P5：路线 B 升级到 in-tree（参考 `registration_design.md` §6 的 B → A 迁移）**。

---

## 7. 需要立刻验证的两个假设（写代码前）

在动手 m2 之前花半天跑两个最小验证：

1. **`AsyncLLM.from_vllm_config` 返回的对象，在 worker_extension_cls 注入后，`engine.collective_rpc("hello_world", ...)` 能调到 extension 的方法**。这是 verl 现有 vllm rollout 的核心契约，应该没问题，但 recipe 侧第一次跑必须先确认。
2. **`create_runtime(discovery_backend="mem", ...)` 在已经存在 asyncio event loop 的 Ray actor 里能不能起来**。如果 dynamo Rust runtime 抢 event loop，路线 A 的 m3 就过不去——这一关决定整个项目走 A 还是 C。

测试代码可以放 `recipe/dynamo/tests/smoke_test_dynamo_in_actor.py`。

---

## 8. 参考文件

| 路径 | 用处 |
|---|---|
| `dynamo/components/src/dynamo/vllm/main.py:114-183` | worker 入口完整流程，是路线 A m2-m3 的搬运对象 |
| `dynamo/components/src/dynamo/vllm/main.py:416-598` | `setup_vllm_engine` 实现，是注入 `worker_extension_cls` 的钩子点 |
| `dynamo/components/src/dynamo/vllm/worker_factory.py:103-300` | `WorkerFactory` + `_create_decode_worker`，需要从这里抓 `engine_client` |
| `dynamo/components/src/dynamo/frontend/__main__.py` | Frontend 入口，subprocess 用 |
| `dynamo/examples/backends/vllm/launch/agg.sh` | Frontend + worker 的最简 CLI 启动样例 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py:194-373` | `vLLMHttpServer.launch_server`，路线 A 的"vllm 那一半"模板 |
| `verl/workers/rollout/vllm_rollout/vllm_async_server.py:885-972` | `vLLMReplica.launch_servers`，`DynamoReplica` 的搬运目标 |
| `verl/workers/rollout/vllm_rollout/utils.py:129+` | verl 现有的 worker_extension_cls 实现 |
