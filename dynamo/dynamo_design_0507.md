dynamo集成到verl的设计文档：
dynamo: /lustre/fsw/general_sa/sopyang/dynamo
verl: /lustre/fsw/general_sa/sopyang/rl/verl_0211/verl

recipe/dynamo/dynamo_async_server.py 放DynamoHttpServer类
参考/lustre/fsw/general_sa/sopyang/rl/RL/nemo_rl/models/generation/dynamo/dynamo_generation.py 在/lustre/fsw/general_sa/sopyang/rl/verl_0211/verl/recipe/dynamo里实现dynamo_async_server和dynamo_rollout.py

注意：
1. dynamo frontend，如何启动，healthcheck, config都参考nemo-rl里的实现
2. 如果可以，也不需要具体做forward pass，做GPU占位和subprocess看门狗就行
3. 不要参考dynamo_bk里的实现，这里的有可能是错误的

---

# Design Doc — verl × Dynamo (subprocess + watchdog)

**Last updated:** 2026-05-07
**Status:** Draft, scope = generation-only smoke (no train, no weight update)
**Reference:** `nemo_rl/models/generation/dynamo/dynamo_generation.py` 是唯一参考实现

---

## 0. TL;DR

verl 跑 RL，rollout 后端用 NVIDIA Dynamo（frontend HTTP + KV-aware router）。

- verl 主仓不动；所有改动在 `recipe/dynamo/` 下。
- `DynamoHttpServer` 是 Ray actor，**只做两件事**：
  1. **GPU 占位**——`@ray.remote(num_gpus=N)` 让 Ray 把整 node 资源锁给本 actor，
     这样 verl 主流程的其他 worker 不会和 dynamo 子进程抢卡。
  2. **subprocess 看门狗**——拉起并监护 `etcd` / `nats-server` /
     `python -m dynamo.frontend` / `python -m dynamo.vllm × N` 子进程。
- 实际的 forward pass、KV cache、权重持有，全部由 `dynamo.vllm` 子进程做。
  actor 进程**不**嵌 vLLM AsyncLLM。
- `ServerAdapter`（`dynamo_rollout.py`）= verl trainer rank 侧的薄客户端。
  generate 请求经 `_server_address`（→ frontend HTTP）派发到 router；wake_up /
  sleep / set_global_steps / clear_kv_cache 通过 `ray.get_actor("dynamo_server_*")`
  调到 actor 上，由 actor 转发给子进程或直接 no-op。
- v1 范围：只验证 dynamo 链路能起、frontend healthy、看门狗正确报错。
  weight update / forward pass 走通的训练 v2 再做。

---

## 1. 范围

### In-scope

- `recipe/dynamo/__init__.py` — 空 docstring，无 module-level 副作用。
- `recipe/dynamo/dynamo_async_server.py` — `DynamoHttpServer` actor + `DynamoReplica`。
- `recipe/dynamo/dynamo_rollout.py` — `ServerAdapter`。
- `recipe/dynamo/config/dynamo_trainer.yaml` — 继承 `ppo_trainer`，仅改
  `rollout.name=dynamo` / `rollout.mode=async`。
- `recipe/dynamo/main_dynamo.py` — hydra 入口，复用 verl `TaskRunner`。

### Out-of-scope（v1）

- weight update（`init_collective` / `update_weights_via_ipc_zmq` / `update_weights_from_collective` 全部 stub `assert False`，与 nemo-rl `DynamoVllmGeneration` 一致）。
- 训练 step（generate-only smoke）。
- 跨 node、planner、autoscaling、disagg prefill/decode。
- KV-event publisher 与 KV router 调优。

---

## 2. 进程拓扑

单 node，8 GPU，TP=2，DP=4 时：

```
┌── Ray cluster, 1 node, 8 GPU ───────────────────────────────────────────┐
│                                                                         │
│  ┌─ Ray actor: DynamoHttpServer (no forward pass) ─────────────────┐    │
│  │  @ray.remote(num_gpus=8)  ← GPU placeholder                    │    │
│  │                                                                │    │
│  │  Subprocesses (managed via _SUBPROCESS_REGISTRY):              │    │
│  │    1. etcd                       (service discovery)           │    │
│  │    2. nats-server                (event plane)                 │    │
│  │    3. python -m dynamo.vllm × 4  (each TP=2)                   │    │
│  │    4. python -m dynamo.frontend  (HTTP router on :frontend_port)│   │
│  │                                                                │    │
│  │  Watchdog loop: poll subprocess.poll() every 5s; raise if any │    │
│  │  subprocess exits unexpectedly.                                │    │
│  └────────────────────────────────────────────────────────────────┘    │
│           │                                                             │
│           │ env: CUDA_VISIBLE_DEVICES=<TP-slice>                        │
│           │      ETCD_ENDPOINTS, NATS_SERVER, DYN_NAMESPACE             │
│           ▼                                                             │
│   ┌─ dynamo.vllm[GPU 0,1] ─┐  ┌─ dynamo.vllm[GPU 2,3] ─┐  ...           │
│   │  TP=2 vLLM             │  │  TP=2 vLLM             │                │
│   │  register_model()      │  │  register_model()      │                │
│   │  serve_endpoint        │  │  serve_endpoint        │                │
│   └────────────────────────┘  └────────────────────────┘                │
│           ▲                       ▲                                     │
│           └───────── etcd discovery ───────────┐                        │
│                                                ▼                        │
│   ┌─ dynamo.frontend ─────────────────────────────┐                     │
│   │  HTTP :frontend_port (OpenAI-compatible)      │                     │
│   │  --router-mode kv  --discovery-backend etcd   │                     │
│   └───────────────────────────────────────────────┘                     │
│           ▲                                                             │
└───────────│─────────────────────────────────────────────────────────────┘
            │ HTTP (chat/completions)
            │
   verl trainer ranks → ServerAdapter
       (server_address = host:frontend_port)
```

要点：

1. actor `num_gpus=N` 让 Ray 整 node 锁给本 actor。子进程通过环境变量
   `CUDA_VISIBLE_DEVICES=<TP-slice>` 实际使用 GPU。
2. service discovery 用 etcd（与 nemo-rl 一致）；不用 file_kv。
3. frontend 与所有 vllm worker 在**同一 actor 进程**起子进程，避免跨进程
   IP/port 协商。

---

## 3. 文件与类划分

```
recipe/dynamo/
├── __init__.py                    # 空 docstring，无副作用
├── dynamo_async_server.py
│   ├── DynamoHttpServer           # Ray actor，verl-side 接口 + subprocess 管理
│   └── DynamoReplica(RolloutReplica)
│                                  #   - init_standalone() 自起 placement group
│                                  #   - launch_servers() 起 actor + 调 launch_server
├── dynamo_rollout.py
│   └── ServerAdapter              # 极简：继承 vLLMServerAdapter，覆盖
│                                  #   _get_server_name_prefix()
├── config/
│   └── dynamo_trainer.yaml        # 继承 ppo_trainer，rollout.name=dynamo
└── main_dynamo.py                 # hydra 入口
```

verl 主仓改动：**0 行**——`_ROLLOUT_REGISTRY` 与 `RolloutReplicaRegistry` 的
entry（已在主仓 commit `88eb8993` 中）继续复用，`("dynamo","async")` 指向
`recipe.dynamo.dynamo_rollout.ServerAdapter`，`RolloutReplicaRegistry["dynamo"]`
指向 `recipe.dynamo.dynamo_async_server.DynamoReplica`。

---

## 4. `DynamoHttpServer`

### 4.1 类签名

直接对齐 verl `vLLMHttpServer.__init__` 的签名（这样 `vLLMReplica.launch_servers`
里那段 `self.server_class.options(...).remote(config=..., model_config=..., ...)`
不用改），但内部不嵌 vLLM AsyncLLM。

```python
class DynamoHttpServer:
    """Ray actor: GPU placeholder + dynamo subprocess watchdog.

    Unlike vLLMHttpServer (which holds an in-process AsyncLLM), this actor
    does not own a model. It only:
      1. Reserves GPU bundles via @ray.remote(num_gpus=...)
      2. Spawns etcd / NATS / dynamo.frontend / dynamo.vllm subprocesses
      3. Watchdogs them; raises on unexpected exit.
      4. Tears down cleanly on shutdown.

    All inference traffic goes through the dynamo.frontend subprocess; the
    actor never sees a generate() call. ServerAdapter on the trainer side
    talks directly to the frontend's HTTP port via get_server_address.
    """

    def __init__(
        self,
        config,                        # RolloutConfig
        model_config,                  # HFModelConfig
        rollout_mode,                  # RolloutMode
        workers,                       # list[ActorHandle], unused but kept for parity
        replica_rank: int,
        node_rank: int,
        gpus_per_node: int,
        nnodes: int,
        cuda_visible_devices: str,
    ):
        ...
```

注意：`workers` 参数收着但不用——`vLLMReplica.launch_servers` 会传，我们保持
签名兼容；不嵌 worker 时这个 list 没意义。

### 4.2 子进程注册表（teardown 顺序）

来自 nemo-rl `dynamo_generation.py:60-65` 的模式，加一项 vllm workers：

```python
# (attr_name, display_name, stop_timeout_seconds)
_SUBPROCESS_REGISTRY: list[tuple[str, str, int]] = [
    ("_frontend_process", "frontend", 15),
    ("_vllm_processes",   "vllm_workers", 30),  # list, not single Popen
    ("_nats_process",     "NATS", 10),
    ("_etcd_process",     "etcd", 10),
]
```

teardown 顺序：先停消费方（frontend），再停被消费方（vllm worker），最后停
基础设施（NATS、etcd）。这样不会出现「frontend 还在派请求时 worker 已死」
或「worker 还在 publish 时 NATS 已挂」。

### 4.3 `launch_server` 流程

对应 nemo-rl 各方法 1:1：

| 步骤 | nemo-rl 引用 | 描述 |
|---|---|---|
| 1. 选端口 | `_get_free_port_local` | 用 verl `get_free_port`，预留 etcd/etcd-peer/NATS/frontend |
| 2. 起 etcd | `_start_etcd` (`dynamo_generation.py:270`) | tmpdir 数据目录、`--listen-client-urls` 等 |
| 3. 等 etcd healthy | `_wait_for_etcd` | 轮询 `/health`，30s timeout |
| 4. 起 NATS | `_start_nats` | `nats-server -p <port>` |
| 5. 等 NATS 端口可连 | `_wait_for_port` | 30s timeout |
| 6. 起 vllm workers | 简化版 `DynamoWorkerPool._create_workers` | 每个 dp shard 一个 `python -m dynamo.vllm` 子进程；CUDA_VISIBLE_DEVICES 切片传入 |
| 7. 起 frontend | `_start_frontend` (`dynamo_generation.py:351`) | `python -m dynamo.frontend --router-mode kv ...` |
| 8. healthcheck frontend | `_healthcheck_frontend` | 轮询 `/health`，等 expected_workers 注册满 |
| 9. 暴露 server_address | 设置 `self._server_address` / `self._server_port=frontend_port` | trainer 拿到的 URL = frontend |
| 10. 启动看门狗 | 新增 `_start_watchdog` | `asyncio.create_task(_watchdog_loop)` |

### 4.4 子进程环境变量（统一来源）

来自 `dynamo_generation.py:212-221` 的 `_dynamo_env_vars`，原样照搬：

```python
def _dynamo_env_vars(self) -> dict[str, str]:
    return {
        "ETCD_ENDPOINTS": f"http://{self._host}:{self._etcd_port}",
        "NATS_SERVER":    f"nats://{self._host}:{self._nats_port}",
        "DYN_NAMESPACE":  self._namespace,                  # 默认 "verl_dynamo"
        "DYN_DISCOVERY_BACKEND": "etcd",
        "DYN_SDK_DISABLE_ANSI_LOGGING": "1",
        "DYN_LOG": (
            "dynamo_llm::http::service::metrics=warn,"
            "dynamo_runtime::pipeline::network::ingress::push_handler=warn,"
            "dynamo_llm::http::service::service_v2=warn,info"
        ),
    }
```

### 4.5 vllm worker 子进程的拉起

不复制 nemo-rl 的 `WorkerFactory`/`RayWorkerBuilder` 那套（对动态 scaling 没需求）。
直接：

```python
def _start_vllm_workers(self):
    tp_size = self.config.tensor_model_parallel_size
    cvd = self._cuda_visible_devices.split(",")
    assert len(cvd) % tp_size == 0
    n_workers = len(cvd) // tp_size

    self._vllm_processes: list[subprocess.Popen] = []
    for i in range(n_workers):
        worker_cvd = ",".join(cvd[i*tp_size : (i+1)*tp_size])
        env = {
            **os.environ, **self._dynamo_env_vars(),
            "CUDA_VISIBLE_DEVICES": worker_cvd,
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_SKIP_P2P_CHECK": "1",
        }
        cmd = [
            sys.executable, "-m", "dynamo.vllm",
            "--model", self.model_config.local_path,
            "--tensor-parallel-size", str(tp_size),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--max-model-len", str(self.config.max_model_len or 4096),
            "--dtype", self.config.dtype,
            "--seed", str(i + self.replica_rank * 1024),
        ]
        if self.model_config.trust_remote_code:
            cmd.append("--trust-remote-code")
        if self.config.enforce_eager:
            cmd.append("--enforce-eager")
        self._vllm_processes.append(subprocess.Popen(cmd, env=env))
```

### 4.6 看门狗

```python
def _check_subprocesses(self):
    for attr, name, _ in _SUBPROCESS_REGISTRY:
        proc = getattr(self, attr, None)
        if proc is None:
            continue
        if isinstance(proc, list):  # vllm workers
            for i, p in enumerate(proc):
                if p.poll() is not None:
                    raise RuntimeError(
                        f"dynamo {name}[{i}] exited rc={p.returncode}")
        elif proc.poll() is not None:
            raise RuntimeError(f"dynamo {name} exited rc={proc.returncode}")

async def _watchdog_loop(self):
    while not self._shutdown_requested:
        try:
            self._check_subprocesses()
        except RuntimeError:
            logger.exception("dynamo subprocess died, actor will crash")
            raise
        await asyncio.sleep(5.0)
```

异常退出 → 抛 RuntimeError → Ray actor crash → trainer job fail（fail-fast）。

### 4.7 verl-required 接口

verl 主流程会调下面这些方法，每一个都有明确语义：

| 方法 | 调用方 | v1 实现 |
|---|---|---|
| `get_master_address()` | `vLLMReplica.launch_servers` | 返回 `(host, port, dp_rpc_port)` 凑数；nnodes=1 时不会用上 |
| `get_server_address()` | `RolloutReplica.launch_servers` 末尾、`ServerAdapter` 经 server_address URL | 返回 `(host, frontend_port)` |
| `launch_server(master_address, master_port, dp_rpc_port)` | `vLLMReplica.launch_servers` | 触发 §4.3 整套启动流程；non-master 节点 v1 不支持 |
| `generate(...)` | agent loop 经 `server.generate.remote(...)` | **raise NotImplementedError**（agent loop 走 HTTP 到 server_address，不会调到 actor 这里） |
| `wake_up()` / `sleep()` | `RolloutReplica` | no-op（v1 不做 memory occupation 控制） |
| `clear_kv_cache()` | `ServerAdapter.update_weights` | no-op（v1 不做权重更新） |
| `set_global_steps(n)` | `ServerAdapter.update_weights` | 只缓存到 `self.global_steps` |
| `abort_all_requests()` / `resume_generation()` | `RolloutReplica` | no-op（dynamo 暂无 abort API） |
| `start_profile()` / `stop_profile()` | profiler | no-op |
| `wait_for_requests_to_drain()` | `vLLMReplica.sleep` | no-op |
| `collective_rpc(method, ...)` | `ServerAdapter._execute_method` | **raise NotImplementedError**（v1 没有 in-process engine） |
| `shutdown()` | replica teardown | 逐个 SIGTERM 子进程 + cleanup tmp dirs |

### 4.8 序列化

参考 `dynamo_generation.py:551-563`，Ray actor 的 `__getstate__` 把
subprocess handle 和 watchdog task 屏蔽掉（不可 pickle）：

```python
def __getstate__(self):
    state = self.__dict__.copy()
    for attr, _, _ in _SUBPROCESS_REGISTRY:
        state[attr] = None
    state["_watchdog_task"] = None
    return state
```

---

## 5. `DynamoReplica`

继承 `RolloutReplica`，关键覆盖点：

```python
class DynamoReplica(RolloutReplica):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_class = ray.remote(DynamoHttpServer)

    def _get_server_name_prefix(self) -> str:
        return "dynamo_"

    def rollout_worker_use_gpu(self) -> bool:
        # CheckpointEngineWorker 占位（trainer 侧 GPU pool）；DynamoHttpServer
        # 自己再 num_gpus=N 占一遍。v1 用 standalone 模式，不冲突。
        return False

    async def launch_servers(self):
        # nnodes=1 only in v1.
        # 要求 self.workers 已就位（init_standalone 走过）；CUDA_VISIBLE_DEVICES
        # 用 worker actor 实际拿到的 GPU 列表。
        ...
```

`launch_servers` 主体复用 vLLMReplica（fetch worker node_id + cvd → 起 server actor → 调 `launch_server`），唯一区别：
- `name` 用 `dynamo_server_{r}_{n}` 前缀（与 `_get_server_name_prefix()` 对齐）。
- v1 nnodes==1 校验：多 node 暂不支持。

---

## 6. `ServerAdapter`

trainer rank 上每个 process 一个，做 actor handle lookup 和 RPC 转发。
v1 极简——继承 vLLM 的 `ServerAdapter`，**只**重写 actor name prefix：

```python
from verl.workers.rollout.vllm_rollout.vllm_rollout import ServerAdapter as _VllmServerAdapter

class ServerAdapter(_VllmServerAdapter):
    def _get_server_name_prefix(self) -> str:
        return "dynamo_"
```

这样：
- agent loop 通过 `replica.server_address`（由 `DynamoReplica` 设置成 frontend
  URL）发 chat completions HTTP 请求 → 命中 dynamo router。
- `update_weights` / `wake_up` / `sleep` / `clear_kv_cache` 通过
  `ray.get_actor("dynamo_server_{r}_{n}")` 调到 `DynamoHttpServer`，actor 内部
  按 §4.7 表 no-op 或转发。

---

## 7. 配置

`config/dynamo_trainer.yaml`（继承 `ppo_trainer`）：

```yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

actor_rollout_ref:
  rollout:
    name: dynamo
    mode: async
    engine_kwargs:
      dynamo:
        namespace: verl_dynamo
        router_mode: kv         # round-robin | kv | random | least-loaded
        frontend_http_port: 0   # 0 = auto

trainer:
  project_name: verl-dynamo
```

读取：actor 内通过 `self.config.engine_kwargs.dynamo.<key>` 拿到，与 nemo-rl
`config["dynamo_cfg"]` 字段名 1:1。

---

## 8. v1 验证清单

每个里程碑配一个明确的 pass criterion：

| Stage | 验证 |
|---|---|
| s1: `import recipe.dynamo.dynamo_async_server` | 不抛错（`__init__` 无 module-level import 副作用） |
| s2: actor 起得来 | `DynamoReplica.launch_servers()` 跑完，`server_address` 是 `host:port` 字符串 |
| s3: etcd / NATS / frontend 都健康 | `curl http://<host>:<frontend_port>/health` 200 |
| s4: 至少 1 个 vllm worker 注册 | `/health` JSON 里 `instances` 非空 |
| s5: 一次 chat completion 跑通 | `curl POST /v1/chat/completions` 返回 token |
| s6: 看门狗 | `kill -9 frontend_pid` → 5s 内 actor 抛 RuntimeError → job fail |
| s7: shutdown 无残留 | `pgrep -f dynamo.frontend` / `pgrep -f dynamo.vllm` / `pgrep -f etcd` 都没东西 |

---

## 9. 已知限制（v1）

1. **不支持训练**：weight update 全部 stub，只能跑 generate-only smoke。
2. **单 node**：`nnodes>1` 暂不支持（nemo-rl 的跨 node placement group 没移植）。
3. **fail-fast**：任一 subprocess 异常退出 → 整 job 挂；没有 retry。
4. **etcd / nats-server 二进制**：必须在 PATH 里（与 nemo-rl 同要求）。
5. **venv**：用 `sys.executable`（actor 同 venv 起 dynamo subprocess）；
   nemo-rl 用 `create_local_venv_on_each_node` 隔离，v1 暂不做。

---

## 10. 参考

- 实现模板：`/lustre/fsw/general_sa/sopyang/rl/RL/nemo_rl/models/generation/dynamo/dynamo_generation.py`
- verl 接口背景：
  - `verl/workers/rollout/replica.py` — `RolloutReplica` 基类
  - `verl/workers/rollout/vllm_rollout/vllm_async_server.py` — `vLLMHttpServer` & `vLLMReplica`
  - `verl/workers/rollout/vllm_rollout/vllm_rollout.py` — `ServerAdapter`（基类）
  - `verl/workers/rollout/base.py:88` — `("dynamo","async") -> recipe.dynamo.dynamo_rollout.ServerAdapter`
  - `verl/workers/rollout/replica.py:390` — `_load_dynamo` 注册 entry

---

## 11. Multi-node 可行性与约束（v3 范围）

v1 单 node generate-only smoke 跑通后，v3 把它扩到多 node。本节登记可行
性结论、必须遵守的约束、要改的地方。

### 11.1 硬约束

**TP group 不能跨 node。** 原因：
- `cudaIpcOpenMemHandle` 是单 host 内的 GPU IPC，跨 node 无定义。
- ZMQ `ipc:///tmp/...sock` 是 UNIX socket 文件，跨 node 不可见。
- → 每个 vLLM TP group 必须整组落在同一 node。

实际约束：`config.tensor_model_parallel_size <= gpus_per_node`。违反就在
`DynamoReplica.__init__` 里 assert，并提示「改 DP scaling」。

DP **可以** 跨 node（dp shard 之间无 IPC 同步）；PP 也可以（PP 走
NCCL，不走 IPC）。

### 11.2 actor 主从分工

`DynamoReplica.launch_servers` 复用 vLLMReplica 的 nnodes 循环——一个
`DynamoHttpServer` actor / node。`launch_server(master_address,
master_port, dp_rpc_port)` 区分主从：

```
node 0 (master, replica_rank=0, node_rank=0):
  ├─ _start_etcd      ← 仅 master 起
  ├─ _start_nats      ← 仅 master 起
  ├─ _start_vllm_workers (本 node 的 dp shards)
  ├─ _start_frontend  ← 仅 master 起
  └─ _healthcheck_frontend(expected_workers = nnodes * dp_per_node)

node 1..N-1 (slave, node_rank>0):
  ├─ ETCD_ENDPOINTS=http://<master_ip>:<etcd_port>   ← launch_server 传入
  ├─ NATS_SERVER=nats://<master_ip>:<nats_port>
  ├─ _start_vllm_workers (本 node 的 dp shards)
  └─ 不起 frontend / etcd / nats
```

`get_master_address()` 在 master 返回 `(master_ip, etcd_port, nats_port)`
的封装；`get_server_address()` 在所有 node 都返回 `(master_ip,
frontend_port)`，trainer rank 全打 master frontend。

### 11.3 `VERL_DYNAMO_RANK_OFFSET` 按 node-local 计算

trainer 侧 `local_rank = rollout_rank % local_world_size` 本来就是
node-local（`local_world_size == gpus_per_node`，每 node 内 0..N-1）。所
以 dynamo.vllm 子进程的 RANK_OFFSET 也按本 node 的 dp shard idx 算，不
跨 node 累加：

```python
# DynamoHttpServer._start_vllm_workers (multi-node aware)
n_dp_local = self.gpus_per_node // self.config.tensor_model_parallel_size
for shard_idx_local in range(n_dp_local):
    env["VERL_DYNAMO_RANK_OFFSET"] = str(shard_idx_local * tp_size)
    env["VERL_REPLICA_RANK"] = str(self.replica_rank)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(
        cvd[shard_idx_local*tp_size : (shard_idx_local+1)*tp_size]
    )
    Popen([..., "-m", "recipe.dynamo._dynamo_vllm_with_control", ...], env=env)
```

每 node 的 `(replica_rank, global_rank)` 对应本 node 的 trainer
ranks，IPC socket 文件天然 node-local，不会冲突。

### 11.4 拓扑示意（2 nodes × 8 GPU，TP=2，per-node DP=4）

```
┌── node 0 (master) ──────────────────────────────────────────┐
│  DynamoHttpServer (replica_rank=0, node_rank=0)             │
│  ├─ etcd :ETCD_PORT    (advertise master_ip)                │
│  ├─ nats-server :NATS_PORT                                  │
│  ├─ dynamo.vllm × 4 (GPU 0-1, 2-3, 4-5, 6-7)                │
│  │     ENV: ETCD_ENDPOINTS, NATS_SERVER → master_ip         │
│  │          VERL_DYNAMO_RANK_OFFSET = 0,2,4,6               │
│  └─ dynamo.frontend :FRONTEND_PORT                          │
│       --discovery-backend etcd  --router-mode kv            │
└─────────────────────────────────────────────────────────────┘
             ▲ etcd / nats / frontend 暴露给整集群
             │
┌── node 1 (slave) ───────────────────────────────────────────┐
│  DynamoHttpServer (replica_rank=0, node_rank=1)             │
│  ├─ (no etcd, no nats, no frontend)                         │
│  └─ dynamo.vllm × 4 (GPU 0-1, 2-3, 4-5, 6-7)                │
│        ENV: ETCD_ENDPOINTS, NATS_SERVER → master_ip         │
│             VERL_DYNAMO_RANK_OFFSET = 0,2,4,6  (本 node 内) │
└─────────────────────────────────────────────────────────────┘

verl trainer rank 0..15:
  rank 0..7  在 node 0：generate → master frontend (localhost)
                       weight update → 本 node IPC × 4 dp shards
  rank 8..15 在 node 1：generate → master frontend (cross-node 1 跳)
                       weight update → 本 node IPC × 4 dp shards
```

### 11.5 性能影响

| 路径 | single-node | multi-node | 说明 |
|---|---|---|---|
| generate / 请求 | trainer→frontend localhost→worker localhost | trainer→frontend (跨 node) → worker (dynamo TCP 跨 node) | 加 1-2 跳 NIC RTT；m4 KV router 设计就是为这个场景 |
| weight update | per-node CUDA IPC | per-node CUDA IPC（每 node 各自跑） | **0 退化**——CUDA IPC 不跨 node，weight 同步天然 per-node 并行 |
| collective_rpc 触发 | 单 actor 内本地 ZMQ | 单 actor 内本地 ZMQ + slave actor 的本地 ZMQ（跨 node Ray RPC 1 次） | µs 级 |
| 启动 | 单 actor 起全部 | master 起 etcd/nats/frontend，slave 起 workers | slave 在 `_wait_for_etcd` 阻塞直到 master ready |

### 11.6 落地清单

1. `DynamoHttpServer.get_master_address()` 返回 `(master_ip, etcd_port,
   nats_port)`，复用 vLLMHttpServer 既定签名（master_port 位用 etcd
   port，dp_rpc_port 位用 nats port）。
2. `DynamoHttpServer.launch_server(master_address, master_port, dp_rpc_port)`
   按 node_rank 分支：
   - `node_rank == 0`：起 etcd + nats + workers + frontend。
   - `node_rank > 0`：拼 `ETCD_ENDPOINTS` / `NATS_SERVER` env，仅起 workers。
3. `DynamoReplica.launch_servers`：复用 vLLMReplica 的 nnodes 循环（`for
   node_rank in range(nnodes)`），先建所有 actor，调
   `master.get_master_address`，把结果传给所有 `launch_server`。
4. `_healthcheck_frontend(expected_workers)`：master 等待
   `nnodes * dp_per_node` 个 worker 全注册。
5. `DynamoReplica.__init__`：
   ```python
   assert self.config.tensor_model_parallel_size <= self.gpus_per_replica_node, (
       f"TP={self.config.tensor_model_parallel_size} 必须 <= gpus_per_node="
       f"{self.gpus_per_replica_node}（CUDA IPC 不跨 node）；改成 DP 扩"
   )
   ```
6. `DynamoHttpServer.shutdown` 顺序：master 多了 frontend / etcd / nats
   要停（`_SUBPROCESS_REGISTRY` 已经按从前到后排好）。

### 11.7 不可行 / 不覆盖的场景

- **TP > gpus_per_node**：dynamo.vllm 默认 MP executor 不支持跨 node TP；
  绕路是切到 `dynamo.sglang` backend（支持 Ray 跨 node TP），但那是另一
  条 work item，不在本设计范围。
- **跨 node 单 vllm worker 同步 weight**（PP 拆 node + 单 worker 跨 node）：
  CUDA IPC 不存在，得切 NCCL broadcast 或 NIXL；本设计不覆盖。
- **etcd / NATS 跨 SLURM 节点防火墙**：要确保 master node 的 etcd/nats
  端口对集群其它 node 可达；SLURM 默认 OK，K8s 要 expose service。
