# Agentic Recipe

基于 verl PPO 训练流程的扩展 recipe，引入一个 **OpenAI 兼容的 LLM 代理服务**（proxy server），将远程第三方 Agent 的推理请求桥接到 verl 的 vLLM rollout 服务，从而支持 *agentic* 风格的 RL 训练。

## 1. 适用场景

当 rollout 不再是单轮 prompt → response，而是需要由一个**外部 Agent**（例如 SWE-Agent、ReAct/Tool-Use Agent 等）驱动多轮交互、调用工具、消费环境反馈时，使用本 recipe：

- 训练侧（verl）继续使用标准 PPO/GRPO 训练循环；
- 推理侧通过 HTTP 暴露 OpenAI 兼容接口，给远程 Agent（运行在 [Harbor](../../harbor) 或本地 Trial 中）调用；
- 代理服务负责会话路由、消息记录、tool 解析、tokenization，回收完整 trajectory 给 verl 用于训练。

## 2. 目录结构

```
recipe/agentic/
├── agentic_main.py           # 入口：继承 PPO TaskRunner，init_workers 后启动 proxy server
├── agentic-qwen2.5-3b.sh     # 端到端启动脚本（Qwen2.5 + SWE-Agent 示例）
├── mcp-tools.sh              # MCP 工具相关的辅助脚本
├── swe-agent.yaml            # agent_loop 配置示例（指向 RemoteAgentLoop）
├── config/
│   └── agentic_trainer.yaml  # Hydra 配置：继承 ppo_trainer + proxy_server 配置
├── agent_loop/
│   ├── remote_agent_loop.py  # RemoteAgentLoop：通过 Harbor/本地 Trial 驱动远程 Agent
│   └── config.py             # RemoteAgentConfig：所有参数从环境变量读取
├── proxyserver/              # 代理服务实现
│   ├── ray_actor.py          # ProxyServerActor（Ray named actor，head 节点）
│   ├── proxy_server.py       # FastAPI/uvicorn 入口
│   ├── server.py             # OpenAI 兼容 HTTP endpoints
│   ├── relay.py              # 请求中继 / token 化 / tool 解析
│   ├── recorder.py           # session 记录与 trajectory 回放
│   ├── vllm_provider.py      # 桥接 verl AsyncLLMServerManager
│   ├── worker_client.py      # 独立部署模式下的 WebSocket 推理 worker 客户端
│   ├── models.py             # SessionRecord 等数据模型
│   ├── Dockerfile / deploy.yaml  # 独立部署所需镜像与 K8s 资源
└── serversdk/                # 远程 Agent 调用 verl/proxy 的客户端 SDK
    ├── client.py             # AgentRunClient
    └── models.py
```

## 3. 运行模式

`RemoteAgentLoop` 与代理服务支持两种部署模式（详见 [`agent_loop/remote_agent_loop.py`](agent_loop/remote_agent_loop.py)）：

| 模式 | 启用方式 | 说明 |
| --- | --- | --- |
| **Ray actor 模式**（默认） | 不设置 `PROXY_SERVER_URL` | proxy 由 `agentic_main.py` 在 `init_workers()` 后通过 `start_proxy_server()` 创建为 Ray named actor，运行在 head 节点。 |
| **独立 proxy 模式** | 设置 `PROXY_SERVER_URL=http://proxy:port` | proxy 作为独立服务（可用 [`proxyserver/Dockerfile`](proxyserver/Dockerfile) + [`proxyserver/deploy.yaml`](proxyserver/deploy.yaml) 部署）；agent loop 自动启动 `InferenceWorkerClient` 通过 WebSocket 把 verl 的 vLLM 服务接入 proxy。 |

远程 Agent 收到的 `base_url` 形如：

```
http://${LLM_PROXY_IP}:${port}/${trial_id}/v1
```

其中 `LLM_PROXY_IP` 必须是 Ray 集群外部（例如 Harbor 服务）能访问的 LLM proxy 对外 IP。当 LLM proxy 与 trainer 同机部署时，它就是 trainer 节点本机的对外 IP；独立 proxy 部署时则填独立 proxy 服务的对外 IP。

## 4. 配置优先级与关键参数

`RemoteAgentLoop` / proxy 的参数同时支持三种来源，优先级从高到低：

1. **命令行 override**（`remote_agent.xxx=...`、`proxy_server.llm_proxy_ip=...`）；Hydra 直接写入配置。
2. **shell 环境变量**（`REMOTE_AGENT_NAME` 等）；[`agentic_main.py`](agentic_main.py) 中使用 `os.environ.setdefault` 仅在未设时才从 yaml 注入。
3. **yaml 默认值**（[`config/agentic_trainer.yaml`](config/agentic_trainer.yaml) 中的 `remote_agent` / `proxy_server.llm_proxy_ip`）；在 driver 启动时被转换成环境变量并通过 Ray `runtime_env.env_vars` 传递给所有 worker。

yaml 字段与环境变量的对应关系：

| yaml 字段 | 环境变量 | 说明 |
| --- | --- | --- |
| `proxy_server.llm_proxy_ip` | `LLM_PROXY_IP` | LLM proxy 对远程 Agent 可达的 IP（proxy 与 trainer 同机时即 trainer 本机 IP） |
| `remote_agent.agent_name` | `REMOTE_AGENT_NAME` | Harbor 上注册的 agent 名 |
| `remote_agent.agent_import_path` | `REMOTE_AGENT_IMPORT_PATH` | 本地 Trial 模式下 agent 的 import path |
| `remote_agent.model_name` | `REMOTE_MODEL_NAME` | 透传给远程 agent 的 model name |
| `remote_agent.agent_kwargs` | `REMOTE_AGENT_KWARGS` | dict，yaml 中直接用嵌套对象 |
| `remote_agent.max_retries` / `retry_base_delay` | `REMOTE_AGENT_MAX_RETRIES` / `REMOTE_AGENT_RETRY_BASE_DELAY` | 重试控制 |
| `remote_agent.task_path_template` | `REMOTE_AGENT_TASK_PATH_TEMPLATE` | 任务路径模板 |
| `remote_agent.environment_overrides` | `REMOTE_AGENT_ENVIRONMENT_OVERRIDES` | dict |
| `remote_agent.environment_kwargs` | `REMOTE_AGENT_ENVIRONMENT_KWARGS` | dict，传给环境构造器 |
| `remote_agent.use_local_trial` | `REMOTE_AGENT_USE_LOCAL_TRIAL` | true 表示本地 Trial 模式 |
| `remote_agent.proxy_server_url` | `PROXY_SERVER_URL` | 设置后走独立 proxy 模式 |
| `remote_agent.harbor_server_url` / `harbor_timeout` | `HARBOR_SERVER_URL` / `HARBOR_TIMEOUT` | Harbor 服务连接 |

> 调试场景可随时用 `export REMOTE_AGENT_USE_LOCAL_TRIAL=true` 临时覆盖 yaml，
> 不需修改配置文件。`PYTHONPATH` 因为需要在 Python 启动前生效，仍保留在 shell 脚本中。

## 5. 快速开始

### 5.1 准备依赖

```bash
# 在 verl 仓库根目录下
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
# Harbor 端依赖
uv pip install -e ./harbor
```

### 5.2 启动训练（SWE-Agent + Qwen2.5 示例）

直接使用提供的端到端脚本：

```bash
bash recipe/agentic/agentic-qwen2.5-3b.sh
```

脚本核心命令等价于（参数全部走 Hydra override，`PYTHONPATH` 除外）：

```bash
PYTHONPATH=/home/verl/harbor/src/harbor:$PYTHONPATH \
python3 -m recipe.agentic.agentic_main \
    proxy_server.llm_proxy_ip=10.0.30.11 \
    remote_agent.agent_name=swe-agent \
    remote_agent.model_name=openai/qwen-max \
    remote_agent.use_local_trial=true \
    remote_agent.task_path_template='/home/verl/dataset-tasks/{instance_id}' \
    'remote_agent.agent_kwargs={total_cost_limit: 0, per_instance_cost_limit: 0}' \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.agent.default_agent_loop=remote_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=recipe/agentic/swe-agent.yaml \
    ... # 其它 PPO/GRPO 配置项
```

要点：

- `actor_rollout_ref.rollout.mode=async`：rollout 必须使用异步模式；
- `actor_rollout_ref.rollout.agent.default_agent_loop=remote_agent`：使用本 recipe 提供的 `RemoteAgentLoop`；
- `agent_loop_config_path`：指向 [`swe-agent.yaml`](swe-agent.yaml)（也可以替换为自定义 yaml）。

### 5.3 自定义 proxy 监听端口

修改 [`config/agentic_trainer.yaml`](config/agentic_trainer.yaml) 或在命令行 override：

```yaml
proxy_server:
  host: "0.0.0.0"
  port: 0           # 0 表示随机端口
  tool_format: "hermes"
```

```bash
python3 -m recipe.agentic.agentic_main \
    proxy_server.port=9123 \
    proxy_server.tool_format=hermes \
    ...
```

## 6. 从 Harbor 本地任务目录构建数据集

除了手工准备好的 parquet，agentic recipe 还支持直接从两个 **本地 Harbor 任务目录**（分别对应 train / val）构建数据集，适用于已经将 Harbor 任务同步到本地磁盘的场景。

### 6.1 目录约定

输入目录下的每个子目录被视为一个 Harbor task，**必须**包含如下两个文件：

```
<train_root>/
    task-001/
        task.toml          # TOML 格式任务配置（会被写入 extra_info.task_metadata）
        instruction.md     # 作为 user prompt 内容
        environment/       # 可选，不被读取
        tests/             # 可选，不被读取
    task-002/
        ...
<val_root>/
    ...
```

不同时包含 `task.toml` 与 `instruction.md` 的子目录会被跳过。

### 6.2 生成的 row schema

与 verl 默认 RL row 一致：

```python
{
    "data_source": "harbor",
    "prompt": [{"role": "user", "content": <instruction.md 内容>}],
    "ability": "agent",
    "reward_model": {"style": "rule", "ground_truth": <task_id>},
    "extra_info": {
        "split": "train" | "val",
        "index": <从 0 起的序号>,
        "task_id": <子目录名>,
        "local_path": <任务本地路径>,
        "task_metadata": <task.toml 中的 metadata 表>,
    },
    "instance_id": <子目录名>,         # 顶层字段，会原样透传给 RemoteAgentLoop.run 的 kwargs
    "local_task_path": <任务本地路径>,  # 顶层字段，含任务目录绝对路径
}
```

实现参见 [`dataset/local_harbor.py`](dataset/local_harbor.py)，本地扫描不依赖 OSS / `oss2` SDK。

### 6.3 配置项

在 [`config/agentic_trainer.yaml`](config/agentic_trainer.yaml) 中的 `data` 节点下：

| 字段 | 说明 |
| --- | --- |
| `data.train_harbor_dir` | 训练集任务根目录（含多个子任务目录）。为 `null` 时不启用，使用原 `data.train_files` |
| `data.val_harbor_dir` | 验证集任务根目录。为 `null` 时不启用，使用原 `data.val_files` |
| `data.harbor_cache_dir` | 生成的 parquet 缓存路径，默认 `~/.cache/verl/agentic/harbor` |
| `data.harbor_data_source` | row 中 `data_source` 字段值，默认 `harbor` |
| `data.harbor_ability` | row 中 `ability` 字段值，默认 `agent` |
| `data.harbor_overwrite` | `true` 时强制重新生成 parquet（默认复用缓存） |
| `data.harbor_train_limit` / `data.harbor_val_limit` | 可选上限，仅取前 N 个任务用于快速调试 |

启动后 [agentic_main.py](agentic_main.py) 会在加载 dataset 之前调用 `_materialize_harbor_datasets`：

1. 扫描 `train_harbor_dir` / `val_harbor_dir`，调用 `build_verl_parquets` 生成 `train-<sig>.parquet` / `val-<sig>.parquet`（文件名中 `<sig>` 是输入参数的哈希，同样输入仅生成一次）；
2. 通过 `OmegaConf.update` 覆盖 `data.train_files` 与 `data.val_files`；
3. 后续逻辑走 verl 原生的 `create_rl_dataset(parquet_paths, ...)`，无需任何定制 Dataset。

### 6.4 示例

```bash
PYTHONPATH=/home/verl/harbor/src/harbor:$PYTHONPATH \
python3 -m recipe.agentic.agentic_main \
    data.train_harbor_dir=/var/harbor/tasks/train \
    data.val_harbor_dir=/var/harbor/tasks/val \
    data.harbor_cache_dir=/var/cache/verl/agentic/harbor \
    proxy_server.llm_proxy_ip=10.0.30.11 \
    remote_agent.agent_name=swe-agent \
    remote_agent.use_local_trial=true \
    ... # 其它 PPO/GRPO 配置项
```

你不必再传 `data.train_files` / `data.val_files` —— 这两项会在 driver 启动时被生成的 parquet 路径覆盖。若同时传了 parquet 与 Harbor 目录，后者会赢。

### 6.5 RemoteAgentLoop 自动解析 task_path

[`RemoteAgentLoop`](agent_loop/remote_agent_loop.py) 中的 `_resolve_task_path()` 会按以下顺序解析单条 rollout 的 `task_path`：

1. **`local_task_path`**：dataset row 顶层字段（由本地 Harbor builder 写入），若指向存在的目录直接采用，无需再拼接路径；
2. **`<harbor_root>/<instance_id>`**：依次在 `data.train_harbor_dir`、`data.val_harbor_dir`、以及 env `HARBOR_DATASET_DIRS`（`:` 或 `,` 分隔）下查找；
3. **`task_path_template.format(instance_id=...)`**：兜底，保留旧的 `REMOTE_AGENT_TASK_PATH_TEMPLATE` 工作流。

意味着只要数据集是用本地 Harbor builder 生成的，启动时无须再设置 `remote_agent.task_path_template`，agent loop 会用 `instance_id` 自动定位任务目录。在独立 proxy 模式下若 worker 看不到 trainer config，也可改用 env：

```bash
export HARBOR_DATASET_DIRS=/var/harbor/tasks/train:/var/harbor/tasks/val
```

## 7. 工作流程

```
                 ┌─────────────────────┐
                 │  RayPPOTrainer      │  init_workers() → server_handles
                 └─────────┬───────────┘
                           │ start_proxy_server(server_handles, ...)
                           ▼
       ┌────────────────────────────────────────┐
       │  ProxyServerActor (Ray named actor)    │  HTTP/WebSocket
       │  - OpenAI 兼容 endpoints               │◄────────────────┐
       │  - session 路由 / 记录 trajectory       │                 │
       └─────────┬──────────────────────────────┘                 │
                 │ AsyncLLM (vLLM)                                │
                 ▼                                                │
       verl rollout workers (vLLM TP)                             │
                                                                  │
                                                                  │ OpenAI base_url
                                                                  │
                                                       ┌──────────┴────────────┐
                                                       │ Remote Agent (Harbor) │
                                                       │ e.g. SWE-Agent, ReAct │
                                                       └───────────────────────┘
```

1. `agentic_main.TaskRunner.run()` 启动标准 PPO 流程；
2. 在 `trainer.init_workers()` 之后、`trainer.fit()` 之前调用 `start_proxy_server`，注入 `server_handles`；
3. 每个 rollout 步骤里 `RemoteAgentLoop` 向 proxy 注册 session、把 `base_url` 交给远程 Agent；
4. 远程 Agent 通过 OpenAI 协议调用 proxy，proxy 再把请求转发给 verl 的 vLLM 服务；
5. 会话结束后 `RemoteAgentLoop` 从 proxy 拉取完整 token 序列、tool 调用记录，构造 `AgentLoopOutput` 返回训练器。

## 8. 相关参考

- 入口实现：[`agentic_main.py`](agentic_main.py)
- 远程 agent loop：[`agent_loop/remote_agent_loop.py`](agent_loop/remote_agent_loop.py)
- Proxy actor 启动：[`proxyserver/ray_actor.py`](proxyserver/ray_actor.py)
- Harbor 端 Agent 实现：[`../../harbor`](../../harbor)
- 基础 PPO 训练入口：[`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py)
