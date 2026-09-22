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

### 5.2 安装容器工具（Docker 或 crane + BuildKit）

本地 Trial 模式下 Harbor 环境需要构建/拉取容器镜像。可选择安装 Docker，或使用轻量的 crane + BuildKit 组合：

**方式一：Docker**

按照 [Docker 官方文档](https://docs.docker.com/engine/install/) 安装即可。

**方式二：crane + BuildKit（无需 Docker daemon）**

```bash
# 安装 crane（容器镜像操作工具）
VERSION=v0.20.2
curl -L "https://github.com/google/go-containerregistry/releases/download/${VERSION}/go-containerregistry_Linux_x86_64.tar.gz" -o crane.tar.gz
tar -xzf crane.tar.gz
mv crane /usr/local/bin/

# 安装 buildctl（BuildKit 客户端）
BUILDKIT_VERSION=v0.13.2
curl -sL "https://github.com/moby/buildkit/releases/download/${BUILDKIT_VERSION}/buildkit-${BUILDKIT_VERSION}.linux-amd64.tar.gz" \
  | tar -xz -C /usr/local bin/buildctl
```

### 5.3 下载数据集

使用 Harbor CLI 将任务数据集下载到本地：

```bash
harbor datasets download --output /home/verl/swe-bench-verified
```

下载后的目录结构应符合 [6.1 目录约定](#61-目录约定)。

### 5.4 登录容器镜像仓库

Harbor 环境需要拉取基础镜像，确保已登录目标 registry：

```bash
# 方式一：通过 docker login
docker login <registry-url>

# 方式二：通过 crane login
crane auth login <registry-url> -u <username> -p <password>

# 方式三：直接编辑 ~/.docker/config.json
cat > ~/.docker/config.json <<'EOF'
{
  "auths": {
    "<registry-url>": {
      "auth": "<base64(username:password)>"
    }
  }
}
EOF
```

> **提示**：crane 和 BuildKit 均会读取 `~/.docker/config.json` 中的认证信息，因此三种方式任选其一即可。

### 5.5 启动训练（SWE-Agent + Qwen2.5 示例）

脚本 [`agentic-qwen2.5-3b.sh`](agentic-qwen2.5-3b.sh) 通过 `ENVIRONMENT` 环境变量选择运行环境，支持 **Docker**（本地）和 **ACK**（Kubernetes）两种模式。脚本会自动设置对应的 `environment_import_path` 和 `environment_kwargs`。

#### 通用参数

| 变量 | 必填 | 默认值 | 说明 |
| --- | :---: | --- | --- |
| `MODEL_PATH` | **是** | — | HuggingFace 模型本地路径或名称，例如 `/var/model/Qwen2.5-7B-Instruct` |
| `ENVIRONMENT` | 否 | `docker` | 环境模式：`docker`（本地 Docker）或 `ack`（Kubernetes） |
| `REGISTRY` | 否 | *(空)* | 容器镜像 registry 地址，ACK 模式下用于推送/拉取 sandbox 镜像，例如 `my-registry.example.com/swebench` |

#### ACK (Kubernetes) 环境参数

以下参数仅在 `ENVIRONMENT=ack` 时生效，对应 `ACKEnvironment.__init__` 的参数：

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `NAMESPACE` | `default` | Kubernetes namespace |
| `KUBECONFIG` | *(空，使用默认 ~/.kube/config)* | kubeconfig 文件路径 |
| `IMAGE_PULL_SECRET` | *(空)* | K8s image pull secret 名称，用于从私有 registry 拉取镜像 |
| `SERVICE_ACCOUNT` | *(空)* | K8s service account 名称 |
| `USE_BUILDKIT` | `false` | 是否使用 BuildKit 构建 sandbox 镜像（替代 DinD）。设为 `true` 时需同时设置 `BUILDKIT_ADDRESS` |
| `BUILDKIT_ADDRESS` | *(空)* | BuildKit 服务地址，例如 `tcp://buildkit-service:1234`。仅当 `USE_BUILDKIT=true` 时生效 |
| `USE_SANDBOX_CLAIM` | `false` | 是否使用 [OpenKruise](https://openkruise.io/) SandboxClaim 获取预热的 sandbox 容器，避免每次冷启动，显著加速环境初始化 |
| `CLAIM_TIMEOUT` | `300` | SandboxClaim 等待超时（秒）。仅当 `USE_SANDBOX_CLAIM=true` 时生效 |
| `SANDBOXSET_REPLICAS` | `5` | SandboxSet 预热副本数。仅当 `USE_SANDBOX_CLAIM=true` 时生效 |

> Docker 模式不需要上述任何 K8s 参数。脚本会自动使用 `DockerEnvironment` 并传入空的 `environment_kwargs`。

#### 完整示例

**Docker 本地环境（最简配置）：**

```bash
MODEL_PATH=/var/model/Qwen2.5-7B-Instruct \
bash recipe/agentic/agentic-qwen2.5-3b.sh
```

**ACK 环境（使用 BuildKit 构建镜像）：**

```bash
MODEL_PATH=/var/model/Qwen2.5-7B-Instruct \
ENVIRONMENT=ack \
REGISTRY=my-registry.example.com/swebench \
KUBECONFIG=/root/.kube/config \
USE_BUILDKIT=true \
BUILDKIT_ADDRESS=tcp://buildkit-service:1234 \
IMAGE_PULL_SECRET=acr-registry \
bash recipe/agentic/agentic-qwen2.5-3b.sh
```

**ACK 环境（使用 OpenKruise SandboxClaim 预热容器）：**

```bash
MODEL_PATH=/var/model/Qwen2.5-7B-Instruct \
ENVIRONMENT=ack \
REGISTRY=my-registry.example.com/swebench \
KUBECONFIG=/root/.kube/config \
USE_SANDBOX_CLAIM=true \
SANDBOXSET_REPLICAS=10 \
IMAGE_PULL_SECRET=acr-registry \
bash recipe/agentic/agentic-qwen2.5-3b.sh
```

#### 要点

- `actor_rollout_ref.rollout.mode=async`：rollout 必须使用异步模式；
- `actor_rollout_ref.rollout.agent.default_agent_loop=remote_agent`：使用本 recipe 提供的 `RemoteAgentLoop`；
- `agent_loop_config_path`：指向 [`swe-agent.yaml`](swe-agent.yaml)（也可以替换为自定义 yaml）。

### 5.6 自定义 proxy 监听端口

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
                 │  RayPPOTrainer      │  init_workers() → LLMServerManager + LB
                 └─────────┬───────────┘
                           │ start_proxy_server(load_balancer, ...)
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
2. 在 `trainer.init_workers()` 之后、`trainer.fit()` 之前调用 `start_proxy_server`，把 `trainer.llm_server_manager.global_load_balancer` 这个 LB actor handle 注入 proxy；
3. 每个 rollout 步骤里 `RemoteAgentLoop` 向 proxy 注册 session、把 `base_url` 交给远程 Agent；
4. 远程 Agent 通过 OpenAI 协议调用 proxy，proxy 再把请求转发给 verl 的 vLLM 服务；
5. 会话结束后 `RemoteAgentLoop` 从 proxy 拉取完整 token 序列、tool 调用记录，构造 `AgentLoopOutput` 返回训练器。

## 8. 将 Agentic 能力集成到你自己的算法

Agentic recipe 在标准 PPO 训练之上增加了两个组件：

1. **Proxy Server** — 桥接远程 Agent 的 OpenAI 兼容 HTTP 请求到 verl 的 vLLM rollout 服务，同时记录每次 LLM 调用的 token IDs 和 logprobs。
2. **Remote Agent Loop**（`RemoteAgentLoop`）— 每条样本的 rollout 编排：将任务分发给远程 Agent 框架（如 Harbor），并从 proxy 的录制数据重建 verl 兼容的 trajectory。

```
┌─────────────────────────────────────────────────────┐
│  Your TaskRunner                                    │
│                                                     │
│  1. init_workers()   ← 标准 verl 初始化              │
│  2. start_proxy()    ← agentic: 桥接 LLM 流量        │
│  3. fit()            ← 标准 verl 训练                 │
└─────────────────────────────────────────────────────┘

Per-sample rollout:
┌──────────────┐     HTTP/OpenAI     ┌──────────────┐
│ Remote Agent │ ◄──────────────────► │ LLM Proxy    │
│ (Harbor,     │                      │ (records      │
│  SWE-agent,  │                      │  token_ids +  │
│  custom)     │                      │  logprobs)    │
└──────────────┘                      └──────┬───────┘
                                             │ routes to
                                      ┌──────▼───────┐
                                      │ verl vLLM    │
                                      │ rollout      │
                                      │ servers      │
                                      └──────────────┘
```

根据你的需求，有三种集成方式：

### 8.1 方式一：直接使用 Agentic Recipe

如果你的算法基于 PPO，只需自定义远程 Agent 或环境，可以直接用现有 recipe + 配置覆盖。

**步骤 1：创建继承 agentic trainer 的 YAML 配置。**

```yaml
# my_config.yaml
hydra:
  searchpath:
    - file://verl/trainer/config

defaults:
  - ppo_trainer
  - _self_

# Proxy server 设置
proxy_server:
  host: "0.0.0.0"
  port: 0
  tool_format: "hermes"
  llm_proxy_ip: <你的外部 IP>

# 远程 Agent 设置
remote_agent:
  agent_name: my_custom_agent
  agent_import_path: "my_package.agents:MyAgent"
  use_local_trial: true
  environment_import_path: "harbor.environments.docker.docker:DockerEnvironment"
```

**步骤 2：运行。**

```bash
python -m recipe.agentic.agentic_main --config-name my_config
```

这会使用内置的 `RemoteAgentLoop`（注册名 `"remote_agent"`）和 agentic `TaskRunner`，自动管理 proxy server 生命周期。

### 8.2 方式二：在你自己的 TaskRunner 中添加 Proxy Server

如果你有自定义的 `TaskRunner`（例如非 PPO 算法），需要自己在 `init_workers()` 和 `fit()` 之间启动 proxy server。

**步骤 1：继承 base TaskRunner 并插入 proxy 启动。**

```python
# my_recipe/main.py
import os
from verl.trainer.main_ppo import TaskRunner as BaseTaskRunner

class MyAgenticTaskRunner(BaseTaskRunner):
    """在自定义 TaskRunner 中添加 LLM proxy server。"""

    def run(self, config):
        from pprint import pprint
        from omegaconf import OmegaConf
        from verl.utils.fs import copy_to_local
        from verl.utils import hf_tokenizer, hf_processor
        from verl.utils.dataset.rl_dataset import collate_fn
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.trainer.ppo.ray_trainer import RayPPOTrainer
        from verl.trainer.ppo.utils import need_critic, need_reference_policy
        from verl.utils.config import validate_config

        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)

        # --- 标准 verl 初始化（与 base TaskRunner 相同）---
        actor_rollout_cls, ray_worker_group_cls = self.add_actor_rollout_worker(config)
        self.add_critic_worker(config)
        self.add_reward_model_resource_pool(config)
        self.add_teacher_model_resource_pool(config)
        self.add_ref_policy_worker(config, actor_rollout_cls)

        validate_config(
            config=config,
            use_reference_policy=need_reference_policy(config),
            use_critic=need_critic(config),
        )

        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
        resource_pool_manager = self.init_resource_pool_mgr(config)

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor,
            is_train=True, max_samples=config.data.get("train_max_samples", -1),
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor,
            is_train=False, max_samples=config.data.get("val_max_samples", -1),
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        trainer = RayPPOTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=self.role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )
        trainer.init_workers()

        # --- Agentic: 启动 LLM proxy server ---
        self._start_proxy(trainer, config)

        # --- 你的自定义逻辑（如有）---

        trainer.fit()

    def _start_proxy(self, trainer, config):
        """在 workers 初始化之后启动 LLM proxy server。"""
        load_balancer = trainer.llm_server_manager.global_load_balancer
        proxy_cfg = config.get("proxy_server", {})
        standalone_proxy_url = os.environ.get("PROXY_SERVER_URL")

        if standalone_proxy_url:
            # 外部 proxy 模式：仅注册 load balancer
            from recipe.agentic.proxyserver.ray_actor import start_lb_registry
            start_lb_registry(load_balancer=load_balancer)
            print(f"[agentic] standalone proxy: {standalone_proxy_url}")
        else:
            # 本地 proxy 模式：启动完整 HTTP proxy（Ray actor）
            from recipe.agentic.proxyserver.ray_actor import start_proxy_server
            proxy_url = start_proxy_server(
                load_balancer=load_balancer,
                model_path=config.actor_rollout_ref.model.path,
                host=proxy_cfg.get("host", "0.0.0.0"),
                port=proxy_cfg.get("port", 0),
                tool_format=proxy_cfg.get("tool_format", "hermes"),
            )
            print(f"[agentic] proxy server started at {proxy_url}")
```

**步骤 2：在你的 YAML 配置中添加 proxy_server 和 remote_agent 配置。**

```yaml
# 追加到你现有的配置中
proxy_server:
  host: "0.0.0.0"
  port: 0
  tool_format: "hermes"
  llm_proxy_ip: <你的外部 IP>

remote_agent:
  agent_import_path: "my_package.agents:MyAgent"
  use_local_trial: true
```

**步骤 3：设置 agent loop 使用 `remote_agent`。**

`RemoteAgentLoop` 在 verl 的 agent loop registry 中注册名为 `"remote_agent"`，在 rollout 配置中指定：

```yaml
actor_rollout_ref:
  rollout:
    agent:
      default_agent_loop: remote_agent
      agent_loop_config_path: null
```

## 9. 相关参考

- 入口实现：[`agentic_main.py`](agentic_main.py)
- 远程 agent loop：[`agent_loop/remote_agent_loop.py`](agent_loop/remote_agent_loop.py)
- Proxy actor 启动：[`proxyserver/ray_actor.py`](proxyserver/ray_actor.py)
- Harbor 端 Agent 实现：[`../../harbor`](../../harbor)
- 基础 PPO 训练入口：[`verl/trainer/main_ppo.py`](../../verl/trainer/main_ppo.py)
