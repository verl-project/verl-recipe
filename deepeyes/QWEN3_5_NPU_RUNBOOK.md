# Qwen3.5-4B / 9B DeepEyes GRPO 全流程 Runbook（Ascend NPU）

本文档描述从裸机到一次完整 DeepEyes 多轮视觉工具 GRPO 训练的全部步骤，每一步都给出
可直接执行的命令、预期结果和失败处理方式。文末的验证清单用于逐步确认环境与代码。

- 适用代码：本仓库 `main`（`3c6d9b43`）+ `recipe/deepeyes` 的 Qwen3.5 适配。
- 适用硬件：Atlas 800T A2 / 900 A2（910B）单机 8 卡；多机需先组建跨机 Ray 集群，再设置 `NNODES`（见 §8.1）。
- 关键开关：`MODEL_ID`、`MODEL_PATH`、`TRAIN_FILE`、`VAL_FILE`、`LLM_AS_A_JUDGE_BASE`、
  `LLM_AS_A_JUDGE_API_KEY`、`LLM_AS_A_JUDGE_MODEL`。

---

## 0. 数据流总览

```
DeepEyes parquet (prompt/images/reward_model/extra_info)
        │  scripts/prepare_data.py：修小图、去 Chart、切 train/val
        ▼
CustomRLHFDatasetQwen35.__getitem__
   ├─ raw_prompt        = [system, user(<image> + 问题 + 格式要求)]
   └─ tools_kwargs      = {image_zoom_in_tool: {create_kwargs: {image: PIL}}}
        │
        ▼
AgentLoop (ToolAgentLoop)
   ├─ CT builder：processor.apply_chat_template(messages, tools=...) → token ids
   ├─ vLLM-Ascend 异步生成（prompt_ids + multi_modal_data）
   ├─ Qwen3XMLToolParser(qwen3_coder) 解析 <tool_call><function=...>
   ├─ ImageZoomInTool 裁剪 bbox → ToolResponse(image=[crop])
   ├─ CT builder 增量渲染 <tool_response> 与裁剪图 → 进入下一轮
   └─ 最多 max_assistant_turns / max_user_turns 轮
        ▼
Reward (recipe/deepeyes/deepeyes.py::compute_score)
   ├─ 0.8 * acc   ← LLM-as-a-judge（LLM_AS_A_JUDGE_BASE）
   ├─ 0.2 * format(<think>/<answer> 标签平衡)
   └─ 1.2 * tool  （调用过工具且答对）
        ▼
GRPO → FSDP2 训练（Qwen3.5 混合注意力：GDN + full attention）
```

---

## 1. 版本矩阵

### 1.1 本机实测环境

| 项 | 实测值 |
| --- | --- |
| OS / 架构 | Ubuntu 22.04.5 LTS / aarch64 |
| NPU | 8 × 910B3（HBM 64 GB/卡） |
| CANN | 9.1.0（`/usr/local/Ascend/cann-9.1.0`） |
| Driver | 26.0.rc1（`ascendhal 7.35.23`） |
| Python | 3.11 |
| torch / torch_npu | 2.9.0+cpu / 2.9.0 |
| 内存 / 磁盘 | ~2 TB / 约 100 GB 可用（装完整个栈+4B 模型后约剩 50 GB） |

### 1.2 软件栈（与仓库 Qwen3.5 Ascend 镜像保持一致）

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| vllm | 0.18.0 | NPU 上以 `VLLM_TARGET_DEVICE=empty` 安装（不需要 CUDA 编译） |
| vllm-ascend | 0.18.0 | aarch64 wheel，含 NPU 自定义算子 |
| torch / torch_npu | 2.9.0 | vllm-ascend 0.18.0 的硬性要求 |
| transformers | `5.3.0.dev0 @ cc7ab9be`（镜像同款） | 必须包含 `transformers.models.qwen3_5`；仓库镜像见 `docker/ascend/Dockerfile.ascend_8.5.2_a2_qwen3-5` |
| numpy | < 2.0 | vllm-ascend 要求；verl NPU 依赖本身也 pin 了 `<2.0.0` |
| triton-ascend | 3.2.2（`requirements-npu.txt`） | vllm-ascend 0.18.0 的 wheel 元数据写 3.2.0；若启动报错回退 3.2.0 |
| verl | 本仓库 `main` | `pip install -e . --no-deps`（见 2.3 的原因） |

> 版本冲突说明：vllm 0.18.0 的 wheel 元数据写 `transformers<5`，而仓库镜像实际安装的是
> `5.3.0.dev0`（运行时可用）。verl `main` 的 `requirements.txt` 写 `transformers>=5.5.3`，
> 这是 GPU 路径的 pin；Qwen3.5 NPU 示例脚本头部明确标注 `transformers@cc7ab9be`，以该组合为准。
> 因此安装 verl 时使用 `--no-deps`，避免 resolver 把 torch/numpy/transformers 改坏。

---

## 2. 阶段 1：安装

### 2.1 创建独立环境

不要复用已有环境（避免改动他人依赖）：

```bash
conda create -y -n qwen35 python=3.11
conda activate qwen35

# 确认 CANN 环境变量（登录 shell 通常已 source）
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

### 2.2 安装推理栈（vllm + vllm-ascend + transformers）

```bash
# 固定 torch/torchvision/torchaudio（版本必须与 vllm-ascend 0.18.0 对齐）
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 torch_npu==2.9.0

# 方案 A（推荐，与官方镜像一致）：源码安装 pure-python 版 vllm
export DEPS_DIR=${DEPS_DIR:-$HOME/src}   # 持久化目录，勿用 /tmp
mkdir -p "$DEPS_DIR"
git clone --depth 1 --branch v0.18.0 https://github.com/vllm-project/vllm.git "$DEPS_DIR/vllm"
cd "$DEPS_DIR/vllm" && pip install -r requirements/build.txt && VLLM_TARGET_DEVICE=empty pip install -v -e . && cd -

# 方案 B（快速验证可试）：PyPI aarch64 wheel；注意其元数据 pin torch==2.10.0，
# 需 --no-deps 后手动补齐依赖，ABI 风险高于方案 A
# pip install --no-deps vllm==0.18.0

# vllm-ascend（会按需安装 numpy<2、triton-ascend、xgrammar、numba 等）
#
# 注意：直接 pip install vllm-ascend==0.18.0 会连带安装 arctic-inference==0.1.1，
# 后者是 sdist 且 build 隔离要求 torch==2.7.0 + CMake 编译，与本栈冲突且 Qwen3.5
# 用不到。推荐用 --no-deps 安装后手动补齐运行依赖（实测步骤）：
pip install --no-deps vllm-ascend==0.18.0
pip install "numpy==1.26.4" pandas scipy psutil pyyaml regex \
    "xgrammar>=0.1.32,<1.0.0" "compressed_tensors>=0.11.0" msgpack quart \
    "numba==0.61.2" "opencv-python-headless<=4.11.0.86" \
    memfabric_hybrid==1.0.8 memcache_hybrid==1.0.4 "fastapi<0.124.0" \
    triton-ascend==3.2.2 cmake decorator einops pybind11 setuptools-scm \
    "setuptools>=64" wheel pandas-stubs
# vllm 自身的运行依赖（见 $DEPS_DIR/vllm/requirements/common.txt），去掉
# torch/transformers/opencv/numpy 等已单独锁定的行后安装：
pip install -r <(grep -v -E "^(opencv-python-headless|numpy|torch|transformers|tokenizers)" \
    "$DEPS_DIR/vllm/requirements/common.txt")

# transformers：与仓库镜像一致的 commit（含 qwen3_5）
git init "$DEPS_DIR/transformers" && cd "$DEPS_DIR/transformers" && \
    git remote add origin https://github.com/huggingface/transformers.git && \
    git fetch --depth 1 origin cc7ab9be508ce6ed3637bba9e50367b29b742dc6 && \
    git checkout FETCH_HEAD && cd -
# 5.3.0.dev0 要求 huggingface-hub<2 且 tokenizers<=0.23.0；先钉依赖再用 --no-deps 安装
pip install "huggingface-hub>=1.3.0,<2.0" "tokenizers>=0.22.0,<=0.23.0"
pip install -e "$DEPS_DIR/transformers" --no-deps
python -c "import transformers; print(transformers.__version__); import transformers.models.qwen3_5"
```

**必须复核 torch 版本。** pip 在解析 `xgrammar` 等依赖时可能把 torch 升级成更新的
CPU 版本（实测会装 `torch-2.14.1+cpu`），从而让 `torch_npu 2.9.0` 失效。每次安装完
依赖后都执行：

```bash
pip install --no-deps torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 torch_npu==2.9.0
pip list | grep -E "^(torch|torch-npu|vllm|vllm-ascend|transformers|numpy|triton-ascend|pandas)"
python -c "import torch, torch_npu; print(torch.__version__, torch_npu.__version__, torch.npu.is_available())"
```

### 2.3 安装 verl（本仓库）

```bash
cd /path/to/verl
pip install -r requirements-npu.txt
pip install tensorboard nvtx        # 不在 requirements-npu.txt 中，但训练/日志需要
pip install -e . --no-deps          # --no-deps：保持 numpy<2 / transformers 5.3.0.dev0
python -c "import verl; print(verl.__version__)"
```

> `requirements-npu.txt` 里的 `triton-ascend==3.2.2` 会覆盖 vllm-ascend 声明的 3.2.0。
> 若 `import vllm_ascend` 或起服务时报 triton 版本不匹配，改为 `pip install triton-ascend==3.2.0`。

### 2.4 快速自检

```bash
python - <<'PY'
import torch, torch_npu, transformers, vllm, vllm_ascend
print("torch", torch.__version__, "| npu", torch.npu.is_available(), "| devices", torch.npu.device_count())
print("transformers", transformers.__version__)
print("vllm", vllm.__version__, "| vllm_ascend", vllm_ascend.__version__)
from transformers.models.qwen3_5 import Qwen3_5ForConditionalGeneration  # noqa: F401
print("qwen3_5 OK")
PY
```

---

## 3. 阶段 2：模型与数据

### 3.1 下载模型

```bash
export HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}   # 国内可换镜像
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ~/models/Qwen3.5-4B
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ~/models/Qwen3.5-9B
```

磁盘：4B ≈ 8 GB，9B ≈ 18 GB。先只下 4B 完成全流程，再下 9B。

### 3.2 准备数据

> **缓存目录**：`datasets` 在缓存目录所在分区剩余空间不足时会直接拒绝
> 加载（`OSError: Not enough disk space`），而且它默认把缓存放在 `~/.cache`。本机
> `/root/.cache` 只有 69 GB 且已满。训练前把缓存指到大盘：
>
> ```bash
> export HF_HOME=/path/to/big-disk/.cache/huggingface
> export HF_DATASETS_CACHE=$HF_HOME/datasets
> export RAY_TMPDIR=/path/to/big-disk/ray_tmp
> mkdir -p "$HF_DATASETS_CACHE" "$RAY_TMPDIR"
> ```
>
> 本地 parquet/模型路径不会被复制到缓存（`is_non_local()` 为 False 时直接使用原路径）。

```bash
cd /path/to/verl
python3 recipe/deepeyes/scripts/prepare_data.py --output-dir ~/data/deepeyes --stats-only   # 先体检
python3 recipe/deepeyes/scripts/prepare_data.py --output-dir ~/data/deepeyes               # 再落盘
```

预期：`~/data/deepeyes/train.parquet`、`val.parquet`，并打印样本数、修复图片数、
最小边分位数、估计 vision token 分位数、`data_source` 分布。
默认排除 Chart（`data_v0.8_visual_toolbox_v2.parquet`），`--include-chart` 可保留。

小样本快速验证（不写全量）：

```bash
python3 recipe/deepeyes/scripts/prepare_data.py --local-parquet /path/shard.parquet \
    --max-samples 512 --val-size 32 --output-dir /tmp/deepeyes-smoke
```

### 3.3 数据抽查（可选）

```bash
python - <<'PY'
import datasets
ds = datasets.load_dataset("parquet", data_files="~/data/deepeyes/train.parquet", split="train")
row = ds[0]
print(row.keys())
print("image bytes:", len(row["images"][0]["bytes"]), "| gt:", row["reward_model"]["ground_truth"])
print("user:", row["prompt"][1]["content"][:200])
PY
```

---

## 4. 阶段 3：离线语义验证（不需要起服务）

这一步把最容易出问题的三件事在纯 CPU/NPU-import 层面确认掉：**聊天模板往返**、
**tool call 解析**、**多模态位置编码**。

### 4.1 模板 + 图像 token 渲染

```bash
python - <<'PY'
from transformers import AutoProcessor
from PIL import Image

proc = AutoProcessor.from_pretrained("~/models/Qwen3.5-4B")
tools = [{
    "type": "function",
    "function": {
        "name": "image_zoom_in_tool",
        "description": "Zoom in on a region.",
        "parameters": {"type": "object",
                       "properties": {"bbox_2d": {"type": "array", "items": {"type": "number"}},
                                      "label": {"type": "string"}},
                       "required": ["bbox_2d"]},
    },
}]
messages = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [{"type": "image", "image": Image.new("RGB", (640, 480))},
                                 {"type": "text", "text": "Is the car on the left? <answer>...</answer>"}]},
]
text = proc.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False,
                                enable_thinking=True)
print(text[-400:])   # 末尾应为 <|im_start|>assistant\n<think>\n
inputs = proc(text=[text], images=[messages[1]["content"][0]["image"]], return_tensors="pt")
print("tokens", inputs["input_ids"].shape, "| image tokens",
      int((inputs["input_ids"] == proc.image_token_id).sum()), "| grid", inputs["image_grid_thw"])
PY
```

预期：生成段以 `<|im_start|>assistant\n<think>\n` 结尾；`image_grid_thw` 形如 `[[1, h/32, w/32]]`，
图像 token 数 = `h*w/1024`。

### 4.2 tool call 解析（Qwen3XMLToolParser）

```bash
python - <<'PY'
from transformers import AutoTokenizer
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema

tok = AutoTokenizer.from_pretrained("~/models/Qwen3.5-4B")
parser = ToolParser.get_tool_parser("qwen3_coder", tok)
text = ("I need a closer look.</think>\n"
        "<tool_call>\n<function=image_zoom_in_tool>\n"
        "<parameter=bbox_2d>\n[226, 399, 265, 464]\n</parameter>\n"
        "<parameter=label>\nwhite van\n</parameter>\n"
        "</function>\n</tool_call>")
ids = tok.encode(text, add_special_tokens=False)
schema = OpenAIFunctionToolSchema.model_validate({
    "type": "function",
    "function": {"name": "image_zoom_in_tool",
                 "parameters": {"type": "object",
                                "properties": {"bbox_2d": {"type": "array", "items": {"type": "number"}},
                                               "label": {"type": "string"}},
                                "required": ["bbox_2d"]}}})
import asyncio
content, calls = asyncio.run(parser.extract_tool_calls(ids, [schema]))
print(repr(content)); print(calls)
PY
```

预期：`content` 为 `I need a closer look.</think>`，`calls[0].arguments` 是
`{"bbox_2d": [226, 399, 265, 464], "label": "white van"}` 的 JSON 串。

### 4.3 位置编码绑定

```bash
python - <<'PY'
import torch
from transformers import AutoProcessor
from verl.utils import hf_processor   # verl 的封装：会绑定 get_rope_index

proc = hf_processor("~/models/Qwen3.5-4B")
print("processor:", type(proc).__name__, "| has get_rope_index:", hasattr(proc, "get_rope_index"))
PY
```

预期：`Qwen3VLProcessor`，`has get_rope_index: True`（verl 会把
`Qwen3VLModel.get_rope_index` 绑到 processor 上；Qwen3.5 与 Qwen3-VL 的 mrope 规则一致）。

### 4.4 数据集类

```bash
python - <<'PY'
from omegaconf import OmegaConf
from verl.utils.dataset.rl_dataset import RLHFDataset
from verl.utils import hf_tokenizer, hf_processor
from recipe.deepeyes.deepeyes import CustomRLHFDatasetQwen35

cfg = OmegaConf.create({"max_prompt_length": 8192, "truncation": "error",
                        "image_key": "images", "filter_overlong_prompts": False})
ds = CustomRLHFDatasetQwen35(data_files="~/data/deepeyes/train.parquet",
                             tokenizer=hf_tokenizer("~/models/Qwen3.5-4B"),
                             processor=hf_processor("~/models/Qwen3.5-4B"), config=cfg)
item = ds[0]
print([m["role"] for m in item["raw_prompt"]])
print(item["tools_kwargs"]["image_zoom_in_tool"]["create_kwargs"]["image"].size)
assert "<think>...</think>" not in str(item["raw_prompt"][1]["content"])
PY
```

预期：`['system', 'user']`，image size 与原始图一致，user 指令中不再含字面
`<think>...</think>`。

---

## 5. 阶段 4：单实例推理（vLLM-Ascend）

先只验证推理栈和模型加载，不接训练。

> **共享机器先看空闲卡**：`npu-smi info` 可以看到每张卡的 HBM 占用和进程。
> 本机 0/1 卡被其他进程占用（18 GB / 58 GB），因此下面示例故意使用 2/3 卡；
> 训练时同理，用 `ASCEND_RT_VISIBLE_DEVICES` 挑空闲卡并同步调整 `NDEVICES_PER_NODE`。

```bash
npu-smi info            # 先确认哪些卡空闲
ss -ltn | grep -E ':(8000|18000|18901)' || true   # 再确认端口空闲
export ASCEND_RT_VISIBLE_DEVICES=2,3
vllm serve ~/models/Qwen3.5-4B \
    --served-model-name qwen35-4b \
    --tensor-parallel-size 2 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.6 \
    --port 18000
```

另开终端：

```bash
curl -s localhost:18000/v1/models | python -m json.tool
curl -s localhost:18000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen35-4b",
  "messages": [{"role": "user", "content": "用一句话解释什么是强化学习。"}],
  "max_tokens": 64
}' | python -m json.tool | head -30
```

多模态 + tool 生成验证（发给同一服务）：

```bash
python - <<'PY'
import base64, requests
from PIL import Image
import io
buf = io.BytesIO(); Image.new("RGB", (640, 480), (200, 40, 40)).save(buf, format="JPEG")
b64 = base64.b64encode(buf.getvalue()).decode()
tools = [{"type": "function", "function": {
    "name": "image_zoom_in_tool", "description": "Zoom in on a region.",
    "parameters": {"type": "object",
                   "properties": {"bbox_2d": {"type": "array", "items": {"type": "number"}},
                                  "label": {"type": "string"}},
                   "required": ["bbox_2d"]}}}]
resp = requests.post("http://localhost:18000/v1/chat/completions", json={
    "model": "qwen35-4b",
    "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
        {"type": "text", "text": "图里有什么？需要时调用 image_zoom_in_tool。"}]}],
    "tools": tools, "max_tokens": 256, "temperature": 0,
}, timeout=300)
msg = resp.json()["choices"][0]["message"]
print(msg.get("content")); print("tool_calls:", msg.get("tool_calls"))
PY
```

预期：服务能加载 Qwen3.5（混合 GDN 架构），图像输入不报错；模型可能返回
`<tool_call><function=image_zoom_in_tool>...` 文本（取决于 vLLM 版本的 tool 解析开关，
训练走的是文本解析，不依赖服务端 tool_calls 字段）。

---

## 6. 阶段 5：judge 服务与 reward

### 6.1 部署 judge

judge 只需 OpenAI 兼容接口（`/v1/models` 与 `/v1/chat/completions`），推荐部署在独立推理机；
与训练同机分卡仅用于 smoke。

**独立推理机（全量训练）**

在 judge 机器上监听所有网卡（模型与 TP 按显存选择）：

```bash
export ASCEND_RT_VISIBLE_DEVICES=0,1
vllm serve ~/models/Qwen2.5-32B-Instruct \
    --served-model-name judge --host 0.0.0.0 --tensor-parallel-size 2 \
    --max-model-len 8192 --gpu-memory-utilization 0.9 --port 18901 \
    --api-key <key>          # 不需要鉴权时省略
```

训练机器只配置地址，不启动本地服务，也不为 judge 保留 NPU：

```bash
export LLM_AS_A_JUDGE_BASE=http://<judge-ip>:18901/v1
export LLM_AS_A_JUDGE_API_KEY=<key>    # 与服务端 --api-key 一致；无鉴权时省略
export LLM_AS_A_JUDGE_MODEL=judge      # 可选；服务端不提供 /models 时必填

curl -s -H "Authorization: Bearer ${LLM_AS_A_JUDGE_API_KEY}" \
    "${LLM_AS_A_JUDGE_BASE}/models"
```

要求训练机器能访问 judge 的 18901 端口，且 `LLM_AS_A_JUDGE_BASE` 包含 `/v1`。

> reward 打分在 Ray actor 中执行。单机由 `ray.init()` 自动拉起 Ray 时 shell 变量会被继承；
> 预先启动的多机 Ray 集群不会继承，脚本会把 `LLM_AS_A_JUDGE_BASE`、`LLM_AS_A_JUDGE_MODEL`
> 注入 `ray_kwargs.ray_init.runtime_env.env_vars`。`LLM_AS_A_JUDGE_API_KEY` 不写入命令行和日志，
> 需在每台节点 `ray start` 前导出，或通过私有配置传入 runtime_env。

**同机分卡（仅 smoke）**

```bash
export ASCEND_RT_VISIBLE_DEVICES=6,7
vllm serve ~/models/Qwen3.5-9B \
    --served-model-name judge --tensor-parallel-size 2 \
    --max-model-len 8192 --gpu-memory-utilization 0.6 --port 18901
export LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18901/v1
```

此时训练脚本需把 `NDEVICES_PER_NODE` 减为剩余卡数，并让 Ray 只看到训练卡。

**吞吐**：reward 默认由 `reward.num_workers=8` 个 Ray worker 分片并行打分，每个 worker 内部
再并发调用 judge。judge 侧按 `train_batch_size × rollout.n / 步时长` 估算 QPS 并配置
`--max-num-seqs` 或多副本；judge 较慢时可用 `reward.num_workers=<n>` 下调并发。

> 资源紧张时可直接把 `LLM_AS_A_JUDGE_BASE` 指向 Qwen3.5-4B 服务；此时 accuracy 项质量下降，
> 但流程可完整跑通。未设置该变量时 accuracy 恒为 0，只剩 format/tool 两项。

### 6.2 reward 单测

```bash
python - <<'PY'
from recipe.deepeyes.deepeyes import compute_score
sol = ("思考过程</think>\n<tool_call>\n<function=image_zoom_in_tool>\n"
       "<parameter=bbox_2d>\n[226, 399, 265, 464]\n</parameter>\n</function>\n</tool_call>\n"
       "<answer>yes</answer>")
print(compute_score("vstar", sol, "yes", {"question": "is it?"}))
PY
```

预期返回 `{'score': ~2.0, 'acc': 1.0, 'format': 0.0, 'tool_used': 1.0, ...}`
（judge 可用时 acc=1）。

---

## 7. 阶段 6：训练 smoke（1 step）

先跑最小闭环：小 batch、小 n、短序列、1 step。

```bash
cd /path/to/verl
export LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18901/v1
export HF_HOME=/path/to/big-disk/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export RAY_TMPDIR=/path/to/big-disk/ray_tmp

# 共享机器：训练用 4,5,6,7 四张空闲卡（judge 占 2,3）
ASCEND_RT_VISIBLE_DEVICES=4,5,6,7 \
MODEL_PATH=~/models/Qwen3.5-4B \
TRAIN_FILE=/tmp/deepeyes-smoke/train.parquet \
VAL_FILE=/tmp/deepeyes-smoke/val.parquet \
TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 ROLLOUT_N=2 \
MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=2048 \
MAX_NUM_BATCHED_TOKENS=8192 ROLLOUT_GPU_MEM_UTIL=0.4 \
SAVE_FREQ=-1 TEST_FREQ=-1 TOTAL_EPOCHS=1 \
NDEVICES_PER_NODE=4 GEN_TP=1 RAY_NOSET_VISIBLE_DEVICES=0 \
bash recipe/deepeyes/run_deepeyes_grpo_qwen3_5_4b.sh \
    trainer.total_training_steps=1 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=3
```

> 使用独立 judge 时去掉 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7` 和 `RAY_NOSET_VISIBLE_DEVICES=0`：
> 训练机使用全部 8 卡（`NDEVICES_PER_NODE=8`、`GEN_TP=2`），judge 不占用本地显存。

smoke 通过的标准（日志关键字）：

1. 4 个 vLLM-Ascend engine 起来，权重同步完成（`update_weights` / `wake_up` 无报错）。
2. `AgentLoop` 产生轨迹，`num_turns > 1`（说明真的调用了 zoom 工具）。
3. reward 有限值且 `tool_used` 有非零样本。
4. 训练 step 完成，loss/kl 有限，模型保存路径可写。
5. 无 `tokenization_sanity_check` 报错；若报错，见 §9。

> **启动耗时**：`data.filter_overlong_prompts_workers` 上游默认是 1，全量 47k 数据逐条渲染
> 需要数小时。本配方脚本默认设为 32（可用 `FILTER_WORKERS` 覆盖），`--stats-only` 的数据
> 体检也建议先跑，避免把超大图带进训练。

常见显存调整顺序：`ROLLOUT_GPU_MEM_UTIL` ↓ → `MAX_PROMPT_LENGTH` ↓ →
`TRAIN_BATCH_SIZE` ↓ → `GEN_TP` ↑ → `FSDP_SIZE` 调小（多机分片）。

---

## 8. 阶段 7：正式训练与监控

```bash
export LLM_AS_A_JUDGE_BASE=http://<judge>:18901/v1
MODEL_PATH=~/models/Qwen3.5-9B \
bash recipe/deepeyes/run_deepeyes_grpo_qwen3_5_9b.sh
```

监控项：

| 指标 | 期望 |
| --- | --- |
| `critic/rewards/mean` | 从 0 附近上升；`acc` 分项逐步上升 |
| `reward_extra_info/tool_used` | 早期上升，稳定在较高水平 |
| `response_length/mean` | 稳定，不应持续顶到 `max_response_length` |
| `actor/entropy` | 缓慢下降，不应骤降为 0 |
| `rollout/num_turns`（若记录） | 与工具调用数相关：`tool_calls ≈ num_turns/2 - 1` |

checkpoint：`trainer.default_local_dir` 下按 `global_step_*` 保存；
`actor.checkpoint.save_contents` 控制内容（`model`,`hf_model`,`optimizer`,`extra`）。

### 8.1 推理与训练分机部署

本配方涉及两类推理：

- **judge（reward）**：只通过 `LLM_AS_A_JUDGE_BASE` 访问，独立推理机开箱可用（§6.1）。
  训练机不需要为 judge 留卡，按整机 8 卡启动训练即可。
- **策略 rollout**：vLLM-Ascend 与 FSDP2 在训练卡上共置（hybrid engine），本配方使用的
  `main_ppo` 同步训练器不支持将 rollout 放到独立机器。

若必须把策略推理与训练分机部署，verl 提供 V1 `separate_async` 模式（独立 rollout 资源池）：

```bash
trainer.v1.trainer_mode=separate_async \
trainer.v1.separate_async.parameter_sync_step=4 \
actor_rollout_ref.rollout.nnodes=<rollout 节点数> \
actor_rollout_ref.rollout.n_gpus_per_node=8 \
actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
data.train_batch_size=64 \
actor_rollout_ref.actor.ppo_mini_batch_size=16
```

`checkpoint_engine.backend=nccl` 在 Ascend 上对应 HCCL 实现。该模式需要跨机权重同步；
本配方的脚本与本文档尚未在 Ascend + Qwen3.5 多轮 agent loop 上验证，上线前需单独跑通 1 step。
常规多机扩展仍是每节点 colocate：head 节点 `ray start --head`，其余节点
`ray start --address=<head>`，再设置 `NNODES=<节点数>`、`NDEVICES_PER_NODE=8`，
rollout 与训练共享每节点的 8 卡。

---

## 9. 故障排查

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `Unsupported processor type: Qwen3VLProcessor` | verl 版本过旧 | 使用本仓库 `main`（`verl/utils/tokenizer/tokenizer.py` 已支持） |
| `KeyError: qwen3_5` / `transformers.models.qwen3_5` 不存在 | transformers 太旧 | 换 `cc7ab9be` 或 ≥5.5.3 |
| `ModuleNotFoundError: vllm_ascend` 或 triton 版本不匹配 | triton-ascend 3.2.2 vs 3.2.0 | `pip install triton-ascend==3.2.0` |
| vLLM 起服务报 `torch` / ABI 错误 | 用了 PyPI 的 vllm wheel（pin torch 2.10） | 按 §2.2 方案 A 源码安装（`VLLM_TARGET_DEVICE=empty`） |
| `ValueError: Multimodal inputs require a Continuous Token builder...` | processor 未加载 | 确认 `MODEL_PATH` 是完整 HF 目录（含 preprocessor_config.json） |
| `Multimodal prompt ... exceeding rollout.prompt_length` | 图像 token 太多 | 调小 `data.mm_processor_kwargs.max_pixels` 或调大 `MAX_PROMPT_LENGTH` |
| `filter dataset len: 0` 且日志刷 `AttributeError: 'CustomRLHFDatasetQwen35' object has no attribute '_build_messages'` | `custom_cls.path` 写成文件路径时，verl 用 `custom_module_<hash>` 动态加载且不注册 `sys.modules`；`datasets.filter` 用 dill 序列化回调会把类按值重建，继承来的方法丢失 | `custom_cls.path` 改用包路径 `pkg://recipe.deepeyes.deepeyes`（reward 路径同理），模块可正常 import、按引用 pickle |
| worker 创建时 `c10_npu::SetDevice(...) error code is 107001` | 共享机器上用了非 0 起的卡子集（如 `ASCEND_RT_VISIBLE_DEVICES=4,5,6,7`）且保持 `RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1`：Ray 返回物理卡号，verl 调 `set_device(4)`，而 torch_npu 只接受可见列表内的 0..3 | 设 `RAY_NOSET_VISIBLE_DEVICES=0` 且 `GEN_TP=1`（Ray 给每个 actor 单卡，无需手工 set_device）；整机 0 起全卡时保持默认 NOSET=1 以支持 TP>1 |
| `tokenization_sanity_check` 警告/报错 | 模板逐轮回渲与 rollout token 有空白差异 | 加 `actor_rollout_ref.rollout.multi_turn.tokenization_sanity_check_mode=ignore_strippable`（或 `disable`） |
| reward 恒为 0 | judge 未设置/不可达/鉴权失败 | `curl -H "Authorization: Bearer $LLM_AS_A_JUDGE_API_KEY" $LLM_AS_A_JUDGE_BASE/models`；查看 `LLM_AS_A_JUDGE_BASE` |
| judge 返回 401/403 | 服务端启用了 `--api-key` | 设置 `LLM_AS_A_JUDGE_API_KEY` |
| judge 返回 400/422 | 服务端不接受 `chat_template_kwargs` | 客户端会自动重试一次不带该字段；仍失败则检查服务端请求体 |
| 独立 judge 首次探测失败 | 网络未通或 judge 尚未启动 | 客户端每 60 秒重试；也可设置 `LLM_AS_A_JUDGE_MODEL` 跳过 `/models` 探测 |
| 工具调用全部失败 | bbox 越界/过小 | 正常会被裁剪放大；若密集失败，检查 `min_dimension=32` 与数据图尺寸 |
| NPU OOM（训练侧） | offload/长度/批大小 | `TRAIN_BATCH_SIZE` ↓、`MAX_PROMPT_LENGTH` ↓、确认 param/optimizer offload 开启 |
| NPU OOM（rollout 侧） | KV cache | `ROLLOUT_GPU_MEM_UTIL` ↓、`GEN_TP` ↑、`MAX_NUM_BATCHED_TOKENS` ↓ |

---

## 10. 验证清单（逐项执行后勾选）

| 编号 | 验证项 | 命令/判据 | 状态 |
| --- | --- | --- | --- |
| V0 | 硬件/CANN/网络/磁盘体检 | `npu-smi info`、`df -h`、§2.1 | ✅ 通过（8×910B3 / CANN 9.1.0 / driver 26.0.rc1 / aarch64 wheel 可达） |
| V1 | 安装推理栈并 `import` 自检 | §2.4 | ✅ 通过（torch 2.9.0 / torch_npu 2.9.0 / npu=True / transformers 5.3.0.dev0 / vllm 0.18.0 / vllm-ascend 0.18.0） |
| V2 | verl 可导入、recipe 可导入 | `import verl`、`from recipe.deepeyes.deepeyes import ...` | ✅ 通过（verl 0.10.0.dev0，recipe 三个模块均可导入） |
| V3 | 模板渲染 + 图像 token 网格 | §4.1 | ✅ 通过（结尾为 `assistant\n<think>\n`；640×480 → grid [[1,30,40]]、300 image tokens；assistant 段字符串级往返一致） |
| V4 | `qwen3_coder` 解析 XML tool call | §4.2 | ✅ 通过（bbox_2d=[226,399,265,464]、label="white van"） |
| V5 | `get_rope_index` 绑定 | §4.3 | ✅ 通过（Qwen3VLProcessor，get_rope_index=True） |
| V6 | `CustomRLHFDatasetQwen35` 真实数据 | §4.4 | ✅ 通过（真实 shard 取 512 条：480 train/32 val；图像、tools_kwargs、prompt 改写、模板渲染均正确） |
| V7 | vLLM-Ascend 单实例（文本+图像） | §5 | ✅ 通过（Qwen3.5-4B TP=2 加载约 40GB/卡；文本与图像请求 200；带 `tools` 的请求需服务端加 `--enable-auto-tool-choice --tool-call-parser qwen3_coder`，训练路径不依赖） |
| V8 | judge + reward 联调 | §6 | ✅ 通过（真实服务当 judge：正确工具答案 score=2.0；无工具/格式错误答案 score=0.6） |
| V9 | 1-step 训练闭环 | §7 | ✅ 通过（4 卡 4B：reward mean 0.325 / max 2.0，num_turns 4–6，KL 0.0036，rollout-train logprob corr 0.9988，grad_norm 1.81，单步 355 s） |
| V10 | 9B 模型加载/smoke | 复制 §7 改 `MODEL_PATH` | ✅ V10a（9B TP=2 服务：文本/图像/judge 正常，39.4 GB/卡）；V10b（1-step，2048 token/3 轮 → 全部截断、reward 退化）；V10c（8192 token/5 轮 → reward mean 0.35 / max 1.8，grad_norm 8.25，num_turns 6–10） |
| V11 | judge 独立推理机接入 | §6.1、§11 | ✅ 本地 mock 服务通过（API Key、422 回退、无 `/models`、不可达重试）；未接第二台物理机 |

> 验证过程中如修改了环境或代码，请同步更新本表的“状态”列，保持文档与实际一致。


---

## 11. 本次验证记录（2026-10-26，8×910B3 / CANN 9.1.0）

- 环境：独立 venv `$HOME/qwen35-env`（torch 2.9.0 + torch_npu 2.9.0 +
  vllm 0.18.0 `VLLM_TARGET_DEVICE=empty` + vllm-ascend 0.18.0 + transformers 5.3.0.dev0@cc7ab9be
  + verl `main`）。已有 conda 环境未改动。
- 数据：DeepEyes `data_0.1.2_visual_toolbox_v2.parquet`（0.99 GB）取 512 条 → 480 train / 32 val；
  修复图片 0 张（vstar 子集最小边 ≥306 px），估计 vision token：mean 204 / p99 450 / max 768。
- V3–V8 全部通过；V9 用 4 卡（4/5/6/7，judge 服务占 2/3）跑通 1 step：
  `TRAIN_BATCH_SIZE=8, ROLLOUT_N=2, max_prompt 4096, max_response 2048, GEN_TP=1,
  RAY_NOSET_VISIBLE_DEVICES=0`。
- V9 关键指标：reward mean 0.325（min -0.2 / max 2.0，max 说明“答对且用工具”的轨迹存在）；
  num_turns 4–6（≈1.8 次工具调用/轨迹）；response mean 754 / max 1871；KL 0.0036；
  rollout 与训练的 logprob Pearson 0.9988（模板往返一致性良好）；
  actor 单卡峰值 8.96 GB（reserved 12.67 GB）；单步 355 s（gen 197 s + update_actor 109 s）。
- 运行中修复的 3 个问题（详见 §9）：`python3` 解释器、`custom_cls` 文件路径导致 filter 丢方法、
  共享机器卡子集导致 `SetDevice 107001`。
- 收尾时 `DataLoader worker ... Killed` 出现在 `atexit` 的 `_npu_shutdown` 中，属解释器退出清理，
  不影响已完成的 step（进度条已 100%）。

### 9B（V10）

- V10a：`Qwen3.5-9B` 以 TP=2 在 2/3 卡起 vLLM-Ascend，2/3 卡各 39.4 GB；文本、图像问答均正常，
  同时作为 judge 时 `compute_score` 返回 2.0。
- V10b（smoke 限制过紧）：`max_response 2048 / 3 轮` 下 8 条轨迹全部在轮数或长度上限被截断
  （`num_turns` 全 6、`response_length/clip_ratio 0.375`），reward 全为 -0.2 → advantage=0、
  `grad_norm=0`。**这不是栈问题**，栈本身完整跑完 1 step。
- V10c（接近真实生成上限）：`max_response 8192 / 5 轮 / batch 4 / n=2`：
  reward mean 0.35（min -0.2 / max 1.8），`num_turns` 6–10，response mean 1052，
  `grad_norm 8.25`、loss 0.043、KL 0.00024、rollout-train logprob corr 0.9995，
  actor 单卡 10.74 GB（reserved 13.0 GB），单步 237 s，4 卡 9B 显存无压力。
- 结论：9B 配方在 4 卡 910B 上可直接训练；正式训练建议 8 卡 + `train_batch_size=64`
  + `max_response_length=16384` + 5 轮（脚本默认值）。

V9 实际使用的启动命令（可复现）：

```bash
cd /path/to/verl
export HF_HOME=/path/to/big-disk/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export RAY_TMPDIR=/path/to/big-disk/ray_tmp
export LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18000/v1
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7
export PYTHON=$HOME/qwen35-env/bin/python
export RAY_NOSET_VISIBLE_DEVICES=0

MODEL_PATH=$HOME/models/Qwen3.5-4B \
TRAIN_FILE=/tmp/deepeyes-smoke/train.parquet \
VAL_FILE=/tmp/deepeyes-smoke/val.parquet \
TRAIN_BATCH_SIZE=8 PPO_MINI_BATCH_SIZE=8 ROLLOUT_N=2 \
MAX_PROMPT_LENGTH=4096 MAX_RESPONSE_LENGTH=2048 \
MAX_NUM_BATCHED_TOKENS=8192 ROLLOUT_GPU_MEM_UTIL=0.4 \
SAVE_FREQ=-1 TEST_FREQ=-1 TOTAL_EPOCHS=1 \
NDEVICES_PER_NODE=4 GEN_TP=1 LOGGER="['console']" \
bash recipe/deepeyes/run_deepeyes_grpo_qwen3_5_4b.sh \
    trainer.total_training_steps=1 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=3 \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=3
```

### judge 远程接入（2026-10-29）

本地 mock 服务验证了三种独立推理机场景，未接入第二台物理机：

- `LLM_AS_A_JUDGE_API_KEY`：`/models` 返回 401，携带 Bearer 后通过。
- 服务端拒绝 `chat_template_kwargs`（HTTP 422）：客户端自动重试不带该字段，`acc=1.0`，
  后续请求不再携带。
- 不提供 `/models`：设置 `LLM_AS_A_JUDGE_MODEL` 后跳过探测，`acc=1.0`。
- judge 不可达：`acc=0` 且每 60 秒重试，不会永久禁用。
