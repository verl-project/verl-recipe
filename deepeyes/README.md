# DeepEyes: Incentivizing "Thinking with Images" via Reinforcement Learning
## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repository, install mode (rolling `main`, pinned release tag, or pinned git commit), and copy-pastable `pip` / `git` instructions where they exist.


This directory contains the implementation for reproducing the DeepEyes paper within the verl framework, supporting multi-turn visual tool calls. This implementation is based on the original [DeepEyes paper](https://arxiv.org/abs/2505.14362) and its [official implementation](https://github.com/Visual-Agent/DeepEyes), integrated with the multi-modal and multi-turn capabilities of the verl framework.

Two model families are supported:

| Model | Config | Entry script | Backend |
| --- | --- | --- | --- |
| Qwen2.5-VL-7B (original recipe) | `configs/deepeyes_multiturn_grpo.yaml` | `run_deepeyes_grpo.sh` | SGLang (GPU) |
| Qwen3.5-4B | `configs/deepeyes_multiturn_grpo_qwen3_5.yaml` | `run_deepeyes_grpo_qwen3_5_4b.sh` | vLLM-Ascend (NPU) |
| Qwen3.5-9B | `configs/deepeyes_multiturn_grpo_qwen3_5.yaml` | `run_deepeyes_grpo_qwen3_5_9b.sh` | vLLM-Ascend (NPU) |

## Qwen3.5-4B / Qwen3.5-9B on Ascend NPU

Requirements: CANN 8.5.2+, torch/torch_npu 2.9.0, vllm 0.18.0, vllm-ascend 0.18.0,
transformers `cc7ab9be` (contains `qwen3_5`). Installation, data preparation, judge
setup and troubleshooting are documented in [`QWEN3_5_NPU_RUNBOOK.md`](QWEN3_5_NPU_RUNBOOK.md).

```bash
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ~/models/Qwen3.5-4B
huggingface-cli download Qwen/Qwen3.5-9B --local-dir ~/models/Qwen3.5-9B
python3 recipe/deepeyes/scripts/prepare_data.py --output-dir ~/data/deepeyes

# A judge server must be running; see the runbook.
LLM_AS_A_JUDGE_BASE=http://127.0.0.1:18901/v1 \
MODEL_PATH=~/models/Qwen3.5-4B \
bash recipe/deepeyes/run_deepeyes_grpo_qwen3_5_4b.sh
```

Differences from the Qwen2.5-VL recipe:

| Topic | Qwen2.5-VL | Qwen3.5 |
| --- | --- | --- |
| Chat template | custom ChatML in the YAML | model's native template |
| Tool call format | JSON | XML `<function=...><parameter=...>` |
| Tool parser | `hermes` | `multi_turn.format=qwen3_coder` |
| Thinking | model emits `<think>...</think>` | template injects `<think>`, so the dataset removes the duplicated instruction |
| Dataset class | `CustomRLHFDataset` | `CustomRLHFDatasetQwen35` |
| Rollout engine | SGLang | vLLM-Ascend |
| Training engine | FSDP | FSDP2, `use_remove_padding=False` |

## Reproducing the Experiment

> **Note on the 'Chart' Dataset:**
> 
> The provided preprocessing script intentionally excludes `data_v0.8_visual_toolbox_v2.parquet`, which contains the 'Chart' data. This subset consists of very high-resolution images, often resembling large figures composed of multiple sub-plots, much like those found in academic papers.
>
> Consequently, even after using the zoom-in tool, the resulting cropped images remain large. This poses a significant risk of causing Out-of-Memory (OOM) errors, which can abruptly terminate the training process. 
> 
> **We strongly recommend against training on the 'Chart' dataset on a single node.**

> **Note on the 'thinklite' Dataset:**
> Many images in the `thinklite` dataset have a very low resolution, with either a height or width below 28 pixels. This fails to meet the minimum input size required by the Qwen-2.5VL image processor and would cause errors during data loading.
>
> To mitigate this, we upscale these low-resolution images to satisfy the processor's requirements. However, please be aware that because the original resolution is low, subsequent `crop` operations by the zoom-in tool might frequently trigger exceptions, which could in turn affect the model's tool-use performance.

First, launch an inference service to act as a judge for reward calculation. You can use the following script as a reference:

```bash
python -m sglang.launch_server --model-path /path/to/Qwen2.5-72B-Instruct \
    --port 18901 \
    --tp-size 8 \
    --context-length 32768 \
    --trust-remote-code \
    --log-requests false
```

Next, you can start the training:

```bash
bash recipe/deepeyes/run_deepeyes_grpo.sh
```

## Performance

See [Comment](https://github.com/volcengine/verl/pull/2398#issuecomment-3157142856) for more details.

Note: AgentLoop does not directly record num_tool_calls, but records num_turns. In our scenario, you can calculate the number of tool calls by num_tool_calls = num_turns / 2 - 1.

## References and Acknowledgements

- [DeepEyes Paper](https://arxiv.org/abs/2505.14362)
- [DeepEyes Official Implementation](https://github.com/Visual-Agent/DeepEyes)

---
If you need further details for reproduction or encounter any issues, feel free to open an issue or contact the maintainers. 
