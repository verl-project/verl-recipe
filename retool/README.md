# Retool

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repository, install mode (rolling `main`, pinned release tag, or pinned git commit), and copy-pastable `pip` / `git` instructions where they exist.

[ReTool: Reinforcement Learning for Strategic Tool Use in LLMs](https://arxiv.org/abs/2504.11536)

## Overview
- Models:
  - GPU: [Qwen/Qwen2.5-32B-Instruct](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
  - TPU (`torch_tpu` + `torchtitan`): [Qwen/Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B)
- SFT dataset: [JoeYing/ReTool-SFT](https://huggingface.co/datasets/JoeYing/ReTool-SFT)
- RL dataset: [BytedTsinghua-SIA/DAPO-Math-17k](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)
- Val datasets: [Maxwell-Jia/AIME_2024](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024), [yentinglin/aime_2025](https://huggingface.co/datasets/yentinglin/aime_2025)

## How it works

 Retool's workflow is divided into two key phases:

1.  Cold Start and Supervised Fine Tuning (SFT)
    
     The data generation pipeline builds a high-quality dataset containing code-enhanced inference trajectories, and supervised fine-tuning enables the model to master basic Tool call (e.g., code execution) and analysis of the execution results.
    
2.  Dynamic Interaction and Policy Optimization (RL).
    
     With the verl Reinforcement Learning framework, the model dynamically inserts code blocks during inference and interacts with the tool environment in real-time, generating a hybrid trajectory of natural language thinking and code snippets, executing Python code when tool call markers are detected, and feeding execution results (success outputs/errors) back to the model for guiding the subsequent inference. This "think-execute-feedback" cycle, together with the design of rewards based on the accuracy of the final answer, enables the model to independently optimize the Tool call strategy, and improves the reasoning efficiency and computational accuracy.

## Installation

```bash
pip install verl==0.6.1
```

## SFT
1. Data preparation
```bash
python3 recipe/retool/retool_sft_preprocess.py
```

2. Training (GPU - Qwen2.5-32B)
```bash
bash recipe/retool/run_qwen2-32b_sft.sh
```

After 6 epoches, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.24
- val-aux/num_turns/mean: 7.2

3. Training (TPU - Qwen3-30B-A3B MoE)
```bash
bash recipe/retool/run_qwen3_30b_a3b_sft_tpu.sh
```

## RL

### GRPO / DAPO (GPU - Qwen2.5-32B)
```bash
bash recipe/retool/run_qwen2-32b_dapo.sh
```

After 150 steps, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.6
- val-aux/num_turns/mean: 10

### GRPO / DAPO (TPU - Qwen3-30B-A3B MoE)
```bash
bash recipe/retool/run_qwen3_30b_a3b_dapo_tpu.sh
```

### PPO (GPU - Qwen2.5-32B)

```bash
bash recipe/retool/run_qwen2-32b_ppo.sh
```

After 250 steps, validation metrics:
- val-core/aime_2025/acc/mean@30: 0.55
- val-aux/num_turns/mean: 8.3

## Tool backends

The RL scripts run the model's code through the tool named in `actor_rollout_ref.rollout.multi_turn.tool_config_path`. The default is [Sandbox Fusion](sandbox_fusion_tool_config.yaml), which you host yourself. [`daytona_tool_config.yaml`](daytona_tool_config.yaml) and [`e2b_tool_config.yaml`](e2b_tool_config.yaml) use hosted sandboxes instead.

### E2B

[E2B](https://e2b.dev) runs the code in cloud sandboxes, so there is no code execution service to deploy.

```bash
pip install e2b
export E2B_API_KEY="..."

bash recipe/retool/run_qwen2-32b_dapo.sh \
    actor_rollout_ref.rollout.multi_turn.tool_config_path=recipe/retool/e2b_tool_config.yaml
```

`E2B_API_KEY` must be set in the environment of the Ray workers, not only in the shell that starts training.

How it works:

- Each agent loop worker keeps a pool of `pool_size` sandboxes and reuses them, because E2B limits how fast sandboxes can be created (1 per second on Hobby, 5 on Pro). Sandboxes are created on demand, `create_interval` seconds apart.
- Every tool call runs in fresh Linux namespaces inside one sandbox: read-only root, empty `/tmp`, no network, an unprivileged user. Variables, files and processes do not carry over to the next call, the same as with Sandbox Fusion.
- The model gets stdout followed by stderr. Code that runs longer than `default_timeout` seconds is killed and the output ends with `TimeoutError: Code execution timed out.`
- If a sandbox fails, the call is retried once on a new sandbox. If that fails too, the model gets `Code execution is temporarily unavailable; the code was not run.` and a warning is logged.
- A sandbox that gets no calls for `sandbox_timeout` seconds is removed by E2B and replaced on the next call. Nothing has to be shut down after training.

Size the pool to your plan: `pool_size` × `actor_rollout_ref.rollout.agent.num_workers` must stay within the plan's concurrent sandbox limit (10 on Hobby, 100 on Pro; see [E2B limits](https://e2b.dev/docs/billing)). The default `pool_size: 8` with 8 workers uses up to 64 sandboxes. On Hobby, set `pool_size: 1`.

The default `code-interpreter-v1` template includes numpy, sympy and scipy. To use other packages, build a [custom template](https://e2b.dev/docs/template/quickstart) and set `template`.

Tested with `verl==0.9.1`. The tests in [`tests/test_e2b_sandbox_tool.py`](tests/test_e2b_sandbox_tool.py) need no GPU or network: `pytest recipe/retool/tests/test_e2b_sandbox_tool.py` from the verl root.
