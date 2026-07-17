# Recipe: Async On-Policy Knowledge Distillation Trainer on Ascend NPU

## Required `verl` version

See [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) for the upstream repository, install mode (rolling `main`, pinned release tag, or pinned git commit), and copy-pastable `pip` / `git` instructions where they exist.

This recipe extends the [GKD (On-Policy Knowledge Distillation)](../gkd/megatron/README.md) recipe with **Ascend NPU support** and **FSDP/FSDP2 backend support**, enabling knowledge distillation training on Huawei Ascend NPUs and providing an alternative to the Megatron-based training backend.

## 1. Background

On-policy knowledge distillation (KD) trains a student policy to imitate a stronger teacher using samples drawn from the student's current policy. For each on-policy rollout the teacher returns soft, top-k token distributions and the student is optimized with a token-wise sparse KL objective that focuses learning on the teacher's high-probability modes. Because training examples come from the student's own state distribution, KD reduces distributional mismatch relative to off-policy distillation or supervised fine-tuning (SFT), improving stability and sample efficiency. Compared with reinforcement learning, KD avoids high-variance reward-based optimization and complex reward design by providing dense, informative per-token targets, which typically yields faster convergence and simpler scaling.

Built on verl's Ray-based single-controller components, we initially assembled a strictly on-policy KD pipeline where rollout generation, teacher knowledge acquisition, and policy optimization ran in lockstep. In practice, this synchronous design proved highly inefficient: the three stages had to wait for one another, creating pipeline bubbles and underutilized GPUs. To address this, we extend the asynchronous schedulers introduced by the One-Step-Off Policy pipeline to overlap these phases. This overlap preserves the same distillation objective while trading some strict on-policy guarantees for substantial gains in end-to-end throughput and hardware utilization.

## 2. What This Recipe Adds

This recipe is a direct fork of the original `recipe/gkd` with the following additions and modifications:

### 2.1 Ascend NPU Adaptation

To run GKD training on Ascend NPUs, the following key adaptations were made:

- **Device auto-detection**: `main_gkd.py` calls `auto_set_ascend_device_name(config)` to automatically set `config.trainer.device = npu` when running on Ascend hardware, so no manual config changes are needed.
- **HCCL communication backend**: `distributed_util.py` replaces NCCL with HCCL (via `vllm_ascend.distributed.device_communicators.pyhccl`) for weight synchronization between actor and rollout workers on NPU.
- **NPU-aware weight sync group creation**: `ray_trainer.py` and `megatron_workers.py` use a separate weight-sync group creation path for NPU, bypassing the Ray collective group approach (which relies on NCCL) and instead using a direct IP/port-based stateless process group.
- **Device name propagation**: All worker modules use `get_device_name()` (from `verl.utils.device`) instead of hardcoded `"cuda"`, ensuring tensors and autocast operations target the correct device.

### 2.2 Teacher vLLM API Backend

The original GKD recipe only supports an embedded vLLM engine as the teacher backend, which requires loading the teacher model into a worker process. This recipe adds a **vLLM API backend** (`vllm_api`) that connects to an existing vLLM serve API server via its OpenAI-compatible completions API, instead of embedding the vLLM engine in the worker:

- **`teacher/vllm_api_backend.py`**: Implements `VLLMAPIBackend` class that connects to an external `vllm serve` instance, retrieves top-k logprobs through the completions API, and handles batch splitting to avoid OOM on the vLLM server side.
- **`teacher/start_server_vllm_api.sh`**: Startup script for the API backend mode — first waits for the vLLM serve API to be ready, then launches the proxy and worker with `--backend vllm_api`.
- **`teacher/worker.py`**: Extended to support `--backend vllm_api` with `--api-base` and `--serve-model` arguments alongside the original `--backend vllm_engine`.

This is particularly useful on Ascend NPU where vLLM-Ascend is deployed as a standalone inference service, or when the teacher model is already running as a shared API server that multiple training jobs can consume simultaneously.

### 2.3 FSDP/FSDP2 Backend Support

The original GKD recipe only supports the Megatron training backend. This recipe adds a full FSDP/FSDP2 backend alongside Megatron, enabling users who prefer FSDP's simpler deployment model or who are working with models not yet supported by Megatron:

- **`fsdp_workers.py`**: Implements `FSDPOnPolicyDistillActorWorker` and `FSDPOnPolicyDistillRolloutWorker`, reusing the disaggregated weight-sync path from `recipe.one_step_off_policy.fsdp_workers` and overriding `update_actor` / `async_generate_sequences` for the KD objective.
- **`fsdp_kl_loss.py`**: FSDP-adapted version of the KL distillation loss. Since FSDP does not shard the vocab dimension across tensor-parallel ranks, the logits tensor on every rank already contains the full vocab dimension, allowing standard softmax / KL computation directly (no vocab-parallel cross-entropy needed).
- **Dual config files**: `config/on_policy_distill_trainer.yaml` (FSDP) and `config/on_policy_distill_megatron_trainer.yaml` (Megatron) provide separate Hydra configurations for each backend.
- **Backend selection**: `main_gkd.py`'s `create_role_worker_mapping()` dispatches to the correct worker classes based on `actor_rollout_ref.actor.strategy` (`megatron`, `fsdp`, or `fsdp2`).

## 3. Distillation Overview and Objective

This recipe centers on on-policy knowledge distillation: the student policy learns from a stronger teacher on samples generated by the current policy (on-policy). For each input prompt, the student (actor) generates responses; the teacher provides top-k token distributions, and the student is trained to match them token-wise.

Core components:

1. Teacher signal: top-k log-probabilities and token indices per valid token position.
2. Student objective: sparse, token-level KL divergence between student logits and teacher top-k distribution.

Objective: encourage student probabilities $Q$ to cover teacher modes $P$ using token-wise $\mathrm{KL}(P\,\|\,Q)$ computed on the teacher's top-k support.

## 4. Efficient System Design

### 4.1 Schedulers (One-Step / Two-Step Off-Policy)

The native (serial) on-policy distillation process is shown in the figure below.

![Zero-Step-Off Scheduler](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/docs/zero-step-off-distill.png)

This recipe supports optional schedulers that overlap generation, teacher querying, and updates to improve throughput without changing the distillation objective.

#### 4.1.1 One-Step-Off-Policy

![One-Step-Off Scheduler](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/docs/one-step-off-distill.png)

- Warm-up: 2 steps.
- Overlap pattern: rollout while actor update; weight sync while teacher retrieving.
- Timing keys: `sync_rollout_weights`, `wait_prev_gen`, `wait_prev_teacher`.

#### 4.1.2 Two-Step-Off-Policy

![Two-Step-Off Scheduler](https://raw.githubusercontent.com/eric-haibin-lin/verl-community/refs/heads/main/docs/two-step-off-distill.png)

- Warm-up: 3 steps.
- Overlap pattern: rollout, actor update while teacher retrieving; interleave weight sync.
- Timing keys: `sync_rollout_weights`, `max(wait_prev_gen, wait_prev_prev_teacher)`.

Tip: Use `two_step_off` when teacher takes much more time than sync; `one_step_off` for simpler overlapping.

### 4.2 Weights sync between actor and rollout

We initially followed the weight synchronization path from the One-Step-Off-Policy recipe (Ray collective broadcast across all actor and rollout ranks, plus Megatron-side allgather of parameter shards). In practice this became the dominant bottleneck, so we made three changes:

1. Batch-and-bulk load on the rollout side: instead of streaming tensors one-by-one, we stage a bundle of parameter tensors and issue a single batched load into the rollout engine.
2. Batch-and-bulk broadcast between the actor and rollout: instead of streaming tensors one-by-one, we stage a bundle of parameter tensors and issue a single batched broadcast between the actor and rollout workers.
3. Replace allgather with gather-to-root in Megatron: parameter shards are gathered to actor rank 0, and that root serves as the single source for broadcasting to rollout ranks.

## 5. High-Level Data & Control Flow

```
Driver (TaskRunner)
  ├─ Initialize Ray, tokenizer, datasets, worker groups
  ├─ Build ResourcePoolManager (actor vs rollout GPU/NPU layouts)
  ├─ Trainer.fit()
      ├─ init_workers(): build actor + rollout groups, broadcast weight metadata,
      │   create weight-sync group (NCCL on GPU, HCCL on NPU)
      ├─ continuous_iterator(): epochs → batches
      ├─ scheduler (see Section 4)
        • _async_gen_next_batch(): optional weight sync + non-blocking rollout
        • _async_get_teacher_knowledge(): submit teacher requests, store future
        ├─ For each step:
            • Sync rollout weights
            • Retrieve (batch, gen_output, teacher_output) from futures
            • Merge gen + teacher outputs → DataProto
            • Compute metrics (response length stats, timing, throughput)
            • Update actor (forward_backward_batch + KL loss + optimizer step)
            • (Optional) save checkpoint
```

## 6. Key Components

### 6.1 `OnPolicyDistillTrainer` (`ray_trainer.py`)
- Creates `GenerationBatchFuture` objects holding rollout and teacher futures.
- Adds scheduling + teacher integration + modified metric emission (KL, timing, MFU).
- NPU-aware weight-sync group creation using HCCL instead of NCCL.

### 6.2 Actor Worker (Megatron backend)
- `MegatronOnPolicyDistillActorWorker.update_policy()` orchestrates micro-batch forward/backward.
- KL Loss injection via `logits_processor` during forward on pipeline last stage.

### 6.3 Actor Worker (FSDP backend)
- `FSDPOnPolicyDistillActorWorker.update_actor()` performs KL distillation update with FSDP-sharded model.
- Uses `fsdp_kl_loss.py` for KL computation on full vocab logits.

### 6.4 Rollout Worker (vLLM)
- Pure inference mode (`init_model` builds model; no optimizer).
- `async_generate_sequences` returns a Ray future for overlapping.

### 6.5 Teacher Service (`teacher/`)
- Proxy + worker architecture (ZMQ REQ/REP) for batched top-k retrieval.
- `TeacherClient.submit()` returns a `Future`; aggregator composes micro-batches.
- Configurable temperature, max tokens, only-response mode.
- Two backend modes:
  - **`vllm_engine`** (original): Embeds a vLLM engine instance in the worker process. Use `start_server_vllm_engine.sh` to launch.
  - **`vllm_api`** (new): Connects to an existing vLLM serve API server via OpenAI-compatible completions API. Use `start_server_vllm_api.sh` to launch.

### 6.6 KL Loss
- **Megatron** (`megatron_kl_loss.py`): Performs normalization & stable per-token probability construction across TP shards. Gradient is (student_probs - teacher_sparse_probs) scaled by upstream grad.
- **FSDP** (`fsdp_kl_loss.py`): Direct softmax / KL on full vocab logits (no TP vocab sharding needed).

### 6.7 Distributed Utility (`distributed_util.py`)
- `vllm_stateless_init_process_group()` selects NCCL or HCCL backend based on `is_npu_available`.

## 7. Configuration Highlights

### FSDP backend (`on_policy_distill_trainer.yaml`)

| Section | Purpose | Notable Keys |
|---------|---------|-------------|
| actor_rollout_ref.actor.strategy | Backend selection | `fsdp` or `fsdp2` |
| actor_rollout_ref.teacher | Teacher server | server_ip, server_port, n_server_workers |
| actor_rollout_ref.actor.fsdp_config | FSDP settings | model_dtype, param_offload, optimizer_offload |
| trainer | Global training control | total_epochs, save_freq, scheduler, device (`npu` on Ascend) |
| rollout | Resource split for rollout | n_gpus_per_node, nnodes |

### Megatron backend (`on_policy_distill_megatron_trainer.yaml`)

| Section | Purpose | Notable Keys |
|---------|---------|-------------|
| actor_rollout_ref.actor.megatron | Megatron parallelism | pipeline_model_parallel_size, tensor_model_parallel_size, expert_model_parallel_size |
| actor_rollout_ref.teacher | Teacher server | server_ip, server_port, n_server_workers |
| trainer | Global training control | total_epochs, save_freq, scheduler, device (`npu` on Ascend) |
| rollout | Resource split for rollout | n_gpus_per_node, nnodes |

**Remember to set `trainer.n_gpus_per_node`, `trainer.nnodes`, `rollout.n_gpus_per_node` and `rollout.nnodes` to allocate NPU resources. On Ascend, `trainer.device` should be set to `npu`.**

## 8. Usage Examples

### 8.1 Environment Setup

For setting up the Ascend NPU environment for verl, please refer to [ascend_quick_start.rst (in Chinese)](../../docs/ascend_tutorial/ascend_quick_start.rst).

Key dependencies:
- torch_npu (matching PyTorch version)
- vllm + vllm-ascend (for rollout on NPU)
- CANN toolkit
- MindSpeed + Megatron-LM (for Megatron backend only)

### 8.2 Launch Teacher Server

Before training, you need a teacher server to provide logp information. The teacher service supports two backend modes:

#### vLLM Engine backend (embedded)

Launches a vLLM engine inside the worker process. The teacher model is loaded directly by the worker.

```bash
cd recipe/gkd_ascend/teacher
bash start_server_vllm_engine.sh
```

You can also start a multi-node teacher server: start the main node using `start_server_vllm_engine.sh`, then start slave nodes using `join_server_vllm_engine.sh` (remember to set `$PROXY_IP` and `$PROXY_BACKEND_PORT` of the main node).

#### vLLM API backend (remote serve)

Connects to an existing vLLM serve API server. Start the vLLM server separately first:

```bash
vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 --port 8000 --max-logprobs 256
```

Then launch the teacher worker connecting to it:

```bash
cd recipe/gkd_ascend/teacher
bash start_server_vllm_api.sh
```

In `start_server_vllm_api.sh`, configure `MODEL_API` (the vLLM serve URL, e.g. `http://0.0.0.0:8000`) and `SERVE_MODEL_NAME`.

Verify the teacher server is reachable with:

```bash
telnet localhost 15555
```

### 8.3 FSDP Backend (Qwen3-4B Example)

```bash
export BACKEND=fsdp2
bash run_4b_fsdp.sh
```

Or run directly:

```bash
python3 -u -m main_gkd --config-path=config --config-name on_policy_distill_trainer \
    data.train_files=openai-gsm8k/train.parquet \
    data.val_files=openai-gsm8k/test.parquet \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.path=/path/to/Qwen3-4B/ \
    actor_rollout_ref.teacher.server_ip=127.0.0.1 \
    actor_rollout_ref.teacher.server_port=15555 \
    trainer.device=npu \
    trainer.n_gpus_per_node=4 rollout.n_gpus_per_node=2 \
    trainer.scheduler=one_step_off
```

### 8.4 Megatron Backend (Qwen3-4B Example)

```bash
export BACKEND=megatron
bash run_4b_megatron.sh
```

Or run directly:

```bash
python3 -m main_gkd --config-path=config --config-name on_policy_distill_megatron_trainer \
    data.train_files=openai-gsm8k/train.parquet \
    data.val_files=openai-gsm8k/test.parquet \
    actor_rollout_ref.model.path=/path/to/Qwen3-4B/ \
    actor_rollout_ref.teacher.server_ip=127.0.0.1 \
    actor_rollout_ref.teacher.server_port=15555 \
    trainer.device=npu \
    trainer.n_gpus_per_node=4 rollout.n_gpus_per_node=2 \
    trainer.scheduler=one_step_off
```

## 9. Metrics & Monitoring

Emitted metrics include (prefixes may vary):

- Timing: `timing/wait_prev_gen`, `timing/sync_rollout_weights`, `timing/get_teacher_knowledge`, `timing/update_actor`.
- Sequence stats: `response_seq_len/*` (avg, max, min, counts).
- Performance: `perf/mfu/actor`, `perf/max_memory_allocated_gb`, `perf/cpu_memory_used_gb`.
- Distillation: `actor/kl_loss`, `actor/grad_norm`, `actor/lr`.

Interpretation Tips:

- High `wait_prev_teacher` → scale `n_server_workers` and allocate more teacher NPUs or reduce per-request batch size, or just use `two_step_off`.
- High `wait_prev_gen` with uniform lengths → allocate more rollout NPUs.
- High `sync_rollout_weights` → check HCCL env / network congestion and try to modify `actor_rollout_ref.rollout.update_weights_bucket_megabytes`.

## 10. Functional Support Summary

| Category | Supported |
|----------|-----------|
| Train engine | Megatron, FSDP, FSDP2 |
| Rollout engine | vLLM (via vLLM-Ascend on NPU) |
| Hardware | Ascend NPU, NVIDIA GPU |
| Teacher backend | vLLM engine (embedded), vLLM API (remote serve) |
| Distillation signal | Teacher top-k logprobs & indices |
| Scheduling | one_step_off, two_step_off |

## 11. Quick Checklist Before Running

- Ascend NPU environment set up (CANN, torch_npu, vllm-ascend installed).
- Teacher server reachable (`telnet <ip> <port>`).
- `actor_rollout_ref.model.path` contains the correct model config artifacts.
- `train_files` points to a parquet dataset compatible with this recipe's dataset loader.
- `trainer.device` set to `npu` (auto-detected by `auto_set_ascend_device_name`).
- HCCL environment vars set for multi-node communication.