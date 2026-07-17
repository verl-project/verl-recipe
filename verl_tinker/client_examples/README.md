# Tinker Server Client Examples

This directory contains client-side smoke workloads for the VeRL-backed Tinker
server. The examples use `tinker` and `tinker-cookbook` from a separate `uv`
environment and connect to an already-running server over HTTP.

## Setup

The client examples are their own `uv` project with their own environment. From
the repository root, change into this directory before installing or running
client code:

```bash
cd verl_tinker/client_examples
uv sync
```

This installs only client-side dependencies. It does not install core `verl` or
the `verl_tinker` server package.

## Start A Server

Install the server from the repository root, then change into the `verl_tinker`
recipe directory before launching it:

```bash
./install_verl.sh --recipe verl_tinker
cd verl_tinker

python -m verl_tinker.start \
  --config configs/quick_start/actor_rollout.yaml
```

For SFT-only tests that do not need sampling, use:

```bash
python -m verl_tinker.start \
  --config configs/quick_start/actor.yaml
```

For the dedicated-teacher OPD example on one eight-GPU node, use:

```bash
python -m verl_tinker.start \
  --config configs/advance/qwen3_1b7_actor_qwen3_30b_a3b_teacher.yaml
```

## Run A Workload

In another shell, change into the client examples directory so `uv run` uses the
client environment:

```bash
cd verl_tinker/client_examples
uv run run_single_test.py \
  --base-url http://127.0.0.1:8000/ \
  --test-name sft_tulu3
```

The runner waits for `/api/v1/healthz`, sets
`TINKER_API_KEY=tml-verl-tinker-local`, runs the selected workload, and asks the
server to shut down when the workload exits.

## Workloads

Set `--test-name` to one of:

- `sft_tulu3`
- `sft_norobot`
- `sft_norobot_no_rollout`
- `sdft_single_task`
- `rl_gsm8k`
- `sft_rl_gsm8k`
- `opd_deepmath`

If `--test-name` is not set, the runner defaults to `sft_tulu3`.

Useful arguments:

- `--base-url`: server URL. Defaults to `http://127.0.0.1:8000/`.
- `--model-name`: model name sent to the Tinker Cookbook.
- `--tokenizer-name-or-path`: tokenizer path override. Defaults to
  `--model-name`.
- `--api-key`: Tinker API key compatibility value.
- `--test-name`: workload selector.
- `--teacher-model-name`: teacher requested by OPD workloads. Defaults to
  `Qwen/Qwen3-30B-A3B`.
- `--patch-hdfs-tokenizer-import`: opt in to the unsupported HDFS tokenizer
  monkey patch described below. Disabled by default.

Run the one-step OPD smoke workload with:

```bash
uv run run_single_test.py \
  --test-name opd_deepmath \
  --model-name Qwen/Qwen3-1.7B \
  --teacher-model-name Qwen/Qwen3-30B-A3B
```

### HDFS tokenizer compatibility patch

The Tinker SDK and Tinker Cookbook assume tokenizer identifiers are local paths
or Hugging Face model IDs. They also split model identifiers at `:`, which
breaks `hdfs://` URIs. If a test server exposes an HDFS tokenizer URI, the
client runner can install a monkey patch before running the workload:

```bash
uv run run_single_test.py \
  --test-name opd_deepmath \
  --patch-hdfs-tokenizer-import
```

This is intentionally disabled by default and is a fragile compatibility
workaround, not supported HDFS integration. It patches private Tinker SDK and
Cookbook functions plus `AutoTokenizer.from_pretrained`, so upstream package
updates can break it. Prefer configuring a normal Hugging Face tokenizer ID or
a local tokenizer path whenever possible. On a cache miss, the workaround
copies only tokenizer/config files from HDFS into `/tmp`; it excludes model
weight files.

Some workloads download Hugging Face datasets. Configure the standard Hugging
Face cache environment variables if you need offline or pre-populated datasets.
