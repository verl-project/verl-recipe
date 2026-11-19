# Dynamo tests

Run the CPU regression suite from a verl checkout with this repository at
`recipe/`, using a prepared environment from the [installation guide](../INSTALL.md):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q \
    -p pytest_asyncio.plugin --import-mode=importlib recipe/dynamo/tests
```

[smoke/](smoke/) contains GPU generation/training smokes, the NIXL refit smoke,
and the Slurm KV-metrics validation job. Configure model/data paths and resources
using the [README smoke examples](../README.md#launch).

The [KV-metrics Slurm job](smoke/validate_vllm_kv_metrics.sbatch) uses the sibling
[container bootstrap](smoke/container_bootstrap.sh) for its legacy ReTool environment.

The launcher argument checks in [test_smoke_launchers.py](test_smoke_launchers.py)
need only pytest and hydra-core; they run the Bash scripts with a captured Python
child process and do not start training.

[test_thunderagent_vllm024_launcher.py](test_thunderagent_vllm024_launcher.py)
checks the base, Uni-Agent and fully async launchers' arguments, GPU environment
and child exit status.

[test_verl_logging_compat.py](test_verl_logging_compat.py) checks the fully async
logging adapter, including the pristine upstream trainer's setup path with INFO
enabled. The upstream setup test skips when verl is absent.

[test_uniagent_gateway.py](test_uniagent_gateway.py) uses Uni-Agent's actual
Gateway with a fake model backend to check token alignment, program cleanup,
retries, timeouts and cancellation. Install the Uni-Agent/verl versions in the
[0.24 guide](../THUNDERAGENT_VLLM024.md); this module skips when Uni-Agent is absent.

[check_continuations.py](check_continuations.py) checks all recipe shell/Slurm
scripts recursively by default; optional arguments select file/glob paths:

```bash
python3 recipe/dynamo/tests/check_continuations.py
```

The NIXL bandwidth microbenchmark is in [benchmarks/](../benchmarks/nixl_bench.py).
