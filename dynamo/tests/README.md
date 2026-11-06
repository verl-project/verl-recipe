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
Supply a compatible image path with `CONTAINER`; this job is separate from the
[validated Uni-Agent vLLM 0.29 launchers](../THUNDERAGENT_VLLM.md).

The launcher argument checks in [test_smoke_launchers.py](test_smoke_launchers.py)
need only pytest and hydra-core; they run the Bash scripts with a captured Python
child process and do not start training.

[test_thunderagent_vllm_launcher.py](test_thunderagent_vllm_launcher.py)
composes the real YAML for the Uni-Agent colocate and fully async launchers,
checks the environment forwarded to Ray, and preserves the child's exit status.
It requires the prepared verl configuration package as well as pytest and hydra-core.

[test_verl_logging_compat.py](test_verl_logging_compat.py) checks the fully async
logging adapter, including the pristine upstream trainer's setup path with INFO
enabled. The upstream setup test skips when verl is absent.

[test_uniagent_gateway.py](test_uniagent_gateway.py) uses Uni-Agent's actual
Gateway with a fake model backend to check token alignment, program cleanup,
retries, timeouts and cancellation. Install the Uni-Agent/verl versions in the
[Claude configuration](../README.md#subagent-programs-claude-code); this module
skips when Uni-Agent is absent. Install the pinned package, or put its checkout
on `PYTHONPATH` next to verl (for example `PYTHONPATH=/workspace/verl:/workspace`
with Uni-Agent at `/workspace`). Pass `-rfs` to see the skip reasons.

[test_uniagent_subagents.py](test_uniagent_subagents.py) checks all three real HTTP
routes, identical sibling and nested agent trajectories, lifecycle hooks, resumed
agents, paused engines and retained cleanup failures. Gateway tests also exercise
the real framework failure path and cancellation acknowledgement from a CPU Ray actor.
These checks do not run Claude CLI or GPU inference.

[check_continuations.py](check_continuations.py) checks all recipe shell/Slurm
scripts recursively by default; optional arguments select file/glob paths:

```bash
python3 recipe/dynamo/tests/check_continuations.py
```

The NIXL bandwidth microbenchmark is in [benchmarks/](../benchmarks/nixl_bench.py).
