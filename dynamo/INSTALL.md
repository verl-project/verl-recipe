# Installation

For **Uni-Agent + Rust ThunderAgent + V1 async**, start with the pinned
**vLLM 0.29** image and its bundled verl in the
[usage guide](THUNDERAGENT_VLLM.md#environment). The recipe's launchers use that
prepared environment.

## Source install

Use the activated **Python 3.12** Linux environment from the guide, with
verl and **vLLM 0.29.0** or **SGLang 0.5.19** already installed. Install the Python build tools
into that environment:

```bash
python3 -m pip install uv 'maturin[patchelf]'
```

Rust **1.96.1** is selected by the pinned checkout's `rust-toolchain.toml`.
For system libraries, the compiler and libclang, follow the
[official source-build guide](https://docs.dynamo.nvidia.com/dynamo/advanced-customizations/building-from-source).
Build in a Linux environment compatible with the training container.

```bash
export DYNAMO_SRC=/path/to/dynamo
cd "$DYNAMO_SRC"
git checkout b1429b8bf67a47f4657fb716b2fab369c172f5e5
python3 -m uv build --wheel
cd "$DYNAMO_SRC/lib/bindings/python"
python3 -m maturin build --release --locked --features kv-indexer --out "$DYNAMO_SRC/dist"
python3 -m uv pip install --python "$(command -v python3)" \
    --reinstall-package ai-dynamo --reinstall-package ai-dynamo-runtime \
    "$DYNAMO_SRC"/dist/ai_dynamo*.whl 'protobuf>=6.33.5,<7'
export PYTHONPATH="$DYNAMO_SRC/components/src${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import dynamo._core; print(dynamo._core.__file__)'
python3 -m dynamo.frontend --help
```

`uv build` produces the `ai_dynamo` Python wheel; `maturin build` produces the
`ai_dynamo_runtime` wheel containing `dynamo._core`. Both go into `dist/` here;
keep only this build's two wheels there. The install command replaces those two
packages even if a previous build has the same version number, while resolving
Dynamo dependencies. `--python` selects the active interpreter explicitly.

`PYTHONPATH` loads frontend/backend Python code from the same checkout. Repeat
that export in the training job, with the checkout mounted at `DYNAMO_SRC`.
Keep `etcd` and `nats-server` on `PATH`, then use the
[Uni-Agent launch commands](THUNDERAGENT_VLLM.md#workload-and-launch).

## Build features

The pinned [Dynamo build](https://github.com/ai-dynamo/dynamo/blob/b1429b8bf67a47f4657fb716b2fab369c172f5e5/lib/bindings/python/Cargo.toml)
includes the builtin ThunderAgent plugin. `--features kv-indexer` keeps Cargo's
default `custom-policy` feature and adds the standalone indexer; do not disable
default features. This recipe's text-only frontend KV routing does not use the other
optional services in the full build. Enable routing at runtime with `router_mode=kv` and
`enable_kv_events=true`; the build flag alone does not configure the router.

The source-build commands describe the pinned source's features. The H100 RL
validation used matching prebuilt Python/native wheels; it did not rebuild them
with these commands.

## Earlier engine benchmarks

The earlier vLLM 0.28.0 / SGLang 0.5.19 results used Dynamo `8c5a737` and the
[full build recorded with the benchmark](https://github.com/verl-project/verl-recipe/blob/5451026758c3f1dc3ebc2a46a0e52b14406f0f72/dynamo/INSTALL.md#dfw-benchmark-build);
that revision predates the builtin Rust ThunderAgent plugin. Use the recorded
environment to reproduce those results, in separate environments for the two
engines. The current native ThunderAgent preset requires the build above.
