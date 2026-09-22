# Installation Guide (Ascend NPU)

This recipe is developed and validated inside a Docker container on **Ascend A3**
NPUs. Start from an Ascend CANN base image, run a container with the NPU devices
mounted, then build the framework stack against the verified version matrix below.

Commands assume an **`aarch64`** host.

---

## 1. Verified environment

The recipe is tested with the following stack:

| Component | Version |
|---|---|
| OS | Ubuntu 22.04 (`aarch64`) |
| CANN | **8.5.2** (toolkit + `nnal/atb`) — verified on the `8.5.2-a3-ubuntu22.04-py3.11` base image |
| Python | 3.11 |
| PyTorch | `torch==2.8.0` (CPU wheel) + `torch_npu==2.8.0` |
| torchvision | 0.23.0 |
| transformers | 4.57.6 |
| vLLM | 0.13.0 — built with `VLLM_TARGET_DEVICE=empty` (`vllm-project/vllm` @ `v0.13.0`) |
| vllm-ascend | 0.13.0rc3 + patches (`vllm-project/vllm-ascend` @ `894c0de3`) |
| Megatron-LM | `core_v0.14.0` (`megatron-core==0.14.0`) |
| MindSpeed | 0.14.1 (`gitcode Ascend/MindSpeed` @ `71b9e055`) |
| mbridge | 0.15.1 (`ISEEKYAN/mbridge` @ `4389fcc`) |
| Apex (Ascend) | `0.1+ascend` |
| triton-ascend | **3.2.0** (kept — DR.Kernel generates/evaluates Triton-on-NPU kernels) |
| Ray | 2.54.0 |
| verl | `verl-project/verl` @ `284d0388` (PR #6020) — see [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt) |
| KernelGYM server | redis 5.0.1, fastapi 0.123.10, uvicorn 0.42.0 |

---

## 2. Launch the NPU container

Start from an Ascend **CANN 8.5.2** base image (Ubuntu 22.04, Python 3.11). Run
the container with host networking, the NPU devices, and the standard Ascend
driver/tooling bind-mounts:

```bash
IMAGE=quay.io/ascend/cann:8.5.2-a3-ubuntu22.04-py3.11

docker run -it --name verl_drkernel \
    --network host --privileged --shm-size=128g \
    --device=/dev/davinci0 --device=/dev/davinci1 \
    --device=/dev/davinci2 --device=/dev/davinci3 \
    --device=/dev/davinci4 --device=/dev/davinci5 \
    --device=/dev/davinci6 --device=/dev/davinci7 \
    --device=/dev/davinci8 --device=/dev/davinci9 \
    --device=/dev/davinci10 --device=/dev/davinci11 \
    --device=/dev/davinci12 --device=/dev/davinci13 \
    --device=/dev/davinci14 --device=/dev/davinci15 \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v /var/log/npu/:/usr/slog \
    -v /path/to/models:/home/model \
    -v /path/to/datasets:/data \
    "${IMAGE}" /bin/bash
```

Adjust the number of `--device=/dev/davinciN` entries to the cards on the node and
`--shm-size` to host RAM. If your base image already ships the framework stack
(§3) and CANN env (§4), skip to §5.

---

## 3. Build the framework stack from source

Run these **inside** the container:

```bash
# 3.1 system deps
apt-get update -y && apt-get install -y --no-install-recommends \
    gcc g++ cmake libnuma-dev wget git curl jq vim build-essential
# Pin setuptools < 81: 81+ drops the bundled `pkg_resources`, which `torch_npu`'s
# `torchair` imports at load time. Without it, `import torchair` (and therefore
# any `verl`/MindSpeed import) fails with `ModuleNotFoundError: No module named
# 'pkg_resources'`.
pip install --upgrade pip packaging "setuptools<81"

# 3.2 PyTorch + NPU backend
pip install torch==2.8.0 torch_npu==2.8.0 torchvision==0.23.0 transformers==4.57.6

# 3.3 triton-ascend — KEEP IT (do not uninstall). Ships from the Ascend channel.
pip install triton-ascend==3.2.0

# 3.4 vLLM (empty target) + vLLM-Ascend
git clone --branch v0.13.0 --depth 1 https://github.com/vllm-project/vllm.git
cd vllm        && VLLM_TARGET_DEVICE=empty pip install -v -e . && cd ..
git clone https://github.com/vllm-project/vllm-ascend.git
cd vllm-ascend && git checkout 894c0de3 && pip install -v -e . && cd ..

# Re-pin torch_npu to exactly 2.8.0.
pip install --force-reinstall --no-deps torch_npu==2.8.0

# 3.5 Megatron-LM + MindSpeed (for the optional Megatron backend)
git clone --branch core_v0.14.0 --depth 1 https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && pip install -v -e . && cd ..
git clone https://gitcode.com/Ascend/MindSpeed.git
cd MindSpeed   && git checkout 71b9e055 && pip install -e . && cd ..

# 3.6 mbridge
pip install git+https://github.com/ISEEKYAN/mbridge.git@4389fcc450c5f90f0cf22e9c77e3d49e2c643e24

# 3.7 Apex (Ascend)
git clone -b master https://gitcode.com/Ascend/apex.git
cd apex && bash scripts/build.sh --python=3.11
pip install apex/dist/apex-0.1+ascend-cp311-cp311-linux_aarch64.whl && cd ..
```

> **Note (changed from the old guide):** earlier setups *uninstalled* `triton` /
> `triton-ascend`. This recipe keeps **`triton-ascend==3.2.0`**, because DR.Kernel
> generates and evaluates Triton kernels on the NPU. Do not remove it, and do not
> install the upstream `triton` wheel alongside it.

---

## 4. CANN environment variables

Source these in every shell that runs training/rollout (the launcher scripts also
source them):

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # CANN 8.5.2
source /usr/local/Ascend/nnal/atb/set_env.sh
```

---

## 5. Install verl + the recipe

Install verl at the commit pinned in [`REQUIRED_VERL.txt`](REQUIRED_VERL.txt)
(upstream `verl-project/verl` @ `284d0388`, the squash-merge of PR 6020). Then place this recipe under `recipe/drkernel` so
it is importable as `recipe.drkernel.*`:

```bash
git clone https://github.com/verl-project/verl.git
cd verl
git checkout 284d03888e5161521e4b4c0bdf502ab7175ffda7
pip install -r requirements-npu.txt
pip install -v -e .
# copy this recipe in:  recipe/drkernel/  (matches recipe.drkernel.* imports)
```

---

## 6. Install the KernelGYM reward server

The DR.Kernel reward server is upstream
[hkust-nlp/KernelGYM](https://github.com/hkust-nlp/KernelGYM) with our NPU
adaptation applied on top. The recipe ships only that adaptation — a patch plus
new NPU toolkits — in `recipe/drkernel/kernelgym_npu/`. Set it up from the repo
root:

```bash
cd recipe/drkernel/kernelgym_npu
bash setup_kernelgym.sh                 # clone upstream KernelGYM (pinned commit), apply patch + overlay
cd KernelGYM && bash setup.sh           # install python deps + Redis
cd ../../../..
```

See [`recipe/drkernel/kernelgym_npu/README.md`](kernelgym_npu/README.md) for what
the patch/overlay contain and for deployment details.
