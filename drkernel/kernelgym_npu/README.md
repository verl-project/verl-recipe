# KernelGYM on Ascend NPU (patch + overlay)

The DR.Kernel recipe uses [**KernelGYM**](https://github.com/hkust-nlp/KernelGYM)
as its kernel-evaluation reward server. Rather than vendoring the whole KernelGYM
codebase, this directory ships **only our NPU adaptation** on top of the upstream
repo:

| Path | Content |
|---|---|
| `patches/kernelgym-npu-support.patch` | Modifications to existing upstream files: NPU device handling (`cuda` → `npu`, ACL runtime), triton-ascend detection/compilation/profiling, worker & monitoring adaptations, CANN-container-friendly `setup.sh`/`requirements.txt` |
| `overlay/` | Files **new** to this port, copied verbatim into a KernelGYM checkout: the `sandbox_v3` NPU-kernel evaluation toolkit, the `ascend_opt_gen_agent` toolkit, smoke/validation tests, and an `.env.example` template |
| `setup_kernelgym.sh` | One-shot setup: clone upstream at the pinned commit, apply the patch, copy the overlay |

Pinned upstream commit:
[`3a84417f8c0efaadb215ef638b37d12e71ed20f3`](https://github.com/hkust-nlp/KernelGYM/commit/3a84417f8c0efaadb215ef638b37d12e71ed20f3).

> **Note:** the upstream KernelGYM repo also contains a `drkernel/` training
> implementation. It is **not used** here — this recipe ships its own verl-based
> port (the rest of `recipe/drkernel/`). Only KernelGYM's server/evaluation stack
> is used, patched for NPU.

## Setup

```bash
cd recipe/drkernel/kernelgym_npu
bash setup_kernelgym.sh            # clones into ./KernelGYM (gitignored), patches, overlays
cd KernelGYM
bash setup.sh                      # python deps + redis
cp .env.example .env               # then edit API_HOST / GPU_DEVICES
bash start_all_with_monitor.sh     # start server + workers + monitor
```

To place the checkout elsewhere, pass a target dir
(`bash setup_kernelgym.sh /path/to/KernelGYM`) and point the training scripts at
it via `KERNELGYM_DIR=/path/to/KernelGYM`.

For server architecture, API reference, and deployment details see the README of
the patched checkout (`KernelGYM/README.md`).
