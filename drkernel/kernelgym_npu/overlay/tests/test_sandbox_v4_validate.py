# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""Standalone AST unit test for the sandbox_v4 decoy validator.

Torch-free: it imports only the two ``validate_triton_impl`` modules (pure
``ast`` static analysis), bypassing the heavy ``kernelgym/__init__.py`` (which
imports the server + torch) via lightweight package shims. So it runs on a dev
box without torch/torch_npu.

What it pins down:
  * sandbox_v4 ACCEPTS a genuine Triton kernel whose forward() also calls the
    vendor/extern ops (mm, F.conv2d, F.scaled_dot_product_attention,
    torch.linalg.*, torch.fft.*, and the ``@`` operator).
  * sandbox_v4 ACCEPTS whitelisted submodule calls — self.conv(x),
    self.conv_transpose(x), self.fc(x) — when the attribute is bound to a
    Conv*/Linear/Attention module in __init__ (the idiomatic form the model
    emits, mirroring the reference Model).
  * sandbox_v4 still REJECTS partial-PyTorch (softmax / .relu() / torch.add),
    zero-Triton implementations, and non-whitelisted submodule calls (a
    norm/pool/activation layer, or self.<x>(x) not bound to a whitelisted
    layer in __init__).
  * sandbox_v3 is unchanged — it still rejects matmul/conv and ALL self.<x>(x)
    (proving the injected-validator seam left v3 behavior intact).

Exit code: 0 = all checks passed, 1 = at least one failed.

    # from a patched KernelGYM checkout (recipe/drkernel/kernelgym_npu/setup_kernelgym.sh):
    python tests/test_sandbox_v4_validate.py
"""

from __future__ import annotations

import importlib
import pathlib
import sys
import types

# ---------------------------------------------------------------------------
# Bootstrap: register lightweight package shims so the absolute imports inside
# the validator modules resolve WITHOUT executing kernelgym/__init__.py (which
# pulls in the server and torch). Each shim is a bare module object with
# __path__ set, so Python finds the leaf submodule by file but never runs the
# package __init__.
# ---------------------------------------------------------------------------
_KG_ROOT = pathlib.Path(__file__).resolve().parents[1] / "kernelgym"


def _shim(name: str, path: pathlib.Path) -> None:
    if name in sys.modules:
        return
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod


_shim("kernelgym", _KG_ROOT)
_shim("kernelgym.toolkit", _KG_ROOT / "toolkit")
_shim("kernelgym.toolkit.sandbox_v3", _KG_ROOT / "toolkit" / "sandbox_v3")
_shim("kernelgym.toolkit.sandbox_v4", _KG_ROOT / "toolkit" / "sandbox_v4")

v3 = importlib.import_module("kernelgym.toolkit.sandbox_v3.validate_triton_impl")
v4 = importlib.import_module("kernelgym.toolkit.sandbox_v4.validate_triton_impl")


# ---------------------------------------------------------------------------
# Source builders
# ---------------------------------------------------------------------------

# A real @triton.jit kernel (uses tl.*) launched exactly once. Appended to a
# forward() body that has produced a tensor named ``t``.
_LAUNCH = (
    "        y = torch.empty_like(t)\n"
    "        n = t.numel()\n"
    "        _k[(triton.cdiv(n, 1024),)](t, y, n, BLOCK=1024)\n"
    "        return y\n"
)


def make_src(forward_body: str, init_body: str = "        pass") -> str:
    return (
        "import torch\n"
        "import torch.nn as nn\n"
        "import torch.nn.functional as F\n"
        "import triton\n"
        "import triton.language as tl\n"
        "\n"
        "@triton.jit\n"
        "def _k(x_ptr, y_ptr, n, BLOCK: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offs = pid * BLOCK + tl.arange(0, BLOCK)\n"
        "    mask = offs < n\n"
        "    val = tl.load(x_ptr + offs, mask=mask)\n"
        "    tl.store(y_ptr + offs, val, mask=mask)\n"
        "\n"
        "class ModelNew(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        f"{init_body}\n"
        "    def forward(self, x):\n"
        f"{forward_body}"
    )


# --- ACCEPT cases: extern op as the hard sub-op + one Triton kernel ---------
SRC_CONV = make_src(
    "        t = F.conv2d(x, self.w)\n" + _LAUNCH, init_body="        self.w = nn.Parameter(torch.randn(8, 4, 3, 3))"
)
SRC_MM_AT = make_src("        t = torch.mm(x, x)\n        t = x @ x\n" + _LAUNCH)
SRC_SDPA = make_src("        t = F.scaled_dot_product_attention(x, x, x)\n" + _LAUNCH)
SRC_LINALG_FFT = make_src("        t = torch.linalg.inv(x)\n        t = torch.fft.rfft(t)\n" + _LAUNCH)

# --- ACCEPT cases: whitelisted nn.Module submodule call + one Triton kernel --
# self.<layer>(x) where <layer> is bound to a whitelisted module type in
# __init__ — the idiomatic form the model emits (e.g. self.conv_transpose(x),
# the exact pattern that was being mislabeled as decoy).
SRC_SELF_CONV = make_src("        t = self.conv(x)\n" + _LAUNCH, init_body="        self.conv = nn.Conv2d(4, 8, 3)")
SRC_SELF_CONVT = make_src(
    "        t = self.conv_transpose(x)\n" + _LAUNCH,
    init_body="        self.conv_transpose = nn.ConvTranspose3d(64, 128, 3, stride=2, padding=1)",
)
SRC_SELF_CONV_TORCHNN = make_src(
    "        t = self.conv(x)\n" + _LAUNCH, init_body="        self.conv = torch.nn.Conv2d(4, 8, 3)"
)
SRC_SELF_LINEAR = make_src("        t = self.fc(x)\n" + _LAUNCH, init_body="        self.fc = nn.Linear(16, 32)")
# Attribute name collides with a FORBIDDEN_TENSOR_METHODS entry ("linear"); the
# self.<name>(x) allowlist must win over the forbidden-method check.
SRC_SELF_LINEAR_NAMED = make_src(
    "        t = self.linear(x)\n" + _LAUNCH, init_body="        self.linear = nn.Linear(16, 32)"
)

# --- REJECT cases -----------------------------------------------------------
SRC_SOFTMAX = make_src("        t = torch.softmax(x, dim=-1)\n" + _LAUNCH)  # Type 3
SRC_RELU_METHOD = make_src("        t = x.relu()\n" + _LAUNCH)  # Type 3
SRC_ADD = make_src("        t = torch.add(x, x)\n" + _LAUNCH)  # Type 3
SRC_SELF_NORM = make_src(
    "        t = self.gn(x)\n" + _LAUNCH,  # Type 3
    init_body="        self.gn = nn.GroupNorm(4, 8)",
)
SRC_SELF_UNDECLARED = make_src("        t = self.conv(x)\n" + _LAUNCH)  # Type 3 (no __init__ binding)
SRC_NO_TRITON = (  # Type 1
    "import torch\n"
    "import torch.nn as nn\n"
    "class ModelNew(nn.Module):\n"
    "    def forward(self, x):\n"
    "        return torch.mm(x, x)\n"
)


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
_FAILURES: list[str] = []


def _check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        _FAILURES.append(name)


def _expect_valid(name: str, src: str) -> None:
    r = v4.validate(src)
    _check(name, r["valid"] is True, f"regression_type={r['regression_type']}")


def _expect_decoy(name: str, src: str, rtype: int) -> None:
    r = v4.validate(src)
    calls = [v.get("call") for v in r["checks"]["no_forbidden_torch_ops"]["violations"]]
    _check(
        name,
        (not r["valid"]) and r["regression_type"] == rtype,
        f"valid={r['valid']} rtype={r['regression_type']} violations={calls}",
    )


def main() -> int:
    print("sandbox_v4 — ACCEPT (extern op + real Triton kernel):")
    _expect_valid("conv2d + triton relu", SRC_CONV)
    _expect_valid("mm + @ operator", SRC_MM_AT)
    _expect_valid("scaled_dot_product_attention", SRC_SDPA)
    _expect_valid("torch.linalg.inv + torch.fft.rfft", SRC_LINALG_FFT)

    print("sandbox_v4 — ACCEPT (whitelisted submodule call + real Triton kernel):")
    _expect_valid("self.conv(x) [nn.Conv2d]", SRC_SELF_CONV)
    _expect_valid("self.conv_transpose(x) [nn.ConvTranspose3d]", SRC_SELF_CONVT)
    _expect_valid("self.conv(x) [torch.nn.Conv2d]", SRC_SELF_CONV_TORCHNN)
    _expect_valid("self.fc(x) [nn.Linear]", SRC_SELF_LINEAR)
    _expect_valid("self.linear(x) [nn.Linear; name == forbidden method]", SRC_SELF_LINEAR_NAMED)

    print("sandbox_v4 — REJECT (still forbidden):")
    _expect_decoy("torch.softmax outside kernel (Type 3)", SRC_SOFTMAX, 3)
    _expect_decoy("x.relu() method (Type 3)", SRC_RELU_METHOD, 3)
    _expect_decoy("torch.add elementwise (Type 3)", SRC_ADD, 3)
    _expect_decoy("self.gn(x) GroupNorm submodule (Type 3)", SRC_SELF_NORM, 3)
    _expect_decoy("self.conv(x) undeclared in __init__ (Type 3)", SRC_SELF_UNDECLARED, 3)
    _expect_decoy("pure torch.mm, no @triton.jit (Type 1)", SRC_NO_TRITON, 1)

    print("sandbox_v3 — regression (extern ops + all submodule calls rejected, seam intact):")
    _check("v3 rejects F.conv2d", v3.validate(SRC_CONV)["valid"] is False)
    _check("v3 rejects torch.mm", v3.validate(SRC_MM_AT)["valid"] is False)
    _check("v3 rejects self.conv(x)", v3.validate(SRC_SELF_CONV)["valid"] is False)
    # And the contrast that motivates v4: v3 rejects what v4 accepts.
    _check("v4 accepts conv that v3 rejects", v4.validate(SRC_CONV)["valid"] and not v3.validate(SRC_CONV)["valid"])
    _check(
        "v4 accepts self.conv(x) that v3 rejects",
        v4.validate(SRC_SELF_CONV)["valid"] and not v3.validate(SRC_SELF_CONV)["valid"],
    )

    print()
    if _FAILURES:
        print(f"RESULT: {len(_FAILURES)} check(s) FAILED: {_FAILURES}")
        return 1
    print("RESULT: all checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
