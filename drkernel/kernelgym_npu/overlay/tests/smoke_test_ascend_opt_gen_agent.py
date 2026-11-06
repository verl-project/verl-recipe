# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""Smoke test for the AscendOptGenAgent toolkit.

Exercises four scenarios against the live KernelGym server, each with
``toolkit="ascend_opt_gen_agent"`` so the dispatch path hits the new
methodology (allclose precision check, multi-shape via
``get_input_groups``, no triton-decoy detection).

Run inside the head-node docker container where the server is reachable
on the configured port. Adjust ``SERVER_URL`` if your deployment binds
to a different host/port (default 8002 matches tests/example1.py and
example2.py; the production server in this repo binds to 10907 — set
``KERNELGYM_URL=http://127.0.0.1:10907`` to switch).

Exits 0 if every scenario matches expectations, non-zero on any failure.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

import httpx

SERVER_URL = os.environ.get("KERNELGYM_URL", "http://127.0.0.1:8002")
DEVICE = os.environ.get("KERNELGYM_DEVICE", "npu:0")
REQUEST_TIMEOUT_S = float(os.environ.get("KERNELGYM_TIMEOUT_S", "600"))


# ---------------------------------------------------------------------------
# Test payloads
# ---------------------------------------------------------------------------

# A trivial reference: ReLU on a single tensor. Using torch ops on the impl
# side keeps the harness independent of any Triton-Ascend kernel compiling.
REF_RELU_SINGLE = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

def get_init_inputs():
    return []

def get_inputs():
    return [torch.randn(256, 256, device='npu')]
"""

# Same reference but with get_input_groups() — exercises the multi-shape
# path that is the AscendOptGenAgent methodology's headline feature.
REF_RELU_MULTI = """
import torch

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

def get_init_inputs():
    return []

def get_input_groups():
    return [
        [torch.randn(128, device='npu')],
        [torch.randn(64, 64, device='npu')],
        [torch.randn(32, 32, 4, device='npu')],
    ]
"""

# Known-good impl — calls torch.relu directly, must match exactly.
IMPL_RELU_GOOD = """
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)
"""

# Wrong impl — returns torch.tanh, which for any negative input gives a
# value in (-1, 0) while ReLU gives exactly 0. allclose must reject.
IMPL_WRONG = """
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)
"""

# Syntax-broken impl. Has a class ModelNew so validate_code() lets it
# through, but exec() blows up. Triggers the compiled=False branch in
# pipeline.py phase 2.
IMPL_BROKEN = """
import torch

class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x  # missing close-paren on purpose
"""


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


async def evaluate(task_id: str, reference_code: str, kernel_code: str) -> dict[str, Any]:
    timeout = httpx.Timeout(REQUEST_TIMEOUT_S)
    async with httpx.AsyncClient(timeout=timeout) as client:
        rsp = await client.post(
            f"{SERVER_URL}/evaluate",
            json={
                "task_id": task_id,
                "toolkit": "ascend_opt_gen_agent",
                "backend_adapter": "kernelbench",
                "backend": "triton",
                "reference_code": reference_code,
                "kernel_code": kernel_code,
                "entry_point": "Model",
                "device": DEVICE,
                "num_correct_trials": 1,
                "num_perf_trials": 20,
                "force_refresh": True,
            },
        )
        rsp.raise_for_status()
        return rsp.json()


# ---------------------------------------------------------------------------
# Assertions (one per scenario)
# ---------------------------------------------------------------------------


def _summarize(result: dict[str, Any]) -> str:
    return (
        f"status={result.get('status')!s} "
        f"compiled={result.get('compiled')!s} "
        f"correctness={result.get('correctness')!s} "
        f"kernel_runtime={result.get('kernel_runtime')!s} "
        f"speedup={result.get('speedup')!s}"
    )


def _methodology(result: dict[str, Any]) -> str:
    metadata = result.get("metadata") or {}
    return metadata.get("methodology", "<missing>")


def _ascend_meta(result: dict[str, Any]) -> dict[str, Any]:
    metadata = result.get("metadata") or {}
    return metadata.get("ascend_opt_gen_agent") or {}


def assert_case1(result: dict[str, Any]) -> None:
    """Single-shape happy path — compiled + correct + methodology tag set."""
    assert _methodology(result) == "ascend_opt_gen_agent", f"methodology tag missing or wrong: {_methodology(result)}"
    assert result.get("compiled") is True, f"expected compiled=True, got {_summarize(result)}"
    assert result.get("correctness") is True, f"expected correctness=True, got {_summarize(result)}"
    ascend = _ascend_meta(result)
    assert ascend.get("total_cases") == 1, f"expected 1 case, got {ascend}"
    assert ascend.get("passed_cases") == 1, f"expected 1 passed, got {ascend}"


def assert_case2(result: dict[str, Any]) -> None:
    """Multi-shape happy path — every shape passes."""
    assert _methodology(result) == "ascend_opt_gen_agent"
    assert result.get("compiled") is True, _summarize(result)
    assert result.get("correctness") is True, _summarize(result)
    ascend = _ascend_meta(result)
    assert ascend.get("total_cases") == 3, f"expected 3 cases, got {ascend}"
    assert ascend.get("passed_cases") == 3, f"expected all passed, got {ascend}"


def assert_case3(result: dict[str, Any]) -> None:
    """Wrong output — compiled but correctness must be False with failure detail."""
    assert _methodology(result) == "ascend_opt_gen_agent"
    assert result.get("compiled") is True, f"expected compiled=True, got {_summarize(result)}"
    assert result.get("correctness") is False, f"expected correctness=False, got {_summarize(result)}"
    ascend = _ascend_meta(result)
    failures = ascend.get("failures") or []
    assert len(failures) >= 1, f"expected at least one failure entry, got {ascend}"
    err_msg = failures[0].get("error_msg", "")
    assert "验证失败" in err_msg, f"expected allclose failure marker in error_msg, got: {err_msg[:200]}"


def assert_case4(result: dict[str, Any]) -> None:
    """Broken syntax — must be compiled=False, surface the error."""
    assert _methodology(result) == "ascend_opt_gen_agent"
    assert result.get("compiled") is False, f"expected compiled=False, got {_summarize(result)}"
    # correctness may be False or None; either is acceptable for a compile failure
    assert result.get("correctness") in (False, None), _summarize(result)
    err_msg = (result.get("error_message") or "").lower()
    assert err_msg, f"expected non-empty error_message, got {result!r}"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

CASES = [
    ("case1_single_shape_good", REF_RELU_SINGLE, IMPL_RELU_GOOD, assert_case1),
    ("case2_multi_shape_good", REF_RELU_MULTI, IMPL_RELU_GOOD, assert_case2),
    ("case3_wrong_output", REF_RELU_SINGLE, IMPL_WRONG, assert_case3),
    ("case4_broken_syntax", REF_RELU_SINGLE, IMPL_BROKEN, assert_case4),
]


async def main() -> int:
    print(f"[smoke] server={SERVER_URL} device={DEVICE}")
    failures = 0
    for name, ref, impl, check in CASES:
        print(f"\n[smoke] -- {name} --")
        try:
            result = await evaluate(name, ref, impl)
        except Exception as e:
            print(f"[smoke] {name} ERROR during POST: {type(e).__name__}: {e}")
            failures += 1
            continue

        print(f"[smoke] response: {_summarize(result)}")
        print(f"[smoke] methodology={_methodology(result)} ascend_meta={json.dumps(_ascend_meta(result))[:300]}")
        try:
            check(result)
            print(f"[smoke] {name} PASS")
        except AssertionError as e:
            print(f"[smoke] {name} FAIL: {e}")
            print(f"[smoke] full result: {json.dumps(result, indent=2)[:2000]}")
            failures += 1

    print(f"\n[smoke] done — {len(CASES) - failures}/{len(CASES)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
