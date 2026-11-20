# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""Precision-comparison primitives for the AscendOptGenAgent methodology.

This is a faithful port of the comparison logic in
``eval_AscendOptGenAgent_original/verify_latest.py`` (the upstream's
post-revision file). It differs from the original KernelBench-style
comparator (see ``toolkit/kernelbench/correctness.py``) in two ways:

* Pass criterion is the per-element ``allclose`` rule:
      ``|actual - golden| <= atol + rtol * |golden|``
  (must hold for every element — equivalent to ``torch.allclose``).
* Dtype-specific (rtol, atol) pairs come from the NPU-Benchmark table
  documented in the source: FP32 (2^-13, 1e-5), FP16 (2^-10, 1e-3),
  BF16 (2^-7, 1e-2).

The old MERE/MARE rule (which judged on aggregate mean + max relative
error) is gone — relative error is still computed here but only for
log-readability in failure messages, never for the pass/fail verdict.
"""

from __future__ import annotations

from typing import Any

import torch

# Tolerance table matches verify_latest.py exactly. Any unknown dtype
# falls back to the FP32 entry — this is also verify_latest.py's behavior.
_DEFAULT_TOL: dict[str, float] = {
    "rtol": 2 ** (-13),
    "atol": 1e-5,
}


def get_allclose_tolerance(data_type: Any) -> dict[str, float]:
    """Return the (rtol, atol) tolerance dict for ``data_type``.

    Accepts a ``torch.dtype`` or a string identifier — the string path is
    used when the surrounding code only has a dtype name (e.g. coming
    from JSON metadata). String keys are normalised by stripping a
    ``torch.`` prefix and lower-casing, so ``"torch.float16"`` and
    ``"float16"`` both resolve to the FP16 row.
    """
    if isinstance(data_type, str):
        key = data_type.lower().replace("torch.", "")
        str_to_tol: dict[str, dict[str, float]] = {
            "float32": {"rtol": 2 ** (-13), "atol": 1e-5},
            "float": {"rtol": 2 ** (-13), "atol": 1e-5},
            "float16": {"rtol": 2 ** (-10), "atol": 1e-3},
            "half": {"rtol": 2 ** (-10), "atol": 1e-3},
            "bfloat16": {"rtol": 2 ** (-7), "atol": 1e-2},
        }
        return str_to_tol.get(key, _DEFAULT_TOL)

    dtype_to_tol: dict[torch.dtype, dict[str, float]] = {
        torch.float32: {"rtol": 2 ** (-13), "atol": 1e-5},
        torch.float16: {"rtol": 2 ** (-10), "atol": 1e-3},
        torch.bfloat16: {"rtol": 2 ** (-7), "atol": 1e-2},
    }
    return dtype_to_tol.get(data_type, _DEFAULT_TOL)


def _check_accuracy_allclose(golden: torch.Tensor, actual: torch.Tensor, data_type: Any) -> None:
    """Raise AssertionError unless every element satisfies the allclose rule.

    ``golden`` / ``actual`` must be 1-D, finite, on CPU (the caller in
    :func:`compare` already filters NaN/Inf and flattens). ``data_type``
    is the dtype of the framework (golden) tensor and selects the
    (rtol, atol) pair. The diagnostic block in the failure message
    matches verify_latest.py's format byte-for-byte so existing
    log-parsing tools keep working.
    """
    golden_f = golden.float()
    actual_f = actual.float()

    if golden_f.shape != actual_f.shape:
        raise AssertionError(f"验证失败，输出形状不一致: golden={golden_f.shape}, actual={actual_f.shape}")

    numel = golden_f.numel()
    if numel == 0:
        return

    tol = get_allclose_tolerance(data_type)
    rtol = tol["rtol"]
    atol = tol["atol"]

    diff = (actual_f - golden_f).abs()
    golden_abs = golden_f.abs()

    allowed_error = atol + rtol * golden_abs
    close_mask = diff <= allowed_error
    allclose_ok = bool(close_mask.all().item())

    if allclose_ok:
        return

    failed_close_mask = ~close_mask
    failed_close_count = int(failed_close_mask.sum().item())
    pass_rate = 1.0 - failed_close_count / max(numel, 1)

    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()
    max_allowed_err = allowed_error.max().item()
    mean_allowed_err = allowed_error.mean().item()

    # Diagnostic relative error — NOT part of the pass/fail rule. The
    # denominator floor (atol/rtol) keeps the log readable near zero.
    rel_denom_floor = atol / rtol
    rel_denom = torch.clamp(golden_abs, min=rel_denom_floor)
    relative_error = diff / rel_denom
    max_rel_err = relative_error.max().item()
    mean_rel_err = relative_error.mean().item()

    failed_indices = torch.where(failed_close_mask)[0]
    num_failed_to_show = min(10, len(failed_indices))

    topk = min(10, numel)
    top_rel_values, top_rel_indices = torch.topk(relative_error, k=topk)

    error_msg = (
        "验证失败，输出不一致:\n"
        f"  dtype={data_type}\n"
        f"  numel={numel}\n"
        f"  allclose_ok={allclose_ok}\n"
        f"  pass_rate={pass_rate:.6%}\n"
        f"  failed_close_count={failed_close_count}/{numel}\n"
        "\n"
        "阈值配置:\n"
        f"  rtol={rtol:.12e}\n"
        f"  atol={atol:.12e}\n"
        f"  rel_denom_floor=atol/rtol={rel_denom_floor:.12e}  # 仅用于日志中的相对误差\n"
        "\n"
        "误差统计:\n"
        f"  max_abs_err={max_abs_err:.12e}\n"
        f"  mean_abs_err={mean_abs_err:.12e}\n"
        f"  max_rel_err={max_rel_err:.12e}  # 仅日志\n"
        f"  mean_rel_err={mean_rel_err:.12e}  # 仅日志\n"
        f"  max_allowed_err={max_allowed_err:.12e}\n"
        f"  mean_allowed_err={mean_allowed_err:.12e}\n"
    )

    if failed_close_count > 0:
        error_msg += f"\n前 {num_failed_to_show} 个 allclose 失败点:\n"
        for i in range(num_failed_to_show):
            idx = failed_indices[i].item()
            error_msg += (
                f"  位置[{idx}]: "
                f"golden={golden_f[idx].item():.12e}, "
                f"actual={actual_f[idx].item():.12e}, "
                f"abs_err={diff[idx].item():.12e}, "
                f"allowed={allowed_error[idx].item():.12e}, "
                f"rel_err={relative_error[idx].item():.12e}\n"
            )

    error_msg += f"\n相对误差最大的前 {topk} 个点，注意仅用于诊断，不参与判定:\n"
    for i in range(topk):
        idx = top_rel_indices[i].item()
        error_msg += (
            f"  位置[{idx}]: "
            f"golden={golden_f[idx].item():.12e}, "
            f"actual={actual_f[idx].item():.12e}, "
            f"abs_err={diff[idx].item():.12e}, "
            f"allowed={allowed_error[idx].item():.12e}, "
            f"rel_err={relative_error[idx].item():.12e}\n"
        )

    raise AssertionError(error_msg)


def compare(fw_out: torch.Tensor, impl_out: Any, data_type: Any) -> None:
    """Compare framework and impl outputs using the allclose rule.

    Mirrors ``verify_latest.py::compare``:
      1. Move both tensors to CPU and flatten.
      2. Reject any shape mismatch.
      3. Reject mismatched NaN/Inf masks (and Inf-sign mismatches).
      4. On the finite subset, run the per-element allclose check via
         :func:`_check_accuracy_allclose`.

    Raises ``AssertionError`` on any failure; returns silently on success.
    """
    fw_flat = fw_out.flatten().detach().cpu()
    impl_flat = impl_out.flatten()
    if isinstance(impl_flat, torch.Tensor):
        impl_flat = impl_flat.detach().cpu()
    else:
        impl_flat = torch.tensor(impl_flat, dtype=fw_flat.dtype)

    size = fw_flat.numel()

    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(f"验证失败，输出形状不一致: framework={fw_flat.shape}, impl={impl_flat.shape}")

    fw_nan_mask = torch.isnan(fw_flat)
    impl_nan_mask = torch.isnan(impl_flat)
    if not torch.equal(fw_nan_mask, impl_nan_mask):
        fw_nan_count = fw_nan_mask.sum().item()
        impl_nan_count = impl_nan_mask.sum().item()
        raise AssertionError(
            f"验证失败，NaN 位置不匹配: Framework={fw_nan_count}/{size}, Implementation={impl_nan_count}/{size}"
        )

    fw_inf_mask = torch.isinf(fw_flat)
    impl_inf_mask = torch.isinf(impl_flat)
    if not torch.equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = fw_inf_mask.sum().item()
        impl_inf_count = impl_inf_mask.sum().item()
        raise AssertionError(
            f"验证失败，Inf 位置不匹配: Framework={fw_inf_count}/{size}, Implementation={impl_inf_count}/{size}"
        )
    if fw_inf_mask.any():
        if not torch.equal(
            torch.sign(fw_flat[fw_inf_mask]),
            torch.sign(impl_flat[impl_inf_mask]),
        ):
            raise AssertionError("验证失败，Inf 符号不匹配")

    finite_mask = torch.isfinite(fw_flat) & torch.isfinite(impl_flat)
    finite_count = finite_mask.sum().item()
    if finite_count == 0:
        print("警告: 所有值都是非有限值，跳过精度检查")
        return

    fw_finite = fw_flat[finite_mask]
    impl_finite = impl_flat[finite_mask]

    if fw_finite.dtype == torch.bool:
        if not torch.equal(fw_finite, impl_finite):
            raise AssertionError(f"验证失败，布尔值不匹配: dtype={data_type}")
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)

    _check_accuracy_allclose(fw_finite, impl_finite, data_type)
