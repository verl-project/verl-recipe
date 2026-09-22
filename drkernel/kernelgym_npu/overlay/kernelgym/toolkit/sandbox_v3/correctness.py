# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""KernelBench correctness helpers (toolkit layer)."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from kernelgym.toolkit.kernelbench.exec_types import (
    KernelExecResult,
    get_error_name,
    set_seed,
)


def get_limit(data_type):
    """根据数据类型获取精度阈值"""
    import torch

    if data_type == torch.float16:
        return 0.004
    elif data_type == torch.bfloat16:
        return 0.03
    elif data_type == torch.int8:
        return 0.01
    else:
        return 0.02


def is_env_true(env_name: str, default: bool = False) -> bool:
    """
    通用判断环境变量是否为真值
    :param env_name: 环境变量名称
    :param default: 环境变量不存在时的默认值（默认False）
    :return: 布尔值结果
    """
    # 读取并清洗环境变量
    import os

    value = os.getenv(env_name, "").strip().lower()
    # 匹配真值
    true_values = {"true", "1", "yes", "on", "y"}
    return value in true_values if value else default


def compare(fw_out, impl_out, limit, data_type):
    """对比框架输出和实现输出"""
    import torch

    fw_flat = fw_out.flatten().detach().cpu()
    impl_flat = impl_out.flatten()
    if isinstance(impl_flat, torch.Tensor):
        impl_flat = impl_flat.detach().cpu()
    else:
        impl_flat = torch.tensor(impl_flat, dtype=fw_flat.dtype)

    size = fw_flat.numel()

    if fw_flat.shape != impl_flat.shape:
        raise AssertionError(
            f"Validation failed: output shape mismatch, framework={fw_flat.shape}, impl={impl_flat.shape}"
        )

    fw_nan_mask = torch.isnan(fw_flat)
    impl_nan_mask = torch.isnan(impl_flat)
    if not torch.equal(fw_nan_mask, impl_nan_mask):
        fw_nan_count = fw_nan_mask.sum().item()
        impl_nan_count = impl_nan_mask.sum().item()
        raise AssertionError(
            f"Validation failed: NaN position mismatch, Framework={fw_nan_count}/{size}, "
            f"Implementation={impl_nan_count}/{size}"
        )

    fw_inf_mask = torch.isinf(fw_flat)
    impl_inf_mask = torch.isinf(impl_flat)
    if not torch.equal(fw_inf_mask, impl_inf_mask):
        fw_inf_count = fw_inf_mask.sum().item()
        impl_inf_count = impl_inf_mask.sum().item()
        raise AssertionError(
            f"Validation failed: Inf position mismatch, Framework={fw_inf_count}/{size}, "
            f"Implementation={impl_inf_count}/{size}"
        )
    if fw_inf_mask.any():
        if not torch.equal(
            torch.sign(fw_flat[fw_inf_mask]),
            torch.sign(impl_flat[impl_inf_mask]),
        ):
            raise AssertionError("Validation failed: Inf sign mismatch")

    finite_mask = torch.isfinite(fw_flat) & torch.isfinite(impl_flat)
    finite_count = finite_mask.sum().item()
    if finite_count == 0:
        print("Warning: All values are non-finite, skipping precision check")
        return

    fw_finite = fw_flat[finite_mask]
    impl_finite = impl_flat[finite_mask]

    if fw_finite.dtype == torch.bool:
        if not torch.equal(fw_finite, impl_finite):
            raise AssertionError(f"Validation failed: boolean value mismatch, dtype={data_type}")
        return

    if impl_finite.dtype != fw_finite.dtype:
        impl_finite = impl_finite.to(fw_finite.dtype)

    # sandbox_v3 hard-wires the NPU-kernel path (ORIGIN_MODE=off): always use
    # the allclose-style precision check.
    _check_accuracy_allclose(fw_finite, impl_finite, data_type)


def register_and_format_exception(
    exception_type: str,
    exception_msg: Exception | str,
    metadata: dict,
    verbose: bool = False,
    truncate: bool = False,
    max_length: int = 200,
):
    if verbose:
        print(f"[Exception {exception_type}] {str(exception_msg)} ")

    metadata[exception_type] = exception_msg
    return metadata


def get_allclose_tolerance(data_type):
    """根据数据类型获取 allclose 样式精度阈值。

    判定标准：
        abs(actual - golden) <= atol + rtol * abs(golden)

    当前采用阈值：
        FLOAT32:
            rtol = 2^{-13} = 1.220703125e-4
            atol = 1e-5

        FLOAT16:
            rtol = 2^{-10} = 9.765625e-4
            atol = 1e-3

        BFLOAT16:
            rtol = 2^{-7} = 7.8125e-3
            atol = 1e-2
    """
    import torch

    default_tol = {
        "rtol": 2 ** (-13),
        "atol": 1e-5,
    }

    if isinstance(data_type, str):
        key = data_type.lower().replace("torch.", "")
        str_to_tol = {
            "float32": {
                "rtol": 2 ** (-13),
                "atol": 1e-5,
            },
            "float": {
                "rtol": 2 ** (-13),
                "atol": 1e-5,
            },
            "float16": {
                "rtol": 2 ** (-10),
                "atol": 1e-3,
            },
            "half": {
                "rtol": 2 ** (-10),
                "atol": 1e-3,
            },
            "bfloat16": {
                "rtol": 2 ** (-7),
                "atol": 1e-2,
            },
        }
        return str_to_tol.get(key, default_tol)

    dtype_to_tol = {
        torch.float32: {
            "rtol": 2 ** (-13),
            "atol": 1e-5,
        },
        torch.float16: {
            "rtol": 2 ** (-10),
            "atol": 1e-3,
        },
        torch.bfloat16: {
            "rtol": 2 ** (-7),
            "atol": 1e-2,
        },
    }

    return dtype_to_tol.get(data_type, default_tol)


def _check_accuracy_allclose(golden, actual, data_type):
    """执行 allclose 精度验证。

    判定标准：
        abs(actual - golden) <= atol + rtol * abs(golden)

    Args:
        golden: 参考输出，通常是 PyTorch framework 输出
        actual: 被测实现输出，通常是 Triton-Ascend 输出
        data_type: 数据类型，用于获取对应阈值

    Raises:
        AssertionError: 当精度验证未通过时
    """
    import torch

    golden_f = golden.float()
    actual_f = actual.float()

    if golden_f.shape != actual_f.shape:
        raise AssertionError(
            f"Validation failed: output shape mismatch, golden={golden_f.shape}, actual={actual_f.shape}"
        )

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

    if not allclose_ok:
        failed_close_mask = ~close_mask
        failed_close_count = int(failed_close_mask.sum().item())
        pass_rate = 1.0 - failed_close_count / max(numel, 1)

        max_abs_err = diff.max().item()
        mean_abs_err = diff.mean().item()
        max_allowed_err = allowed_error.max().item()
        mean_allowed_err = allowed_error.mean().item()

        # 为了日志可读，计算一个诊断用相对误差。
        # 注意：该 relative_error 只用于错误信息展示，不参与判定。
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
            "Validation failed: output mismatch:\n"
            f"  dtype={data_type}\n"
            f"  numel={numel}\n"
            f"  allclose_ok={allclose_ok}\n"
            f"  pass_rate={pass_rate:.6%}\n"
            f"  failed_close_count={failed_close_count}/{numel}\n"
            "\n"
            "Threshold Configuration:\n"
            f"  rtol={rtol:.12e}\n"
            f"  atol={atol:.12e}\n"
            f"  rel_denom_floor=atol/rtol={rel_denom_floor:.12e}  # For relative error in logs only\n"
            "\n"
            "Error Statistics:\n"
            f"  max_abs_err={max_abs_err:.12e}\n"
            f"  mean_abs_err={mean_abs_err:.12e}\n"
            f"  max_rel_err={max_rel_err:.12e}  # For logs only\n"
            f"  mean_rel_err={mean_rel_err:.12e}  # For logs only\n"
            f"  max_allowed_err={max_allowed_err:.12e}\n"
            f"  mean_allowed_err={mean_allowed_err:.12e}\n"
        )

        if failed_close_count > 0:
            error_msg += f"\nFirst {num_failed_to_show} allclose failure points:\n"
            for i in range(num_failed_to_show):
                idx = failed_indices[i].item()
                error_msg += (
                    f"  Position [{idx}]: "
                    f"golden={golden_f[idx].item():.12e}, "
                    f"actual={actual_f[idx].item():.12e}, "
                    f"abs_err={diff[idx].item():.12e}, "
                    f"allowed={allowed_error[idx].item():.12e}, "
                    f"rel_err={relative_error[idx].item():.12e}\n"
                )
        error_msg += (
            f"\nTop {topk} points with maximum relative error (note: for diagnosis only, not used for judgment):\n"
        )
        for i in range(topk):
            idx = top_rel_indices[i].item()
            error_msg += (
                f"  Position [{idx}]: "
                f"golden={golden_f[idx].item():.12e}, "
                f"actual={actual_f[idx].item():.12e}, "
                f"abs_err={diff[idx].item():.12e}, "
                f"allowed={allowed_error[idx].item():.12e}, "
                f"rel_err={relative_error[idx].item():.12e}\n"
            )
        raise AssertionError(error_msg)


def run_and_check_correctness(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    device: Any = None,
) -> KernelExecResult:
    pass_count = 0

    torch.manual_seed(seed)
    correctness_trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(num_correct_trials)]

    with torch.no_grad():
        for trial in range(num_correct_trials):
            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            inputs = get_inputs_fn()
            inputs = [x.npu(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]

            set_seed(trial_seed)
            model = original_model_instance.npu(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.npu(device=device)
            if verbose:
                print(f"device: {device}")
                print(f"inputs: {inputs}")

            output = model(*inputs)
            torch.npu.synchronize(device=device)

            try:
                output_new = model_new(*inputs)
                torch.npu.synchronize(device=device)

                # 标准化输出格式
                if not isinstance(output, (list, tuple)):
                    output = [output]
                if not isinstance(output_new, (list, tuple)):
                    output_new = [output_new]

                # 验证输出数量
                if len(output) != len(output_new):
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Inconsistent number of outputs from the use case: framework={len(output)}, impl={len(output_new)}",  # noqa: E501
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

                # 比较每个输出
                for i, (fw_out, impl_out) in enumerate(zip(output, output_new, strict=False)):
                    if fw_out is None or impl_out is None:
                        metadata = register_and_format_exception(
                            "correctness_issue",
                            f"Output {i} of the test case is None: framework={len(output)}, impl={len(output_new)}",
                            metadata,
                        )
                        metadata["correctness_issue_name"] = "correctness_issue"
                        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

                    if isinstance(fw_out, torch.Tensor) and isinstance(impl_out, torch.Tensor):
                        try:
                            data_type = fw_out.dtype
                            limit = get_limit(data_type)
                            compare(fw_out, impl_out, limit, data_type)
                        except AssertionError:
                            max_diff = torch.max(torch.abs(fw_out - impl_out)).item()
                            avg_diff = torch.mean(torch.abs(fw_out - impl_out)).item()
                            metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                            metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                            metadata["correctness_issue"] = "Output mismatch"
                            return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
                pass_count += 1
            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")

                metadata = register_and_format_exception("runtime_error", e, metadata, truncate=False)
                metadata["runtime_error_name"] = get_error_name(e)
                return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

    if verbose:
        print(f"[Eval] Pass count: {pass_count}, num_correct_trials: {num_correct_trials}")

    metadata["correctness_trials"] = f"({pass_count} / {num_correct_trials})"

    if pass_count == num_correct_trials:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def run_and_check_correctness_fornpukernel(
    original_model_instance: nn.Module,
    new_model_instance: nn.Module,
    get_inputs_fn: callable,
    metadata: dict,
    num_correct_trials: int,
    verbose: bool = False,
    seed: int = 42,
    device: Any = None,
) -> KernelExecResult:
    pass_count = 0

    torch.manual_seed(seed)
    print("torch.no_grad() out")
    all_input_groups = get_inputs_fn()  # 取一次
    total_cases = len(all_input_groups)  # 总用例数
    correctness_trial_seeds = [torch.randint(0, 2**32 - 1, (1,)).item() for _ in range(total_cases)]
    print(f"torch.no_grad() out1 total_cases:{total_cases}")
    with torch.no_grad():
        print("torch.no_grad() in", flush=True)
        for trial in range(total_cases):
            trial_seed = correctness_trial_seeds[trial]
            if verbose:
                print(f"[Eval] Generating Random Input with seed {trial_seed}")

            set_seed(trial_seed)
            # 每个trial取【一组输入 [x,y]】，而不是全量用例
            inputs = all_input_groups[trial]
            inputs = [x.npu(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]

            set_seed(trial_seed)
            model = original_model_instance.npu(device=device)

            set_seed(trial_seed)
            model_new = new_model_instance.npu(device=device)
            if verbose:
                print(f"device: {device}")
                print(f"inputs: {inputs}")

            output = model(*inputs)
            torch.npu.synchronize(device=device)

            try:
                output_new = model_new(*inputs)
                torch.npu.synchronize(device=device)

                # 标准化输出格式
                if not isinstance(output, (list, tuple)):
                    output = [output]
                if not isinstance(output_new, (list, tuple)):
                    output_new = [output_new]

                # 验证输出数量
                if len(output) != len(output_new):
                    metadata = register_and_format_exception(
                        "correctness_issue",
                        f"Inconsistent number of outputs from the use case: framework={len(output)}, impl={len(output_new)}",  # noqa: E501
                        metadata,
                    )
                    metadata["correctness_issue_name"] = "correctness_issue"
                    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

                # 比较每个输出
                for i, (fw_out, impl_out) in enumerate(zip(output, output_new, strict=False)):
                    if fw_out is None or impl_out is None:
                        metadata = register_and_format_exception(
                            "correctness_issue",
                            f"Output {i} of the test case is None: framework={len(output)}, impl={len(output_new)}",
                            metadata,
                        )
                        metadata["correctness_issue_name"] = "correctness_issue"
                        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

                    if isinstance(fw_out, torch.Tensor) and isinstance(impl_out, torch.Tensor):
                        try:
                            data_type = fw_out.dtype
                            limit = get_limit(data_type)
                            compare(fw_out, impl_out, limit, data_type)
                        except AssertionError:
                            max_diff = torch.max(torch.abs(output - output_new)).item()
                            avg_diff = torch.mean(torch.abs(output - output_new)).item()
                            metadata.setdefault("max_difference", []).append(f"{max_diff:.6f}")
                            metadata.setdefault("avg_difference", []).append(f"{avg_diff:.6f}")
                            metadata["correctness_issue"] = "Output mismatch"
                            return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
                pass_count += 1
            except Exception as e:
                print("[Error] Exception happens during correctness check")
                print(f"Error in launching kernel for ModelNew: {e}")

                metadata = register_and_format_exception("runtime_error", e, metadata, truncate=False)
                metadata["runtime_error_name"] = get_error_name(e)
                return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

    if verbose:
        print(f"[Eval] Pass count: {pass_count}, total_cases: {total_cases}")

    metadata["correctness_trials"] = f"({pass_count} / {total_cases})"
    print(f"pass_count: {pass_count}, total_cases: {total_cases},", flush=True)
    if pass_count == total_cases:
        return KernelExecResult(compiled=True, correctness=True, metadata=metadata)
    return KernelExecResult(compiled=True, correctness=False, metadata=metadata)
