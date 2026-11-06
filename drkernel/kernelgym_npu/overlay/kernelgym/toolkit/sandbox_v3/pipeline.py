# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""KernelBench evaluation pipeline (task-level, toolkit layer)."""

from __future__ import annotations

import os
from typing import Any, Optional

import torch
from kernelgym.toolkit.kernelbench import triton_detect as detect
from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult, get_error_name, set_seed
from kernelgym.toolkit.sandbox_v3.correctness import run_and_check_correctness_fornpukernel
from kernelgym.toolkit.sandbox_v3.loading import (
    graceful_eval_cleanup,
    load_custom_model,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
)
from kernelgym.toolkit.sandbox_v3.profiling import measure_single
from kernelgym.toolkit.sandbox_v3.validate_triton_impl import fillKernelExecResult


def _run_correctness_step(
    original_model,
    custom_model,
    get_inputs,
    metadata: dict[str, Any],
    num_correct_trials: int,
    verbose: bool,
    seed_num: int,
    device: torch.device | int,
) -> KernelExecResult:
    if verbose:
        print("[Eval] Checking Correctness")
    try:
        # sandbox_v3 hard-wires the NPU-kernel path (NPUKERNEL_MODE=on): always
        # use the multi-shape correctness check.
        return run_and_check_correctness_fornpukernel(
            original_model,
            custom_model,
            get_inputs,
            metadata=metadata,
            num_correct_trials=num_correct_trials,
            verbose=verbose,
            seed=seed_num,
            device=device,
        )
    except Exception as e:
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)


def _run_triton_detection_step(
    *,
    enable_triton_detection: bool,
    is_triton: bool,
    kernel_exec_result: KernelExecResult,
    custom_model,
    custom_model_src: str,
    get_inputs,
    metadata: dict[str, Any],
    seed_num: int,
    device: torch.device | int,
    verbose: bool,
    backend: str,
    decoy_validator=fillKernelExecResult,
):
    if not enable_triton_detection:
        return False
    try:
        print("Begin Triton usage detection")
        if kernel_exec_result and kernel_exec_result.correctness:
            torch.npu.synchronize(device=device)
            set_seed(seed_num)
            inputs = get_inputs()[0]  # first input group (timing/triton use one shape)
            inputs = [x.npu(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
            model_new = custom_model.npu(device=device)
            torch.npu.synchronize(device=device)

            used, matches = detect.detect_triton_usage_for_module(
                model_new,
                *inputs,
                warmup=1,
                steps=1,
                use_npu=True,
                return_matches=True,
            )
            metadata["triton_profiler_used"] = used
            metadata["triton_profiler_matches"] = matches
            print(f"Triton usage detection result: {used}")
            print(f"Triton usage detection matches: {matches}")
            if not used and is_triton:
                print("[Eval] Backend is 'triton' but no Triton usage detected, marking as decoy")
                kernel_exec_result.decoy_kernel = True
                kernel_exec_result.runtime = -1.0
                return True
                if not used:
                    print(
                        f"[Eval] No Triton usage detected, but backend is '{backend}', continuing to performance measurement"  # noqa: E501
                    )
        # sandbox_v3 hard-wires the NPU-kernel path (ORIGIN_MODE=off): always
        # run the AST-based Triton-implementation check. ``decoy_validator``
        # defaults to sandbox_v3's ``fillKernelExecResult``; sandbox_v4 injects
        # its own validator (relaxed extern-op whitelist) via this seam.
        decoy_validator(kernel_exec_result, custom_model_src)
    except Exception as e:
        if verbose:
            print(f"[Eval] Error in Triton usage detection: {e}")
        metadata["error_in_triton_detection"] = e
    return False


def _run_performance_step(
    *,
    kernel_exec_result: KernelExecResult,
    custom_model,
    get_inputs,
    metadata: dict[str, Any],
    num_perf_trials: int,
    verbose: bool,
    seed_num: int,
    device: torch.device | int,
    enable_profiling: bool,
):
    try:
        if kernel_exec_result and kernel_exec_result.correctness:
            if verbose:
                print("[Eval] Measuring Performance as Sample is Correct")

            torch.npu.synchronize(device=device)
            set_seed(seed_num)
            inputs = get_inputs()[0]  # first input group (timing/triton use one shape)
            inputs = [x.npu(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
            model_new = custom_model.npu(device=device)
            torch.npu.synchronize(device=device)

            impl_operators, impl_latency_ms, impl_peak_memory = measure_single(
                model_new,
                inputs,
                warmup=3,
                repeats=num_perf_trials,
                profile_name="model_new_profile_case",
                device=device,
            )

            kernel_exec_result.runtime = impl_latency_ms
            kernel_exec_result.metadata["kernels"] = impl_operators
            kernel_exec_result.metadata["peak_memory_mb"] = impl_peak_memory
    except Exception as e:
        if verbose:
            print(f"[Eval] Error in Measuring Performance: {e}")
        kernel_exec_result.metadata["error_during_performance"] = e


def eval_kernel_against_ref(
    original_model_src: str,
    custom_model_src: str,
    seed_num: int = 42,
    num_correct_trials: int = 1,
    num_perf_trials: int = 10,
    verbose: bool = True,
    measure_performance: bool = True,
    build_dir: os.PathLike = None,
    device: torch.device | int = (torch.npu.current_device() if torch.npu.is_available() else None),  # noqa: B008
    backend: str = "cuda",
    entry_point: str = "Model",
    enable_profiling: bool = True,
    enable_triton_detection: bool = True,
    backend_adapter: Optional[Any] = None,
    decoy_validator=fillKernelExecResult,
) -> KernelExecResult:
    assert torch.npu.is_available(), "CUDA is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,
        threshold=10,
        edgeitems=3,
        linewidth=80,
    )

    torch.npu.set_device(device)
    is_triton = backend == "triton"
    metadata: dict[str, Any] = {}
    metadata["hardware"] = torch.npu.get_device_name()
    metadata["device"] = str(device)

    if is_triton:
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert device.type == "npu", "CUDA is not availible on device, cannot run Eval"
            device_num = device.index
        else:
            raise ValueError(f"device must be an int or torch.device, got {type(device)}")
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device_num)
    context = {}

    if verbose:
        print(f"[Eval] Start Evalulation! on device: {device}")
        print("[Eval] Loading Original Model")

    Model, get_init_inputs, get_inputs = load_original_model_and_inputs(original_model_src, context, entry_point)
    set_seed(seed_num)
    init_inputs = get_init_inputs()
    init_inputs = [x.npu(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

    print(f"[DEBUG] init inputs: {init_inputs}")

    if (
        len(init_inputs) > 1
        and hasattr(init_inputs[0], "__len__")
        and not isinstance(init_inputs[0], (str, torch.Tensor))
        and len(init_inputs[0]) == 0
    ):
        init_inputs = init_inputs[1]

    with torch.no_grad():
        set_seed(seed_num)

        if type(init_inputs) is list:
            original_model = Model(*init_inputs)
        else:
            original_model = Model(**init_inputs)

        assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")
    if verbose:
        print("[Eval] Loading and Compiling New Model with Custom CUDA Kernel")

    tempfile_handle = None
    backend_handle = None
    backend_session = None

    def _cleanup():
        if backend_session is not None:
            backend_session.close()
            return
        if backend_adapter is not None and backend_handle is not None:
            backend_adapter.cleanup(backend_handle)
            return
        graceful_eval_cleanup(context, device, tempfile_handle)

    try:
        os.environ["TORCH_USE_CUDA_DSA"] = "1"
        if backend_adapter is not None:
            artifact = backend_adapter.compile(
                custom_model_src,
                device=device,
                backend=backend,
                entry_point=f"{entry_point}New",
                build_dir=build_dir,
            )
            if not artifact.get("compiled"):
                error = artifact.get("error", "Unknown compile error")
                if "lock" in str(error) or "No such file or directory" in str(error):
                    print(f"[Eval] Lock file error during compilation, Please retry. Error: {error}")
                    _cleanup()
                    return None
                metadata["compilation_error_name"] = "compile_error"
                metadata["compilation_error"] = error
                _cleanup()
                return KernelExecResult(compiled=False, metadata=metadata)

            backend_handle = backend_adapter.load(
                artifact,
                device=device,
                context=context,
                build_dir=build_dir,
            )
            backend_session = backend_adapter.open_session(backend_handle, device=device)
            tempfile_handle = backend_handle.get("tempfile_handle")
        else:
            if is_triton:
                ModelNew, tempfile_handle = load_custom_model_with_tempfile(
                    custom_model_src, entry_point=f"{entry_point}New"
                )
                if verbose:
                    print("[Eval] Model with Triton Loaded")
            else:
                ModelNew = load_custom_model(custom_model_src, context, build_dir)
        torch.npu.synchronize(device=device)
    except Exception as e:
        print(f"Failed to compile custom CUDA kernel: Record as compilation failure. \nError: {e}")

        if "lock" in str(e) or "No such file or directory" in str(e):
            print(f"[Eval] Lock file error during compilation, Please retry. Error: {e}")
            _cleanup()
            return None
        metadata["compilation_error_name"] = get_error_name(e)
        metadata["compilation_error"] = e
        _cleanup()
        return KernelExecResult(compiled=False, metadata=metadata)

    try:

        def _create_custom_model():
            if backend_session is not None:
                return backend_session.create_model(
                    init_inputs,
                    no_grad=True,
                    synchronize=False,
                )
            if type(init_inputs) is list:
                return ModelNew(*init_inputs)
            return ModelNew(**init_inputs)

        with torch.no_grad():
            set_seed(seed_num)
            custom_model = _create_custom_model()

            assert hasattr(custom_model, "forward")
            torch.npu.synchronize(device=device)
        if verbose:
            print("[Eval] New Model with Custom CUDA Kernel Loaded")
    except RuntimeError as e:
        print(f"Failed to load custom CUDA kernel; Compiled but not able to run, count as runtime error. \nError: {e}")
        _cleanup()
        metadata["runtime_error"] = e
        metadata["runtime_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=True, correctness=False, metadata=metadata)

    kernel_exec_result = None

    kernel_exec_result = _run_correctness_step(
        original_model,
        custom_model,
        get_inputs,
        metadata,
        num_correct_trials,
        verbose,
        seed_num,
        device,
    )

    print(f"enable_triton_detection={enable_triton_detection}")
    decoy_detected = _run_triton_detection_step(
        enable_triton_detection=enable_triton_detection,
        is_triton=is_triton,
        kernel_exec_result=kernel_exec_result,
        custom_model=custom_model,
        custom_model_src=custom_model_src,
        get_inputs=get_inputs,
        metadata=metadata,
        seed_num=seed_num,
        device=device,
        verbose=verbose,
        backend=backend,
        decoy_validator=decoy_validator,
    )
    if decoy_detected:
        _cleanup()
        return kernel_exec_result

    if measure_performance:
        _run_performance_step(
            kernel_exec_result=kernel_exec_result,
            custom_model=custom_model,
            get_inputs=get_inputs,
            metadata=metadata,
            num_perf_trials=num_perf_trials,
            verbose=verbose,
            seed_num=seed_num,
            device=device,
            enable_profiling=enable_profiling,
        )

    _cleanup()
    return kernel_exec_result


def eval_reference_only(
    original_model_src: str,
    seed_num: int = 42,
    num_perf_trials: int = 10,
    verbose: bool = False,
    device: torch.device | int = (torch.npu.current_device() if torch.npu.is_available() else None),  # noqa: B008
    entry_point: str = "Model",
    reference_backend: Optional[str] = None,
    backend_adapter: Optional[Any] = None,
) -> KernelExecResult:
    assert torch.npu.is_available(), "NPU is not available, cannot run Eval"
    torch.set_printoptions(
        precision=4,
        threshold=10,
        edgeitems=3,
        linewidth=80,
    )

    torch.npu.set_device(device)
    metadata: dict[str, Any] = {}
    metadata["hardware"] = torch.npu.get_device_name(device)
    metadata["device"] = str(device)

    context: dict[str, Any] = {}

    if verbose:
        print(f"[Eval] Start Evaluation! on device: {device}")
        print("[Eval] Loading Original Model")

    try:
        Model, get_init_inputs, get_inputs = load_original_model_and_inputs(original_model_src, context, entry_point)
        set_seed(seed_num)
        init_inputs = get_init_inputs()
        init_inputs = [x.npu(device=device) if isinstance(x, torch.Tensor) else x for x in init_inputs]

        with torch.no_grad():
            set_seed(seed_num)
            if type(init_inputs) is list:
                original_model = Model(*init_inputs)
            else:
                original_model = Model(**init_inputs)
            assert hasattr(original_model, "forward")
        if verbose:
            print("[Eval] Original Model Loaded")

    except Exception as e:
        print(f"Failed to load original model: {e}")
        metadata["model_load_error"] = e
        metadata["model_load_error_name"] = get_error_name(e)
        return KernelExecResult(compiled=False, correctness=False, metadata=metadata)

    kernel_exec_result = KernelExecResult(compiled=True, correctness=True, metadata=metadata)

    try:
        if verbose:
            print("[Eval] Measuring Performance of Original Model")

        torch.npu.synchronize(device=device)
        set_seed(seed_num)
        inputs = get_inputs()[0]  # first input group (reference timing uses one shape)
        inputs = [x.npu(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]
        model = original_model.npu(device=device)
        if reference_backend:
            backend_name = reference_backend.lower()
            metadata["reference_backend"] = backend_name
            print(f"[Eval] reference_backend={backend_name}")
            if backend_name in ("torch_compile", "torch-compile", "compile"):
                try:
                    if not hasattr(torch, "compile"):
                        raise RuntimeError("torch.compile is not available")
                    model = torch.compile(model)
                    metadata["reference_backend_compiled"] = True
                    print("[Eval] torch.compile succeeded")
                except Exception as e:
                    metadata["reference_backend_error"] = str(e)
                    print(f"[Eval] torch.compile failed: {e}")
                    return KernelExecResult(compiled=False, correctness=False, metadata=metadata)
        torch.npu.synchronize(device=device)

        impl_operators, impl_latency_ms, impl_peak_memory = measure_single(
            model, inputs, warmup=3, repeats=num_perf_trials, profile_name="model_profile_case", device=device
        )

        kernel_exec_result.runtime = impl_latency_ms
        kernel_exec_result.runtime = impl_latency_ms
        kernel_exec_result.metadata["kernels"] = impl_operators
        kernel_exec_result.metadata["peak_memory"] = impl_peak_memory
        if verbose:
            print(f"[Eval] Performance Stats: {impl_latency_ms}")
        kernel_exec_result.runtime = impl_latency_ms
        kernel_exec_result.runtime_stats = {"operators": impl_operators, "peak_memory": impl_peak_memory}
    except Exception as e:
        if verbose:
            print(f"[Eval] Error in Measuring Performance: {e}")
        kernel_exec_result.metadata["error_during_performance"] = e

    graceful_eval_cleanup(context, device, None)
    return kernel_exec_result
