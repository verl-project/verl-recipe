# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""AscendOptGenAgent evaluation pipeline (multi-shape correctness + timing).

This is the methodology adapter that swaps in the comparison logic from
``eval_AscendOptGenAgent_original/verify.py`` while reusing all the
KernelBench infrastructure (module loading, NPU timing, backend adapter,
cleanup). The single public entry point ``eval_kernel_against_ref_ascend``
mirrors the structure of
``kernelgym.toolkit.kernelbench.pipeline.eval_kernel_against_ref`` so it
slots into the same scheduler / toolkit / subprocess-isolation harness.

Key behavioural differences from the KernelBench pipeline:

* **Multi-shape correctness.** If the reference module defines
  ``get_input_groups()``, every shape is evaluated independently and the
  ``correctness`` field reports strict all-pass / any-fail. Otherwise we
  fall back to a single call to ``get_inputs()``.
* **Strict MERE/MARE pass criterion** (see ``precision.py``).
* **No Triton-decoy detection / no torch.profiler.** Those concepts come
  from KernelBench's methodology; AscendOptGenAgent does not use them.
* **Compile semantics.** ``compiled=True`` iff the custom-kernel module
  loaded successfully. Forward-time runtime errors keep ``compiled=True``
  and set ``correctness=False`` — matching ``verify.py``'s behaviour
  where the script still completes and prints "验证失败" for any
  per-shape failure.
"""

from __future__ import annotations

import gc
import os
import traceback
from typing import Any, Optional

import torch
from kernelgym.toolkit.ascend_opt_gen_agent import precision
from kernelgym.toolkit.kernelbench.exec_types import (
    KernelExecResult,
    get_error_name,
    set_seed,
)
from kernelgym.toolkit.kernelbench.loading import (
    graceful_eval_cleanup,
    load_custom_model,
    load_custom_model_with_tempfile,
    load_original_model_and_inputs,
)
from kernelgym.toolkit.kernelbench.timing import (
    get_timing_stats,
    time_execution_with_cuda_event,
)

# Seed verify.py uses for every shape. We hard-code this (rather than
# accept a ``seed_num`` argument) so the methodology stays faithful to the
# original — AscendOptGenAgent results are not averaged across seeds.
_ASCEND_SEED = 0

ERROR_MSG_LIMIT = 2000


def _truncate_error(msg: str, limit: int = ERROR_MSG_LIMIT) -> str:
    if msg is None:
        return ""
    if len(msg) <= limit:
        return msg
    half = limit // 2
    return f"{msg[:half]}\n... [truncated {len(msg) - limit} chars] ...\n{msg[-half:]}"


def _describe_inputs(inputs: list[Any]) -> list[dict[str, Any]]:
    """Structured per-input description (mirrors verify.py::describe_input).

    Returned values are JSON-serialisable so they can be stashed in
    ``metadata`` and ferried through Redis to the trainer side.
    """
    descs: list[dict[str, Any]] = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            descs.append(
                {
                    "type": "tensor",
                    "shape": list(x.shape),
                    "dtype": str(x.dtype),
                }
            )
        else:
            try:
                val = x if isinstance(x, (int, float, bool, str)) else repr(x)
            except Exception:
                val = "<unrepr>"
            descs.append({"type": "scalar", "value": val})
    return descs


def _cleanup_npu_memory() -> None:
    """Per-shape NPU memory release (mirrors verify.py::cleanup_npu_memory)."""
    try:
        torch.npu.empty_cache()
    except Exception:
        pass
    gc.collect()


def _resolve_input_groups(context: dict[str, Any], get_inputs_fn: Any) -> tuple[list[list[Any]], int]:
    """Resolve verify.py's two input modes.

    Returns ``(groups, total_cases)``. If the user's reference module
    defines ``get_input_groups()`` we treat that as the multi-shape source
    of truth; otherwise we wrap a single ``get_inputs()`` call.
    """
    get_input_groups_fn = context.get("get_input_groups")
    if callable(get_input_groups_fn):
        groups = get_input_groups_fn()
        return groups, len(groups)
    if callable(get_inputs_fn):
        return [get_inputs_fn()], 1
    raise AttributeError("Reference module must provide get_inputs() or get_input_groups()")


def _compare_outputs(
    framework_output: Any,
    impl_output: Any,
    case_idx: int,
    total_cases: int,
) -> None:
    """Normalise model outputs to a list and run precision.compare per tensor.

    Raises AssertionError on the first mismatch, just like verify.py.
    """
    if not isinstance(framework_output, (list, tuple)):
        framework_output = [framework_output]
    if not isinstance(impl_output, (list, tuple)):
        impl_output = [impl_output]

    if len(framework_output) != len(impl_output):
        raise AssertionError(
            f"[用例 {case_idx}/{total_cases}] 输出数量不一致: "
            f"framework={len(framework_output)}, impl={len(impl_output)}"
        )

    for i, (fw_out, impl_out) in enumerate(zip(framework_output, impl_output, strict=False)):
        if fw_out is None or impl_out is None:
            raise AssertionError(
                f"[用例 {case_idx}/{total_cases}] 输出 {i} 为 None: framework={fw_out is None}, impl={impl_out is None}"
            )
        if isinstance(fw_out, torch.Tensor) and isinstance(impl_out, torch.Tensor):
            try:
                precision.compare(fw_out, impl_out, fw_out.dtype)
            except AssertionError as e:
                raise AssertionError(f"[用例 {case_idx}/{total_cases}] 输出 {i}: {str(e)}") from e
        else:
            # Non-tensor outputs (e.g. scalar returns) must be exactly equal.
            # verify_latest.py made this a hard check — previously they
            # were silently accepted.
            if fw_out != impl_out:
                raise AssertionError(
                    f"[用例 {case_idx}/{total_cases}] 输出 {i} 非 Tensor 值不一致: framework={fw_out}, impl={impl_out}"
                )


def _instantiate_impl_model(
    *,
    backend_session: Any,
    ModelNew: Any,
    init_inputs: Any,
):
    """Build a ModelNew instance, going through the backend adapter if any."""
    if backend_session is not None:
        return backend_session.create_model(
            init_inputs,
            no_grad=True,
            synchronize=False,
        )
    if isinstance(init_inputs, list):
        return ModelNew(*init_inputs)
    return ModelNew(**init_inputs)


def _to_device(inputs: list[Any], device: torch.device) -> list[Any]:
    return [x.npu(device=device) if isinstance(x, torch.Tensor) else x for x in inputs]


def eval_kernel_against_ref_ascend(
    original_model_src: str,
    custom_model_src: str,
    *,
    num_perf_trials: int = 100,
    measure_performance: bool = True,
    device: torch.device | int = (torch.npu.current_device() if torch.npu.is_available() else None),  # noqa: B008
    backend: str = "triton",
    entry_point: str = "Model",
    build_dir: Optional[os.PathLike] = None,
    backend_adapter: Optional[Any] = None,
    verbose: bool = False,
) -> KernelExecResult:
    """AscendOptGenAgent-style evaluation entry point.

    Returns a :class:`KernelExecResult` populated with:

    * ``compiled`` — True iff the custom kernel module loaded.
    * ``correctness`` — True iff every input shape passed MERE/MARE.
    * ``runtime`` — mean kernel runtime on the first input group when
      ``measure_performance`` is True and correctness passed; else -1.0.
    * ``metadata`` — includes ``ascend_opt_gen_agent`` dict with
      ``passed_cases``, ``total_cases``, ``failures``.

    The function never raises (catches everything and surfaces it via
    ``compiled`` / ``correctness`` / ``metadata``), so the subprocess
    worker can serialize a failure result instead of crashing.
    """
    assert torch.npu.is_available(), "NPU is not available, cannot run Eval"

    torch.set_printoptions(precision=4, threshold=10, edgeitems=3, linewidth=80)
    torch.npu.set_device(device)

    is_triton = backend == "triton"
    metadata: dict[str, Any] = {
        "hardware": torch.npu.get_device_name(),
        "device": str(device),
        "methodology": "ascend_opt_gen_agent",
    }

    if is_triton:
        if isinstance(device, int):
            device_num = device
        elif isinstance(device, torch.device):
            assert device.type == "npu", "NPU is not available on device, cannot run Eval"
            device_num = device.index
        else:
            raise ValueError(f"device must be an int or torch.device, got {type(device)}")
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = str(device_num)

    context: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Phase 1: load reference (torch) module and resolve input providers.
    # ------------------------------------------------------------------
    try:
        Model, get_init_inputs_fn, get_inputs_fn = load_original_model_and_inputs(
            original_model_src, context, entry_point
        )
        if Model is None:
            raise RuntimeError(f"Failed to locate entry point '{entry_point}' in reference code")
        input_groups, total_cases = _resolve_input_groups(context, get_inputs_fn)
    except Exception as e:
        metadata["compilation_error_name"] = get_error_name(e)
        metadata["compilation_error"] = str(e)
        metadata["ascend_opt_gen_agent"] = {
            "passed_cases": 0,
            "total_cases": 0,
            "failures": [],
            "error": "reference_load_failed",
        }
        return KernelExecResult(compiled=False, metadata=metadata)

    # ------------------------------------------------------------------
    # Phase 2: load custom kernel module (failure here ⇒ compiled=False).
    # ------------------------------------------------------------------
    tempfile_handle = None
    backend_handle = None
    backend_session = None
    ModelNew = None

    def _cleanup() -> None:
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
                    if verbose:
                        print(f"[Eval] Lock-file error during compilation: {error}")
                    _cleanup()
                    return None
                metadata["compilation_error_name"] = "compile_error"
                metadata["compilation_error"] = error
                metadata["ascend_opt_gen_agent"] = {
                    "passed_cases": 0,
                    "total_cases": total_cases,
                    "failures": [],
                    "error": "impl_compile_failed",
                }
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
            else:
                ModelNew = load_custom_model(custom_model_src, context, build_dir)
                if ModelNew is None:
                    raise RuntimeError(f"Failed to locate entry point '{entry_point}New' in custom code")
        torch.npu.synchronize(device=device)
    except Exception as e:
        if "lock" in str(e) or "No such file or directory" in str(e):
            if verbose:
                print(f"[Eval] Lock-file error during compilation: {e}")
            _cleanup()
            return None
        metadata["compilation_error_name"] = get_error_name(e)
        metadata["compilation_error"] = str(e)
        metadata["ascend_opt_gen_agent"] = {
            "passed_cases": 0,
            "total_cases": total_cases,
            "failures": [],
            "error": "impl_load_failed",
        }
        _cleanup()
        return KernelExecResult(compiled=False, metadata=metadata)

    # ------------------------------------------------------------------
    # Phase 3: per-shape correctness loop (verify.py semantics).
    # ------------------------------------------------------------------
    failures: list[dict[str, Any]] = []
    passed_cases = 0
    device_obj = device if isinstance(device, torch.device) else torch.device(f"npu:{device}")

    for case_idx, inputs in enumerate(input_groups, start=1):
        input_desc = _describe_inputs(inputs)
        framework_model = None
        impl_model = None
        try:
            # Seed BEFORE get_init_inputs() so random init params are
            # reproducible across both models (matches verify.py).
            set_seed(_ASCEND_SEED)
            init_inputs = get_init_inputs_fn()

            # KernelBench task convention: init_inputs may be a list, a
            # tuple, or [[], <kwargs>]. Replicate the kernelbench pipeline's
            # handling so we don't regress on existing tasks.
            init_inputs = [x.npu(device=device_obj) if isinstance(x, torch.Tensor) else x for x in init_inputs]
            if (
                len(init_inputs) > 1
                and hasattr(init_inputs[0], "__len__")
                and not isinstance(init_inputs[0], (str, torch.Tensor))
                and len(init_inputs[0]) == 0
            ):
                init_inputs = init_inputs[1]

            with torch.no_grad():
                set_seed(_ASCEND_SEED)
                if isinstance(init_inputs, list):
                    framework_model = Model(*init_inputs)
                else:
                    framework_model = Model(**init_inputs)
                framework_model = framework_model.npu(device=device_obj)

                set_seed(_ASCEND_SEED)
                impl_model = _instantiate_impl_model(
                    backend_session=backend_session,
                    ModelNew=ModelNew,
                    init_inputs=init_inputs,
                )
                # backend_session paths already move the model to the
                # right device; the local-exec path does not.
                if backend_session is None and hasattr(impl_model, "npu"):
                    impl_model = impl_model.npu(device=device_obj)

                fw_inputs = _to_device(inputs, device_obj)
                impl_inputs = _to_device(inputs, device_obj)

                impl_output = impl_model(*impl_inputs)
                framework_output = framework_model(*fw_inputs)
                torch.npu.synchronize(device=device_obj)

            _compare_outputs(framework_output, impl_output, case_idx, total_cases)
            passed_cases += 1
            if verbose:
                print(f"[Eval] case {case_idx}/{total_cases} passed")
        except Exception as e:
            err_detail = traceback.format_exc()
            failures.append(
                {
                    "case_idx": case_idx,
                    "input_desc": input_desc,
                    "error_type": type(e).__name__,
                    "error_msg": _truncate_error(err_detail),
                }
            )
            if verbose:
                print(f"[Eval] case {case_idx}/{total_cases} failed: {type(e).__name__}: {e}")
        finally:
            del framework_model
            del impl_model
            _cleanup_npu_memory()

    correctness = (passed_cases == total_cases) and (total_cases > 0)
    metadata["ascend_opt_gen_agent"] = {
        "passed_cases": passed_cases,
        "total_cases": total_cases,
        "failures": failures,
    }
    # Mirror kernelbench's expected metadata key for the schema layer.
    metadata["correctness_trials"] = f"({passed_cases} / {total_cases})"
    if not correctness and failures:
        # Surface the first failure as a top-level error string so
        # ``KernelEvaluationResult.from_kernel_exec_result`` can attach it
        # to ``error_message``.
        metadata["runtime_error"] = failures[0]["error_msg"]
        metadata["runtime_error_name"] = failures[0]["error_type"]

    kernel_exec_result = KernelExecResult(
        compiled=True,
        correctness=correctness,
        metadata=metadata,
    )

    # ------------------------------------------------------------------
    # Phase 4: performance (only if correctness fully passed).
    # ------------------------------------------------------------------
    if measure_performance and correctness:
        try:
            # Use the FIRST input group for timing (matches verify.py +
            # benchmark.py which time a single representative shape).
            timing_inputs = input_groups[0]

            set_seed(_ASCEND_SEED)
            timing_init_inputs = get_init_inputs_fn()
            timing_init_inputs = [
                x.npu(device=device_obj) if isinstance(x, torch.Tensor) else x for x in timing_init_inputs
            ]
            if (
                len(timing_init_inputs) > 1
                and hasattr(timing_init_inputs[0], "__len__")
                and not isinstance(timing_init_inputs[0], (str, torch.Tensor))
                and len(timing_init_inputs[0]) == 0
            ):
                timing_init_inputs = timing_init_inputs[1]

            with torch.no_grad():
                set_seed(_ASCEND_SEED)
                timing_impl_model = _instantiate_impl_model(
                    backend_session=backend_session,
                    ModelNew=ModelNew,
                    init_inputs=timing_init_inputs,
                )
                if backend_session is None and hasattr(timing_impl_model, "npu"):
                    timing_impl_model = timing_impl_model.npu(device=device_obj)

                timing_inputs_dev = _to_device(timing_inputs, device_obj)
                torch.npu.synchronize(device=device_obj)

            elapsed_times, _ = time_execution_with_cuda_event(
                timing_impl_model,
                *timing_inputs_dev,
                num_trials=num_perf_trials,
                verbose=verbose,
                device=device_obj,
                enable_profiling=False,
            )
            runtime_stats = get_timing_stats(elapsed_times, device=device_obj)
            kernel_exec_result.runtime = runtime_stats["mean"]
            kernel_exec_result.runtime_stats = runtime_stats
            if verbose:
                print(f"[Eval] Performance Stats: {runtime_stats}")
        except Exception as e:
            if verbose:
                print(f"[Eval] Error in Measuring Performance: {e}")
            kernel_exec_result.metadata["error_during_performance"] = str(e)

    _cleanup()
    return kernel_exec_result


def eval_reference_only_ascend(
    original_model_src: str,
    *,
    num_perf_trials: int = 100,
    device: torch.device | int = (torch.npu.current_device() if torch.npu.is_available() else None),  # noqa: B008
    entry_point: str = "Model",
    reference_backend: Optional[str] = None,
    verbose: bool = False,
) -> KernelExecResult:
    """Reference-only timing pass.

    AscendOptGenAgent doesn't change anything about timing the ground-truth
    Model — this is here only so the toolkit can serve the
    ``reference_timing`` subtask without delegating back to KernelBench.
    Implementation is functionally identical to
    ``kernelbench.pipeline.eval_reference_only`` but uses the AscendOpt
    seed (0) for symmetry.
    """
    from kernelgym.toolkit.kernelbench.pipeline import eval_reference_only

    return eval_reference_only(
        original_model_src,
        seed_num=_ASCEND_SEED,
        num_perf_trials=num_perf_trials,
        verbose=verbose,
        device=device,
        entry_point=entry_point,
        reference_backend=reference_backend,
        backend_adapter=None,
    )
