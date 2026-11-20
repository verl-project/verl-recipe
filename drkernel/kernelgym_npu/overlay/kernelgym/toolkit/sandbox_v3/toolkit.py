# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""sandbox_v3 toolkit wrapper.

Vendored from the Triton-Training-kernelgym-sandbox KernelBench toolkit and
hard-wired to the NPU-kernel evaluation path (NPUKERNEL_MODE=on,
ORIGIN_MODE=off): multi-shape correctness, allclose precision, AST-based
Triton-implementation validation, and operator-level NPU profiling.
"""

from __future__ import annotations

from typing import Any

import torch
from kernelgym.common import ErrorCode
from kernelgym.config import settings
from kernelgym.schema import (
    EvaluationResult,
    EvaluationTask,
    KernelEvaluationResult,
    KernelEvaluationTask,
    ReferenceTimingResult,
    ReferenceTimingTask,
)
from kernelgym.toolkit.kernelbench.exec_types import set_seed
from kernelgym.toolkit.sandbox_v3 import pipeline as sandbox_pipeline
from kernelgym.toolkit.sandbox_v3.validate_triton_impl import fillKernelExecResult
from kernelgym.toolkit.validation import validate_code

from ..base import Toolkit


class SandboxV3Toolkit(Toolkit):
    """Toolkit adapter around the sandbox_v3 (NPU-kernel) evaluation path."""

    name = "sandbox_v3"

    # AST-based decoy validator fed into the pipeline's Triton-detection step.
    # Subclasses (e.g. sandbox_v4) override this to swap the forbidden-op
    # whitelist while reusing the rest of the evaluation path unchanged.
    _decoy_validator = staticmethod(fillKernelExecResult)

    def __init__(self) -> None:
        pass

    def _resolve_eval_flags(self, task: Any) -> tuple[bool, bool, bool]:
        run_correctness = task.run_correctness
        if run_correctness is None:
            run_correctness = True

        run_triton_detection = task.run_triton_detection
        if run_triton_detection is None:
            run_triton_detection = task.enable_triton_detection
        if run_triton_detection is None:
            run_triton_detection = task.backend == "triton"

        run_performance = task.run_performance
        if run_performance is None:
            run_performance = task.measure_performance
        if run_performance is None:
            run_performance = True

        return run_correctness, run_triton_detection, run_performance

    def evaluate(self, task: dict[str, Any], backend=None, **kwargs: Any) -> dict[str, Any]:
        task_type = task.get("task_type", "evaluation")
        if task_type == "evaluation":
            result = self.evaluate_kernel(EvaluationTask.from_dict(task), backend_adapter=backend)
        elif task_type == "reference_timing":
            result = self.evaluate_reference_timing(
                ReferenceTimingTask.from_dict(task),
                backend_adapter=backend,
            )
        elif task_type == "kernel_evaluation":
            result = self.evaluate_kernel_only(
                KernelEvaluationTask.from_dict(task),
                verbose_errors=task.get("verbose_errors", True),
                enable_profiling=task.get("enable_profiling", settings.enable_profiling),
                backend_adapter=backend,
            )
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        return result.to_dict()

    def evaluate_kernel(self, task: EvaluationTask, backend_adapter=None) -> EvaluationResult:
        device = torch.device(task.device)
        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return EvaluationResult(
                task_id=task.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                reference_runtime=0.0,
                kernel_runtime=0.0,
                speedup=0.0,
                metadata={"validation_error": ref_error},
                status="failed",
                error_message=f"Reference code validation failed: {ref_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        kernel_entry_point = f"{task.entry_point}New"
        kernel_valid, kernel_error = validate_code(task.kernel_code, kernel_entry_point)
        if not kernel_valid:
            return EvaluationResult(
                task_id=task.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                reference_runtime=0.0,
                kernel_runtime=0.0,
                speedup=0.0,
                metadata={"validation_error": kernel_error},
                status="failed",
                error_message=f"Kernel code validation failed: {kernel_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        try:
            set_seed(42)

            run_correctness, enable_triton_detection, measure_performance = self._resolve_eval_flags(task)
            num_correct_trials = task.num_correct_trials if run_correctness else 0

            enable_profiling = task.enable_profiling
            if enable_profiling is None:
                enable_profiling = settings.enable_profiling

            result = sandbox_pipeline.eval_kernel_against_ref(
                original_model_src=task.reference_code,
                custom_model_src=task.kernel_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=task.num_perf_trials,
                measure_performance=measure_performance,
                verbose=False,
                device=device,
                backend=task.backend,
                entry_point=task.entry_point,
                enable_profiling=bool(enable_profiling),
                enable_triton_detection=enable_triton_detection,
                backend_adapter=backend_adapter,
                decoy_validator=self._decoy_validator,
            )

            if not run_correctness:
                if result.metadata is None:
                    result.metadata = {}
                result.metadata["correctness_skipped"] = True

            reference_runtime = sandbox_pipeline.eval_reference_only(
                original_model_src=task.reference_code,
                num_perf_trials=task.num_perf_trials,
                verbose=False,
                device=device,
                entry_point=task.entry_point,
                backend_adapter=backend_adapter,
            ).runtime

            if result.metadata is None:
                result.metadata = {}
            result.metadata.update(
                {
                    "device": str(device),
                    "npu_name": torch.npu.get_device_name(device),
                    "backend": task.backend,
                    "num_correct_trials": num_correct_trials,
                    "num_perf_trials": task.num_perf_trials,
                }
            )

            return EvaluationResult.from_kernel_exec_result(task.task_id, result, reference_runtime)

        except Exception as e:
            from kernelgym.utils.error_classifier import classify_error

            error_code = classify_error(str(e), "runtime")
            return EvaluationResult(
                task_id=task.task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                reference_runtime=0.0,
                kernel_runtime=0.0,
                speedup=0.0,
                metadata={"error": str(e)},
                status="failed",
                error_message=f"Evaluation failed: {str(e)}",
                error_code=error_code,
            )

    def evaluate_reference_timing(self, task: ReferenceTimingTask, backend_adapter=None) -> ReferenceTimingResult:
        device = torch.device(task.device)

        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return ReferenceTimingResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                reference_runtime=0.0,
                metadata={"validation_error": ref_error},
                status="failed",
                error_message=f"Reference code validation failed: {ref_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        try:
            set_seed(42)

            if task.reference_backend:
                print(f"[RefTiming] task={task.task_id} reference_backend={task.reference_backend}")

            ref_exec_result = sandbox_pipeline.eval_reference_only(
                original_model_src=task.reference_code,
                num_perf_trials=task.num_perf_trials,
                verbose=False,
                device=device,
                entry_point=task.entry_point,
                reference_backend=task.reference_backend,
                backend_adapter=backend_adapter,
            )
            reference_runtime = ref_exec_result.runtime

            metadata = {
                "device": str(device),
                "npu_name": torch.npu.get_device_name(device),
                "backend": task.backend,
                "num_perf_trials": task.num_perf_trials,
            }
            if ref_exec_result.metadata:
                metadata.update(ref_exec_result.metadata)

            return ReferenceTimingResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                reference_runtime=reference_runtime,
                metadata=metadata,
                status="completed",
            )

        except Exception as e:
            from kernelgym.utils.error_classifier import classify_error

            error_code = classify_error(str(e), "runtime")
            return ReferenceTimingResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                reference_runtime=0.0,
                metadata={"error": str(e)},
                status="failed",
                error_message=f"Reference timing failed: {str(e)}",
                error_code=error_code,
            )

    def evaluate_kernel_only(
        self,
        task: KernelEvaluationTask,
        verbose_errors: bool = True,
        enable_profiling: bool = False,
        backend_adapter=None,
    ) -> KernelEvaluationResult:
        device = torch.device(task.device)
        print(f"evaluate_kernel_only device={device}")
        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return KernelEvaluationResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=0.0,
                metadata={"validation_error": ref_error},
                status="failed",
                error_message=f"Reference code validation failed: {ref_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        kernel_entry_point = f"{task.entry_point}New"
        kernel_valid, kernel_error = validate_code(task.kernel_code, kernel_entry_point)
        print(f"validate_code kernel_valid={kernel_valid}")
        if not kernel_valid:
            return KernelEvaluationResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=0.0,
                metadata={"validation_error": kernel_error},
                status="failed",
                error_message=f"Kernel code validation failed: {kernel_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        try:
            set_seed(42)

            run_correctness, enable_triton_detection, measure_performance = self._resolve_eval_flags(task)
            num_correct_trials = task.num_correct_trials if run_correctness else 0

            result = sandbox_pipeline.eval_kernel_against_ref(
                original_model_src=task.reference_code,
                custom_model_src=task.kernel_code,
                num_correct_trials=num_correct_trials,
                num_perf_trials=task.num_perf_trials,
                measure_performance=measure_performance,
                verbose=False,
                device=device,
                backend=task.backend,
                entry_point=task.entry_point,
                enable_profiling=enable_profiling,
                enable_triton_detection=enable_triton_detection,
                backend_adapter=backend_adapter,
                decoy_validator=self._decoy_validator,
            )
            print(f"eval_kernel_against_ref result={result}")
            if not run_correctness:
                if result.metadata is None:
                    result.metadata = {}
                result.metadata["correctness_skipped"] = True

            if result.metadata is None:
                result.metadata = {}
            result.metadata.update(
                {
                    "device": str(device),
                    "npu_name": torch.npu.get_device_name(device),
                    "backend": task.backend,
                    "num_correct_trials": num_correct_trials,
                    "num_perf_trials": task.num_perf_trials,
                }
            )

            if enable_profiling and "profiling" in result.metadata:
                profiling_metrics = result.metadata["profiling"]
                if profiling_metrics:
                    print(f"[DEBUG] Profiling captured {profiling_metrics.get('kernel_count', 0)} kernels")

            return KernelEvaluationResult.from_kernel_exec_result(
                task.task_id,
                task.base_task_id,
                result,
                verbose_errors=verbose_errors,
            )

        except Exception as e:
            from kernelgym.utils.error_classifier import classify_error

            error_code = classify_error(str(e), "runtime")
            return KernelEvaluationResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=0.0,
                metadata={"error": str(e)},
                status="failed",
                error_message=f"Kernel evaluation failed: {str(e)}",
                error_code=error_code,
            )
