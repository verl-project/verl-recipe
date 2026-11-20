# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""Toolkit adapter for the AscendOptGenAgent evaluation methodology.

Plugs the AscendOptGenAgent pipeline (multi-shape verify.py-style
correctness + NPU-Benchmark MERE/MARE precision check) into the same
toolkit/scheduler/subprocess harness used by ``KernelBenchToolkit``.

Selection is per-request: set ``toolkit: "ascend_opt_gen_agent"`` on the
evaluation request (or set ``DEFAULT_TOOLKIT=ascend_opt_gen_agent`` to
make it the server default). The workflow controller propagates the
toolkit name into both the ``kernel_evaluation`` and ``reference_timing``
subtasks (see ``workflow/kernelbench_helpers.py::_create_paired_tasks``),
so the dispatcher below has to handle all three task types.
"""

from __future__ import annotations

from typing import Any

import torch
from kernelgym.common import ErrorCode
from kernelgym.schema import (
    EvaluationResult,
    EvaluationTask,
    KernelEvaluationResult,
    KernelEvaluationTask,
    ReferenceTimingResult,
    ReferenceTimingTask,
)
from kernelgym.toolkit.kernelbench.exec_types import set_seed
from kernelgym.toolkit.validation import validate_code

from ..base import Toolkit
from .pipeline import eval_kernel_against_ref_ascend, eval_reference_only_ascend


class AscendOptGenAgentToolkit(Toolkit):
    """Toolkit adapter for AscendOptGenAgent-style evaluation."""

    name = "ascend_opt_gen_agent"

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Flag resolution mirrors KernelBenchToolkit so callers that already
    # set run_correctness / run_performance get the same semantics.
    # AscendOptGenAgent does not use triton-decoy detection, so that flag
    # is intentionally dropped.
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_eval_flags(task: Any) -> tuple[bool, bool]:
        run_correctness = task.run_correctness
        if run_correctness is None:
            run_correctness = True

        run_performance = task.run_performance
        if run_performance is None:
            run_performance = task.measure_performance
        if run_performance is None:
            run_performance = True

        return run_correctness, run_performance

    def evaluate(self, task: dict[str, Any], backend=None, **kwargs: Any) -> dict[str, Any]:
        task_type = task.get("task_type", "evaluation")
        if task_type == "evaluation":
            result = self.evaluate_kernel(EvaluationTask.from_dict(task), backend_adapter=backend)
        elif task_type == "reference_timing":
            result = self.evaluate_reference_timing(
                ReferenceTimingTask.from_dict(task),
                backend_adapter=backend,
            )
        elif task_type in ("kernel_evaluation", "kernel"):
            result = self.evaluate_kernel_only(
                KernelEvaluationTask.from_dict(task),
                verbose_errors=task.get("verbose_errors", True),
                backend_adapter=backend,
            )
        else:
            raise ValueError(f"Unknown task_type for ascend_opt_gen_agent: {task_type}")

        return result.to_dict()

    # ------------------------------------------------------------------
    # task_type == "evaluation": kernel correctness + perf + reference run.
    # This path is only used when the caller bypasses the workflow and
    # calls the toolkit directly. The standard kernelbench workflow splits
    # this into kernel_evaluation + reference_timing.
    # ------------------------------------------------------------------
    def evaluate_kernel(self, task: EvaluationTask, backend_adapter=None) -> EvaluationResult:
        device = torch.device(task.device)

        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return self._validation_failed_evaluation_result(
                task.task_id, f"Reference code validation failed: {ref_error}", ref_error
            )

        kernel_entry_point = f"{task.entry_point}New"
        kernel_valid, kernel_error = validate_code(task.kernel_code, kernel_entry_point)
        if not kernel_valid:
            return self._validation_failed_evaluation_result(
                task.task_id, f"Kernel code validation failed: {kernel_error}", kernel_error
            )

        try:
            set_seed(42)
            run_correctness, measure_performance = self._resolve_eval_flags(task)

            kernel_exec = eval_kernel_against_ref_ascend(
                original_model_src=task.reference_code,
                custom_model_src=task.kernel_code,
                num_perf_trials=task.num_perf_trials,
                measure_performance=measure_performance,
                device=device,
                backend=task.backend,
                entry_point=task.entry_point,
                backend_adapter=backend_adapter,
                verbose=False,
            )

            if kernel_exec is None:
                return self._failed_evaluation_result(task.task_id, "Compilation lock-file error; retry needed")

            if not run_correctness and kernel_exec.metadata is not None:
                kernel_exec.metadata["correctness_skipped"] = True

            # Time the reference model separately so we can report speedup.
            try:
                reference_exec = eval_reference_only_ascend(
                    original_model_src=task.reference_code,
                    num_perf_trials=task.num_perf_trials,
                    device=device,
                    entry_point=task.entry_point,
                    reference_backend=task.reference_backend,
                    verbose=False,
                )
                reference_runtime = reference_exec.runtime if reference_exec else 0.0
            except Exception as e:
                reference_runtime = 0.0
                if kernel_exec.metadata is None:
                    kernel_exec.metadata = {}
                kernel_exec.metadata["reference_timing_error"] = str(e)

            if kernel_exec.metadata is None:
                kernel_exec.metadata = {}
            kernel_exec.metadata.update(
                {
                    "device": str(device),
                    "gpu_name": torch.npu.get_device_name(device),
                    "backend": task.backend,
                    "num_perf_trials": task.num_perf_trials,
                }
            )

            return EvaluationResult.from_kernel_exec_result(task.task_id, kernel_exec, reference_runtime)

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
                metadata={"error": str(e), "methodology": "ascend_opt_gen_agent"},
                status="failed",
                error_message=f"Evaluation failed: {str(e)}",
                error_code=error_code,
            )

    # ------------------------------------------------------------------
    # task_type == "reference_timing": pure torch timing of the reference.
    # Methodology-independent — we delegate to the shared timing helper
    # but route through eval_reference_only_ascend so the seed is consistent
    # with the kernel-eval branch.
    # ------------------------------------------------------------------
    def evaluate_reference_timing(self, task: ReferenceTimingTask, backend_adapter=None) -> ReferenceTimingResult:
        device = torch.device(task.device)

        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return ReferenceTimingResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                reference_runtime=0.0,
                metadata={
                    "validation_error": ref_error,
                    "methodology": "ascend_opt_gen_agent",
                },
                status="failed",
                error_message=f"Reference code validation failed: {ref_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        try:
            ref_exec = eval_reference_only_ascend(
                original_model_src=task.reference_code,
                num_perf_trials=task.num_perf_trials,
                device=device,
                entry_point=task.entry_point,
                reference_backend=task.reference_backend,
                verbose=False,
            )
            reference_runtime = ref_exec.runtime if ref_exec else 0.0

            metadata: dict[str, Any] = {
                "device": str(device),
                "gpu_name": torch.npu.get_device_name(device),
                "backend": task.backend,
                "num_perf_trials": task.num_perf_trials,
                "methodology": "ascend_opt_gen_agent",
            }
            if ref_exec is not None and ref_exec.metadata:
                metadata.update(ref_exec.metadata)

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
                metadata={
                    "error": str(e),
                    "methodology": "ascend_opt_gen_agent",
                },
                status="failed",
                error_message=f"Reference timing failed: {str(e)}",
                error_code=error_code,
            )

    # ------------------------------------------------------------------
    # task_type == "kernel_evaluation": correctness + (optional) perf,
    # NO reference timing. This is the primary path under the workflow.
    # ------------------------------------------------------------------
    def evaluate_kernel_only(
        self,
        task: KernelEvaluationTask,
        verbose_errors: bool = True,
        backend_adapter=None,
    ) -> KernelEvaluationResult:
        device = torch.device(task.device)

        ref_valid, ref_error = validate_code(task.reference_code, task.entry_point)
        if not ref_valid:
            return KernelEvaluationResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=0.0,
                metadata={
                    "validation_error": ref_error,
                    "methodology": "ascend_opt_gen_agent",
                },
                status="failed",
                error_message=f"Reference code validation failed: {ref_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        kernel_entry_point = f"{task.entry_point}New"
        kernel_valid, kernel_error = validate_code(task.kernel_code, kernel_entry_point)
        if not kernel_valid:
            return KernelEvaluationResult(
                task_id=task.task_id,
                base_task_id=task.base_task_id,
                compiled=False,
                correctness=False,
                decoy_kernel=False,
                kernel_runtime=0.0,
                metadata={
                    "validation_error": kernel_error,
                    "methodology": "ascend_opt_gen_agent",
                },
                status="failed",
                error_message=f"Kernel code validation failed: {kernel_error}",
                error_code=ErrorCode.VALIDATION_ERROR,
            )

        try:
            set_seed(42)
            run_correctness, measure_performance = self._resolve_eval_flags(task)

            kernel_exec = eval_kernel_against_ref_ascend(
                original_model_src=task.reference_code,
                custom_model_src=task.kernel_code,
                num_perf_trials=task.num_perf_trials,
                measure_performance=measure_performance,
                device=device,
                backend=task.backend,
                entry_point=task.entry_point,
                backend_adapter=backend_adapter,
                verbose=False,
            )

            if kernel_exec is None:
                return KernelEvaluationResult(
                    task_id=task.task_id,
                    base_task_id=task.base_task_id,
                    compiled=False,
                    correctness=False,
                    decoy_kernel=False,
                    kernel_runtime=0.0,
                    metadata={
                        "error": "Compilation lock-file error; retry needed",
                        "methodology": "ascend_opt_gen_agent",
                    },
                    status="failed",
                    error_message="Compilation lock-file error; retry needed",
                    error_code=ErrorCode.COMPILATION_ERROR,
                )

            if not run_correctness and kernel_exec.metadata is not None:
                kernel_exec.metadata["correctness_skipped"] = True

            if kernel_exec.metadata is None:
                kernel_exec.metadata = {}
            kernel_exec.metadata.update(
                {
                    "device": str(device),
                    "gpu_name": torch.npu.get_device_name(device),
                    "backend": task.backend,
                    "num_perf_trials": task.num_perf_trials,
                }
            )

            return KernelEvaluationResult.from_kernel_exec_result(
                task.task_id,
                task.base_task_id,
                kernel_exec,
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
                metadata={
                    "error": str(e),
                    "methodology": "ascend_opt_gen_agent",
                },
                status="failed",
                error_message=f"Kernel evaluation failed: {str(e)}",
                error_code=error_code,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _validation_failed_evaluation_result(
        self, task_id: str, message: str, validation_error: str
    ) -> EvaluationResult:
        return EvaluationResult(
            task_id=task_id,
            compiled=False,
            correctness=False,
            decoy_kernel=False,
            reference_runtime=0.0,
            kernel_runtime=0.0,
            speedup=0.0,
            metadata={
                "validation_error": validation_error,
                "methodology": "ascend_opt_gen_agent",
            },
            status="failed",
            error_message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
        )

    def _failed_evaluation_result(self, task_id: str, message: str) -> EvaluationResult:
        return EvaluationResult(
            task_id=task_id,
            compiled=False,
            correctness=False,
            decoy_kernel=False,
            reference_runtime=0.0,
            kernel_runtime=0.0,
            speedup=0.0,
            metadata={"error": message, "methodology": "ascend_opt_gen_agent"},
            status="failed",
            error_message=message,
            error_code=ErrorCode.COMPILATION_ERROR,
        )
