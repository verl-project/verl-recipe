# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""KernelBench profiling helpers (toolkit layer)."""

from __future__ import annotations

import logging
import os
import shutil
import time
from contextlib import contextmanager
from typing import Any, Optional

import torch
from kernelgym.config import settings

logger = logging.getLogger("kernelgym.toolkit.kernelbench.profiling")


def prepare_model_fn(model: Any, inputs: list[Any], device: Any) -> callable:
    """准备模型用于性能测试，返回测试函数"""
    import torch

    # 执行warmup
    with torch.no_grad():
        _ = model(*inputs)
    torch.npu.synchronize()

    # 返回测试函数
    def test_fn():
        with torch.no_grad():
            _ = model(*inputs)
        torch.npu.synchronize()

    return test_fn


def find_profile_file(profile_path: str, filename: str) -> Optional[str]:
    """在profile目录中查找指定文件"""
    for root, _, files in os.walk(profile_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def cleanup_profile_path(profile_path: str) -> None:
    """清理profile目录"""
    if os.path.exists(profile_path):
        shutil.rmtree(profile_path, ignore_errors=True)


def parse_operator_latency(profile_path: str, active_count: int) -> tuple[Optional[dict[str, float]], Optional[float]]:
    """从 profiling 结果文件中提取算子时延数据，计算平均执行时间。"""
    import pandas as pd

    operator_details_file = find_profile_file(profile_path, "operator_details.csv")

    if not operator_details_file or not os.path.exists(operator_details_file):
        cleanup_profile_path(profile_path)
        return None, None

    try:
        df = pd.read_csv(operator_details_file)
    except Exception:
        cleanup_profile_path(profile_path)
        return None, None

    required_columns = ["Name", "Device Self Duration(us)"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        cleanup_profile_path(profile_path)
        return None, None

    if "Count" not in df.columns:
        return _parse_without_count(df, profile_path, active_count)

    return _parse_with_count(df, profile_path, active_count)


def _parse_without_count(
    df: Any, profile_path: str, active_count: int
) -> tuple[Optional[dict[str, float]], Optional[float]]:
    """处理没有 Count 列的情况：按操作名称直接累加计算。"""
    # 按算子名称分组，累加所有测量周期的 Device Self Duration
    operator_avg_times = {}
    grouped = df.groupby("Name")["Device Self Duration(us)"].sum()
    for op_name_str, total_us in grouped.items():
        # 平均到每次运行（微秒）
        operator_avg_times[op_name_str] = total_us / active_count

    # 汇总所有算子的平均时间，得到完整的 device 侧执行时间
    total_avg_us = sum(operator_avg_times.values())
    total_avg_ms = total_avg_us / 1000.0

    cleanup_profile_path(profile_path)

    return operator_avg_times, round(total_avg_ms, 4)


def _parse_with_count(
    df: Any, profile_path: str, active_count: int
) -> tuple[Optional[dict[str, float]], Optional[float]]:
    """解析有 Count 列的情况：按操作名称分组，累加 Self Duration，计算每次运行的平均时间。"""
    # 筛选出 Count 等于 active_count 的记录（即正式测试阶段的算子）
    valid_ops = df[df["Count"] == active_count].copy()

    if valid_ops.empty:
        cleanup_profile_path(profile_path)
        return None, None

    # 按算子名称分组，累加 Device Self Duration
    operator_avg_times = {}
    grouped = valid_ops.groupby("Name")
    for op_name_str, group in grouped:
        total_us = group["Device Self Duration(us)"].sum()
        avg_us = total_us / active_count
        # 存储单位为微秒（us）
        operator_avg_times[op_name_str] = avg_us

    # 汇总所有算子的 Self Duration，得到一次完整运行的 device 侧总时间
    total_avg_us = sum(operator_avg_times.values())
    # 转换为毫秒
    total_avg_ms = total_avg_us / 1000.0

    cleanup_profile_path(profile_path)

    return operator_avg_times, round(total_avg_ms, 4)


def run_profiler_with_config(test_fn: callable, warmup: int, repeats: int, profile_name: str) -> str:
    """运行NPU profiler并返回生成的性能分析目录路径。"""
    import torch
    import torch_npu

    # 实验性配置
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=None,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
        data_simplification=False,
    )

    # 预热一次确保模型准备就绪
    test_fn()
    torch.npu.synchronize()

    skip_first = 1 + warmup
    total_steps = skip_first + repeats

    # 生成唯一的profile路径
    timestamp = int(time.time() * 1000)
    profile_path = os.path.join(os.getcwd(), f"{profile_name}_{timestamp}")

    # 创建profiler
    with torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU, torch_npu.profiler.ProfilerActivity.CPU],
        schedule=torch_npu.profiler.schedule(wait=0, warmup=warmup, active=repeats, repeat=1, skip_first=skip_first),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        with_flops=False,
        with_modules=False,
        experimental_config=experimental_config,
    ) as prof:
        for _ in range(total_steps):
            test_fn()
            prof.step()
            torch.npu.synchronize()

    return profile_path


def measure_single(
    model: Any, inputs: list[Any], warmup: int, repeats: int, profile_name: str, device: Any
) -> tuple[Optional[dict[str, float]], Optional[float], float]:
    """测量单次性能（warmup + profiling）"""
    import torch

    # 重置峰值内存统计
    torch.npu.reset_peak_memory_stats()

    # 准备测试函数
    test_fn = prepare_model_fn(model, inputs, device)

    try:
        # 运行profiler
        profile_path = run_profiler_with_config(test_fn, warmup, repeats, profile_name)

        # 解析结果
        operators, latency_ms = parse_operator_latency(profile_path, repeats)
    except Exception as e:
        print(f"torch_npu.profiler 获取数据失败: {e}，使用兜底测试机制...")
        operators, latency_ms = None, None

    # 如果profiler获取不到数据或时延为0/无效，使用兜底机制
    if operators is None or latency_ms is None or latency_ms <= 0.0001:
        print(
            f"警告: profiler 无法获取有效时延数据（当前:{latency_ms} ms），将使用 time.perf_counter() 进行兜底测试..."
        )
        return measure_single_fallback(model, inputs, warmup, repeats, device)

    # 获取峰值内存
    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)

    return operators, latency_ms, round(peak_memory, 2)


def measure_single_fallback(
    model: Any, inputs: list[Any], warmup: int, repeats: int, device: Any
) -> tuple[Optional[dict[str, float]], Optional[float], float]:
    """使用time.perf_counter()的兜底测试机制"""
    import statistics
    import time

    import torch

    # 执行warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
    torch.npu.synchronize()

    # 正式测试
    latencies = []
    for _ in range(repeats):
        torch.npu.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(*inputs)
        torch.npu.synchronize()
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # 转换为毫秒

    # 计算平均时延
    avg_latency_ms = statistics.mean(latencies)

    # 获取峰值内存
    peak_memory = torch.npu.max_memory_allocated() / (1024 * 1024)

    # 兜底机制不获取算子级别的时延，返回空字典
    return {}, round(avg_latency_ms, 4), round(peak_memory, 2)


def compute_triton_kernel_coverage(matched_triton_kernels: list[str], profilling_result: dict[str, Any]):
    """Compute the coverage of the matched triton kernels in the profiling result."""

    def _matches_profiler_name(captured: str, profiler_name: str) -> bool:
        cap = captured.lower()
        prof = profiler_name.lower()
        if cap == prof:
            return True
        if cap in prof or prof in cap:
            return True
        return False

    kernels = matched_triton_kernels
    num_custom_kernels = 0
    kernel_names = [kernel.split(" ")[0] for kernel in kernels]

    kernels_in_profiling = profilling_result["kernels"]

    total_time = 0.0
    matched_cuda_time = 0.0
    triton_kernels_in_profiling = []

    for prof_kernel in kernels_in_profiling:
        prof_name = prof_kernel["name"]
        cuda_time = float(prof_kernel["device_time_us"])
        cpu_time = float(prof_kernel["cpu_time_us"])
        total_time += cuda_time + cpu_time

        if any(_matches_profiler_name(kernel_name, prof_name) for kernel_name in kernel_names):
            triton_kernels_in_profiling.append(prof_name)
            num_custom_kernels += 1
            matched_cuda_time += cuda_time

    triton_kernels_not_in_profiling = [
        kernel_name
        for kernel_name in kernel_names
        if not any(_matches_profiler_name(kernel_name, prof_name) for prof_name in triton_kernels_in_profiling)
    ]

    return {
        "num_custom_kernels": num_custom_kernels,
        "num_total_kernels": len(kernels_in_profiling),
        "total_kernel_run_time_in_profiling_us": total_time,
        "custom_kernel_cuda_time_in_profiling_us": matched_cuda_time,
        "triton_kernels_not_in_profiling": triton_kernels_not_in_profiling,
        "triton_kernels_in_profiling": triton_kernels_in_profiling,
    }


@contextmanager
def profiling_context(enabled: bool = True, num_warmup: int = 3, profile_path: str = "./"):
    if not enabled:
        yield None
        return

    try:
        import torch_npu
        import torch_npu.profiler as profiler

        activities = []
        if "cpu" in settings.profiling_activities:
            activities.append(profiler.ProfilerActivity.CPU)
        if "npu" in settings.profiling_activities:
            activities.append(profiler.ProfilerActivity.NPU)

        print(f"[Profiler] Initializing with activities: {[str(a) for a in activities]}")

        if not activities:
            print("[Profiler] No activities configured, profiler will return no data")
            yield None
            return

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            export_type=[torch_npu.profiler.ExportType.Text],
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            mstx=False,  # 原参数名msprof_tx改为mstx，新版本依旧兼容原参数名msprof_tx
            aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone,
            l2_cache=False,
            op_attr=True,
            data_simplification=False,
            record_op_args=False,
            gc_detect_threshold=None,
            host_sys=[],
            sys_io=False,
            sys_interconnection=False,
        )

        prof = profiler.profile(
            activities=activities,
            record_shapes=settings.profiling_record_shapes,
            profile_memory=settings.profiling_profile_memory,
            with_stack=settings.profiling_with_stack,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(profile_path),
            experimental_config=experimental_config,
            # schedule=torch_npu.profiler.schedule(wait=0, warmup=0, active=1, repeat=1, skip_first=1),
        )

        prof.__enter__()
        try:
            print("[Profiler] Profiler started successfully")
            npu_available = torch.npu.is_available()
            npu_visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
            device_info = "npu:unavailable"
            if npu_available:
                try:
                    current_device = torch.npu.current_device()
                    device_name = torch.npu.get_device_name(current_device)
                    device_info = f"npu:{current_device} ({device_name})"
                except Exception as e:
                    device_info = f"npu:unknown (error={e})"
            print(
                "[Profiler] Context pid=%s npu_available=%s device=%s ASCEND_RT_VISIBLE_DEVICES=%s",
                os.getpid(),
                npu_available,
                device_info,
                npu_visible,
            )
            if npu_available:
                try:
                    test = torch.ones((1024,), device="npu")
                    _ = test.sum()
                    torch.npu.synchronize()
                    print("[Profiler] Self-test NPU op executed")
                except Exception as e:
                    print(f"[Profiler] Self-test failed: {e}")
            yield prof
        finally:
            try:
                prof.__exit__(None, None, None)
                print("[Profiler] Profiler stopped successfully")
            except Exception as e:
                print(f"[Profiler] Error during profiler cleanup: {e}")

    except Exception as e:
        logger.warning(f"[Profiler] Failed to initialize profiler: {e}. Continuing without profiling.")
        yield None


def extract_profiling_metrics(base_dir: str) -> dict[str, Any]:
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")

    try:
        import pandas as pd

        op_time_file_path = None
        op_memory_file_path = None
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file == "operator_details.csv":
                    op_time_file_path = os.path.join(root, file)
                elif file == "memory_record.csv":
                    op_memory_file_path = os.path.join(root, file)
                else:
                    continue

        try:
            op_time_df = pd.read_csv(op_time_file_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
            print(f"Failed to read {op_time_file_path}: {e}")

        try:
            pd.read_csv(op_memory_file_path)
        except (pd.errors.EmptyDataError, pd.errors.ParserError, FileNotFoundError) as e:
            print(f"Failed to read {op_memory_file_path}: {e}")

        op_time_df["Input Shapes"] = op_time_df["Input Shapes"].fillna("")
        op_time_agg = op_time_df.groupby(["Name", "Input Shapes"]).mean()
        op_time_agg["count"] = op_time_df.groupby(["Name", "Input Shapes"]).size()
        op_time_agg = op_time_agg.reset_index()

        events = op_time_agg.to_dict("records")
        total_events = len(events)
        npu_device_event_count = 0
        npu_time_event_count = 0
        self_npu_time_event_count = 0

        logger.debug(f"[Profiler] Captured {total_events} total events")

        def _safe_metric(evt: Any, names: tuple[str, ...], default: float = 0.0) -> float:
            for name in names:
                # ===================== 修复点 =====================
                # 1. 如果是字典
                if isinstance(evt, dict):
                    if name in evt:
                        value = evt[name]
                    else:
                        continue
                # 2. 如果是对象（torch profiler event）
                else:
                    if hasattr(evt, name):
                        value = getattr(evt)
                    else:
                        continue
                # ==================================================

                if callable(value):
                    try:
                        value = value()
                    except Exception:
                        continue
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
            return default

        # int 版本也一起修复
        def _safe_int_metric(evt: Any, names: tuple[str, ...], default: int = 0) -> int:
            for name in names:
                if isinstance(evt, dict):
                    if name in evt:
                        value = evt[name]
                    else:
                        continue
                else:
                    if hasattr(evt, name):
                        value = getattr(evt)
                    else:
                        continue

                if callable(value):
                    try:
                        value = value()
                    except Exception:
                        continue
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
            return default

        npu_kernels = []
        total_cpu_time = 0.0
        total_self_cuda_time = 0.0
        for evt in events:
            cpu_time_us = _safe_metric(evt, ("Host Total Duration(us)", "cpu_time_total", "cpu_time"), 0.0)
            total_cpu_time += cpu_time_us
            device_time_us = _safe_metric(
                evt,
                ("Device Total Duration(us)", "device_time_total", "device_time", "cuda_time_total", "cuda_time"),
                0.0,
            )
            self_device_time_us = _safe_metric(
                evt,
                ("Device Self Duration(us)", "self_cuda_time_total", "self_cuda_time"),
                0.0,
            )
            if self_device_time_us > 0.0:
                self_npu_time_event_count += 1
                total_self_cuda_time += self_device_time_us
            if device_time_us <= 0.0:
                continue

            device_type = evt.get("Name", None)
            if "aclnn" in device_type:
                npu_device_event_count += 1
            npu_time_event_count += 1

            kernel_entry = {
                "name": evt.get("Name", "unknown"),
                "device_time_us": device_time_us,
                "cpu_time_us": cpu_time_us,
                "count": _safe_int_metric(evt, ("count",), 0),
            }
            print(kernel_entry)
            memory_usage = _safe_metric(evt, ("cuda_memory_usage",), 0.0)
            if memory_usage > 0.0:
                kernel_entry["cuda_memory_usage"] = memory_usage
            npu_kernels.append(kernel_entry)
        npu_kernels.sort(key=lambda x: x["device_time_us"], reverse=True)

        logger.debug(f"[Profiler] Filtered to {len(npu_kernels)} CUDA kernels (from {len(events)} total)")
        if len(npu_kernels) == 0 and len(events) > 0:
            logger.warning(
                f"[Profiler] Captured events but no CUDA kernels! Event types: {[getattr(evt, 'device_type', 'unknown') for evt in list(events)[:5]]}"  # noqa: E501
            )

        memory_stats = {}
        try:
            if torch.npu.is_available():
                device = torch.npu.current_device()
                memory_stats = {
                    "allocated_mb": torch.npu.memory_allocated(device) / (1024 * 1024),
                    "reserved_mb": torch.npu.memory_reserved(device) / (1024 * 1024),
                    "max_allocated_mb": torch.npu.max_memory_allocated(device) / (1024 * 1024),
                    "max_reserved_mb": torch.npu.max_memory_reserved(device) / (1024 * 1024),
                }
        except Exception as e:
            logger.warning(f"[Profiler] Failed to collect memory stats: {e}")

        profiling_metrics = {
            "kernels": npu_kernels,
            "kernel_count": len(npu_kernels),
            "total_cpu_time_us": total_cpu_time,
            "total_device_time_us": sum(k["device_time_us"] for k in npu_kernels),
            "total_self_device_time_us": total_self_cuda_time,
            "npu_device_event_count": npu_device_event_count,
            "npu_time_event_count": npu_time_event_count,
            "self_npu_time_event_count": self_npu_time_event_count,
            "memory_stats": memory_stats,
        }

        if len(npu_kernels) == 0:
            profiling_metrics["profiling_warning"] = (
                "Profiler captured no CUDA kernels. This may indicate a profiler failure."
            )
        return profiling_metrics

    except Exception as e:
        logger.warning(f"[Profiler] Failed to extract profiling metrics: {e}")
        return {"profiling_error": str(e)}
