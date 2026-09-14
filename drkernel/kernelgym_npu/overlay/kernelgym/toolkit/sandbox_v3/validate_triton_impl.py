#!/usr/bin/env python3
# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
"""Triton 实现退化检测脚本 — 通过 AST 静态分析检查生成代码是否退化为 PyTorch 原生实现。

检测三种退化类型：
  Type 1: 无 @triton.jit kernel，全部使用 PyTorch
  Type 2: 有 @triton.jit kernel 定义但 forward() 未调用
  Type 3: forward() 调用了 kernel 但仍有部分计算使用 torch 接口

用法:
    python validate_triton_impl.py <file_path> [--json]

退出码: 0 = 通过, 1 = 检测到退化
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

# Deferred so this module imports with only the stdlib ``ast`` — ``validate()``
# is pure static analysis and stays torch-free (unit-testable off-device, and
# importable by sandbox_v4 without pulling torch). The concrete type is only
# needed for the annotation on ``fillKernelExecResult``.
if TYPE_CHECKING:
    from kernelgym.toolkit.kernelbench.exec_types import KernelExecResult

# ---------------------------------------------------------------------------
# 白名单：forward() 中允许的 torch 调用和 tensor 方法
# ---------------------------------------------------------------------------

ALLOWED_TORCH_FUNCS = {
    # buffer 分配
    "empty",
    "empty_like",
    "empty_strided",
    "zeros",
    "zeros_like",
    "ones",
    "ones_like",
    "full",
    "full_like",
    # tensor 创建（有时需要用于标量常量 / 索引）
    "tensor",
    "arange",
    "linspace",
    # 类型 / 设备
    "as_tensor",
}

ALLOWED_TENSOR_METHODS = {
    # 形状 / 元信息
    "size",
    "shape",
    "stride",
    "numel",
    "dtype",
    "device",
    "dim",
    "is_contiguous",
    "data_ptr",
    "element_size",
    "storage_offset",
    # 布局操作（不执行计算）
    "contiguous",
    "to",
    "view",
    "view_as",
    "reshape",
    "permute",
    "transpose",
    "expand",
    "expand_as",
    "flatten",
    "unflatten",
    "unsqueeze",
    "squeeze",
    "narrow",
    "clone",
    "detach",
    "t",
    "type",
    "float",
    "half",
    "bfloat16",
    "int",
    "long",
    "bool",
    "double",
    "cpu",
    "npu",
    "cuda",
    "item",
    "tolist",
    # 原地标记
    "requires_grad_",
    "zero_",
    # 切片相关（一般通过 __getitem__ 而非方法，但以防万一）
    "index_select",
}

ALLOWED_TRITON_ATTRS = {
    "cdiv",
    "next_power_of_2",
}

FORBIDDEN_TENSOR_METHODS = {
    # 计算操作
    "sum",
    "mean",
    "max",
    "min",
    "softmax",
    "log_softmax",
    "matmul",
    "mm",
    "bmm",
    "addmm",
    "add",
    "sub",
    "mul",
    "div",
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "silu",
    "elu",
    "leaky_relu",
    "exp",
    "log",
    "log2",
    "log10",
    "sqrt",
    "pow",
    "abs",
    "norm",
    "layer_norm",
    "batch_norm",
    "group_norm",
    "conv1d",
    "conv2d",
    "conv3d",
    "conv_transpose2d",
    "linear",
    "dropout",
    "softplus",
    "hardtanh",
    "hardswish",
}

# forward() 中禁止的 Python 控制流和结构
FORBIDDEN_PYTHON_STMTS = {
    "for": "Python for 循环",
    "while": "Python while 循环",
}


# ---------------------------------------------------------------------------
# AST 辅助函数
# ---------------------------------------------------------------------------


def _decorator_is_triton_jit(decorator):
    """判断装饰器节点是否为 triton.jit 或 @jit（从 triton 导入）。"""
    # @triton.jit
    if isinstance(decorator, ast.Attribute):
        if isinstance(decorator.value, ast.Name) and decorator.value.id == "triton" and decorator.attr == "jit":
            return True
    # @jit（直接导入）
    if isinstance(decorator, ast.Name) and decorator.id == "jit":
        return True
    # @triton.jit 作为 Call（如 @triton.jit 带参数，虽然少见）
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_jit(decorator.func)
    return False


def _decorator_is_triton_autotune(decorator):
    """判断装饰器是否为 triton.autotune。"""
    if isinstance(decorator, ast.Attribute):
        if isinstance(decorator.value, ast.Name) and decorator.value.id == "triton" and decorator.attr == "autotune":
            return True
    if isinstance(decorator, ast.Call):
        return _decorator_is_triton_autotune(decorator.func)
    return False


def _has_triton_decorator(func_node):
    """检查函数是否有 @triton.jit（可能与 @triton.autotune 组合）。"""
    for dec in func_node.decorator_list:
        if _decorator_is_triton_jit(dec):
            return True
    return False


def _resolve_call_name(node):
    """尝试从 ast.Call 节点提取被调用函数的名称字符串。

    返回 (qualifier, attr) 或 (None, name) 或 None。
    例如：torch.empty -> ('torch', 'empty')
          my_func    -> (None, 'my_func')
          self.conv  -> ('self', 'conv')
          kernel[g]  -> 返回 None（kernel launch 通过 Subscript）
    """
    func = node.func if isinstance(node, ast.Call) else node
    if isinstance(func, ast.Attribute):
        if isinstance(func.value, ast.Name):
            return (func.value.id, func.attr)
        # 处理 torch.nn.functional.relu 形式
        if isinstance(func.value, ast.Attribute):
            inner = func.value
            if isinstance(inner.value, ast.Name):
                return (f"{inner.value.id}.{inner.attr}", func.attr)
    if isinstance(func, ast.Name):
        return (None, func.id)
    return None


def _get_subscript_value_name(node):
    """从 kernel[grid](...) 的 Subscript 节点提取 kernel 名称。"""
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name):
            return node.value.id
        if isinstance(node.value, ast.Attribute):
            if isinstance(node.value.value, ast.Name):
                return f"{node.value.value.id}.{node.value.attr}"
    return None


# ---------------------------------------------------------------------------
# 核心检查
# ---------------------------------------------------------------------------


def find_triton_kernels(tree):
    """查找所有 @triton.jit 装饰的函数名，及其是否使用了 tl.* API。"""
    kernels = {}  # name -> {"has_tl_usage": bool, "line": int}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and _has_triton_decorator(node):
            # 检查函数体中是否使用 tl.* API
            has_tl = False
            for child in ast.walk(node):
                if isinstance(child, ast.Attribute):
                    if isinstance(child.value, ast.Name) and child.value.id == "tl":
                        has_tl = True
                        break
            kernels[node.name] = {"has_tl_usage": has_tl, "line": node.lineno}
    return kernels


def find_model_new_forward(tree):
    """找到 ModelNew 类的 forward 方法节点。"""
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelNew":
            for item in node.body:
                if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if item.name == "forward":
                        return item
    return None


def find_wrapper_functions(tree, kernel_names):
    """找到模块级别或类级别的辅助函数，这些函数内部调用了 triton kernel。

    返回函数名集合。
    """
    wrappers = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name not in kernel_names:
            for child in ast.walk(node):
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Subscript):
                    name = _get_subscript_value_name(child.func)
                    if name in kernel_names:
                        wrappers.add(node.name)
                        break
                if isinstance(child, ast.Call):
                    resolved = _resolve_call_name(child)
                    if resolved and resolved[0] is None and resolved[1] in kernel_names:
                        wrappers.add(node.name)
                        break
    return wrappers


def check_kernel_calls_in_forward(forward_node, kernel_names, wrapper_names):
    """检查 forward 中是否调用了 triton kernel（直接或通过 wrapper）。

    返回被调用的 kernel/wrapper 名称集合。
    """
    called = set()
    if forward_node is None:
        return called
    for node in ast.walk(forward_node):
        # kernel[grid](...) 形式
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            name = _get_subscript_value_name(node.func)
            if name in kernel_names:
                called.add(name)
        # 直接调用 kernel_name(...) — 虽然不标准但以防万一
        if isinstance(node, ast.Call):
            resolved = _resolve_call_name(node)
            if resolved:
                qual, attr = resolved
                if qual is None and attr in kernel_names:
                    called.add(attr)
                if qual is None and attr in wrapper_names:
                    called.add(attr)
                # self.wrapper_name(...)
                if qual == "self" and attr in wrapper_names:
                    called.add(attr)
    return called


def _count_kernel_launches_in_forward(forward_node):
    """统计 forward() 中 kernel 启动调用（kernel[grid](...)）的次数。"""
    count = 0
    if forward_node is None:
        return count
    for node in ast.walk(forward_node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript):
            count += 1
    return count


def check_forbidden_torch_ops(forward_node):
    """检查 forward 中是否使用了禁止的 torch 计算操作或 Python 控制流。

    返回违规列表 [{"line": N, "call": str, "reason": str}, ...]
    """
    violations = []
    if forward_node is None:
        return violations

    # --- 规则 A: forward() 中禁止 Python 循环（for/while）---
    # 例外：如果 forward() 中只有一个 kernel 启动，允许简单的固定次数循环
    #      （如 for _ in range(1) 这种无意义循环仍会被检测）
    kernel_launch_count = _count_kernel_launches_in_forward(forward_node)

    for node in ast.walk(forward_node):
        if isinstance(node, ast.For):
            violations.append(
                {
                    "line": node.lineno,
                    "call": "for 循环",
                    "reason": "forward() 中禁止 Python for 循环，核心计算必须在单个 Triton kernel 内完成",
                }
            )
            continue
        if isinstance(node, ast.While):
            violations.append(
                {
                    "line": node.lineno,
                    "call": "while 循环",
                    "reason": "forward() 中禁止 Python while 循环，核心计算必须在单个 Triton kernel 内完成",
                }
            )
            continue

        # --- 检测 @ 运算符（矩阵乘法）---
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            violations.append(
                {
                    "line": node.lineno,
                    "call": "@",
                    "reason": "矩阵乘法 @ 运算符必须在 Triton kernel 中实现",
                }
            )
            continue

        # --- 检测 list.append — 表明在 host 端维护动态状态 ---
        if isinstance(node, ast.Call):
            resolved = _resolve_call_name(node)
            if resolved:
                qual, attr = resolved
                if attr == "append" and qual is not None:
                    violations.append(
                        {
                            "line": node.lineno,
                            "call": f"{qual}.append(...)",
                            "reason": "forward() 中禁止 list.append，动态状态维护必须在 Triton kernel 内完成",
                        }
                    )
                    continue

        if not isinstance(node, ast.Call):
            continue

        # --- kernel launch: kernel[grid](...) —— 允许 ---
        if isinstance(node.func, ast.Subscript):
            continue

        resolved = _resolve_call_name(node)
        if resolved is None:
            continue

        qual, attr = resolved

        # --- torch.xxx(...) ---
        if qual == "torch":
            if attr not in ALLOWED_TORCH_FUNCS:
                violations.append(
                    {
                        "line": node.lineno,
                        "call": f"torch.{attr}",
                        "reason": f"torch.{attr} 是计算操作，必须在 Triton kernel 中实现",
                    }
                )
            continue

        # --- F.xxx(...) / functional.xxx(...) ---
        if qual in ("F", "functional", "torch.nn.functional", "nn.functional"):
            violations.append(
                {
                    "line": node.lineno,
                    "call": f"{qual}.{attr}",
                    "reason": f"{qual}.{attr} 是 PyTorch 计算操作，必须在 Triton kernel 中实现",
                }
            )
            continue

        # --- triton.cdiv 等 —— 允许 ---
        if qual == "triton" and attr in ALLOWED_TRITON_ATTRS:
            continue

        # --- tensor 方法计算操作 ---
        if attr in FORBIDDEN_TENSOR_METHODS:
            # 排除已知安全的 qual（torch/F/triton 已在上面处理）
            if qual not in ("torch", "F", "triton", "functional", "torch.nn.functional", "nn.functional"):
                violations.append(
                    {
                        "line": node.lineno,
                        "call": f"{qual}.{attr}()" if qual else f"{attr}()",
                        "reason": f"{attr} 是计算操作，必须在 Triton kernel 中实现",
                    }
                )
            continue

        # --- self.layer_name(x) —— 禁止 nn.Module 调用 ---
        if qual == "self":
            # 允许 self.forward() 递归，以及属性访问不在这里（ast.Attribute 不是 Call）
            # self.xxx(...) 形式视为 nn.Module 前向调用
            if attr not in ("forward",):
                violations.append(
                    {
                        "line": node.lineno,
                        "call": f"self.{attr}(...)",
                        "reason": f"self.{attr}() 疑似 nn.Module 前向调用，核心计算必须在 Triton kernel 中实现",
                    }
                )
            continue

    # --- 规则 B: 如果 forward() 中 kernel 启动次数 > 1，视为 Type3 退化 ---
    if kernel_launch_count > 1:
        violations.append(
            {
                "line": forward_node.lineno,
                "call": f"kernel 启动 {kernel_launch_count} 次",
                "reason": "forward() 中只能启动一次 Triton kernel，多次启动表明核心计算在 host 端循环中完成",
            }
        )

    return violations


# ---------------------------------------------------------------------------
# 主验证逻辑
# ---------------------------------------------------------------------------


def validate(code, check_forbidden_fn=check_forbidden_torch_ops):
    """对生成代码执行完整的退化检查。

    返回结构化结果 dict。

    ``check_forbidden_fn`` 是 forward() 禁止操作检查器（默认 sandbox_v3 的
    ``check_forbidden_torch_ops``）。sandbox_v4 通过注入自己的检查器来放宽白名单
    （允许 matmul/conv/SDPA 等 extern 算子），从而复用本函数的整体编排逻辑。
    """
    result = {
        "valid": False,
        "checks": {
            "triton_kernel_exists": {"passed": False, "kernels": [], "error": None},
            "kernel_called_from_forward": {"passed": False, "called": [], "error": None},
            "no_forbidden_torch_ops": {"passed": False, "violations": [], "error": None},
        },
        "regression_type": None,
        "suggestion": "",
    }

    # --- 解析 ---
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        result["checks"]["triton_kernel_exists"]["error"] = f"SyntaxError: {e}"
        result["regression_type"] = 1
        result["suggestion"] = "代码存在语法错误，无法解析。"
        return result

    # --- Check 1: kernel 存在性 ---
    kernels = find_triton_kernels(tree)
    kernel_names = set(kernels.keys())
    result["checks"]["triton_kernel_exists"]["kernels"] = [
        {"name": k, "line": v["line"], "has_tl_usage": v["has_tl_usage"]} for k, v in kernels.items()
    ]

    if not kernel_names:
        result["checks"]["triton_kernel_exists"]["error"] = "未找到任何 @triton.jit 装饰的 kernel 函数"
        result["regression_type"] = 1
        result["suggestion"] = (
            "代码中没有 Triton kernel。必须创建至少一个 @triton.jit 装饰的函数，"
            "在其中使用 tl.load/tl.store 实现核心计算逻辑。"
        )
        return result

    # 检查 kernel 是否使用了 tl API
    kernels_without_tl = [k for k, v in kernels.items() if not v["has_tl_usage"]]
    if len(kernels_without_tl) == len(kernels):
        result["checks"]["triton_kernel_exists"]["error"] = (
            f"kernel 函数 {kernels_without_tl} 未使用任何 tl.* API，可能是空壳 kernel"
        )
        result["regression_type"] = 1
        result["suggestion"] = (
            "虽然存在 @triton.jit 装饰的函数，但没有使用 triton.language (tl) API。"
            "kernel 必须使用 tl.load/tl.store 等进行显式内存操作和计算。"
        )
        return result

    result["checks"]["triton_kernel_exists"]["passed"] = True

    # --- Check 2: forward 是否调用 kernel ---
    forward_node = find_model_new_forward(tree)
    if forward_node is None:
        result["checks"]["kernel_called_from_forward"]["error"] = "未找到 ModelNew.forward() 方法"
        result["regression_type"] = 2
        result["suggestion"] = "代码缺少 ModelNew 类或 forward 方法。"
        return result

    wrapper_names = find_wrapper_functions(tree, kernel_names)
    called = check_kernel_calls_in_forward(forward_node, kernel_names, wrapper_names)
    result["checks"]["kernel_called_from_forward"]["called"] = list(called)

    if not called:
        result["checks"]["kernel_called_from_forward"]["error"] = (
            f"@triton.jit kernel {list(kernel_names)} 已定义但 forward() 未调用任何 kernel"
        )
        result["regression_type"] = 2
        result["suggestion"] = (
            f"已定义 kernel {list(kernel_names)} 但 ModelNew.forward() 中未调用。"
            "forward() 必须通过 kernel_name[grid](...) 形式启动 kernel。"
            f"{'也存在 wrapper 函数 ' + str(list(wrapper_names)) + ' 但 forward 也未调用它们。' if wrapper_names else ''}"  # noqa: E501
        )
        return result

    result["checks"]["kernel_called_from_forward"]["passed"] = True

    # --- Check 3: 禁止的 torch 操作 ---
    violations = check_forbidden_fn(forward_node)
    result["checks"]["no_forbidden_torch_ops"]["violations"] = violations

    if violations:
        result["checks"]["no_forbidden_torch_ops"]["error"] = (
            f"forward() 中发现 {len(violations)} 处禁止的 PyTorch 计算操作"
        )
        violation_details = "; ".join(f"第{v['line']}行 {v['call']}" for v in violations[:5])
        result["regression_type"] = 3
        result["suggestion"] = (
            f"forward() 调用了 Triton kernel 但仍使用 PyTorch 进行部分计算: "
            f"{violation_details}。"
            "所有核心计算必须在 @triton.jit kernel 中完成，"
            "forward() 中只允许 buffer 分配（torch.empty 等）和形状操作（.view/.reshape 等）。"
        )
        return result

    result["checks"]["no_forbidden_torch_ops"]["passed"] = True

    # --- 全部通过 ---
    result["valid"] = True
    return result


# ---------------------------------------------------------------------------
# 填充防作弊result
# ---------------------------------------------------------------------------


def fillKernelExecResult(kernel_exec_result: KernelExecResult, code):
    result = validate(code)

    if result["valid"]:
        kernels = result["checks"]["triton_kernel_exists"]["kernels"]
        called = result["checks"]["kernel_called_from_forward"]["called"]
        print("[PASS] Triton 实现验证通过")
        print(f"  - 发现 {len(kernels)} 个 @triton.jit kernel: {', '.join(k['name'] for k in kernels)}")
        print(f"  - forward() 调用: {', '.join(called)}")
        print("  - forward() 中无禁止的 PyTorch 计算操作")
    else:
        rtype = result["regression_type"]
        type_desc = {
            1: "完全无 Triton kernel（纯 PyTorch）",
            2: "有 Triton kernel 但 forward() 未调用",
            3: "部分计算使用 PyTorch（需全部移入 Triton kernel）",
        }
        print(f"[FAIL] 检测到 PyTorch 退化 — Type {rtype}: {type_desc.get(rtype, '未知')}")

        # 显示具体检查结果
        for check_name, check_result in result["checks"].items():
            status = "PASS" if check_result["passed"] else "FAIL"
            print(f"  [{status}] {check_name}")
            if check_result["error"]:
                print(f"         {check_result['error']}")

        if result["checks"]["no_forbidden_torch_ops"]["violations"]:
            print("  违规详情:")
            for v in result["checks"]["no_forbidden_torch_ops"]["violations"]:
                print(f"    第 {v['line']} 行: {v['call']} — {v['reason']}")

        print(f"\n  修复建议: {result['suggestion']}")

        # Surface the decoy detail to the rollout/feedback side via metadata.
        # The coach renderer (`recipe/drkernel/agent/kernel_eval.py::_coach_decoy`)
        # reads `decoy_regression_type` + `decoy_violations` to tell the model
        # which degeneration type tripped and which forward() lines to fix.
        # Only line + call are forwarded — the Chinese `reason`/`suggestion`
        # strings stay server-side (the renderer generates its own English).
        kernel_exec_result.metadata["decoy_regression_type"] = rtype
        kernel_exec_result.metadata["decoy_violations"] = [
            {"line": v.get("line"), "call": v.get("call")}
            for v in result["checks"]["no_forbidden_torch_ops"]["violations"]
        ]
    kernel_exec_result.decoy_kernel = not result["valid"]  # decoy_kernel这个字段就是代表是否作弊，True就是作弊
