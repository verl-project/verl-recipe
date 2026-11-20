# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Kernel 提示生成工具
专门用于生成 kernel 代码的提示模板
"""

from typing import Any, Optional


def generate_kernel_prompt(
    task_type: str,
    reference_code: str,
    operation_name: str,
    input_shapes: Optional[list[str]] = None,
    constraints: Optional[dict[str, Any]] = None,
    performance_target: Optional[str] = None,
) -> str:
    """
    生成 kernel 优化提示

    Args:
        task_type: 任务类型 (e.g., "element_wise", "reduction", "matrix_mult")
        reference_code: 参考 PyTorch 实现
        operation_name: 操作名称
        input_shapes: 输入张量形状
        constraints: 约束条件
        performance_target: 性能目标

    Returns:
        格式化的提示字符串
    """
    prompt_parts = []

    # 标题
    prompt_parts.append(f"# CUDA Kernel Optimization Task: {operation_name}")
    prompt_parts.append("")

    # 任务描述
    task_descriptions = {
        "element_wise": "Optimize element-wise operations for better memory access patterns and parallelization.",
        "reduction": "Implement efficient reduction operations with proper shared memory usage and warp-level primitives.",  # noqa: E501
        "matrix_mult": "Optimize matrix multiplication with tiling, shared memory, and register blocking techniques.",
        "convolution": "Implement efficient convolution operations with optimal memory layout and computation patterns.",  # noqa: E501
        "attention": "Optimize attention mechanisms with fused operations and memory-efficient implementations.",
    }

    description = task_descriptions.get(task_type, "Optimize the given operation for better GPU performance.")
    prompt_parts.append("## Task Description")
    prompt_parts.append(description)
    prompt_parts.append("")

    # 参考实现
    prompt_parts.append("## Reference PyTorch Implementation")
    prompt_parts.append("```python")
    prompt_parts.append(reference_code)
    prompt_parts.append("```")
    prompt_parts.append("")

    # 输入信息
    if input_shapes:
        prompt_parts.append("## Input Information")
        for i, shape in enumerate(input_shapes):
            prompt_parts.append(f"- Input {i + 1}: {shape}")
        prompt_parts.append("")

    # 约束条件
    if constraints:
        prompt_parts.append("## Constraints")
        for key, value in constraints.items():
            prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")

    # 性能目标
    if performance_target:
        prompt_parts.append("## Performance Target")
        prompt_parts.append(f"- {performance_target}")
        prompt_parts.append("")

    # 优化指导
    prompt_parts.append("## Optimization Guidelines")

    if task_type == "element_wise":
        prompt_parts.extend(
            [
                "- Ensure coalesced memory access patterns",
                "- Use appropriate block and grid dimensions",
                "- Consider vectorized loads/stores when possible",
                "- Optimize for high arithmetic intensity",
            ]
        )
    elif task_type == "reduction":
        prompt_parts.extend(
            [
                "- Use shared memory for efficient reduction",
                "- Implement warp-level primitives for final stages",
                "- Consider multiple elements per thread",
                "- Handle boundary conditions efficiently",
            ]
        )
    elif task_type == "matrix_mult":
        prompt_parts.extend(
            [
                "- Use tiling to maximize data reuse",
                "- Implement shared memory blocking",
                "- Optimize register usage",
                "- Consider tensor core utilization if applicable",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "- Maximize memory bandwidth utilization",
                "- Optimize thread block configuration",
                "- Minimize shared memory conflicts",
                "- Use appropriate synchronization primitives",
            ]
        )

    prompt_parts.append("")

    # 生成任务
    prompt_parts.append("## Your Task")
    prompt_parts.append("Generate an optimized CUDA kernel implementation that:")
    prompt_parts.append("1. Achieves significantly better performance than the reference")
    prompt_parts.append("2. Maintains numerical correctness")
    prompt_parts.append("3. Follows CUDA best practices")
    prompt_parts.append("4. Includes proper error handling")
    prompt_parts.append("5. Is well-documented with clear comments")
    prompt_parts.append("")

    # 期望输出格式
    prompt_parts.append("## Expected Output Format")
    prompt_parts.append("```python")
    prompt_parts.append("import torch")
    prompt_parts.append("import torch.nn as nn")
    prompt_parts.append("from torch.utils.cpp_extension import load_inline")
    prompt_parts.append("")
    prompt_parts.append("# Your optimized kernel implementation here")
    prompt_parts.append("cuda_kernel = '''")
    prompt_parts.append("__global__ void optimized_kernel(...) {")
    prompt_parts.append("    // Your CUDA kernel code here")
    prompt_parts.append("}")
    prompt_parts.append("'''")
    prompt_parts.append("")
    prompt_parts.append("# Python wrapper")
    prompt_parts.append("def optimized_operation(inputs):")
    prompt_parts.append("    # Your wrapper code here")
    prompt_parts.append("    return result")
    prompt_parts.append("```")

    return "\n".join(prompt_parts)


def extract_kernel_requirements(prompt: str) -> dict[str, Any]:
    """
    从提示中提取 kernel 要求

    Args:
        prompt: 提示字符串

    Returns:
        提取的要求字典
    """
    requirements = {
        "task_type": "general",
        "operation_name": "unknown",
        "has_constraints": False,
        "has_performance_target": False,
        "input_shapes": [],
        "optimization_guidelines": [],
    }

    # 提取任务类型
    if "element_wise" in prompt.lower():
        requirements["task_type"] = "element_wise"
    elif "reduction" in prompt.lower():
        requirements["task_type"] = "reduction"
    elif "matrix" in prompt.lower():
        requirements["task_type"] = "matrix_mult"
    elif "convolution" in prompt.lower():
        requirements["task_type"] = "convolution"
    elif "attention" in prompt.lower():
        requirements["task_type"] = "attention"

    # 提取操作名称
    import re

    name_pattern = r"# CUDA Kernel Optimization Task: (.+)"
    name_match = re.search(name_pattern, prompt)
    if name_match:
        requirements["operation_name"] = name_match.group(1).strip()

    # 检查是否有约束条件
    if "## Constraints" in prompt:
        requirements["has_constraints"] = True

    # 检查是否有性能目标
    if "## Performance Target" in prompt:
        requirements["has_performance_target"] = True

    # 提取输入形状信息
    input_pattern = r"## Input Information\s*\n(.*?)(?=\n##|\n$)"
    input_match = re.search(input_pattern, prompt, re.DOTALL)
    if input_match:
        input_lines = input_match.group(1).strip().split("\n")
        for line in input_lines:
            if line.strip().startswith("-"):
                requirements["input_shapes"].append(line.strip()[1:].strip())

    return requirements


def create_kernel_evaluation_prompt(
    original_code: str, optimized_code: str, test_inputs: Optional[list[str]] = None
) -> str:
    """
    创建 kernel 评估提示

    Args:
        original_code: 原始参考代码
        optimized_code: 优化后的代码
        test_inputs: 测试输入

    Returns:
        评估提示字符串
    """
    prompt_parts = []

    prompt_parts.append("# Kernel Performance Evaluation")
    prompt_parts.append("")

    prompt_parts.append("## Original Reference Code")
    prompt_parts.append("```python")
    prompt_parts.append(original_code)
    prompt_parts.append("```")
    prompt_parts.append("")

    prompt_parts.append("## Optimized Kernel Code")
    prompt_parts.append("```python")
    prompt_parts.append(optimized_code)
    prompt_parts.append("```")
    prompt_parts.append("")

    if test_inputs:
        prompt_parts.append("## Test Inputs")
        for i, test_input in enumerate(test_inputs):
            prompt_parts.append(f"### Test Case {i + 1}")
            prompt_parts.append("```python")
            prompt_parts.append(test_input)
            prompt_parts.append("```")
            prompt_parts.append("")

    prompt_parts.append("## Evaluation Criteria")
    prompt_parts.append("1. **Correctness**: Does the optimized kernel produce the same results?")
    prompt_parts.append("2. **Performance**: How much speedup is achieved?")
    prompt_parts.append("3. **Memory Usage**: Is memory usage optimized?")
    prompt_parts.append("4. **Code Quality**: Is the code well-structured and documented?")

    return "\n".join(prompt_parts)
