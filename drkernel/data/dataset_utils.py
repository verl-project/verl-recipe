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
Kernel 数据处理工具
用于处理 kernel 代码生成的数据集
"""

import json
import re
from typing import Any, Optional


def process_kernel_data(data_item: dict[str, Any]) -> dict[str, Any]:
    """
    处理单个 kernel 数据项

    Args:
        data_item: 原始数据项

    Returns:
        处理后的数据项
    """
    # 标准化字段名
    processed_item = {
        "prompt": data_item.get("prompt", data_item.get("instruction", "")),
        "reference_code": data_item.get("reference_code", data_item.get("reference", "")),
        "description": data_item.get("description", ""),
        "test_cases": data_item.get("test_cases", []),
        "requirements": data_item.get("requirements", {}),
        "data_source": data_item.get("data_source", "kernel"),
        "difficulty": data_item.get("difficulty", "medium"),
        "tags": data_item.get("tags", []),
    }

    # 确保必要字段存在
    if not processed_item["prompt"]:
        processed_item["prompt"] = "Generate an optimized CUDA kernel implementation."

    if not processed_item["reference_code"]:
        processed_item["reference_code"] = "# No reference code provided"

    return processed_item


def create_kernel_prompt(
    description: str,
    reference_code: str,
    requirements: Optional[dict[str, Any]] = None,
    test_cases: Optional[list[dict[str, Any]]] = None,
) -> str:
    """
    创建 kernel 生成提示

    Args:
        description: 任务描述
        reference_code: 参考实现
        requirements: 性能和约束要求
        test_cases: 测试用例

    Returns:
        格式化的提示字符串
    """
    prompt_parts = []

    # 任务描述
    prompt_parts.append("# Task Description")
    prompt_parts.append(description)
    prompt_parts.append("")

    # 参考实现
    if reference_code and reference_code.strip():
        prompt_parts.append("# Reference Implementation")
        prompt_parts.append("```python")
        prompt_parts.append(reference_code)
        prompt_parts.append("```")
        prompt_parts.append("")

    # 性能要求
    if requirements:
        prompt_parts.append("# Performance Requirements")
        for key, value in requirements.items():
            prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")

    # 测试用例
    if test_cases:
        prompt_parts.append("# Test Cases")
        for i, test_case in enumerate(test_cases):
            prompt_parts.append(f"## Test Case {i + 1}")
            if "input" in test_case:
                prompt_parts.append(f"Input: {test_case['input']}")
            if "expected_output" in test_case:
                prompt_parts.append(f"Expected Output: {test_case['expected_output']}")
            if "description" in test_case:
                prompt_parts.append(f"Description: {test_case['description']}")
            prompt_parts.append("")

    # 生成任务
    prompt_parts.append("# Your Task")
    prompt_parts.append("Please generate an optimized CUDA kernel implementation that:")
    prompt_parts.append("1. Achieves better performance than the reference implementation")
    prompt_parts.append("2. Maintains correctness and numerical stability")
    prompt_parts.append("3. Follows CUDA best practices")
    prompt_parts.append("4. Is well-documented with comments")
    prompt_parts.append("")
    prompt_parts.append("# Generated Kernel Implementation")
    prompt_parts.append("```python")

    return "\n".join(prompt_parts)


def extract_kernel_requirements(prompt: str) -> dict[str, Any]:
    """
    从提示中提取 kernel 要求

    Args:
        prompt: 提示字符串

    Returns:
        提取的要求字典
    """
    requirements = {}

    # 提取性能要求
    perf_pattern = r"# Performance Requirements\s*\n(.*?)(?=\n#|\n$)"
    perf_match = re.search(perf_pattern, prompt, re.DOTALL)
    if perf_match:
        perf_lines = perf_match.group(1).strip().split("\n")
        for line in perf_lines:
            if line.strip().startswith("-"):
                parts = line.strip()[1:].split(":")
                if len(parts) >= 2:
                    key = parts[0].strip()
                    value = ":".join(parts[1:]).strip()
                    requirements[key] = value

    # 提取测试用例
    test_pattern = r"# Test Cases\s*\n(.*?)(?=\n#|\n$)"
    test_match = re.search(test_pattern, prompt, re.DOTALL)
    if test_match:
        requirements["has_test_cases"] = True

    # 提取任务描述
    desc_pattern = r"# Task Description\s*\n(.*?)(?=\n#|\n$)"
    desc_match = re.search(desc_pattern, prompt, re.DOTALL)
    if desc_match:
        requirements["description"] = desc_match.group(1).strip()

    return requirements


def load_kernel_dataset(dataset_path: str) -> list[dict[str, Any]]:
    """
    加载 kernel 数据集

    Args:
        dataset_path: 数据集路径

    Returns:
        处理后的数据集
    """
    dataset = []

    try:
        with open(dataset_path, encoding="utf-8") as f:
            # 支持 JSON Lines 格式
            if dataset_path.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        data_item = json.loads(line)
                        processed_item = process_kernel_data(data_item)
                        dataset.append(processed_item)
            else:
                # JSON 格式
                data = json.load(f)
                if isinstance(data, list):
                    for data_item in data:
                        processed_item = process_kernel_data(data_item)
                        dataset.append(processed_item)
                else:
                    processed_item = process_kernel_data(data)
                    dataset.append(processed_item)

    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        return []

    return dataset


def validate_kernel_code(kernel_code: str) -> dict[str, Any]:
    """
    验证 kernel 代码的基本格式

    Args:
        kernel_code: kernel 代码字符串

    Returns:
        验证结果
    """
    result = {"valid": True, "errors": [], "warnings": []}

    # 基本检查
    if not kernel_code.strip():
        result["valid"] = False
        result["errors"].append("Empty kernel code")
        return result

    # 检查是否包含必要的导入
    required_imports = ["torch", "cuda"]
    for imp in required_imports:
        if imp not in kernel_code:
            result["warnings"].append(f"Missing import: {imp}")

    # 检查是否包含 kernel 函数定义
    if "def " not in kernel_code:
        result["warnings"].append("No function definition found")

    # 检查是否包含 CUDA 特定代码
    cuda_keywords = ["cuda", "gpu", "device", "kernel", "block", "thread"]
    if not any(keyword in kernel_code.lower() for keyword in cuda_keywords):
        result["warnings"].append("No CUDA-specific keywords found")

    return result
