# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""GSM8K reward function: extract final numerical answer and compare with ground truth."""

import re


def extract_answer(text):
    """Extract the final numerical answer from a model response."""
    # Look for #### pattern (GSM8K standard format)
    match = re.search(r"####\s*([\-\d,\.]+)", text)
    if match:
        return match.group(1).replace(",", "").strip()

    # Fallback: extract last number in text
    numbers = re.findall(r"[\-\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()

    return None


def extract_gt_answer(ground_truth):
    """Extract answer from GSM8K ground truth string (after ####)."""
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("ground_truth", "")
    match = re.search(r"####\s*([\-\d,\.]+)", str(ground_truth))
    if match:
        return match.group(1).replace(",", "").strip()
    return str(ground_truth).strip()


def compute_score(data_source, solution_str, ground_truth=None, extra_info=None):
    """Return 1.0 if predicted answer matches ground truth, 0.0 otherwise."""
    if ground_truth is None:
        return 0.0

    pred = extract_answer(solution_str)
    gt = extract_gt_answer(ground_truth)

    if pred is None:
        return 0.0

    try:
        return 1.0 if float(pred) == float(gt) else 0.0
    except (ValueError, TypeError):
        return 1.0 if pred == gt else 0.0
