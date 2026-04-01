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

"""
Reward function
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from mbpp_exec_isolated import MBPP_EXEC_TIMEOUT_SEC, run_mbpp_eval_with_timeout  # noqa: E402


def _extract_python_from_completion(text: str) -> str:
    """Extract executable Python from model output (PRIME-style).

    Uses the same rule as ``verl.utils.reward_score.prime_code.compute_score``:
    take the segment after the last `` ```python `` marker, then before the next `` ``` ``.

    If `` ```python `` is not present, returns ``text.strip()`` so plain code (no fence)
    still runs—matching prior ``mbpp_reward`` behavior (still subject to ``exec`` risks).
    """
    stripped = text.strip()
    if "```python" in stripped:
        return stripped.split("```python")[-1].split("```")[0].strip()
    return stripped


def mbpp_reward(data_source, solution_str, ground_truth=None, extra_info=None):
    """
    verl-compatible reward: fraction of MBPP assert strings that pass after ``exec``.

    ``solution_str`` may be plain code or a completion containing a last `` ```python `` …
    `` ``` `` block; the executable body is taken via :func:`_extract_python_from_completion`
    (PRIME-aligned). ``extra_info`` must include ``test_list`` and ``test_setup`` (see
    ``create_mbpp_dataset`` export).

    Evaluation runs in a **spawned subprocess** with a **{timeout}s** wall-clock limit
    (see :data:`mbpp_exec_isolated.MBPP_EXEC_TIMEOUT_SEC`); on timeout or worker failure
    the reward is ``0.0``.
    """.format(timeout=int(MBPP_EXEC_TIMEOUT_SEC))

    test_list = extra_info["test_list"]
    test_setup = extra_info["test_setup"]
    code = _extract_python_from_completion(solution_str)

    return run_mbpp_eval_with_timeout(code, test_setup, test_list, MBPP_EXEC_TIMEOUT_SEC)
