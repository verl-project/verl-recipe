# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
SWE Agent Dataset Generator.

Supports two data sources:
1. Simple test cases (for quick validation)
2. SWE-bench Lite (for full evaluation)

Data format is VERL-compatible:
- prompt: Minimal chat messages (satisfies framework's ``raw_prompt`` requirement).
          The *real* system/instance templates are applied at runtime by
          SWE-Agent via ``swe_agent_config.yaml`` — this avoids duplicating
          prompt templates between data preparation and runtime.
- reward_model: Evaluation configuration
- extra_info: Contains problem_statement, repo_content, expected_patch,
              and data-affine overrides (sandbox_overrides / agent_overrides).
- agent_name: "swe_agent"

Data-affine override mechanism:
  extra_info may contain two special dicts that override swe_agent_config.yaml
  defaults at runtime (per-instance granularity):

  - sandbox_overrides: e.g. {"docker_image": "...", "max_steps": 50}
  - agent_overrides:   e.g. {"templates": {"system_template": "..."}}

  See swe_agent_config.yaml for the full list (marked [DATA-AFFINE]).
"""

import argparse
import json
import os
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Prompt helper
# ---------------------------------------------------------------------------


def _make_minimal_prompt(problem_statement: str) -> list[dict[str, str]]:
    """Create a minimal prompt that satisfies VERL's ``raw_prompt`` requirement.

    The real system/instance templates are injected by SWE-Agent at runtime
    (via swe_agent_config.yaml).  This prompt is only used for:
      - ``_agent_loop_postprocess`` (stores ``raw_prompt`` in extra_fields)
      - The reward loop (reconstructs the chat for RM scoring)

    Args:
        problem_statement: Problem description text.

    Returns:
        Minimal conversation in ``[{role, content}]`` format.
    """
    return [{"role": "user", "content": problem_statement}]


# ---------------------------------------------------------------------------
# Simple test data
# ---------------------------------------------------------------------------


_SIMPLE_CASES_TRAIN = [
    {
        "problem_statement": "rename 1.txt to 2.txt",
        "repo_content": {"1.txt": "Hello World"},
        "expected_patch": "diff --git a/1.txt b/2.txt\nsimilarity index 100%\nrename from 1.txt\nrename to 2.txt",
    },
    {
        "problem_statement": "Create a new file called hello.py that prints 'Hello, World!'",
        "repo_content": {},
        "expected_patch": (
            "diff --git a/hello.py b/hello.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/hello.py\n"
            "@@ -0,0 +1 @@\n"
            "+print('Hello, World!')"
        ),
    },
    {
        "problem_statement": "Fix the bug in calculator.py: the add function should return a + b, not a - b",
        "repo_content": {"calculator.py": "def add(a, b):\n    return a - b"},
        "expected_patch": (
            "diff --git a/calculator.py b/calculator.py\n"
            "--- a/calculator.py\n"
            "+++ b/calculator.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def add(a, b):\n"
            "-    return a - b\n"
            "+    return a + b"
        ),
    },
    {
        "problem_statement": "Delete the file named remove_me.txt from the repository",
        "repo_content": {"remove_me.txt": "This file should be deleted", "keep.txt": "Keep this"},
        "expected_patch": (
            "diff --git a/remove_me.txt b/remove_me.txt\n"
            "deleted file mode 100644\n"
            "--- a/remove_me.txt\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-This file should be deleted"
        ),
    },
    {
        "problem_statement": (
            "Add a docstring to the function greet in greet.py. The docstring should say 'Return a greeting message.'"
        ),
        "repo_content": {"greet.py": 'def greet(name):\n    return f"Hello, {name}!"'},
        "expected_patch": (
            "diff --git a/greet.py b/greet.py\n"
            "--- a/greet.py\n"
            "+++ b/greet.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def greet(name):\n"
            '+    """Return a greeting message."""\n'
            '     return f"Hello, {name}!"'
        ),
    },
    {
        "problem_statement": (
            "Fix the off-by-one error in range.py: the loop should print numbers from 1 to 5 inclusive, not 1 to 4"
        ),
        "repo_content": {"range.py": "for i in range(1, 5):\n    print(i)"},
        "expected_patch": (
            "diff --git a/range.py b/range.py\n"
            "--- a/range.py\n"
            "+++ b/range.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-for i in range(1, 5):\n"
            "+for i in range(1, 6):\n"
            "     print(i)"
        ),
    },
    {
        "problem_statement": 'Create a file called config.json with the content: {"debug": true, "version": 1}',
        "repo_content": {},
        "expected_patch": (
            "diff --git a/config.json b/config.json\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/config.json\n"
            "@@ -0,0 +1 @@\n"
            '+{"debug": true, "version": 1}'
        ),
    },
    {
        "problem_statement": (
            "In math_utils.py, the multiply function incorrectly uses addition. Fix it to use multiplication."
        ),
        "repo_content": {
            "math_utils.py": "def multiply(a, b):\n    return a + b\n\ndef divide(a, b):\n    return a / b"
        },
        "expected_patch": (
            "diff --git a/math_utils.py b/math_utils.py\n"
            "--- a/math_utils.py\n"
            "+++ b/math_utils.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def multiply(a, b):\n"
            "-    return a + b\n"
            "+    return a * b"
        ),
    },
    {
        "problem_statement": "Move the file old_dir/data.csv to new_dir/data.csv (create new_dir if it does not exist)",
        "repo_content": {"old_dir/data.csv": "name,value\nalice,100\nbob,200"},
        "expected_patch": (
            "diff --git a/old_dir/data.csv b/new_dir/data.csv\n"
            "similarity index 100%\n"
            "rename from old_dir/data.csv\n"
            "rename to new_dir/data.csv"
        ),
    },
    {
        "problem_statement": "Replace all occurrences of 'foo' with 'bar' in replace_me.txt",
        "repo_content": {"replace_me.txt": "foo is great\nI love foo\nfoo foo foo"},
        "expected_patch": (
            "diff --git a/replace_me.txt b/replace_me.txt\n"
            "--- a/replace_me.txt\n"
            "+++ b/replace_me.txt\n"
            "@@ -1,3 +1,3 @@\n"
            "-foo is great\n"
            "-I love foo\n"
            "-foo foo foo\n"
            "+bar is great\n"
            "+I love bar\n"
            "+bar bar bar"
        ),
    },
    {
        "problem_statement": "Add a shebang line (#!/usr/bin/env python3) at the top of script.py",
        "repo_content": {"script.py": "import sys\nprint(sys.argv)"},
        "expected_patch": (
            "diff --git a/script.py b/script.py\n"
            "--- a/script.py\n"
            "+++ b/script.py\n"
            "@@ -1,2 +1,3 @@\n"
            "+#!/usr/bin/env python3\n"
            " import sys\n"
            " print(sys.argv)"
        ),
    },
    {
        "problem_statement": "Fix the syntax error in broken.py: there is a missing colon after 'if True'",
        "repo_content": {"broken.py": "if True\n    print('works')"},
        "expected_patch": (
            "diff --git a/broken.py b/broken.py\n"
            "--- a/broken.py\n"
            "+++ b/broken.py\n"
            "@@ -1,2 +1,2 @@\n"
            "-if True\n"
            "+if True:\n"
            "     print('works')"
        ),
    },
]

_SIMPLE_CASES_TRAIN_HARD = [
    {
        "problem_statement": (
            "The function `parse_csv` in `parser.py` crashes on empty input. "
            "Fix it to return an empty list when the input string is empty or None."
        ),
        "repo_content": {
            "parser.py": (
                "def parse_csv(text):\n"
                "    rows = []\n"
                "    for line in text.strip().split('\\n'):\n"
                "        rows.append(line.split(','))\n"
                "    return rows\n"
            )
        },
        "expected_patch": (
            "diff --git a/parser.py b/parser.py\n"
            "--- a/parser.py\n"
            "+++ b/parser.py\n"
            "@@ -1,5 +1,7 @@\n"
            " def parse_csv(text):\n"
            "+    if not text:\n"
            "+        return []\n"
            "     rows = []\n"
            "     for line in text.strip().split('\\n'):\n"
            "         rows.append(line.split(','))\n"
            "     return rows"
        ),
    },
    {
        "problem_statement": (
            "In `utils.py`, the `flatten` function only handles one level of nesting. "
            "Fix it to recursively flatten arbitrarily nested lists."
        ),
        "repo_content": {
            "utils.py": (
                "def flatten(lst):\n"
                "    result = []\n"
                "    for item in lst:\n"
                "        if isinstance(item, list):\n"
                "            result.extend(item)\n"
                "        else:\n"
                "            result.append(item)\n"
                "    return result\n"
            )
        },
        "expected_patch": (
            "diff --git a/utils.py b/utils.py\n"
            "--- a/utils.py\n"
            "+++ b/utils.py\n"
            "@@ -3,7 +3,7 @@\n"
            "     for item in lst:\n"
            "         if isinstance(item, list):\n"
            "-            result.extend(item)\n"
            "+            result.extend(flatten(item))\n"
            "         else:\n"
            "             result.append(item)\n"
            "     return result"
        ),
    },
    {
        "problem_statement": (
            "Add a `__repr__` method to the `Point` class in `geometry.py` that returns 'Point(x=<x>, y=<y>)' format."
        ),
        "repo_content": {
            "geometry.py": (
                "class Point:\n"
                "    def __init__(self, x, y):\n"
                "        self.x = x\n"
                "        self.y = y\n"
                "\n"
                "    def distance(self, other):\n"
                "        return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5\n"
            )
        },
        "expected_patch": (
            "diff --git a/geometry.py b/geometry.py\n"
            "--- a/geometry.py\n"
            "+++ b/geometry.py\n"
            "@@ -4,4 +4,7 @@\n"
            "         self.y = y\n"
            " \n"
            "+    def __repr__(self):\n"
            "+        return f'Point(x={self.x}, y={self.y})'\n"
            "+\n"
            "     def distance(self, other):\n"
            "         return ((self.x - other.x)**2 + (self.y - other.y)**2)**0.5"
        ),
    },
    {
        "problem_statement": (
            "The `is_palindrome` function in `strings.py` is case-sensitive. "
            "Fix it to be case-insensitive and also ignore spaces."
        ),
        "repo_content": {"strings.py": ("def is_palindrome(s):\n    return s == s[::-1]\n")},
        "expected_patch": (
            "diff --git a/strings.py b/strings.py\n"
            "--- a/strings.py\n"
            "+++ b/strings.py\n"
            "@@ -1,2 +1,3 @@\n"
            " def is_palindrome(s):\n"
            "-    return s == s[::-1]\n"
            "+    s = s.lower().replace(' ', '')\n"
            "+    return s == s[::-1]"
        ),
    },
    {
        "problem_statement": (
            "In `app.py`, the `read_config` function does not handle the case where the "
            "config file does not exist. Add a try/except that returns an empty dict if "
            "FileNotFoundError is raised."
        ),
        "repo_content": {
            "app.py": (
                "import json\n\ndef read_config(path):\n    with open(path) as f:\n        return json.load(f)\n"
            )
        },
        "expected_patch": (
            "diff --git a/app.py b/app.py\n"
            "--- a/app.py\n"
            "+++ b/app.py\n"
            "@@ -2,4 +2,7 @@\n"
            " \n"
            " def read_config(path):\n"
            "-    with open(path) as f:\n"
            "-        return json.load(f)\n"
            "+    try:\n"
            "+        with open(path) as f:\n"
            "+            return json.load(f)\n"
            "+    except FileNotFoundError:\n"
            "+        return {}"
        ),
    },
    {
        "problem_statement": (
            "The `Stack` class in `stack.py` is missing a `peek` method. Add a `peek` method "
            "that returns the top element without removing it, or raises IndexError if empty."
        ),
        "repo_content": {
            "stack.py": (
                "class Stack:\n"
                "    def __init__(self):\n"
                "        self._items = []\n"
                "\n"
                "    def push(self, item):\n"
                "        self._items.append(item)\n"
                "\n"
                "    def pop(self):\n"
                "        return self._items.pop()\n"
                "\n"
                "    def is_empty(self):\n"
                "        return len(self._items) == 0\n"
            )
        },
        "expected_patch": (
            "diff --git a/stack.py b/stack.py\n"
            "--- a/stack.py\n"
            "+++ b/stack.py\n"
            "@@ -9,4 +9,9 @@\n"
            "     def pop(self):\n"
            "         return self._items.pop()\n"
            " \n"
            "+    def peek(self):\n"
            "+        if self.is_empty():\n"
            "+            raise IndexError('peek from empty stack')\n"
            "+        return self._items[-1]\n"
            "+\n"
            "     def is_empty(self):\n"
            "         return len(self._items) == 0"
        ),
    },
    {
        "problem_statement": (
            "In `validators.py`, the `validate_email` function only checks for '@'. "
            "Fix it to also require at least one '.' after the '@'."
        ),
        "repo_content": {"validators.py": ("def validate_email(email):\n    return '@' in email\n")},
        "expected_patch": (
            "diff --git a/validators.py b/validators.py\n"
            "--- a/validators.py\n"
            "+++ b/validators.py\n"
            "@@ -1,2 +1,4 @@\n"
            " def validate_email(email):\n"
            "-    return '@' in email\n"
            "+    if '@' not in email:\n"
            "+        return False\n"
            "+    return '.' in email.split('@')[1]"
        ),
    },
    {
        "problem_statement": (
            "The `counter.py` module has a global counter but no `reset` function. "
            "Add a `reset()` function that sets the counter back to 0."
        ),
        "repo_content": {
            "counter.py": (
                "count = 0\n"
                "\n"
                "def increment():\n"
                "    global count\n"
                "    count += 1\n"
                "\n"
                "def get_count():\n"
                "    return count\n"
            )
        },
        "expected_patch": (
            "diff --git a/counter.py b/counter.py\n"
            "--- a/counter.py\n"
            "+++ b/counter.py\n"
            "@@ -6,3 +6,7 @@\n"
            " \n"
            " def get_count():\n"
            "     return count\n"
            "+\n"
            "+def reset():\n"
            "+    global count\n"
            "+    count = 0"
        ),
    },
]

_SIMPLE_CASES_VAL = [
    {
        "problem_statement": "Rename the file report.txt to summary.txt",
        "repo_content": {"report.txt": "Quarterly earnings report"},
        "expected_patch": (
            "diff --git a/report.txt b/summary.txt\n"
            "similarity index 100%\n"
            "rename from report.txt\n"
            "rename to summary.txt"
        ),
    },
    {
        "problem_statement": "Create a new file called goodbye.py that prints 'Goodbye, World!'",
        "repo_content": {},
        "expected_patch": (
            "diff --git a/goodbye.py b/goodbye.py\n"
            "new file mode 100644\n"
            "--- /dev/null\n"
            "+++ b/goodbye.py\n"
            "@@ -0,0 +1 @@\n"
            "+print('Goodbye, World!')"
        ),
    },
    {
        "problem_statement": "Fix the bug in subtract.py: the subtract function should return a - b, not a + b",
        "repo_content": {"subtract.py": "def subtract(a, b):\n    return a + b"},
        "expected_patch": (
            "diff --git a/subtract.py b/subtract.py\n"
            "--- a/subtract.py\n"
            "+++ b/subtract.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def subtract(a, b):\n"
            "-    return a + b\n"
            "+    return a - b"
        ),
    },
    {
        "problem_statement": "Delete the file temp.log from the repository",
        "repo_content": {"temp.log": "DEBUG: old log entries", "app.py": "print('app')"},
        "expected_patch": (
            "diff --git a/temp.log b/temp.log\n"
            "deleted file mode 100644\n"
            "--- a/temp.log\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-DEBUG: old log entries"
        ),
    },
    {
        "problem_statement": (
            "The `clamp` function in `math_helpers.py` should restrict a value to be "
            "between `lo` and `hi`. Currently it returns `value` unchanged. Fix it."
        ),
        "repo_content": {"math_helpers.py": ("def clamp(value, lo, hi):\n    return value\n")},
        "expected_patch": (
            "diff --git a/math_helpers.py b/math_helpers.py\n"
            "--- a/math_helpers.py\n"
            "+++ b/math_helpers.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def clamp(value, lo, hi):\n"
            "-    return value\n"
            "+    return max(lo, min(hi, value))"
        ),
    },
    {
        "problem_statement": (
            "Add a `to_dict` method to the `User` class in `models.py` that returns "
            "{'name': self.name, 'age': self.age}."
        ),
        "repo_content": {
            "models.py": (
                "class User:\n    def __init__(self, name, age):\n        self.name = name\n        self.age = age\n"
            )
        },
        "expected_patch": (
            "diff --git a/models.py b/models.py\n"
            "--- a/models.py\n"
            "+++ b/models.py\n"
            "@@ -3,1 +3,4 @@\n"
            "         self.name = name\n"
            "         self.age = age\n"
            "+\n"
            "+    def to_dict(self):\n"
            "+        return {'name': self.name, 'age': self.age}"
        ),
    },
    {
        "problem_statement": (
            "In `text.py`, the `word_count` function splits only on spaces. "
            "Fix it to split on any whitespace (tabs, newlines, multiple spaces)."
        ),
        "repo_content": {"text.py": ("def word_count(text):\n    return len(text.split(' '))\n")},
        "expected_patch": (
            "diff --git a/text.py b/text.py\n"
            "--- a/text.py\n"
            "+++ b/text.py\n"
            "@@ -1,2 +1,2 @@\n"
            " def word_count(text):\n"
            "-    return len(text.split(' '))\n"
            "+    return len(text.split())"
        ),
    },
    {
        "problem_statement": (
            "The `safe_divide` function in `calc.py` does not handle division by zero. "
            "Fix it to return 0.0 when the divisor is zero."
        ),
        "repo_content": {"calc.py": ("def safe_divide(a, b):\n    return a / b\n")},
        "expected_patch": (
            "diff --git a/calc.py b/calc.py\n"
            "--- a/calc.py\n"
            "+++ b/calc.py\n"
            "@@ -1,2 +1,4 @@\n"
            " def safe_divide(a, b):\n"
            "+    if b == 0:\n"
            "+        return 0.0\n"
            "     return a / b"
        ),
    },
]


def generate_simple_test_data(
    num_samples: int,
    split: str,
    agent_name: str = "swe_agent",
) -> pd.DataFrame:
    """Generate simple test data for quick validation.

    Train and validation use completely disjoint problem pools.
    Training pool includes both easy and harder cases.
    """
    if split == "train":
        pool = _SIMPLE_CASES_TRAIN + _SIMPLE_CASES_TRAIN_HARD
    else:
        pool = _SIMPLE_CASES_VAL

    rows: list[dict[str, Any]] = []
    for idx in range(num_samples):
        case = pool[idx % len(pool)]
        rows.append(
            {
                "prompt": _make_minimal_prompt(case["problem_statement"]),
                "data_source": "swe_agent_simple",
                "ability": "software_engineering",
                "reward_model": {
                    "style": "swe_agent",
                    "ground_truth": case["expected_patch"],
                },
                "extra_info": {
                    "index": idx,
                    "split": split,
                    "repo_content": case["repo_content"],
                    "expected_patch": case["expected_patch"],
                    "problem_statement": case["problem_statement"],
                    "sandbox_overrides": {"max_steps": 10, "max_turns": 8},
                },
                "agent_name": agent_name,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SWE-bench Lite
# ---------------------------------------------------------------------------


def load_swebench_lite(
    swebench_path: str,
    split: str,
    agent_name: str = "swe_agent",
) -> pd.DataFrame:
    """Load SWE-bench Lite dataset."""

    if swebench_path.endswith(".json"):
        with open(swebench_path) as f:
            data = json.load(f)
    elif swebench_path.endswith(".jsonl"):
        data = []
        with open(swebench_path) as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(f"Unsupported file format: {swebench_path}")

    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(data):
        instance_id = item.get("instance_id", f"instance_{idx}")
        problem_statement = item.get("problem_statement", "")

        # Per-instance sandbox overrides
        sandbox_overrides: dict[str, Any] = {}
        if item.get("docker_image"):
            sandbox_overrides["docker_image"] = item["docker_image"]
            # SWE-bench eval images have repos pre-installed at /testbed
            sandbox_overrides["use_preexisting_repo"] = True
            sandbox_overrides["preexisting_repo_name"] = "testbed"
            sandbox_overrides["preexisting_repo_reset"] = True
        if item.get("max_steps"):
            sandbox_overrides["max_steps"] = item["max_steps"]

        # Per-instance agent overrides (templates)
        # NOTE: Always include a placeholder to avoid pyarrow empty-struct error
        agent_overrides: dict[str, Any] = {"_placeholder": True}
        if item.get("system_template"):
            agent_overrides.setdefault("templates", {})["system_template"] = item["system_template"]
        if item.get("instance_template"):
            agent_overrides.setdefault("templates", {})["instance_template"] = item["instance_template"]

        rows.append(
            {
                "prompt": _make_minimal_prompt(problem_statement),
                "data_source": "swe_bench_verified",
                "ability": "software_engineering",
                "reward_model": {
                    "style": "swe_bench",
                    "instance_id": instance_id,
                    "test_patch": item.get("test_patch", ""),
                    "gold_patch": item.get("patch", ""),
                    "ground_truth": {"gold_patch": item.get("patch", "")},
                },
                "extra_info": {
                    "index": idx,
                    "split": split,
                    "instance_id": instance_id,
                    "repo": item.get("repo", ""),
                    "base_commit": item.get("base_commit", ""),
                    "problem_statement": problem_statement,
                    "hints_text": item.get("hints_text", ""),
                    "created_at": item.get("created_at", ""),
                    "version": item.get("version", ""),
                    "FAIL_TO_PASS": item.get("FAIL_TO_PASS", ""),
                    "PASS_TO_PASS": item.get("PASS_TO_PASS", ""),
                    "sandbox_overrides": sandbox_overrides,
                    "agent_overrides": agent_overrides,
                },
                "agent_name": agent_name,
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="SWE Agent Dataset Generator")
    parser.add_argument(
        "--mode",
        choices=["simple", "swebench"],
        default="simple",
        help="Data generation mode",
    )
    parser.add_argument("--train_size", type=int, default=100)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--swebench_train", type=str, default=None)
    parser.add_argument("--swebench_test", type=str, default=None)
    parser.add_argument("--output_dir", default="data/swe_agent")
    parser.add_argument("--agent_name", default="swe_agent")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Check pyarrow
    try:
        import importlib.util

        if importlib.util.find_spec("pyarrow") is None:
            raise ImportError("pyarrow not found")
    except ImportError as err:
        raise ImportError("pyarrow is required for parquet support. Install with: pip install pyarrow") from err

    if args.mode == "simple":
        print("Generating simple test data...")
        train_df = generate_simple_test_data(args.train_size, "train", args.agent_name)
        test_df = generate_simple_test_data(args.test_size, "test", args.agent_name)
    else:
        print("Loading SWE-bench Lite data...")
        if args.swebench_train is None or args.swebench_test is None:
            raise ValueError("--swebench_train and --swebench_test are required for swebench mode")
        train_df = load_swebench_lite(args.swebench_train, "train", args.agent_name)
        test_df = load_swebench_lite(args.swebench_test, "test", args.agent_name)

    train_path = os.path.join(args.output_dir, "train.parquet")
    test_path = os.path.join(args.output_dir, "test.parquet")
    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    print("\nDataset generation completed!")
    print(f"Train: {len(train_df)} samples -> {train_path}")
    print(f"Test:  {len(test_df)} samples -> {test_path}")


if __name__ == "__main__":
    main()
