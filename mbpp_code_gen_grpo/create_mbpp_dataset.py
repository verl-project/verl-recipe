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
MBPP (sanitized) RL Parquet export from Hugging Face ``google-research-datasets/mbpp`` (config ``sanitized``).

Output layout (under ``--data_path``):

- ``{output_rl_subdir}/{split}.parquet`` — ``prompt``, ``data_source``, ``ability``, ``reward_model``, ``extra_info``.

Trainer conventions:

- ``data_source`` is ``mbpp_sanitized`` (``DATA_SOURCE``). Use ``mbpp_reward`` from this recipe's
  ``reward_function.py`` (path set by ``train_grpo.sh`` via Hydra).
- RL prompt: includes only the first ``floor(N/2)`` tests in the prompt (hint without full leakage);
  all ``N`` tests go to ``extra_info`` so the reward function evaluates on the complete set.
- User text = MBPP ``prompt`` + optional global suffix (``--global-prompt`` / ``--global-prompt-file``),
  blocks joined by ``\\n\\n``; a fenced-code-block format instruction is always appended.
- ``extra_info`` for ``mbpp_reward``: ``test_list`` (all tests), ``test_setup``
  (``test_imports`` joined by newlines, or ``""``), ``task_id``, and ``source_file`` when present on the HF example.

Default splits: ``train`` (120 examples, used for RL training) and ``test`` (257 examples,
used for validation during training). The ``validation`` split (43 examples) is skipped by
default because it is too small for reliable evaluation.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import pandas as pd
from datasets import load_dataset

MBPP_DATASET = "google-research-datasets/mbpp"
MBPP_CONFIG = "sanitized"

DATA_SOURCE = "mbpp_sanitized"

DEFAULT_SPLITS = ("train", "test")

CODE_FORMAT_INSTRUCTION = (
    "Write your complete solution in a single markdown fenced code block "
    "using the language tag ```python (open with ```python on its own line, close with ```)."
)

TESTS_HINT = "Your implementation should pass tests like the following (more tests will be used for evaluation):"


def format_test_block(tests: list[str]) -> str:
    """Format a list of assert strings as a fenced code block."""
    return "```python\n" + "\n".join(tests) + "\n```"


def compose_user_text(
    mbpp_prompt: str,
    global_suffix: str | None = None,
    tests_to_show: list[str] | None = None,
    tests_label: str = "Your implementation must pass the following tests:",
) -> str:
    parts = []
    parts.append(mbpp_prompt.rstrip())

    if tests_to_show:
        parts.append(f"{tests_label}\n\n{format_test_block(tests_to_show)}")

    if global_suffix and global_suffix.strip():
        parts.append(global_suffix.strip())

    parts.append(CODE_FORMAT_INSTRUCTION)
    return "\n\n".join(parts)


def example_to_rl_row(
    example: dict[str, Any],
    global_suffix: str | None,
) -> dict[str, Any]:
    mbpp_prompt = example["prompt"]
    task_id = example["task_id"]
    test_list = list(example["test_list"])
    test_setup = "\n".join(example.get("test_imports") or [])

    n_hint = len(test_list) // 2
    rl_hint_tests = test_list[:n_hint] if n_hint > 0 else None
    rl_user_content = compose_user_text(
        mbpp_prompt,
        global_suffix=global_suffix,
        tests_to_show=rl_hint_tests,
        tests_label=TESTS_HINT,
    )

    extra_info: dict[str, Any] = {
        "test_list": test_list,
        "test_setup": test_setup,
        "task_id": task_id,
    }
    if example.get("source_file") is not None:
        extra_info["source_file"] = example["source_file"]

    return {
        "prompt": [{"role": "user", "content": rl_user_content}],
        "data_source": DATA_SOURCE,
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": None},
        "extra_info": extra_info,
    }


def build_parquets(
    data_path: str,
    splits: list[str],
    global_suffix: str | None,
    output_rl_subdir: str,
) -> None:
    data_path = os.path.expanduser(data_path)
    ds = load_dataset(MBPP_DATASET, MBPP_CONFIG)

    rl_root = os.path.join(data_path, output_rl_subdir)
    os.makedirs(rl_root, exist_ok=True)

    for split in splits:
        if split not in ds:
            raise ValueError(f"Split {split!r} not in dataset. Available: {list(ds.keys())}")

        rl_rows: list[dict[str, Any]] = []
        for example in ds[split]:
            rl_row = example_to_rl_row(example, global_suffix=global_suffix)
            rl_rows.append(rl_row)

        rl_df = pd.DataFrame(rl_rows)
        rl_df.to_parquet(os.path.join(rl_root, f"{split}.parquet"))
        print(f"Wrote {len(rl_df)} rows to {os.path.join(rl_root, f'{split}.parquet')}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export MBPP sanitized to RL Parquet files.")
    p.add_argument("--data_path", type=str, required=True, help="Output root (rl/ subdir created).")
    p.add_argument(
        "--global-prompt",
        type=str,
        default="",
        help="Optional text appended to every MBPP task prompt (after blank line).",
    )
    p.add_argument(
        "--global-prompt-file",
        type=str,
        default="",
        help="If set, contents appended like --global-prompt (after --global-prompt if both set).",
    )
    p.add_argument(
        "--splits",
        type=str,
        default=",".join(DEFAULT_SPLITS),
        help="Comma-separated HF split names to export (default: train,test).",
    )
    p.add_argument("--output-rl-subdir", type=str, default="mbpp/rl", help="Subdir under data_path for RL parquet.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    suffix_parts: list[str] = []
    if args.global_prompt.strip():
        suffix_parts.append(args.global_prompt.strip())
    if args.global_prompt_file:
        with open(os.path.expanduser(args.global_prompt_file), encoding="utf-8") as f:
            suffix_parts.append(f.read().strip())
    global_suffix = "\n\n".join(suffix_parts) if suffix_parts else None

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("No splits requested.")

    build_parquets(
        data_path=args.data_path,
        splits=splits,
        global_suffix=global_suffix,
        output_rl_subdir=args.output_rl_subdir,
    )


if __name__ == "__main__":
    main()
