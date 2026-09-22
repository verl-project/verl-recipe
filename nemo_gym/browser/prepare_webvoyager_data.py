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
"""Build NeMo Gym rollout inputs for browser tasks.

Reads a task list (Hugging Face dataset by default, or a local JSONL) and writes
the rollout-input JSONL this recipe trains on. No task data is committed to this
repository.

    python prepare_webvoyager_data.py --output data/webvoyager_train.jsonl
    python prepare_webvoyager_data.py --input tasks.jsonl --output data/train.jsonl

Each input task needs `question` and `start_url`; `verifier_metadata` is optional
and, when present, is scored by the environment's own verifier instead of the
recipe's judge.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterator

SYSTEM_PROMPT = (
    "You are a web agent operating a live browser through one tool. Call browser with "
    "action=observe to see the page (URL, title, and a numbered list of interactive elements "
    "as `[id] role: name`). Use action=navigate/click/type to act — element_id values come "
    "from the most recent observation. Call action=finish with your answer when the task is "
    "complete."
)

DEFAULT_HF_REPO = "lexmount/webvoyager-clean"


def _read_local(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_hf(repo_id: str, split: str) -> Iterator[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - datasets ships with verl
        raise ImportError("reading from Hugging Face needs `datasets`: pip install datasets") from exc
    for row in load_dataset(repo_id, split=split):
        yield dict(row)


def to_rollout_input(task: dict[str, Any]) -> dict[str, Any]:
    question = task.get("question") or task.get("ques") or ""
    start_url = task.get("start_url") or task.get("web") or ""
    if not question or not start_url:
        raise ValueError(f"task needs `question` and `start_url`, got keys {sorted(task)}")

    row: dict[str, Any] = {
        "responses_create_params": {
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        },
        "question": question,
        "initial_url": start_url,
    }
    if task.get("task_id") or task.get("id"):
        row["task_id"] = str(task.get("task_id") or task.get("id"))
    if task.get("verifier_metadata"):
        row["verifier_metadata"] = task["verifier_metadata"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="local JSONL task list; overrides --hf-repo")
    parser.add_argument("--hf-repo", default=DEFAULT_HF_REPO, help=f"Hugging Face dataset (default: {DEFAULT_HF_REPO})")
    parser.add_argument("--hf-split", default="train")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tasks = _read_local(args.input) if args.input else _read_hf(args.hf_repo, args.hf_split)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.output, "w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(to_rollout_input(task), ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} rollout inputs to {args.output}")


if __name__ == "__main__":
    main()
