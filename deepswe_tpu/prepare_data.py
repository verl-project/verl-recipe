# Copyright 2026 Google LLC
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
"""Download R2E-Gym and build non-overlapping train/eval Parquet splits for ``deepswe_tpu``.

Emits rows in the ``uni-agent`` Task Config schema (``prompt``, ``data_source``,
``extra_info.tools_kwargs.task``) consumed by ``deepswe_tpu/run_qwen3_5_35b_deepswe.sh``.

Example::

    PYTHONPATH=uni-agent python3 deepswe_tpu/prepare_data.py \
        --local-save-dir /tmp/r2e_gym \
        --num-train 700 \
        --num-eval 100 \
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Any

from datasets import Dataset, load_dataset
from uni_agent.tasks.r2e_gym.utils import get_gold_patch

DEFAULT_DATA_SOURCE = "R2E-Gym/R2E-Gym-Subset"
DEFAULT_SPLIT = "train"
DEFAULT_NUM_TRAIN = 700
DEFAULT_NUM_EVAL = 100
DEFAULT_SEED = 42


def _parse_expected_output(value: Any) -> str | None:
    """Validate ``expected_output_json`` is a non-empty JSON object and return its string form."""
    if value is None:
        return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, Mapping) or not parsed:
            return None
        return raw
    if isinstance(value, Mapping) and value:
        return json.dumps(dict(value))
    return None


def format_r2e_gym_row(example: Mapping[str, Any], *, data_source: str) -> dict[str, Any] | None:
    """Convert one raw R2E-Gym row to the ``uni-agent`` Task Config format, or ``None`` if invalid."""
    repo_name = str(example.get("repo_name") or "").strip()
    commit_hash = str(example.get("commit_hash") or "").strip()
    problem_statement = str(example.get("problem_statement") or "").strip()
    expected_output_json = _parse_expected_output(example.get("expected_output_json"))
    if not repo_name or not commit_hash or not problem_statement or expected_output_json is None:
        return None

    try:
        gold_patch = get_gold_patch(example)
    except (KeyError, TypeError, ValueError):
        return None

    docker_image = str(example.get("docker_image") or "").strip()
    if not docker_image:
        docker_image = f"namanjain12/{repo_name}_final:{commit_hash}"

    instance_id = f"{repo_name}__{commit_hash[:10]}"
    task_config = {
        "name": "r2e_gym",
        "sandbox": {"image": docker_image},
        "metadata": {
            "repo": repo_name,
            "instance_id": instance_id,
            "base_commit": commit_hash,
            "patch": gold_patch,
            "expected_output_json": expected_output_json,
            "problem_statement": problem_statement,
        },
    }
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": problem_statement}],
        "extra_info": {"tools_kwargs": {"task": task_config}},
    }


def _row_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    task = row["extra_info"]["tools_kwargs"]["task"]
    return str(task["metadata"]["instance_id"]), str(task["sandbox"]["image"])


def _row_repo(row: Mapping[str, Any]) -> str:
    return str(row["extra_info"]["tools_kwargs"]["task"]["metadata"]["repo"])


def select_train_eval_splits(
    examples: Sequence[Mapping[str, Any]],
    *,
    data_source: str,
    num_train: int = DEFAULT_NUM_TRAIN,
    num_eval: int = DEFAULT_NUM_EVAL,
    seed: int = DEFAULT_SEED,
    repos: Collection[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter, deduplicate, shuffle deterministically, and split into disjoint train/eval lists."""
    if num_train < 0 or num_eval < 0:
        raise ValueError(f"num_train and num_eval must be non-negative, got ({num_train}, {num_eval})")

    repo_allowlist = {r.strip().lower() for r in repos if r.strip()} if repos else None

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_images: set[str] = set()
    for example in examples:
        if repo_allowlist is not None:
            raw_repo = str(example.get("repo_name") or "").strip().lower()
            if raw_repo not in repo_allowlist:
                continue
        row = format_r2e_gym_row(example, data_source=data_source)
        if row is None:
            continue
        instance_id, docker_image = _row_identity(row)
        if instance_id in seen_ids or docker_image in seen_images:
            continue
        seen_ids.add(instance_id)
        seen_images.add(docker_image)
        rows.append(row)

    needed = num_train + num_eval
    if len(rows) < needed:
        raise ValueError(
            f"Requested {num_train} train + {num_eval} eval = {needed} non-overlapping examples, "
            f"but only {len(rows)} valid unique examples are available."
        )

    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)

    train_rows = [rows[i] for i in indices[:num_train]]
    eval_rows = [rows[i] for i in indices[num_train:needed]]

    train_ids = {_row_identity(r)[0] for r in train_rows}
    eval_ids = {_row_identity(r)[0] for r in eval_rows}
    train_images = {_row_identity(r)[1] for r in train_rows}
    eval_images = {_row_identity(r)[1] for r in eval_rows}
    assert train_ids.isdisjoint(eval_ids), "Train and eval splits share instance_ids"
    assert train_images.isdisjoint(eval_images), "Train and eval splits share sandbox images"
    return train_rows, eval_rows


def prepare_r2e_gym_datasets(
    *,
    data_source: str = DEFAULT_DATA_SOURCE,
    split: str = DEFAULT_SPLIT,
    num_train: int = DEFAULT_NUM_TRAIN,
    num_eval: int = DEFAULT_NUM_EVAL,
    seed: int = DEFAULT_SEED,
    repos: Collection[str] | None = None,
) -> tuple[Dataset, Dataset]:
    """Load an R2E-Gym dataset from Hugging Face and build disjoint train/eval ``Dataset`` splits."""
    print(f"Loading {data_source} (split={split!r}) from Hugging Face...", flush=True)
    raw_dataset = load_dataset(data_source, split=split)
    print(f"Loaded {len(raw_dataset)} raw instances", flush=True)

    train_rows, eval_rows = select_train_eval_splits(
        raw_dataset,
        data_source=data_source,
        num_train=num_train,
        num_eval=num_eval,
        seed=seed,
        repos=repos,
    )
    return Dataset.from_list(train_rows), Dataset.from_list(eval_rows)


def _format_repo_counts(rows: Sequence[Mapping[str, Any]]) -> str:
    counts = Counter(_row_repo(r) for r in rows)
    return ", ".join(f"{repo}={count}" for repo, count in counts.most_common())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download R2E-Gym and generate non-overlapping train/eval Parquet splits for deepswe_tpu."
    )
    parser.add_argument(
        "--local-save-dir",
        default="~/data/r2e_gym",
        help="Directory where train and eval Parquet files are written.",
    )
    parser.add_argument(
        "--data-source",
        default=DEFAULT_DATA_SOURCE,
        help=f"Hugging Face dataset ID (default: {DEFAULT_DATA_SOURCE}).",
    )
    parser.add_argument(
        "--split",
        default=DEFAULT_SPLIT,
        help=f"Source dataset split to sample from (default: {DEFAULT_SPLIT}).",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=DEFAULT_NUM_TRAIN,
        help=f"Number of training examples to select (default: {DEFAULT_NUM_TRAIN}).",
    )
    parser.add_argument(
        "--num-eval",
        type=int,
        default=DEFAULT_NUM_EVAL,
        help=f"Number of non-overlapping eval examples to select (default: {DEFAULT_NUM_EVAL}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for deterministic train/eval splitting (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--repos",
        default=None,
        help="Optional comma-separated list of repo names to include (e.g. 'numpy,pandas,pillow').",
    )
    parser.add_argument(
        "--train-filename",
        default="train.parquet",
        help="Output filename for the training split (default: train.parquet).",
    )
    parser.add_argument(
        "--eval-filename",
        default="test.parquet",
        help="Output filename for the evaluation split (default: test.parquet).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    repos = [r.strip() for r in args.repos.split(",") if r.strip()] if args.repos else None

    save_dir = os.path.expanduser(args.local_save_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_ds, eval_ds = prepare_r2e_gym_datasets(
        data_source=args.data_source,
        split=args.split,
        num_train=args.num_train,
        num_eval=args.num_eval,
        seed=args.seed,
        repos=repos,
    )

    train_path = os.path.join(save_dir, args.train_filename)
    eval_path = os.path.join(save_dir, args.eval_filename)
    train_ds.to_parquet(train_path)
    eval_ds.to_parquet(eval_path)

    print(f"Wrote {len(train_ds)} train examples to {train_path} ({_format_repo_counts(train_ds)})", flush=True)
    print(f"Wrote {len(eval_ds)} eval examples to {eval_path} ({_format_repo_counts(eval_ds)})", flush=True)


if __name__ == "__main__":
    main()
