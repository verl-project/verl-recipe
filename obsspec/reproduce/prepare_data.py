"""Prepare both ObsSpec datasets and their fixed evaluation manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

# Allow direct execution from the verl root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_data as endless_terminals  # noqa: E402

DEFAULT_REPO_ID = "BytedTsinghua-SIA/DAPO-Math-17k"
DEFAULT_FILENAME = "data/dapo-math-17k.parquet"
PROMPT_PREFIX = (
    "Solve the following math problem step by step. The last line of your response should be of the form "
    "Answer: $Answer (without quotes) where $Answer is the answer to the problem.\n\n"
)
PROMPT_SUFFIX = '\n\nRemember to put your answer on its own line after "Answer:".'
ANSWER_INSTRUCTIONS = (
    "\n\nYou have access to a terminal with Python and common scientific-computing packages. "
    "You may use it to verify calculations, test conjectures, or perform other useful computations. "
    "When you are finished, write the final integer answer to `/home/user/answer.txt`, then call `done`.\n"
)
SYSTEM_PROMPT = (
    "You are a highly capable terminal agent. "
    "Complete the user's task by running commands and verifying the result. "
    "When the task is complete, call done."
)
TASK_TOML = """version = "0.1"

[metadata]
author_name = "BytedTsinghua-SIA"
author_email = "dapo-math@users.noreply.huggingface.co"
difficulty = "hard"
category = "mathematics"
tags = ["dapo", "math", "tool-assisted-reasoning"]

[verifier]
timeout_sec = 120.0

[agent]
timeout_sec = 360.0

[environment]
build_timeout_sec = 1800.0
cpus = 1
memory_mb = 4096
storage_mb = 10240
"""
DOCKERFILE = """FROM python:3.12-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV MPLBACKEND=Agg
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \\
        bash \\
        bc \\
        build-essential \\
        ca-certificates \\
        gfortran \\
        libgmp-dev \\
        liblapack-dev \\
        libmpc-dev \\
        libmpfr-dev \\
        libopenblas-dev \\
        pkg-config \\
    && rm -rf /var/lib/apt/lists/* \\
    && ln -sf /usr/local/bin/python3 /usr/local/bin/python

RUN pip install --no-cache-dir \\
        numpy \\
        scipy \\
        pandas \\
        sympy \\
        mpmath \\
        gmpy2 \\
        networkx \\
        matplotlib \\
        statsmodels \\
        scikit-learn \\
        pulp \\
        ortools \\
        z3-solver \\
        more-itertools

RUN mkdir -p /home/user && chmod 0777 /home/user

WORKDIR /home/user

CMD ["/bin/bash"]
"""
TEST_SH = """#!/usr/bin/env bash
set -euo pipefail

mkdir -p /logs/verifier
rm -f /logs/verifier/reward.txt
python3 /tests/grade.py
"""
GRADE_PY = """from pathlib import Path

from math_dapo import normalize_final_answer


ANSWER_PATH = Path("/home/user/answer.txt")
EXPECTED_PATH = Path("/tests/expected.txt")
REWARD_PATH = Path("/logs/verifier/reward.txt")


def main() -> None:
    reward = 0
    try:
        prediction = ANSWER_PATH.read_text(encoding="utf-8").strip()
        expected = EXPECTED_PATH.read_text(encoding="utf-8").strip()
        if prediction and normalize_final_answer(prediction) == normalize_final_answer(expected):
            reward = 1
    except (OSError, UnicodeError):
        reward = 0
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(f"{reward}\\n", encoding="utf-8")

if __name__ == "__main__":
    main()
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dapo_source_path",
        default=None,
        help="Path to a local DAPO-Math parquet; downloads from Hugging Face when omitted.",
    )
    parser.add_argument(
        "--local_save_dir",
        default="~/data/terminal_obsspec",
        help="Directory for generated tasks and parquet files.",
    )
    parser.add_argument(
        "--local_task_dir",
        default=None,
        help="Directory for Harbor tasks; defaults to <local_save_dir>/dapo-math-tasks.",
    )
    parser.add_argument("--endless_source_dir", default=None)
    parser.add_argument("--endless_repo_id", default=endless_terminals.DEFAULT_REPO_ID)
    parser.add_argument("--repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--filename", default=DEFAULT_FILENAME)
    parser.add_argument("--val_size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite_tasks", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def problem_from_prompt(prompt: str) -> str:
    if prompt.startswith(PROMPT_PREFIX):
        prompt = prompt[len(PROMPT_PREFIX) :]
    if prompt.endswith(PROMPT_SUFFIX):
        prompt = prompt[: -len(PROMPT_SUFFIX)]
    return prompt.strip()


def task_suffix(source_id: str, problem: str) -> str:
    compact_id = re.sub(r"[^a-zA-Z0-9]", "", source_id)[:8].lower()
    if compact_id:
        return compact_id
    return hashlib.sha256(problem.encode("utf-8")).hexdigest()[:8]


def write_task(
    row: dict,
    task_index: int,
    task_root: Path,
    math_dapo: Path,
    overwrite: bool,
) -> bool:
    messages = row.get("prompt") or []
    if len(messages) != 1 or messages[0].get("role") != "user":
        raise ValueError(f"Row {task_index} has an unsupported prompt")
    problem = problem_from_prompt(str(messages[0]["content"]))
    expected = str((row.get("reward_model") or {}).get("ground_truth", "")).strip()
    if not re.fullmatch(r"-?\d+", expected):
        raise ValueError(f"Row {task_index} has non-integer ground truth: {expected!r}")

    raw_source_id = (row.get("extra_info") or {}).get("index")
    source_id = "" if raw_source_id is None else str(raw_source_id)
    task_name = f"task_{task_index:06d}_{task_suffix(source_id, problem)}"
    task_dir = task_root / task_name
    if task_dir.exists():
        if not overwrite:
            return False
        shutil.rmtree(task_dir)

    temporary_dir = task_root / f".{task_name}.tmp"
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    (temporary_dir / "environment").mkdir(parents=True)
    (temporary_dir / "tests").mkdir()
    (temporary_dir / "solution").mkdir()

    (temporary_dir / "environment/Dockerfile").write_text(DOCKERFILE, encoding="utf-8")
    shutil.copyfile(math_dapo, temporary_dir / "tests/math_dapo.py")
    (temporary_dir / "instruction.md").write_text(problem + ANSWER_INSTRUCTIONS, encoding="utf-8")
    (temporary_dir / "task.toml").write_text(TASK_TOML, encoding="utf-8")
    (temporary_dir / "tests/expected.txt").write_text(expected + "\n", encoding="utf-8")
    test_path = temporary_dir / "tests/test.sh"
    test_path.write_text(TEST_SH, encoding="utf-8")
    test_path.chmod(0o755)
    (temporary_dir / "tests/grade.py").write_text(GRADE_PY, encoding="utf-8")

    solve_path = temporary_dir / "solution/solve.sh"
    solve_path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n\n"
        "mkdir -p /home/user\n"
        f"printf '%s\\n' {expected!r} > /home/user/answer.txt\n",
        encoding="utf-8",
    )
    solve_path.chmod(0o755)
    temporary_dir.rename(task_dir)
    return True


def convert_tasks(
    source: Path,
    task_root: Path,
    math_dapo: Path,
    overwrite: bool,
    limit: int | None,
) -> list[Path]:
    seen: dict[str, tuple[str, str]] = {}
    created = 0
    skipped = 0
    duplicate_rows = 0
    task_index = 0
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=256):
        for row in batch.to_pylist():
            raw_source_id = (row.get("extra_info") or {}).get("index")
            source_id = "" if raw_source_id is None else str(raw_source_id)
            messages = row.get("prompt") or []
            problem = problem_from_prompt(str(messages[0]["content"])) if messages else ""
            expected = str((row.get("reward_model") or {}).get("ground_truth", "")).strip()
            previous = seen.get(source_id)
            if previous is not None:
                if previous != (problem, expected):
                    raise ValueError(f"Conflicting duplicate rows for source ID {source_id!r}")
                duplicate_rows += 1
                continue
            seen[source_id] = (problem, expected)

            if limit is not None and task_index >= limit:
                break
            if write_task(row, task_index, task_root, math_dapo, overwrite):
                created += 1
            else:
                skipped += 1
            task_index += 1
            if task_index % 1000 == 0:
                print(f"Converted {task_index} unique tasks.", flush=True)
        if limit is not None and task_index >= limit:
            break

    tasks = sorted(path.parent for path in task_root.glob("task_*/task.toml"))
    if len(tasks) != task_index:
        raise ValueError(
            f"Found {len(tasks)} task directories after converting {task_index} rows; "
            "remove stale tasks or pass --overwrite_tasks"
        )
    print(
        f"Prepared {len(tasks)} Harbor tasks in {task_root} "
        f"({created} created, {skipped} reused, {duplicate_rows} duplicate rows skipped)."
    )
    return tasks


def read_instruction(task_dir: Path) -> str:
    return (task_dir / "instruction.md").read_text(encoding="utf-8").strip()


def build_row(task_dir: Path, task_root: Path, index: int, split: str) -> dict:
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": read_instruction(task_dir)},
        ],
        "data_source": DEFAULT_REPO_ID,
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "split": split,
            "index": index,
            "id": task_dir.relative_to(task_root).as_posix(),
            "task_binary": endless_terminals.archive_task(task_dir),
        },
    }


def write_parquets(
    tasks: list[Path],
    task_root: Path,
    local_save_dir: Path,
    val_size: int,
    seed: int,
) -> tuple[Path, Path]:
    if val_size <= 0 or val_size >= len(tasks):
        raise ValueError(f"val_size must be between 1 and {len(tasks) - 1}")
    val_indices = set(random.Random(seed).sample(range(len(tasks)), val_size))
    train_rows = []
    val_rows = []
    for index, task_dir in enumerate(tasks):
        split = "val" if index in val_indices else "train"
        row = build_row(task_dir, task_root, index, split)
        (val_rows if split == "val" else train_rows).append(row)
        if (index + 1) % 1000 == 0:
            print(f"Packaged {index + 1}/{len(tasks)} tasks.", flush=True)

    train_path = local_save_dir / "dapo-math-train.parquet"
    val_path = local_save_dir / "dapo-math-val.parquet"
    datasets.Dataset.from_list(train_rows).to_parquet(str(train_path))
    datasets.Dataset.from_list(val_rows).to_parquet(str(val_path))
    print(f"Wrote {len(train_rows)} training tasks to {train_path}")
    print(f"Wrote {len(val_rows)} validation tasks to {val_path}")
    return train_path, val_path


def prepare_dapo(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be greater than zero")

    if args.dapo_source_path is None:
        source = Path(
            hf_hub_download(
                repo_id=args.repo_id,
                filename=args.filename,
                repo_type="dataset",
            )
        )
    else:
        source = Path(args.dapo_source_path).expanduser().resolve()

    local_save_dir = Path(os.path.expanduser(args.local_save_dir)).resolve()
    if args.local_task_dir is None:
        task_root = local_save_dir / "dapo-math-tasks"
    else:
        task_root = Path(os.path.expanduser(args.local_task_dir)).resolve()
    recipe_dir = Path(__file__).resolve().parent
    math_dapo = recipe_dir.parents[2] / "verl/utils/reward_score/math_dapo.py"
    for required_path in (source, math_dapo):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    local_save_dir.mkdir(parents=True, exist_ok=True)
    task_root.mkdir(parents=True, exist_ok=True)
    tasks = convert_tasks(
        source,
        task_root,
        math_dapo,
        args.overwrite_tasks,
        args.limit,
    )
    train_path, val_path = write_parquets(
        tasks,
        task_root,
        local_save_dir,
        args.val_size,
        args.seed,
    )

    return train_path, val_path


def copy_manifest(name: str, val_path: Path, output_dir: Path) -> None:
    source = Path(__file__).with_name(name)
    batches = json.loads(source.read_text())
    task_ids = {row["id"] for row in pq.read_table(val_path, columns=["extra_info"])["extra_info"].to_pylist()}
    missing = {task_id for batch in batches for task_id in batch} - task_ids
    if missing:
        raise ValueError(f"{name} contains tasks missing from the validation split: {sorted(missing)}")
    destination = output_dir / name
    if source.resolve() != destination.resolve():
        shutil.copyfile(source, destination)
    print(f"Wrote evaluation manifest to {destination}")


def write_async_parquet(val_path: Path, output_dir: Path) -> None:
    manifest = json.loads((output_dir / "endless_terminals_eval_manifest.json").read_text())
    task_ids = [task_id for batch in manifest for task_id in batch]
    rows = pq.read_table(val_path).to_pylist()
    by_id = {row["extra_info"]["id"]: row for row in rows}
    # Repeat the fixed task order to form 4,096 prompt groups with 16 rollouts each.
    output_path = output_dir / "endless-terminals-async.parquet"
    schema = pa.Table.from_pylist(
        [
            {
                **rows[0],
                "extra_info": {
                    **rows[0]["extra_info"],
                    "async_group_id": 0,
                    "async_rollout_index": 0,
                    "async_job_index": 0,
                },
            }
        ]
    ).schema
    with pq.ParquetWriter(output_path, schema) as writer:
        for group_id in range(4096):
            row = by_id[task_ids[group_id % len(task_ids)]]
            expanded = [
                {
                    **row,
                    "extra_info": {
                        **row["extra_info"],
                        "async_group_id": group_id,
                        "async_rollout_index": rollout_index,
                        "async_job_index": group_id * 16 + rollout_index,
                    },
                }
                for rollout_index in range(16)
            ]
            writer.write_table(pa.Table.from_pylist(expanded, schema=schema))
    print(f"Wrote 65,536 async evaluation rollouts to {output_path}")


def main() -> None:
    args = parse_args()
    _, terminal_val = endless_terminals.prepare_data(
        argparse.Namespace(
            local_dataset_path=args.endless_source_dir,
            local_save_dir=args.local_save_dir,
            repo_id=args.endless_repo_id,
            val_size=args.val_size,
            seed=args.seed,
        )
    )
    _, dapo_val = prepare_dapo(args)
    output_dir = Path(args.local_save_dir).expanduser()
    copy_manifest("endless_terminals_eval_manifest.json", terminal_val, output_dir)
    copy_manifest("dapo_tir_eval_manifest.json", dapo_val, output_dir)
    write_async_parquet(terminal_val, output_dir)


if __name__ == "__main__":
    main()
