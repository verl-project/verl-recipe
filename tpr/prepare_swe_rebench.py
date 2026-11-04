#!/usr/bin/env python3
"""Prepare SWE-ReBench task data for Uni-Agent training."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import Dataset, load_dataset
from uni_agent.tasks.swe_rebench.preprocess import get_image_name


def convert(example: dict) -> dict:
    instance_id = example["instance_id"]
    cfg = example["install_config"]
    metadata = {
        "instance_id": instance_id,
        "repo": example["repo"],
        "base_commit": example["base_commit"],
        "patch": example["patch"],
        "test_patch": example["test_patch"],
        "problem_statement": example["problem_statement"],
        "FAIL_TO_PASS": example["FAIL_TO_PASS"],
        "FAIL_TO_FAIL": example["FAIL_TO_FAIL"],
        "PASS_TO_PASS": example["PASS_TO_PASS"],
        "PASS_TO_FAIL": example["PASS_TO_FAIL"],
        "install": cfg["install"],
        "log_parser": cfg["log_parser"],
        "test_cmd": cfg["test_cmd"],
    }
    return {
        "data_source": "nebius/SWE-rebench",
        "prompt": [{"role": "user", "content": example["problem_statement"]}],
        "extra_info": {
            "tools_kwargs": {
                "task": {
                    "name": "swe_rebench",
                    "sandbox": {"image": get_image_name(instance_id)},
                    "metadata": metadata,
                }
            }
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", default="ASPP__pelita-875")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load_dataset("nebius/SWE-rebench", split="filtered", streaming=True)
    row = next((sample for sample in source if sample["instance_id"] == args.instance_id), None)
    if row is None:
        raise SystemExit(f"instance not found: {args.instance_id}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list([convert(row)]).to_parquet(str(args.output))
    print(f"wrote 1 instance to {args.output.resolve()}")
    print(f"sandbox image: {get_image_name(args.instance_id)}")


if __name__ == "__main__":
    main()
