#!/usr/bin/env python3
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
"""Build DeepEyes train/val parquet files for the Qwen3.5 recipe.

Concatenates the DeepEyes shards (the Chart shard is excluded by default),
upscales images below the processor's 32-pixel factor, pads extreme aspect
ratios, writes train/val parquet and prints a vision-token report.

    python3 recipe/deepeyes/scripts/prepare_data.py --output-dir ~/data/deepeyes
    python3 recipe/deepeyes/scripts/prepare_data.py --stats-only
    python3 recipe/deepeyes/scripts/prepare_data.py --local-parquet /data/deepeyes/*.parquet
"""

import argparse
import io
import json
import statistics
from pathlib import Path

from PIL import Image

DEFAULT_REPO = "ChenShawn/DeepEyes-Datasets-47k"
DEFAULT_FILES = ["data_0.1.2_visual_toolbox_v2.parquet", "data_thinklite_reasoning_acc.parquet"]
CHART_FILE = "data_v0.8_visual_toolbox_v2.parquet"

# patch_size 16 * merge_size 2: the processor rejects images below 32 px per side.
QWEN35_RESIZE_FACTOR = 32


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--hf-repo", default=DEFAULT_REPO, help=f"Hugging Face dataset repo (default: {DEFAULT_REPO}).")
    source.add_argument("--local-parquet", nargs="+", default=None, help="Local parquet files to use instead of HF.")
    parser.add_argument(
        "--files", nargs="+", default=DEFAULT_FILES, help="Shards to download from --hf-repo (Chart is excluded)."
    )
    parser.add_argument("--include-chart", action="store_true", help=f"Also use {CHART_FILE}.")
    parser.add_argument("--output-dir", default=str(Path.home() / "data" / "deepeyes"), help="Output directory.")
    parser.add_argument("--val-size", type=int, default=256, help="Number of validation samples.")
    parser.add_argument("--min-side", type=int, default=QWEN35_RESIZE_FACTOR, help="Upscale images below this side.")
    parser.add_argument("--max-aspect-ratio", type=float, default=200.0, help="Pad images with a larger aspect ratio.")
    parser.add_argument("--max-samples", type=int, default=-1, help="Keep at most this many samples (debugging).")
    parser.add_argument("--num-proc", type=int, default=8, help="Processes used to repair images.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stats-only", action="store_true", help="Only print the report; do not write parquet.")
    return parser.parse_args()


def resolve_files(args: argparse.Namespace) -> list[str]:
    if args.local_parquet:
        return list(args.local_parquet)

    from huggingface_hub import hf_hub_download

    files = list(args.files)
    if args.include_chart and CHART_FILE not in files:
        files.append(CHART_FILE)
    print(f"Downloading {len(files)} shard(s) from {args.hf_repo}: {files}")
    return [hf_hub_download(repo_id=args.hf_repo, repo_type="dataset", filename=name) for name in files]


def _pad_to_aspect_ratio(image: Image.Image, max_ratio: float) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0 or max(width, height) / min(width, height) <= max_ratio:
        return image

    if width > height:
        target_height = int(round(width / max_ratio))
        canvas = Image.new("RGB", (width, target_height), (255, 255, 255))
        canvas.paste(image, (0, (target_height - height) // 2))
    else:
        target_width = int(round(height / max_ratio))
        canvas = Image.new("RGB", (target_width, height), (255, 255, 255))
        canvas.paste(image, ((target_width - width) // 2, 0))
    return canvas


def repair_image(image: dict, min_side: int, max_ratio: float) -> tuple[dict, bool, int, int]:
    """Upscale or pad an image for the Qwen3.5 processor.

    Returns ``(image, changed, width, height)``; sizes are 0 for undecodable bytes.
    """
    raw = image.get("bytes")
    if not raw:
        return image, False, 0, 0

    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except Exception:  # noqa: BLE001 - keep broken bytes so the error surfaces in training
        return image, False, 0, 0

    changed = False
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
        changed = True

    width, height = pil.size
    if min(width, height) < min_side:
        scale = min_side / min(width, height)
        pil = pil.resize((max(min_side, round(width * scale)), max(min_side, round(height * scale))), Image.LANCZOS)
        changed = True

    padded = _pad_to_aspect_ratio(pil, max_ratio)
    if padded is not pil:
        pil = padded
        changed = True

    width, height = pil.size
    if not changed:
        return image, False, width, height

    buffer = io.BytesIO()
    pil.save(buffer, format="JPEG", quality=95, subsampling=0)
    return {**image, "bytes": buffer.getvalue()}, True, width, height


def estimate_vision_tokens(width: int, height: int) -> int:
    """Approximate Qwen3.5 vision tokens: pixels / (patch_size * merge_size) ** 2."""
    if width <= 0 or height <= 0:
        return 0
    return max(1, width // QWEN35_RESIZE_FACTOR) * max(1, height // QWEN35_RESIZE_FACTOR)


def repair_split(dataset, args: argparse.Namespace):
    def repair_row(row: dict) -> dict:
        images = row.get("images") or []
        repaired, was_repaired, token_estimate, min_side = [], False, 0, 0
        for image in images:
            image, changed, width, height = repair_image(image, args.min_side, args.max_aspect_ratio)
            repaired.append(image)
            was_repaired = was_repaired or changed
            if width and height:
                token_estimate += estimate_vision_tokens(width, height)
                min_side = min(min_side, width, height) if min_side else min(width, height)
        return {
            "images": repaired,
            "__deepeyes_repaired": int(was_repaired),
            "__deepeyes_tokens": int(token_estimate),
            "__deepeyes_min_side": int(min_side),
        }

    return dataset.map(repair_row, num_proc=args.num_proc, desc="Repairing images")


def print_report(dataset, args: argparse.Namespace) -> None:
    num_rows = len(dataset)
    repaired = sum(dataset["__deepeyes_repaired"])
    tokens = sorted(dataset["__deepeyes_tokens"])
    min_sides = sorted(side for side in dataset["__deepeyes_min_side"] if side > 0)
    sources: dict[str, int] = {}
    for source in dataset["data_source"]:
        sources[source] = sources.get(source, 0) + 1

    def pct(values: list[int], q: float) -> int:
        if not values:
            return 0
        return values[min(len(values) - 1, int(q * len(values)))]

    print("\n===== DeepEyes data report =====")
    print(f"samples              : {num_rows}")
    print(f"images repaired      : {repaired} ({repaired / max(1, num_rows):.2%} of samples)")
    print(
        f"min image side       : median={int(statistics.median(min_sides)) if min_sides else 0}, "
        f"p5={pct(min_sides, 0.05)}, p95={pct(min_sides, 0.95)}"
    )
    print(
        f"est. vision tokens   : mean={int(statistics.mean(tokens)) if tokens else 0}, "
        f"p50={pct(tokens, 0.5)}, p95={pct(tokens, 0.95)}, p99={pct(tokens, 0.99)}, max={tokens[-1] if tokens else 0}"
    )
    print(f"data_source          : {json.dumps(dict(sorted(sources.items(), key=lambda kv: -kv[1])))}")
    print("================================")


def main() -> None:
    args = parse_args()

    from datasets import concatenate_datasets, load_dataset

    files = resolve_files(args)
    parts = []
    for path in files:
        part = load_dataset("parquet", data_files=path, split="train")
        part = part.add_column("__deepeyes_source_file", [Path(path).name] * len(part))
        parts.append(part)
    dataset = concatenate_datasets(parts).shuffle(seed=args.seed)

    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    dataset = repair_split(dataset, args)
    print_report(dataset, args)

    if args.stats_only:
        return

    dataset = dataset.remove_columns([column for column in dataset.column_names if column.startswith("__deepeyes")])
    val_size = min(args.val_size, len(dataset))
    if val_size >= len(dataset):
        raise ValueError(f"val_size={val_size} leaves no training samples out of {len(dataset)}; use a smaller value.")
    val = dataset.select(range(val_size))
    train = dataset.select(range(val_size, len(dataset)))

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    train.to_parquet(str(train_path))
    val.to_parquet(str(val_path))
    print(f"Wrote {len(train)} train samples -> {train_path}")
    print(f"Wrote {len(val)} val samples   -> {val_path}")


if __name__ == "__main__":
    main()
