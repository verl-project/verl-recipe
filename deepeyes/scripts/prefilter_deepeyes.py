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
"""Pre-filter DeepEyes train/val parquet with verl's overlong-prompt filter.

Training normally runs this filter on every launch, which re-renders and
re-tokenizes all prompts (~15 minutes for 33k rows). Run it once here, then
start training on the filtered files with ``data.filter_overlong_prompts=False``.

    python3 recipe/deepeyes/scripts/prefilter_deepeyes.py \
        --model ~/models/Qwen3.5-9B --data-dir ~/data/deepeyes \
        --tool-config recipe/deepeyes/configs/image_zoom_in_tool_config.yaml

The tool config, tokenizer and chat-template settings must match the training
run; regenerate the filtered files when any of them changes.
"""

import argparse
import os

from omegaconf import OmegaConf

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.dataset.rl_dataset import get_dataset_class


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model or tokenizer path")
    parser.add_argument("--data-dir", required=True, help="directory containing train.parquet/val.parquet")
    parser.add_argument("--tool-config", default=None, help="tool config YAML used by the rollout")
    parser.add_argument("--max-prompt-length", type=int, default=8192)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--image-key", default="images")
    parser.add_argument("--mm-max-pixels", type=int, default=4194304)
    args = parser.parse_args()

    config = OmegaConf.create(
        {
            "custom_cls": {"path": "pkg://recipe.deepeyes.deepeyes", "name": "CustomRLHFDatasetQwen35"},
            "prompt_key": "prompt",
            "image_key": args.image_key,
            "max_prompt_length": args.max_prompt_length,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": args.workers,
            "apply_chat_template_kwargs": {"enable_thinking": True},
            "mm_processor_kwargs": {"max_pixels": args.mm_max_pixels},
            "truncation": "error",
            "tool_config_path": args.tool_config,
        }
    )

    tokenizer = hf_tokenizer(args.model)
    processor = hf_processor(args.model)
    dataset_cls = get_dataset_class(config)

    for split in ("train", "val"):
        src = os.path.join(args.data_dir, f"{split}.parquet")
        dst = os.path.join(args.data_dir, f"{split}.filtered_{args.max_prompt_length}.parquet")
        dataset = dataset_cls(data_files=src, tokenizer=tokenizer, processor=processor, config=config)
        print(f"{split}: kept {len(dataset.dataframe)} rows -> {dst}", flush=True)
        dataset.dataframe.to_parquet(dst)


if __name__ == "__main__":
    main()
