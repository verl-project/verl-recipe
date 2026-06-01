# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        try:
            tensors[key] = torch.stack(val, dim=0)
        except Exception:
            non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
        use_negative=False,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]
        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.use_negative = use_negative

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key

            dataframe = dataframe.filter(
                lambda doc: self.doc2len(prompt_key, doc, tokenizer, processor) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def doc2len(self, prompt_key, doc, tokenizer, processor) -> int:
        prompt = doc[prompt_key]
        if processor is not None:
            messages = self._build_messages(doc)
            content = messages[0].get("content", [])
            if len(content) > 1 and content[0].get("type", "") == "text":
                inputs, _ = processor.preprocess(messages)
                input_ids = inputs["input_ids"]
                input_ids_len = input_ids.size()[1]
            else:
                apply_kwargs = dict(**self.apply_chat_template_kwargs)
                if self.tool_schemas is not None:
                    apply_kwargs["tools"] = self.tool_schemas
                input_ids_len = len(
                    tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **apply_kwargs)
                )
            return input_ids_len
        elif tokenizer.__class__.__name__ == "T5TokenizerFast":
            text = prompt[0].get("content", "")
            text_inputs = tokenizer(
                text,
                padding="max_length",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
            mask_sum = mask.sum(dim=1, keepdim=True)
            input_ids_len = mask_sum.size()[1]
            return input_ids_len
        else:
            apply_kwargs = dict(**self.apply_chat_template_kwargs)
            if self.tool_schemas is not None:
                apply_kwargs["tools"] = self.tool_schemas
            input_ids_len = len(
                tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True, **apply_kwargs)
            )
            return input_ids_len

    def get_seq_len(self, doc, tokenizer, processor) -> int:
        if processor is not None:
            messages = self._build_messages(doc)
            content = messages[0]["content"]
            if len(content) > 1 and content[0].get("type", "") == "text":
                inputs, _ = processor.preprocess(messages)
                input_ids = inputs["input_ids"]
                input_ids_len = input_ids.size()[1]
            else:
                apply_kwargs = dict(**self.apply_chat_template_kwargs)
                if self.tool_schemas is not None:
                    apply_kwargs["tools"] = self.tool_schemas

                input_ids_len = len(
                    tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True, **apply_kwargs)
                )
            return input_ids_len

        elif tokenizer.__class__.__name__ == "T5TokenizerFast":
            prompt_text = doc[self.prompt_key][0].get("content", "")
            text_inputs = tokenizer(
                prompt_text,
                padding="max_length",
                max_length=self.max_prompt_length,
                truncation=True,
                add_special_tokens=True,
                return_special_tokens_mask=True,
                return_tensors="pt",
            )
            input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
            mask_sum = mask.sum(dim=1, keepdim=True)
            input_ids_len = mask_sum.size()[1]
            return input_ids_len

        else:
            apply_kwargs = dict(**self.apply_chat_template_kwargs)
            if self.tool_schemas is not None:
                apply_kwargs["tools"] = self.tool_schemas
            input_ids_len = len(
                tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True, **apply_kwargs)
            )

            return input_ids_len

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        images_path = example.get(self.image_key, "")
                        content_list.append({"type": "image", "image": images_path, "max_pixels": 1024 * 1024})
                        content_list.append({"type": "image_gen", "image": images_path, "max_pixels": 1024 * 1024})
                    elif segment == "<video>":
                        videos_path = example.get(self.video_key, "")
                        content_list.append({"type": "video", "video": videos_path})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list
        else:
            for message in messages:
                content = message["content"]
                content_list = []
                content_list.append({"type": "text", "text": content})
                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        global neg_prompt_embed
        row_dict: dict = self.dataframe[item]
        seq_len = self.get_seq_len(row_dict, self.tokenizer, self.processor)

        messages = self._build_messages(row_dict)

        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["placeholder"] = torch.zeros(1)
        row_dict["messages"] = messages
        row_dict["seq_len"] = seq_len

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()
