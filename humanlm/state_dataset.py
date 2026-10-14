# Copyright 2026 HUMANLM team and/or its affiliates
# Copyright 2026 Bytedance Ltd. and/or its affiliates

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
from typing import Optional

import datasets
from omegaconf import DictConfig, ListConfig
from recipe.humanlm.process_dataset import format_persona
from transformers import PreTrainedTokenizer, ProcessorMixin

import verl.utils.torch_functional as verl_F
from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)


SYS_START = "<|im_start|>system\n"
SYS_END = "<|im_end|>\n"


class StateDataset(RLHFDataset):
    """
    Extends RLHFDataset to support training with multiple "state levels"
    (e.g., different latent user states) by dynamically
    substituting system prompts at runtime.

    Key Features:
        - State augmentation: Multiplies dataset size by swapping system
          prompts according to a state config, allowing one instance to
          appear with different system prompts (e.g., "response", "goal").
        - Train/val splitting: Automaticaly limits validation set size.
        - Multi-role chat templates: Supports custom chat templates with
          "speak_as" parameter.
        - Heterogeneous thinking: Optionally disables thinking mode for
          non-response states during training.

    Args:
        data_files: Path(s) to Parquet file(s) containing prompts.
        tokenizer: HuggingFace tokenizer for text tokenization.
        config: DictConfig with options including:
            - state_config_path: Path to state JSON config.
            - augment_with_states: Whether to multiply dataset by hierarchies.
            - enable_hetero_think: Disable thinking for non-response hierarchies.
            - val_size: Maximum validation set size (default: 2000).
            - eval_state_name: state to use for validation (default: "response").
            - All parent RLHFDataset config options.
        processor: Optional multimodal processor for images/videos.
        is_train: If False, limits dataset to val_size and uses eval_state_name only.

    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        is_train: bool = True,
        max_samples: int = -1,
    ):
        self.speak_as = True if config.kwargs.get("multirole_chat_template_path") else False
        self.state_config_path = config.get("state_config_path", None)
        self.state_system_prompts = {}
        self.augment_with_states = config.get("augment_with_states", False)
        self.enable_hetero_think = config.get("enable_hetero_think", False)
        if isinstance(data_files, (list, ListConfig)):
            _files = list(data_files)
        else:
            _files = [data_files]

        is_train = any(f.endswith("train.parquet") for f in _files)
        self.is_train = is_train
        self.state_names = []
        self.val_size = config.get("val_size", 2000)
        self.dataset = config.get("dataset", None)
        self.eval_state_name = config.get("eval_state_name", "response")

        self.additional_generation_prompt = config.get("additional_generation_prompt", "")

        if self.state_config_path and self.augment_with_states:
            self._load_state_system_prompts()

        super().__init__(data_files, tokenizer, config, processor)

        self._original_len = len(self.dataframe)

    def _load_state_system_prompts(self):
        """Load state config and read system prompt files into memory."""
        import json

        with open(self.state_config_path) as f:
            state_config = json.load(f)

        self.state_names = list(state_config.keys())

        for state_name, cfg in state_config.items():
            system_prompt_path = cfg.get("system_prompt")
            if system_prompt_path:
                if not os.path.isabs(system_prompt_path):
                    # Make relative paths relative to state config location
                    base_dir = os.path.dirname(self.state_config_path)
                    system_prompt_path = os.path.join(base_dir, system_prompt_path)

                if os.path.exists(system_prompt_path):
                    with open(system_prompt_path) as f:
                        self.state_system_prompts[state_name] = f.read().strip()
                    print(f"Loaded system prompt for '{state_name}' from {system_prompt_path}")
                else:
                    print(f"Warning: System prompt file not found: {system_prompt_path}")
        print(self.state_system_prompts)
        print(f"Loaded {len(self.state_system_prompts)} state system prompts")

    def _load_single_state_prompt(self, system_prompt_path):
        if os.path.exists(system_prompt_path):
            with open(system_prompt_path) as f:
                new_system_prompt = f.read().strip()

            print(f"Loaded NEW SYSTEM PROMPT FROM {system_prompt_path}")
        else:
            print(f"Warning: System prompt file not found: {system_prompt_path}")

        return new_system_prompt

    def _replace_system_prompt(self, messages: list, new_system_prompt: str) -> list:
        """Replace or inject a system prompt into the message list."""
        if not messages:
            return [{"role": "system", "content": new_system_prompt}]

        messages = copy.deepcopy(messages)
        if messages[0]["role"] == "system":
            messages[0]["content"] = new_system_prompt
        else:
            messages.insert(0, {"role": "system", "content": new_system_prompt})

        return messages

    def _get_state_system_prompt(self, row_dict: dict, state_name: str) -> Optional[str]:
        """Get the formatted system prompt for a specific state level."""
        if not self.state_system_prompts:
            return None

        template = self.state_system_prompts.get(state_name)
        if template is None:
            return None

        extra_info = row_dict.get("extra_info")
        if extra_info is None:
            extra_info = {}
        elif not isinstance(extra_info, dict):
            extra_info = {}

        format_dict = dict(extra_info)

        if "persona" in format_dict:
            if isinstance(format_dict["persona"], str):
                import json

                try:
                    format_dict["persona"] = json.loads(format_dict["persona"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(format_dict["persona"], dict):
                format_dict["persona"] = format_persona(format_dict["persona"])

        try:
            formatted_prompt = template.format(**format_dict)
            return formatted_prompt
        except KeyError as e:
            print(f"Warning: Missing key {e} for system prompt template")
            return None
        except Exception as e:
            print(f"Warning: Failed to format system prompt: {e}")
            return None

    def _get_new_system_prompt(self, row_dict: dict, template) -> Optional[str]:
        if template is None:
            return None

        extra_info = row_dict.get("extra_info")
        if extra_info is None:
            extra_info = {}

        elif not isinstance(extra_info, dict):
            extra_info = {}
        format_dict = dict(extra_info)

        if "persona" in format_dict:
            if isinstance(format_dict["persona"], str):
                import json

                try:
                    format_dict["persona"] = json.loads(format_dict["persona"])
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(format_dict["persona"], dict):
                format_dict["persona"] = format_persona(format_dict["persona"])
        try:
            formatted_prompt = template.format(**format_dict)
            return formatted_prompt

        except KeyError as e:
            print(f"Warning: Missing key {e} for system prompt template")
            return None
        except Exception as e:
            print(f"Warning: Failed to format system prompt: {e}")
            return None

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        # For eval and test set, select only a portion of the instances
        if not self.is_train:
            self.dataframe = self.dataframe.select(range(min(self.val_size, len(self.dataframe))))

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)

                    # Pass in the name for our custom chat template
                    if isinstance(messages, str):
                        raw_prompt = messages
                    elif self.speak_as:
                        raw_prompt = (
                            self.processor.apply_chat_template(
                                messages,
                                add_generation_prompt=True,
                                tokenize=False,
                                speak_as=doc["extra_info"]["name"],
                                **self.apply_chat_template_kwargs,
                            )
                            + self.additional_generation_prompt
                        )
                    else:
                        raw_prompt = (
                            self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                            + self.additional_generation_prompt
                        )

                    images = (
                        [process_image(image) for image in doc[image_key]]
                        if image_key in doc and doc[image_key]
                        else None
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]]
                        if video_key in doc and doc[video_key]
                        else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])
            else:

                def doc2len(doc) -> int:
                    if self.speak_as:
                        return len(
                            tokenizer.apply_chat_template(
                                doc[prompt_key],
                                add_generation_prompt=True,
                                speak_as=doc["extra_info"]["name"],
                                **self.apply_chat_template_kwargs,
                            )
                        ) + len(tokenizer.encode(self.additional_generation_prompt, add_special_tokens=False))
                    else:
                        return len(
                            tokenizer.apply_chat_template(
                                doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                            )
                        ) + len(tokenizer.encode(self.additional_generation_prompt, add_special_tokens=False))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def __len__(self):
        if self.augment_with_states and self.is_train:
            return self._original_len * len(self.state_names)
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)
        if isinstance(messages, str):
            return messages

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    # Replace system prompts when prompt is a string already
    def _replace_qwen3_system(self, prompt: str, new_system: str) -> str:
        if prompt.startswith(SYS_START):
            end = prompt.find(SYS_END, len(SYS_START))
            if end == -1:
                raise ValueError("No system prompt start found in prompt")

            suffix = prompt[end + len(SYS_END) :]

            return SYS_START + new_system + SYS_END + suffix
        else:
            raise ValueError("no system prompt in prompt to replace")

    def __getitem__(self, item):
        if (self.augment_with_states and self.state_names) and self.is_train:
            original_idx = item % self._original_len
            state_idx = item // self._original_len
            state_name = self.state_names[state_idx]
        elif self.augment_with_states and not self.is_train:
            # if we are augmenting, during validation we only want to generate the response
            original_idx = item
            state_name = self.eval_state_name
        else:
            original_idx = item
            state_name = None

        row_dict: dict = self.dataframe[original_idx]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        # Apply state system prompt
        if self.augment_with_states and state_name:
            if "extra_info" not in row_dict or row_dict["extra_info"] is None:
                row_dict["extra_info"] = {}

            if not isinstance(row_dict, dict):
                raise ValueError("ROW DICT not dict")

            if not isinstance(row_dict["extra_info"], dict):
                raise ValueError("Extra info not dict")

            row_dict["extra_info"]["state_name"] = state_name

            state_system_prompt = self._get_state_system_prompt(row_dict, state_name)
            if state_system_prompt is None:
                raise ValueError(f"Failed to construct state system prompt for '{state_name}'")
            if isinstance(messages, list):
                messages = self._replace_system_prompt(messages, state_system_prompt)
            elif isinstance(messages, str):
                messages = self._replace_qwen3_system(messages, state_system_prompt)

        if not self.is_train:
            row_dict["extra_info"]["state_name"] = "response"

        row_dict["is_val"] = not self.is_train
        row_dict["extra_info"]["is_val"] = not self.is_train

        if self.enable_hetero_think and self.is_train and not state_name == "response":
            apply_chat_template_kwargs = copy.deepcopy(self.apply_chat_template_kwargs)
            apply_chat_template_kwargs["enable_thinking"] = False
        else:
            apply_chat_template_kwargs = self.apply_chat_template_kwargs

        if self.processor is None:
            if apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )

            if isinstance(messages, str):
                raw_prompt = messages
            else:
                raw_prompt = (
                    self.tokenizer.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=False,
                        speak_as=row_dict["extra_info"]["name"],
                        **apply_chat_template_kwargs,
                    )
                    + self.additional_generation_prompt
                )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")
        else:
            raise ValueError("StateDataset not implemented for given processor") from None

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        row_dict["raw_prompts"] = input_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length :]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompt without chat template for later
        row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs

        return row_dict
