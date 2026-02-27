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

import argparse
import ast
import asyncio
import copy
import datetime
import glob
import json
import os
import re
from collections import Counter
from copy import deepcopy
from pathlib import Path

import datasets
import litellm
import polars as pl
from datasets import (Dataset, DatasetDict, Features, Sequence, Value,
                      concatenate_datasets)

from humanlm.utils import parse_messages


def _dist(keys, fill=0.0):
    return {k: float(fill) for k in keys}  


RL_TEMPLATE = {
    "data_source": "",
    "prompt": [
        {"role": "system", "content": "", "name": ""},
        {"role": "user",   "content": "", "name": ""},
    ],
    "ability": "",
    "reward_model": {"style": "", "ground_truth": ""},
    "extra_info": {
        "split": "", "index": 0, "name": "", "post": "",
        "state_name": "", 
        "persona": "", 
        "media_source": "",
        "target_user_id": "",
        "raw_prompt": "",
        "post_metrics": "",
        "comment_metrics": "",
        "post_id": "", "is_top_level": False
    },
}
SFT_TEMPLATE = {
    "prompt": [
        {"role": "system", "content": "", "name": ""},
        {"role": "user",   "content": "", "name": ""},
    ],
    "generation": {"role": "user", "content": "", "name": ""},
}

SFT_FEATURES = Dataset.from_list([SFT_TEMPLATE]).features
RL_FEATURES = Dataset.from_list([RL_TEMPLATE]).features  


def is_valid_persona_json(persona: dict) -> bool:
    """Check if the persona dict has sufficient information"""
        
    total_items = 0
    for key in ['interests', 'values', 'communication', 'statistics']:
        if key in persona:
            if isinstance(persona[key], list):
                total_items += len(persona[key])
    if total_items < 10:
        print('***** persona too small:', persona)
    return total_items >= 10

def format_persona(persona: dict, field_dropout_prob: float = 0.0, item_dropout_prob: float = 0.0, seed: int = 42) -> str:
    """Parse persona dict to a readable string with deterministic field-level and item-level dropout
    
    Args:
        persona: Dictionary containing persona information
        field_dropout_prob: Probability used to compute number of fields to drop (0.0 to 1.0)
        item_dropout_prob: Probability of dropping each item within kept fields (0.0 to 1.0)
        seed: Random seed for deterministic dropout
    """
    import hashlib
    import json
    if isinstance(persona, str):
        return persona

    # Helper: deterministic hash-based decision
    persona_id = json.dumps(persona, sort_keys=True)
    
    def hash_decision(key: str, threshold: float) -> bool:
        """Returns True if hash of key is below threshold"""
        if threshold == 0.0:
            return False
        hash_val = int(hashlib.md5(f"{seed}_{persona_id}_{key}".encode()).hexdigest(), 16)
        return (hash_val % 10000) / 10000.0 < threshold
    
    # Step 1: Field-level dropout - determine which fields to keep
    demographics_has_content = any(v and str(v).strip() != "NA" for k, v in persona.get("demographics", {}).items())
    
    if field_dropout_prob == 0.0:
        kept_fields = {"demographics", "interests", "values", "communication", "statistics"}
    else:
        # Determine pool of fields that can be dropped
        if demographics_has_content:
            all_fields = ["demographics", "interests", "values", "communication", "statistics"]
            max_drop = 4  # Keep at least 1 field
        else:
            all_fields = ["interests", "values", "communication", "statistics"]
            max_drop = 3  # Keep at least 1 field
        
        num_drop = min(int(field_dropout_prob * 5), max_drop)
        num_keep = len(all_fields) - num_drop
        
        # Sort fields by hash and keep top num_keep
        field_hashes = [(int(hashlib.md5(f"{seed}_{persona_id}_field_{f}".encode()).hexdigest(), 16), f) 
                        for f in all_fields]
        field_hashes.sort()
        kept_fields = set(f for _, f in field_hashes[:num_keep])
    
    # Step 2: Build output with item-level dropout on kept fields
    lines = []
    total_items, dropped_items = 0, 0
    
    # Demographics
    if "demographics" not in kept_fields or not demographics_has_content:
        lines.append("Demographics: Missing")
    else:
        demo_items = []
        for k, v in persona["demographics"].items():
            if v and str(v).strip() != "NA":
                total_items += 1
                if hash_decision(f"item_{k}:{v}", item_dropout_prob):
                    dropped_items += 1
                else:
                    demo_items.append(f"  {k}: {v}")
        
        if demo_items:
            lines.append("Demographics:")
            lines.extend(demo_items)
        else:
            lines.append("Demographics: Missing")
    
    # Other aspects
    for aspect in ["interests", "values", "communication", "statistics"]:
        if aspect not in kept_fields:
            lines.append(f"{aspect.capitalize()}: Missing")
        else:
            kept_items = []
            for item in persona.get(aspect, []):
                total_items += 1
                if hash_decision(f"item_{item}", item_dropout_prob):
                    dropped_items += 1
                else:
                    kept_items.append(item)
            
            if kept_items:
                lines.append(f"{aspect.capitalize()}: \n  " + '\n  '.join(kept_items))
            else:
                lines.append(f"{aspect.capitalize()}: Missing")
    
    result = "\n".join(lines)
    # Logging
    if field_dropout_prob > 0.0 or item_dropout_prob > 0.0:
        kept = sorted(kept_fields)
        item_rate = f"{dropped_items/total_items:.1%}" if total_items > 0 else "0.0%"
        print(f"Dropout - Fields: kept {kept}, Items: {dropped_items}/{total_items} " +
              f"({item_rate} vs target {item_dropout_prob:.1%})")
    
        print(result)
        print("---------------------")
    return result


THINKING_GENERATION_PROMPT = """You are generating an internal reasoning trace (thinking) that a real human user would have before writing a response.

Given the user's persona, conversation context, and their actual response, generate a brief, natural internal monologue that reflects what they might be thinking or feeling before responding.

Your thinking should naturally consider:
- **Stance**: Do I agree or disagree with what's being said? How strongly?
- **Sentiment**: What's my emotional valence - positive, negative, or neutral?
- **Emotion**: What specific emotion am I feeling? (e.g., anger, amusement, curiosity, disappointment, excitement, etc.)
- **Content type**: What kind of response am I making? (e.g., sharing an opinion, asking a question, providing evidence, telling a story, etc.)
- **Tone/style**: Should I be serious, humorous, sarcastic, or neutral? How hostile or friendly should I be?

Guidelines:
- Keep it brief (2-4 sentences, ~50-150 words)
- Natural and human-like, not robotic
- May include conflicting thoughts, uncertainties, or hidden considerations
- Reflect the persona's characteristics, values, and communication style
- Should logically lead to the given response
- Focus on internal reasoning about stance, emotion, and how to express yourself
- Don't just restate the response - show the internal process that leads to it

<|The Start of Persona|>
{persona}
<|The End of Persona|>

<|The Start of Context|>
{context}
<|The End of Context|>

<|The Start of Response|>
{response}
<|The End of Response|>

Output the thinking trace directly (no tags or formatting):"""


async def generate_thinking_trace(persona: str, context: str, response: str, model: str = "gpt-4o-mini", **kwargs):
    """Generate a synthetic thinking trace using litellm."""
    prompt = THINKING_GENERATION_PROMPT.format(persona=persona, context=context, response=response)
    
    try:
        resp = await litellm.acompletion(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            temperature=kwargs.pop("temperature", 0.7),
            max_tokens=kwargs.pop("max_tokens", 256),
            num_retries=kwargs.pop("num_retries", 3),
            **kwargs,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


class DatasetMapper:
    """
    `DatasetMapper` is a helper to map the original dataset to fit our training formats.
    Essentially, we need
        - comment_history:
        - poster_id: the user name/id who made the post
        - post: the post content
        - post_prompt: besides `post` content itself, it includes more details like category/subreddit.
        - target_user_id: the user name/id who made the response
        - response: the response content
    """

    def __init__(self, platform: str, is_assistant_chat_mode: bool, raw_template: str, data_source: str, top_state_name: str = None, trajectory_path=None):
        self.platform = platform
        self.is_assistant_chat_mode = is_assistant_chat_mode
        self.raw_template = raw_template
        self.data_source = data_source
        self.top_state_name = top_state_name
        if trajectory_path is not None:
            dset = datasets.load_dataset("json", data_files=trajectory_path, split="train")
            self.tag_dict = {row["index"]: row["tags"] for row in dset}
        else:
            self.tag_dict = None

    def get_response_usr_id(self, example: dict) -> int:
        return example['user_id']

    def get_message_usr_id(self, message):
        raise NotImplementedError

    def get_response_content(self, example: dict) -> int:
        return example['completion']

    def get_message_content(self, message):
        return message["content"]

    def get_response_timestamp(self, example: dict) -> int:
        dt = datetime.datetime.strptime(example["timestamp"], "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()

    def get_message_timestamp(self, message: dict) -> str:
        raise NotImplementedError
    
    def is_assistant_message(self, message, target_user_id) -> bool:
        return target_user_id != self.get_message_usr_id(message)
        
    def make_map_fn_sft(self, split, thinking_sft=False, thinking_model=None, thinking_cache=None):
        def process_fn(example, idx):
            target_user_id = self.get_response_usr_id(example)
            persona = format_persona(
                example["persona"], 
                field_dropout_prob=mapper.field_dropout_prob, 
                item_dropout_prob=mapper.item_dropout_prob
                )
            
            response = self.get_response_content(example)
            
            # TODO: REMOVE TAG WHEN ARGS.NO_TAG
            # If thinking_sft, include thinking trace
            if thinking_sft:
                cache_key = f"{split}_{idx}"
                thinking = thinking_cache[cache_key]
                response = f"<think>\n{thinking}\n</think>\n<response>\n{response}\n</response>"
            elif self.no_tag:
                response = response
            else:
                response = f"<response>\n{response}\n</response>"

            replace_target_user_identifier = True
            target_user_identifier = 'HUMAN'

            values = {
                "name": target_user_id,
                "persona": persona,
                "platform": self.platform,
                "memory": None,
            }
            system_content = self.raw_template.format(**values)  

            if self.is_assistant_chat_mode:
                prompt = [
                    {
                        "role": "system",
                        "name": "",
                        "content": str(system_content),
                    },
                    *[
                        {
                            "role": "user", 
                            "name": target_user_identifier if replace_target_user_identifier else self.get_message_usr_id(m),
                            "content": self.get_message_content(m),
                        } if not self.is_assistant_message(m, target_user_id) else
                        {
                            "role": "assistant", 
                            "content": self.get_message_content(m),
                        } for m in example["prompt"]
                    ]
                ]
            else:
                prompt = [
                    {
                        "role": "system",
                        "name": "",
                        "content": str(system_content),
                    },
                    *[
                        {
                            "role": "user", 
                            "name": target_user_identifier if (
                                target_user_id == self.get_message_usr_id(m) and 
                                replace_target_user_identifier
                            ) else self.get_message_usr_id(m),
                            "content": self.get_message_content(m),
                        } for m in example["prompt"]
                    ]
                ]
            data = {
                "prompt": prompt,
                "generation": {"role": "user", "name": target_user_identifier, "content": str(response)},
            }
            return data

        return process_fn

    def make_map_fn(self, split):
        assert self.platform is not None, "PLATFORM must be defined in subclass"
        print('dropout_prob', self.field_dropout_prob)
        def process_fn(example, idx):
            response = self.get_response_content(example)
            target_user_id = self.get_response_usr_id(example)
            persona = format_persona(
                example["persona"], 
                field_dropout_prob=mapper.field_dropout_prob, 
                item_dropout_prob=mapper.item_dropout_prob
            )
            
            values = {
                "name": target_user_id,
                "persona": persona,
                "platform": self.platform,
                "memory": None,
                
            }
            system_content = self.raw_template.format(**values)  
            
            replace_target_user_identifier = True
            target_user_identifier = 'HUMAN'

            if self.is_assistant_chat_mode:
                prompt = [
                    {
                        "role": "system",
                        "name": "",
                        "content": str(system_content),
                    },
                    
                    *[
                        {
                            "role": "user", 
                            "name": target_user_identifier if replace_target_user_identifier else self.get_message_usr_id(m),
                            "content": self.get_message_content(m),
                        } if not self.is_assistant_message(m, target_user_id) else
                        {
                            "role": "assistant", 
                            "content": self.get_message_content(m),
                        } for m in example["prompt"]
                    ]
                ]
            else:
                prompt = [
                    {
                        "role": "system",
                        "name": "",
                        "content": str(system_content),
                    },
                    *[
                        {
                            "role": "user",
                            "name": target_user_identifier if (
                                target_user_id == self.get_message_usr_id(m) and 
                                replace_target_user_identifier
                            ) else self.get_message_usr_id(m),
                            "content": self.get_message_content(m),
                        } for m in example["prompt"]
                    ]
                ]

            data = {
                "data_source": self.data_source,
                "prompt": prompt,
                "ability": "generation",
                "reward_model": {"style": "custom", "ground_truth": str(response)},
                "extra_info": {
                    "index": int(idx),
                    "split": str(split),
                    "target_user_id": str(target_user_id),
                    "name": target_user_identifier,
                    "persona": json.dumps(example["persona"]),
                    "raw_prompt": json.dumps(prompt),
                    "media_source": str(self.platform),
                    "state_name": str(self.top_state_name),
                    "post_metrics": json.dumps(example["prompt"][0]["metadata"].get("post_metrics")),
                    "comment_metrics": json.dumps(example["metadata"].get("comment_metrics")),
                    "post_id": str(example["post_id"]),
                    "is_top_level": bool(int(example["turn_id"]) == 1)
                },
            }
            return data

        return process_fn


class WildChatMapper(DatasetMapper):
    def __init__(self, platform: str, is_assistant_chat_mode: bool, raw_template: str, data_source: str, top_state_name: str, trajectory_path):
        super().__init__(platform, is_assistant_chat_mode, raw_template, data_source, top_state_name, trajectory_path)

    def get_message_usr_id(self, message):
        return message['role']

    def get_message_timestamp(self, message: dict) -> str:
        dt = datetime.datetime.strptime(message['metadata']["created_utc"], "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()
    
    def is_assistant_message(self, message, target_user_id) -> bool:
        #assistant if 'gpt' is in the role
        return "gpt" in self.get_message_usr_id(message).lower()



class RedditMapper(DatasetMapper):
    def __init__(self, platform: str, is_assistant_chat_mode: bool, raw_template: str, data_source: str, top_state_name: str, trajectory_path):
        super().__init__(platform, is_assistant_chat_mode, raw_template, data_source, top_state_name, trajectory_path)
    
    def get_message_usr_id(self, message):
        return message['metadata']['author']

    def get_message_timestamp(self, message: dict) -> str:
        dt = datetime.datetime.strptime(message['metadata']["created_utc"], "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()


class YoutubeMapper(DatasetMapper):
    def __init__(self, platform: str, is_assistant_chat_mode: bool, raw_template: str, data_source: str, top_state_name: str, trajectory_path):
        super().__init__(platform, is_assistant_chat_mode, raw_template, data_source, top_state_name, trajectory_path)
    
    def get_message_usr_id(self, message):
        if '#video' in message['metadata']['kind']:
            return message['metadata']['snippet']['channelTitle']
        elif '#comment' in message['metadata']['kind']:
            return message['metadata']['snippet']['authorDisplayName']
        else:
            raise ValueError(f"Unknown kind {message['metadata']['kind']}")

    def get_message_timestamp(self, message: dict) -> str:
        if '#video' in message['metadata']['kind']:
            return message['metadata']['snippet']['publishedAt']
        elif '#comment' in message['metadata']['kind']:
            dt = datetime.datetime.strptime(message['metadata']["timestamp"], "%Y-%m-%d %H:%M:%S")
        return dt.timestamp()

    def get_message_content(self, message):
        if '#video' in message['metadata']['kind']:
            content = f"- Title: {message['metadata']['title']}\n\n" + \
                      f"- Description: {message['metadata']['description']}\n\n" + \
                      f"- Transcript: {message['metadata']['transcript']}"
            return content
        elif '#comment' in message['metadata']['kind']:
            return message['content']


class MediumMapper(DatasetMapper):
    def __init__(self, platform: str, is_assistant_chat_mode: bool, raw_template: str, data_source: str, top_state_name: str, trajectory_path):
        super().__init__(platform, is_assistant_chat_mode, raw_template, data_source, top_state_name, trajectory_path)
    
    def get_message_usr_id(self, message):
        return message['role']

    def get_message_timestamp(self, message: dict) -> str:
        return message['metadata']['counts']['published_at']

    def get_message_content(self, message):
        if message['content'].startswith("POLITICS\n\n"):
            return message['content'][len("POLITICS\n\n"):]
        else:
            return message['content']

class AmazonReviewMapper(DatasetMapper):
    def __init__(self, platform: str, is_assistant_chat_mode: bool, raw_template: str, data_source: str, top_state_name: str, trajectory_path):
        super().__init__(platform, is_assistant_chat_mode, raw_template, data_source, top_state_name, trajectory_path)
    
    def get_message_usr_id(self, message):
        return f"Amazon store: {message['metadata']['store']}"

    def get_message_timestamp(self, message: dict) -> str:
        raise NotImplementedError

    def get_message_content(self, message):
        metadata = message['metadata']
        for key in ['details', 'author']:
            if not metadata[key] is None:
                metadata[key] = ast.literal_eval(metadata[key])

        assert isinstance(metadata['description'], list), f'get {type(metadata["description"])} for description'
        assert isinstance(metadata['features'], list), f'get {type(metadata["features"])} for features'
        assert isinstance(metadata['details'], dict), f'get {type(metadata["details"])} for details'
        content = (
            f"- Category: {'->'.join(metadata['categories'])}\n"
            f"- Title: {metadata['title']}\n"
            f"- Subtitle: {metadata['subtitle']}\n"
            f"- Price: {metadata['price']}\n"
        )
        if isinstance(metadata['author'], dict):
            content += f"- Author: {metadata['author']['name']}\n"
            if 'about' in metadata['author']:
                content += f"  " + ' '.join(metadata['author']['about']) + "\n"
        content += (
            "- Description: " + ' '.join(metadata['description']) + "\n"
            "- Features:\n" + ' '.join(metadata['features']) + "\n"
            "- Details:\n" + '\n'.join([f"  {k}: {v}" for k, v in metadata['details'].items()])
        )
        return content


class WildChatMapper2(DatasetMapper):
    """
    Mapper for WildChat-style processed datasets.

    Expected schema for each prompt message: {"content": <str>, "role": <str>, "metadata": <dict|str>}.
    """

    def __init__(self, platform: str, is_assistant_chat_mode: bool, raw_template: str, data_source: str, top_state_name: str, trajectory_path):
        super().__init__(platform, is_assistant_chat_mode, raw_template, data_source, top_state_name, trajectory_path)

    def get_message_usr_id(self, message):
        return message.get("role")

    def get_message_content(self, message):
        return message.get("content")


class EnronMapper(DatasetMapper):
    """
    Mapper for Enron processed datasets.

    Schema (confirmed):
      prompt: list of {"content": str, "role": str, "metadata": str}
    where role is typically the sender email.
    """

    def __init__(self, platform: str, is_assistant_chat_mode: bool, raw_template: str, data_source: str, top_state_name: str, trajectory_path):
        super().__init__(platform, is_assistant_chat_mode, raw_template, data_source, top_state_name, trajectory_path)

    def get_message_usr_id(self, message):
        return message.get("role")

    def get_message_content(self, message):
        return message.get("content")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["reddit", "amazon", "youtube", "medium", "wildchat_english", "enron"], required=True)
    
    # either provide parquet_dir or raw_dataset_repo
    parser.add_argument("--parquet_dir", type=str, required=False, default=None)
    parser.add_argument("--raw_dataset_repo", type=str, required=False, default=None)

    parser.add_argument("--save_data_dir", type=str, required=True)
    parser.add_argument('--save_prompt_dir', default="humanlm/system_prompts/")
    parser.add_argument('--state_config_path', default="humanlm/state_config/r_no_tag.json")
    parser.add_argument("--trajectory_path", default=None) 
    parser.add_argument("--thinking_sft", action="store_true", default=False,
                        help="Generate thinking traces for SFT data (requires --sft)")
    parser.add_argument("--thinking_model", type=str, default="gpt-4o-mini",
                        help="Model to use for generating thinking traces")
    parser.add_argument("--thinking_batch_size", type=int, default=20,
                        help="Batch size for async thinking generation")
    parser.add_argument("--field_dropout_prob", type=float, default=0.6, 
                        help="Persona field dropout probability (0.0 to 1.0)")
    parser.add_argument("--item_dropout_prob", type=float, default=0.6, 
                        help="Persona item dropout probability (0.0 to 1.0)")

    parser.add_argument('--train_subset_percentage', type=int, default=None)
    parser.add_argument('--test_subset_percentage', type=int, default=None)
    parser.add_argument("--sft", action="store_true", default=False)
    parser.add_argument("--no_tag", action="store_true", default=False)
    args = parser.parse_args()
    

    assert not (args.thinking_sft and args.no_tag), "--thinking_sft cannot be used with --no_tag"
    print("Arguments:", args)
    if args.thinking_sft and not args.sft:
        raise ValueError("--thinking_sft requires --sft to be set")

    def format_config_tags(config_json) -> str:
        sections = []
        for key, value in config_json.items():
            sections.append(f"<{key}>\n<{value['desc']}>\n</{key}>")
        return "\n".join(sections)

    state_config = json.load(open(args.state_config_path, "r"))
    state_config_name = args.state_config_path.split("/")[-1][:-len(".json")]
    top_state_name = list(state_config.keys())[0]
    if args.no_tag:
        assert len(state_config) == 1, "no_tag can only be used with single-level state"
        state_template = state_config[top_state_name]['desc']
        additional_notes = ""
    else:
        state_template = format_config_tags(state_config)
        additional_notes = "- Follow the exact order and use the exact XML-style tags\n" \
                           "- Do not output anything outside these XML-style tags"""
    

    # read system prompt template from save_prompt_dir/base.txt
    SYSTEM_PROMPT_TEMPLATE = open(os.path.join(args.save_prompt_dir, "base.txt"), "r").read()
    
    prompt_template = SYSTEM_PROMPT_TEMPLATE.format(state_template=state_template, additional_notes=additional_notes)
    # If the state config filename already encodes no_tag (e.g. r_no_tag.json),
    # don't append a second "_no_tag" suffix.
    no_tag_suffix = ""
    if args.no_tag and not state_config_name.endswith("_no_tag"):
        no_tag_suffix = "_no_tag"

    prompt_name = state_config_name + no_tag_suffix
    new_system_prompt_path = os.path.join(
        args.save_prompt_dir,
        prompt_name + '.txt'
    )
    with open(new_system_prompt_path, "w") as f:
        f.write(prompt_template)
    
    # Generate separate system prompts for each state
    if len(state_config) > 1:
        for name, config in state_config.items():
            single_state_config = {name: config}
            single_state_template = format_config_tags(single_state_config)
            single_prompt_template = SYSTEM_PROMPT_TEMPLATE.format(
                state_template=single_state_template,
                additional_notes=additional_notes
            )
            single_system_prompt_path = os.path.join(
                args.save_prompt_dir,
                f'{state_config_name}_{name}' + '.txt'
            )
            with open(single_system_prompt_path, "w") as f:
                f.write(single_prompt_template)

    mode_name = 'thinking_sft' if args.thinking_sft else ('sft' if args.sft else 'rl')
    args.save_data_dir = os.path.join(
        args.save_data_dir, mode_name, state_config_name + no_tag_suffix
    )
    if args.train_subset_percentage and args.train_subset_percentage:
        args.save_data_dir = os.path.join(
            args.save_data_dir, f'{args.train_subset_percentage}p'
        )
    os.makedirs(args.save_data_dir, exist_ok=True)

    # Create dataset mapper
    # Map dataset names to their default platform names
    DATASET_TO_PLATFORM = {
        "reddit": "Reddit",
        "amazon": "Amazon",
        "youtube": "Youtube",
        "medium": "Medium",
        "wildchat_english": "WildChat",
        "enron": "Enron",
    }
    if args.dataset == "reddit":
        MapperClass = RedditMapper
        is_assistant_chat_mode = False
    elif args.dataset == "amazon":
        MapperClass = AmazonReviewMapper
        is_assistant_chat_mode = False
    elif args.dataset == "youtube":
        MapperClass = YoutubeMapper
        is_assistant_chat_mode = False
    elif args.dataset == "medium":
        MapperClass = MediumMapper
        is_assistant_chat_mode = False
    elif args.dataset == "wildchat_english":
        MapperClass = WildChatMapper
        is_assistant_chat_mode = True
    elif args.dataset == "enron":
        MapperClass = EnronMapper
        is_assistant_chat_mode = False
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
    platform = DATASET_TO_PLATFORM[args.dataset]
    mapper = MapperClass(platform, is_assistant_chat_mode, prompt_template, args.dataset, top_state_name, args.trajectory_path)
    mapper.no_tag = args.no_tag

    # Load all splits first
    split_map = {
        "train": "train",
        "val": "val",
        "test_dropout": "test",
        "test": "test"
    }
    all_splits = ["train", "test_dropout", "test", "val"]
    raw_datasets = {}

    for split in all_splits:
        persona_dropout = (split == "test_dropout")
        mapper.field_dropout_prob = args.field_dropout_prob if persona_dropout else 0.0
        mapper.item_dropout_prob = args.item_dropout_prob if persona_dropout else 0.0
        print(f'split "{split}": persona_dropout={persona_dropout}, args.field_dropout_prob={mapper.field_dropout_prob}, dropout_prob={mapper.item_dropout_prob}')

        hf_split_name = split_map.get(split, split)
        if args.parquet_dir:
            parquet_files = os.listdir(args.parquet_dir)
            parquet_file_paths = [os.path.join(args.parquet_dir, f) for f in parquet_files if hf_split_name in f]
            raw = Dataset.from_parquet(parquet_file_paths)
        else:
            dataset = datasets.load_dataset(args.raw_dataset_repo)
            raw = dataset[hf_split_name]

        print(f'Mapping Dataset "{split}": {len(raw)} rows')
        
        if split == 'train' and args.train_subset_percentage is not None:
            subset_size = int(len(raw) * args.train_subset_percentage / 100)
            raw = raw.shuffle(seed=42)
            raw = raw.select(range(subset_size))
            print(f'Subset to {args.train_subset_percentage}%: {len(raw)} rows')

        if split in ['test', 'test_dropout'] and args.test_subset_percentage is not None:
            subset_size = int(len(raw) * args.test_subset_percentage / 100)
            raw = raw.shuffle(seed=42)
            raw = raw.select(range(subset_size))
            print(f'Subset to {args.test_subset_percentage}%: {len(raw)} rows')
        
        raw_datasets[split] = raw
        
        # remove persona examples when there are very little persona info
        def filter_fn(example):
            try:
                persona = json.loads(example['persona'])
            except Exception as e:
                persona = example['persona']
                
            if isinstance(persona, str):
                return True
            elif isinstance(persona, dict):
                return is_valid_persona_json(persona)
            else:
                raise ValueError

        raw_datasets[split] = raw_datasets[split].filter(filter_fn)
        print(f"{split} valid persona rate: {len(raw_datasets[split])}/{len(raw)}", len(raw_datasets[split]) / len(raw))

    # Generate thinking traces for train and val splits
    thinking_cache = {}
    cache_file = os.path.join(args.save_data_dir, "thinking_cache.json")
    
    if args.thinking_sft:
        def _valid_thinking_trace(x) -> bool:
            return isinstance(x, str) and x.strip() != ""

        # Load existing cache if it exists
        if os.path.exists(cache_file):
            print(f'Loading existing thinking cache from {cache_file}...')
            with open(cache_file, 'r') as f:
                thinking_cache = json.load(f)
            # Drop empty traces so they get regenerated (empty == previous failed LLM calls)
            before = len(thinking_cache)
            thinking_cache = {k: v for k, v in thinking_cache.items() if _valid_thinking_trace(v)}
            dropped = before - len(thinking_cache)
            print(f'Loaded {before} cached thinking traces ({dropped} empty traces dropped)')
        
        # Generate thinking for both train and val splits
        failed_cache_keys = []
        for thinking_split in ['train', 'val']:
            raw = raw_datasets[thinking_split]
            print(f'Generating thinking traces for "{thinking_split}"...')
            raw_list = [raw[i] for i in range(len(raw))]
            tasks = []
            
            for i, ex in enumerate(raw_list):
                cache_key = f"{thinking_split}_{i}"
                # Skip if already in cache
                if cache_key in thinking_cache and _valid_thinking_trace(thinking_cache.get(cache_key)):
                    continue
                    
                ex['metadata'] = json.loads(ex['metadata']) if isinstance(ex['metadata'], str) else ex['metadata']
                if isinstance(ex['prompt'], str):
                    ex['prompt'] = json.loads(ex['prompt'])
                ex['prompt'] = [
                    {
                        "role": p['role'], 
                        "content": p['content'], 
                        "metadata": json.loads(p['metadata']) if isinstance(p['metadata'], str) else p['metadata']
                    } for p in ex['prompt']
                ]
                if isinstance(ex['persona'], str):
                    # WildChat persona is often a plain string; some datasets use JSON.
                    try:
                        ex['persona'] = json.loads(ex['persona'])
                    except json.JSONDecodeError:
                        ex['persona'] = ex['persona']
                
                persona_str = format_persona(
                    ex['persona'], 
                    field_dropout_prob=mapper.field_dropout_prob, 
                    item_dropout_prob=mapper.item_dropout_prob
                )
                context_messages = [
                    {
                        "role": "user",
                        "name": mapper.get_message_usr_id(m),
                        "content": mapper.get_message_content(m),
                    } for m in ex['prompt']
                ]
                context = parse_messages(context_messages, strip_sys_prompt=False)
                response = mapper.get_response_content(ex)
                
                task = generate_thinking_trace(persona_str, context, response, args.thinking_model)
                tasks.append((cache_key, task))
            
            if tasks:
                async def generate_all_thinking():
                    batch_size = args.thinking_batch_size
                    for batch_start in range(0, len(tasks), batch_size):
                        batch = tasks[batch_start:batch_start + batch_size]
                        results = await asyncio.gather(*[t[1] for t in batch])
                        for (cache_key, _), thinking in zip(batch, results):
                            if _valid_thinking_trace(thinking):
                                thinking_cache[cache_key] = thinking
                            else:
                                failed_cache_keys.append(cache_key)
                        
                        # Save cache after each batch
                        with open(cache_file, 'w') as f:
                            json.dump(thinking_cache, f)
                        
                        print(f'Generated thinking for {min(batch_start + batch_size, len(tasks))}/{len(tasks)} examples in {thinking_split} (total cached: {len(thinking_cache)})')
                
                asyncio.run(generate_all_thinking())
            else:
                print(f'All thinking traces for {thinking_split} already cached, skipping generation')
        
        print(f'Final cache: {len(thinking_cache)} thinking traces')
        if failed_cache_keys:
            # Surface failure early so we don't write "thinking_sft" datasets with empty <think> tags.
            unique_failed = sorted(set(failed_cache_keys))
            raise RuntimeError(
                f"Failed to generate non-empty thinking traces for {len(unique_failed)} examples "
                f"(first 20 keys: {unique_failed[:20]}). "
                f"Fix LLM connectivity/credentials and rerun; cache file: {cache_file}"
            )

    # Process each split
    dataset_dict = {}
    for split in all_splits:
        persona_dropout = (split == "test_dropout")
        mapper.field_dropout_prob = args.field_dropout_prob if persona_dropout else 0.0
        mapper.item_dropout_prob = args.item_dropout_prob if persona_dropout else 0.0
        print(f'split "{split}": persona_dropout={persona_dropout}, args.field_dropout_prob={mapper.field_dropout_prob}, dropout_prob={mapper.item_dropout_prob}')

        raw = raw_datasets[split]
        
        use_thinking = args.thinking_sft and split in ['train', 'val']
        map_fn = mapper.make_map_fn_sft(split, use_thinking, args.thinking_model, thinking_cache) if args.sft else mapper.make_map_fn(split)
        features = SFT_FEATURES if args.sft else RL_FEATURES

        def map_fn_wrap(ex, idx):
            ex['metadata'] = json.loads(ex['metadata']) if isinstance(ex['metadata'], str) else ex['metadata']
            if isinstance(ex['prompt'], str):
                ex['prompt'] = json.loads(ex['prompt'])
            ex['prompt'] = [
                {
                    "role": p['role'], 
                    "content": p['content'], 
                    "metadata": json.loads(p['metadata']) if isinstance(p['metadata'], str) else p['metadata']} 
                    for p in ex['prompt']
            ]
            if isinstance(ex['persona'], str):
                try:
                    ex['persona'] = json.loads(ex['persona'])
                except json.JSONDecodeError:
                    ex['persona'] = ex['persona']
            return map_fn(ex, idx)

        mapped_ds = raw.map(
            function=map_fn_wrap,
            with_indices=True,
            remove_columns=raw.column_names,
            load_from_cache_file=False,
            num_proc=1,
            features=features,  
        )
        mapped_ds = mapped_ds.cast(features)
        dataset_dict[split] = mapped_ds

        # Write a small preview JSON (up to 10 rows)
        example_path = os.path.join(args.save_data_dir, f"{split}.example.json")
        mapped_ds.select(range(min(10, len(mapped_ds)))).to_json(example_path)
        print(f'Wrote preview to {example_path}')

        # Write the mapped parquet
        out_path = os.path.join(args.save_data_dir, f"{split}.parquet")
        mapped_ds.to_parquet(out_path)

        print(f'Wrote "{split}" with {len(mapped_ds)} rows to {out_path}')

    print(f"Processing complete! Data saved to {args.save_data_dir}")