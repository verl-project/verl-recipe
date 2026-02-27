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

import asyncio
import copy
import importlib.util
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict, Union

import litellm
import numpy as np
import psutil
import torch
from collections import Counter
import torch.distributed as dist
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager
from verl.workers.reward_manager.registry import REWARD_MANAGER_REGISTRY
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from collections import defaultdict, Counter

METRIC_IMPORT_LOCK = threading.Lock()

async def compute_reward(
    data_source: str,
    generations: list[str],
    ground_truth: str,
    metrics: dict[str, dict[str, Any]],
    extra_info: dict[str, Any] | None = None,
    num_retries: int = 10,
) -> dict[str, list[float]]:
    """Compute rewards for generations using specified metrics."""


    _METRIC_CACHE = {}
    _METRIC_LOAD_LOCK = asyncio.Lock()  # asyncio.Lock, not threading.Lock

    async def load_metric(metric: str):
        if metric in _METRIC_CACHE:
            return _METRIC_CACHE[metric]
        
        async with _METRIC_LOAD_LOCK:  # yields to event loop while waiting
            if metric in _METRIC_CACHE:  # double-check after acquiring
                return _METRIC_CACHE[metric]
            
            metric_path = Path(__file__).parent / "metrics" / f"{metric}.py"
            if not metric_path.exists():
                raise FileNotFoundError(f"Metric not found: {metric_path}")
            
            spec = importlib.util.spec_from_file_location(metric, metric_path)
            if not spec or not spec.loader:
                raise ImportError(f"Could not load metric '{metric}'")
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[metric] = module
            spec.loader.exec_module(module)
            _METRIC_CACHE[metric] = module
        
        return _METRIC_CACHE[metric]

    async def compute_with_retry(fn, is_async: bool, metric_name: str, *args, **kwargs):
        """Execute with exponential backoff retry."""
        for attempt in range(num_retries):
            try:
                return await fn(*args, **kwargs) if is_async else fn(*args, **kwargs)
            except Exception as e:
                if attempt == num_retries - 1:
                    print(f"[Error] Final failure in metric '{metric_name}': {e}", flush=True)
                    return 0.0
                print(f"[Retry {attempt + 1}] metric='{metric_name}' err={e}", flush=True)
                if isinstance(e, litellm.RateLimitError):
                    await asyncio.sleep(2 ** attempt)


    state_name = extra_info.get("state_name") if extra_info else None
    state = extra_info.get("state") if extra_info else None
    if state_name == state[-1]:
        gd_mask = [gen.strip() == ground_truth.strip() for gen in generations]
    else:
        gd_mask = [False] * len(generations)

    # Filter empty generations
    has_content = [bool(gen.strip()) for gen in generations]
    has_content = [hc and (not is_gd) for hc, is_gd in zip(has_content, gd_mask)]
    non_empty = [g for g, keep in zip(generations, has_content) if keep]
    
    if not any(has_content):
        score_lst = [0.0 if not gd_flag else 1.0 for gd_flag in gd_mask]
        reward_dict = {}
        for metric, config in metrics.items():
            # For state_reward_on_response, get the sub-metric keys from its config_path
            # we need state_reward_on_response:score, ... for consistency
            if metric == "state_reward_on_response":
                config_path = config.get("kwargs", {}).get("config_path")
                if config_path:
                    import json
                    sub_config = json.load(open(config_path, "r"))
                    sub_keys = list(sub_config.keys())  
                    # ['stance', 'emotion', 'belief', 'value', 'goal', 'communication']
                    for sub_key in sub_keys:
                        reward_dict[f"{metric}:{sub_key}"] = score_lst.copy()
                    reward_dict[f"{metric}:metrics_info"] = [""] * len(score_lst)
                else:
                    reward_dict[f"{metric}:score"] = score_lst.copy()
                    reward_dict[f"{metric}:metrics_info"] = [""] * len(score_lst)
            else:
                reward_dict[f"{metric}:score"] = score_lst.copy()
                reward_dict[f"{metric}:metrics_info"] = [""] * len(score_lst)
        return reward_dict

    # Compute scores for each metric
    reward_dict = {}
    for metric, config in metrics.items():
        kwargs = config.get("kwargs", {})
        
        module = await load_metric(metric)
        # Use batched function if available
        if hasattr(module, "compute_batch_score"):
            fn = module.compute_batch_score
            scores_and_info = await compute_with_retry(
                fn,
                asyncio.iscoroutinefunction(fn),
                metric,
                data_source,
                non_empty,
                ground_truth,
                extra_info,
                **kwargs,
            )
            assert isinstance(scores_and_info, list), f"Expected list from compute_batch_score, got {type(scores_and_info)}"

        elif hasattr(module, "compute_score"):
            fn = module.compute_score
            is_async = asyncio.iscoroutinefunction(fn)
            scores_and_info = [
                await compute_with_retry(fn, is_async, metric, data_source, pred, ground_truth, extra_info, **kwargs)
                for pred in non_empty
            ]
        else:
            print(f"[Error] Metric '{metric}' missing compute function")
            raise ValueError(f"Metric '{metric}' missing compute function")
        
        out_keys = scores_and_info[0].keys() if isinstance(scores_and_info[0], dict) else (None,)
        for key in out_keys:
            auto_field = '' if key == 'metrics_info' else 0.0
            full_scores_and_info, idx = [], 0
            for keep in has_content:
                full_scores_and_info.append((scores_and_info[idx][key] if key is not None else scores_and_info[idx]) if keep else auto_field)
                idx += keep
                
            if not key == 'metrics_info':
                gd_score = min(1.0, max(full_scores_and_info) + 0.1)
                for i, is_gd in enumerate(gd_mask):
                    if is_gd:
                        full_scores_and_info[i] = gd_score
                
            if key is not None:
                reward_dict[f"{metric}:{key}"] = full_scores_and_info
            else: 
                reward_dict.setdefault(metric, full_scores_and_info)

    return reward_dict


STATE_TEMPLATE = "<{name}>\n{field}\n</{name}>"
HALF_STATE_TEMPLATE = "\n{field}\n</{name}>"
FIELD_PATTERN = re.compile(r"<(?P<name>\w+)>(?P<field>.*?)</\1>", re.DOTALL | re.IGNORECASE)

def parse_any_fields(text: str) -> Dict[str, str]:
    """
    Parse <field>content</field> blocks into a dictionary.
    """
    matches = FIELD_PATTERN.findall(text or "")
    return {name.lower().strip(" \n"): field.strip() for name, field in matches}
    

def parse_fields_strict(text: str, state: List[str]) -> Dict[str, str]:
    """
    Strictly parse a sequence of <field>...</field> blocks:
      - Trim only leading/trailing ' ' and '\n' from the entire input
      - Must START with <state[0]> exactly (case-insensitive)
      - Each field appears EXACTLY once, in EXACT order
      - Only ' ' and '\n' allowed between blocks
      - No extra content before first or after last block
      - PRESERVE inner content exactly (no stripping)
    Returns {} on any violation.
    """
    if text is None:
        return {}

    s = text.strip(" \n")
    if not s:
        return {}

    # Must start with the first tag exactly after trimming
    first_tag_rx = re.compile(rf"^<{re.escape(state[0])}>", re.IGNORECASE)
    if not first_tag_rx.search(s):
        return {}

    # Build a single anchored regex enforcing order and exact-once,
    # allowing only spaces/newlines between blocks.
    parts = []
    for i, field in enumerate(state):
        tag = re.escape(field)
        # Capture inner content EXACTLY as-is (no surrounding \s* in the group)
        block = rf"<{tag}>(?P<g{i}>.*?)</{tag}>"
        parts.append(block)

    # Only spaces/newlines allowed between blocks; anchor ^...$
    between = r"[ \n]*"
    pattern = r"^" + between.join(parts) + r"$"
    rx = re.compile(pattern, re.DOTALL | re.IGNORECASE)

    m = rx.match(s)
    if not m:
        return {}

    # Build result preserving inner content verbatim
    result: Dict[str, str] = {}
    for i, field in enumerate(state):
        content = m.group(f"g{i}")
        # No strip here — preserve leading/trailing whitespace inside the tag
        result[field.strip()] = "" if content is None else content

    return result
    

def parse_fields_strict_thinking(text: str, state: List[str]) -> Dict[str, str]:
    """
    Same as above but we look for "[think text] </think>" before
    """
    if text is None:
        return {}

    s = text.strip(" \n")
    if not s:
        return {}

    if 'think' not in state:
        # Must start with the first tag exactly after trimming
        first_tag_rx = re.compile(rf"^<{re.escape(state[0])}>", re.IGNORECASE)
        if not first_tag_rx.search(s):
            m_think = re.search(r"(</think>|<\\think>)", s, re.IGNORECASE)
            if not m_think:
                return {}
            s = s[m_think.end():].strip(" \n")
            if not s:
                return {}
            if not first_tag_rx.search(s):
                return {}
    
    # Build a single anchored regex enforcing order and exact-once,
    # allowing only spaces/newlines between blocks.
    parts = []
    for i, field in enumerate(state):
        tag = re.escape(field)
        # Capture inner content EXACTLY as-is (no surrounding \s* in the group)
        block = rf"<{tag}>(?P<g{i}>.*?)</{tag}>"
        parts.append(block)

    # Only spaces/newlines allowed between blocks; anchor ^...$
    between = r"[ \n]*"
    pattern = r"^" + between.join(parts) + r"$"
    rx = re.compile(pattern, re.DOTALL | re.IGNORECASE)

    m = rx.match(s)
    if not m:
        return {}

    # Build result preserving inner content verbatim
    result: Dict[str, str] = {}
    for i, field in enumerate(state):
        content = m.group(f"g{i}")
        # No strip here — preserve leading/trailing whitespace inside the tag
        result[field.strip()] = "" if content is None else content

    return result

if "humanlm" not in REWARD_MANAGER_REGISTRY:
    @register("humanlm")
    class HumanLMRewardManager(RewardManagerBase):
        state_map = {}
        _update_state_lock = threading.Lock()

        def __init__(self, config, tokenizer, compute_score=None, **kwargs):
            # Don't call super().__init__() yet - extract kwargs first
            reward_kwargs = {}
            if config is not None:
                rk = config.reward.get("reward_kwargs", {})
                if rk:
                    reward_kwargs = OmegaConf.to_container(rk, resolve=True) or {}

            self.n_rollouts = reward_kwargs.get("n_rollouts", 4)
            #key -> list w/ (data, future)
            self._pending = defaultdict(list)  
            self._lock = None 

            self.tokenizer = tokenizer
            self.compute_score = compute_score or default_compute_score
            self.eval_cache = {}

            # Store both metrics configs for dynamic switching
            self._train_metrics_cfg = reward_kwargs.get("train_metrics", {
                "response": {"state_reward": {"weight": 1.0, "kwargs": {}}}
            })
            self._val_metrics_cfg = reward_kwargs.get("val_metrics", self._train_metrics_cfg)

            self.separate_generation = reward_kwargs.get("separate_generation", False)
            self.strict_format = reward_kwargs.get("strict_format", False)
            self.enable_thinking = reward_kwargs.get("enable_thinking", False)
            self.enable_state = reward_kwargs.get("enable_state", False)
            self.fetch_global_best_state = reward_kwargs.get("fetch_global_best_state", False)
            self.eval_push_to_hub = reward_kwargs.get("eval_push_to_hub", None)
            additional_generation_prompt = reward_kwargs.get("additional_generation_prompt", "")
            self.add_additional_generation_prompt = additional_generation_prompt != ""

            state_config = reward_kwargs.get("state_config", None)
            if state_config:
                self.state_config = json.loads(open(state_config, 'r').read())
                self.state = list(self.state_config.keys())
            else:
                self.state_config = {}
                self.state = []

        def _setup_split(self, is_val: bool):
            self.split = "val" if is_val else "train"
            metrics_cfg = self._val_metrics_cfg if is_val else self._train_metrics_cfg
            
            if isinstance(metrics_cfg, dict):
                self.field_to_metrics = copy.deepcopy(metrics_cfg)
            else:
                self.field_to_metrics = OmegaConf.to_container(metrics_cfg, resolve=True)
            
            self.metric_common_kwargs = self.field_to_metrics.get('common_kwargs', {})
            self.field_to_metrics = {k: v for k, v in self.field_to_metrics.items() if k != 'common_kwargs'}

            if self.state:
                for h in self.state:
                    if h not in self.field_to_metrics and h != 'think':
                        if self.enable_state or self.split == 'train':
                            self.field_to_metrics[h] = {"state_reward": {"weight": 1.0, "kwargs": {}}}

            if self.metric_common_kwargs:
                for field in self.field_to_metrics:
                    for metric in self.field_to_metrics[field]:
                        self.field_to_metrics[field][metric]['kwargs'] = {
                            **self.metric_common_kwargs,
                            **self.field_to_metrics[field][metric].get('kwargs', {})
                        }

            self.field_metric_weights = {
                f"{field}:{metric}": weight_n_kwargs['weight']
                for field in self.field_to_metrics
                for metric, weight_n_kwargs in self.field_to_metrics[field].items()
            }
            print(f'split {self.split} | field_to_metrics {self.field_to_metrics}')

        @property
        def lock(self):
            if self._lock is None:
                self._lock = asyncio.Lock()
            return self._lock
            
        def _stable_prompt_key(self, extra_info: dict, global_step: int = 0) -> str:
            assert 'index' in extra_info, "extra_info must contain 'index' for stable key"
            return f"{global_step}:{extra_info['state_name']}:{extra_info['index']}"

        async def run_single(self, data: DataProto):
            is_val = bool(data.non_tensor_batch["is_val"][0])
            n = 1 if is_val else self.n_rollouts
            global_step = data.meta_info.get("global_steps", 0)

            key = self._stable_prompt_key(data.non_tensor_batch["extra_info"][0], global_step)
            loop = asyncio.get_running_loop()
            future = loop.create_future()

            async with self.lock:
                self._pending[key].append((data, future))
                current_count = len(self._pending[key])

            if current_count >= n:
                asyncio.create_task(self._flush_key(key))

            try:
                return await asyncio.wait_for(asyncio.shield(future), timeout=300.0)
            except asyncio.TimeoutError:
                print(f"[run_single] Timeout waiting for key={key}, flushing remaining", flush=True)
                asyncio.create_task(self._flush_key(key))
                return await future

        async def flush_remaining(self, global_step: int):
            """Flush any stranded pending samples for a given step."""
            async with self.lock:
                step_keys = [k for k in self._pending if k.startswith(f"{global_step}:")]
            
            for key in step_keys:
                async with self.lock:
                    if key not in self._pending or len(self._pending[key]) == 0:
                        continue
                await self._flush_key(key)

        async def _flush_key(self, key: str):
            async with self.lock:
                if key not in self._pending or len(self._pending[key]) == 0:
                    return
                items = self._pending.pop(key)

            datas = [item[0] for item in items]
            futures = [item[1] for item in items]

            # Build a combined DataProto from all accumulated samples
            # Stack tensors along batch dimension
            combined_batch = {}
            for k in datas[0].batch.keys():
                combined_batch[k] = torch.cat([d.batch[k] for d in datas], dim=0)
            
            combined_non_tensor = {}
            for k in datas[0].non_tensor_batch.keys():
                combined_non_tensor[k] = np.concatenate([d.non_tensor_batch[k] for d in datas], axis=0)

            from tensordict import TensorDict
            combined_data = DataProto(
                batch=TensorDict(combined_batch, batch_size=[len(datas)]),
                non_tensor_batch=combined_non_tensor,
                meta_info=datas[0].meta_info,
            )

            is_val = bool(datas[0].non_tensor_batch["is_val"][0])
            self._setup_split(is_val)

            result = await self._compute_rewards_async(combined_data, return_dict=True)

            reward_tensor = result["reward_tensor"]  # shape: (n, response_length)
            reward_extra_info = result.get("reward_extra_info", {})
            reward_dicts = result.get("reward_dicts", [{} for _ in datas])

            # Resolve each future with its individual score
            for i, future in enumerate(futures):
                if future.done():
                    continue

                score = reward_tensor[i].sum().item()

                full_extra_info = {}
                for field in self.field_to_metrics:
                    for metric in self.field_to_metrics[field]:
                        full_extra_info[f"{self.split}/{field}:{metric}:score"] = 0.0
                for h in self.state:
                    if self.separate_generation:
                        full_extra_info[f"{self.split}/valid_rate/{h}"] = 0.0
                    else:
                        full_extra_info[f"{self.split}/valid_rate/all"] = 0.0
                    full_extra_info[f"{self.split}/sample_size_in_batch/{h}"] = 0

                full_extra_info.update(reward_extra_info)

                future.set_result({
                    "reward_score": score,
                    "reward_extra_info": full_extra_info,
                })
                
        def __call__(self, data: DataProto, return_dict: bool = False):
            is_val = data.meta_info.get("validate", False) if hasattr(data, 'meta_info') else False
            self._setup_split(is_val)
            return asyncio.run(self._compute_rewards_async(data, return_dict))
            

        def compute_batch_score_sync(self, *args, **kwargs):
            loop = asyncio.new_event_loop()
            try:
                return loop.run_until_complete(compute_reward(*args, **kwargs))
            finally:
                loop.close()

        
        async def _compute_rewards_async(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
            prompt_ids = data.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
            
            data_source = data.non_tensor_batch["data_source"]
            batch_size = len(data_source)

            extra_info = data.non_tensor_batch["extra_info"]
            if len(self.state_config) == 1:
                extra_info = [
                    {**info, "state_name": "response", "state_desc": self.state_config["response"]["desc"]}
                    for info in extra_info
                ]
            else:
                extra_info = [
                    {**info, "state_desc": self.state_config[info["state_name"]]["desc"]}
                    for info in extra_info
                ]
            ground_truth = [item["ground_truth"] for item in data.non_tensor_batch["reward_model"]]

            global_step = data.meta_info.get("global_steps")
            keys = [self._stable_prompt_key(info, global_step) for info in extra_info]

            state_names = [item["state_name"] for item in extra_info]
            
            #########################################################
            ###################### Get prompts ######################
            attention_mask = data.batch["attention_mask"][:, :prompt_ids.shape[-1]]
            prompt_ids_no_pad = [
                ids[mask.bool()].tolist() for ids, mask in zip(prompt_ids, attention_mask)
            ]
            prompts = self.tokenizer.batch_decode(prompt_ids_no_pad, skip_special_tokens=False)

            #########################################################
            ################ Parse generation fields ################
            raw_generations = self.tokenizer.batch_decode(
                data.batch["responses"],
                skip_special_tokens=True,
            )

            print(f"| raw_generation[0] {repr(raw_generations[0])}", flush=True)

            # add the additional generate tokens
            field_to_metrics_lst = []
            generations, generation_fields = [], []
            valid_mask, valid_mask_by_state = [], {}
            should_have_field_masks = {h: [] for h in self.state}

            index = extra_info[0].get("index", -1) if extra_info else -1
            for i, (g, h) in enumerate(zip(raw_generations, state_names)):
                if self.add_additional_generation_prompt:
                    additional_generation_prompt = f"<{h}>"
                else:
                    additional_generation_prompt = ""
                generations.append(additional_generation_prompt + g)
                
                if self.separate_generation and self.split == 'train':
                    parse_state = [h]
                else:
                    cur_state_idx = self.state.index(h)
                    parse_state = self.state[cur_state_idx:]
                
                # remove 'think' from parse_state if exists
                if 'think' in parse_state:
                    parse_state.remove('think')

                if self.strict_format:
                    if self.enable_thinking:
                        generation_field = parse_fields_strict_thinking(g, parse_state)
                    else:
                        generation_field = parse_fields_strict(g, parse_state)
                else:
                    generation_field = parse_any_fields(g)
                
                if self.split == 'val':
                    if generation_field.get(h, "").strip() == '':
                        generation_field[h] = g.split('</think>')[-1].strip()
                        if (clean_field := generation_field[h].split(f'<{h}>')[-1].strip()) != '':
                            generation_field[h] = clean_field

                if self.separate_generation:
                    field_to_metrics = {h: self.field_to_metrics.get(h)}
                else:
                    field_to_metrics = {h: self.field_to_metrics[h] for h in parse_state}

                for _h in self.state:
                    if self.separate_generation:
                        should_have_field_masks[_h].append(h == _h)
                    else:
                        should_have_field_masks[_h].append(True if _h in parse_state else False)

                field_to_metrics_lst.append(field_to_metrics)
                field_key = list(set(generation_field.keys()).intersection(set(field_to_metrics.keys())))
                
                is_valid = len(field_key) == len(field_to_metrics)
                valid_mask.append(float(is_valid))

                if self.strict_format:
                    generation_fields.append(generation_field if is_valid else {})
                else:
                    generation_fields.append(generation_field)
                
                valid_mask_by_state.setdefault(h, [])
                valid_mask_by_state[h].append(float(is_valid))
                
            counts = Counter(state_names)
            valid_mask = torch.tensor(valid_mask, dtype=torch.bool)
            should_have_field_masks = {field: torch.tensor(should_have_field_masks[field], dtype=torch.bool) for field in should_have_field_masks}
            print(
                f"\n| state_name {counts}\n| generation {generations[0]}\n| generation_fields {generation_fields[0]}\n"
                f"| ground_truth {ground_truth[0]} | all_valid_rate {valid_mask.float().mean().item()}\n| prompts {prompts[0]}",
                flush=True,
            )

            unique_keys = set(keys)
            generation_fields_by_keys = {
                key: [generation_fields[i] for i in range(batch_size) if keys[i] == key]
                for key in unique_keys
            }
            field_to_metrics_by_keys = {
                key: field_to_metrics_lst[i] for i, key in enumerate(keys)
            }

            # Map keys to their data and validate consistency
            key_to_data = {}
            for i, key in enumerate(keys):
                data_info = {
                    'data_source': data_source[i],
                    'ground_truth': ground_truth[i],
                    'extra_info': extra_info[i]
                }
                if key in key_to_data:
                    assert key_to_data[key] == data_info, \
                        "Data source, ground truth, and extra_info must be consistent for the same key"
                else:
                    key_to_data[key] = data_info

            # Create async scoring tasks with deduplication
            tasks = []
            task_metadata = []  # Track (field, key, original_indices, unique_values) for each task
            
            #########################################################
            #################### Compute rewards ####################
            loop = asyncio.get_running_loop()
            for key in generation_fields_by_keys:
                field_to_metrics = field_to_metrics_by_keys[key]

                for field in field_to_metrics:
                    metrics_to_run = {
                        m: v for m, v in field_to_metrics[field].items() 
                    }
                    if not metrics_to_run:
                        continue
                    field_values = [gen_fields.get(field, '').strip() for gen_fields in generation_fields_by_keys[key]]
                    
                    # Deduplicate while preserving mapping
                    unique_values = []
                    value_to_unique_idx = {}
                    original_to_unique_indices = []
                    
                    for val in field_values:
                        if val not in value_to_unique_idx:
                            value_to_unique_idx[val] = len(unique_values)
                            unique_values.append(val)
                        original_to_unique_indices.append(value_to_unique_idx[val])
                    
                    key_extra_info = copy.deepcopy(key_to_data[key]['extra_info'])
                    key_extra_info.update({"state": self.state})
                    key_extra_info.update({"state_name": field})
                    key_extra_info.update({"state_desc": self.state_config[field]['desc']})

                    task = compute_reward(
                        key_to_data[key]["data_source"],
                        unique_values,
                        key_to_data[key]["ground_truth"],
                        field_to_metrics[field],
                        key_extra_info,
                    )

                    tasks.append(task)
                    task_metadata.append((field, key, original_to_unique_indices))

            # Execute tasks and reorganize scores
            print(f"[compute_reward] about to gather {len(tasks)} tasks", flush=True)
            gathered_scores = await asyncio.gather(*tasks)
            print(f"[compute_reward] gather complete", flush=True)

            reward_dicts = [{} for _ in range(batch_size)]
            for task_idx, (field, key, original_to_unique_indices) in enumerate(task_metadata):
                # gathered_scores[task_idx] is a dict: {metric_name: [score1, score2, ...]}
                # where the list has one score per unique value
                metrics_dict = gathered_scores[task_idx]
                
                # Map scores back to original order for this key
                key_indices = [i for i in range(batch_size) if keys[i] == key]
                
                for local_idx, global_idx in enumerate(key_indices):
                    unique_idx = original_to_unique_indices[local_idx]
                    
                    # Initialize field dict if not exists
                    if field not in reward_dicts[global_idx]:
                        reward_dicts[global_idx][field] = {}
                    
                    # Store scores for each metric under this field
                    for metric_name, scores_list in metrics_dict.items():
                        reward_dicts[global_idx][field][metric_name] = scores_list[unique_idx]
            
            # Update eval cache 
            if self.split == "val" and self.eval_push_to_hub:
                for i, (key, prompt, gen, raw_gen, gt, extra) in enumerate(
                    zip(keys, prompts, generation_fields, raw_generations, ground_truth, extra_info)
                ):
                    reward_dict = reward_dicts[i]
                    self.eval_cache[key] = {
                        "prompt": prompt,
                        "raw_generation": raw_gen,
                        "response": gen.get('response', ''),
                        "ground_truth": gt,
                        "metrics": json.dumps({
                            field: {k: v for k, v in m.items() if not k.endswith("metrics_info")}
                            for field, m in reward_dict.items()
                        }),
                        "metrics_info": json.dumps({
                            field: {k: v for k, v in m.items() if k.endswith("metrics_info")}
                            for field, m in reward_dict.items()
                        }),
                        "extra_info": json.dumps(extra),
                    }
            
            # Now reward_dicts[i][field][metric] gives the score for batch_i, field, metric
            # Aggregate scores for each field:metric combination
            if self.separate_generation:
                # Aggregate scores - each sample only has scores for its own state
                scores_by_fm = {}
                scores = torch.zeros(batch_size)

                for i in range(batch_size):
                    h = state_names[i]  # This sample's target state
                    reward_dict = reward_dicts[i]
                    
                    # No valid parse
                    if h not in reward_dict:
                        raise ValueError("[Error] State name not in reward dict")
                    
                    sample_score = 0.0
                    for metric_key, value in reward_dict[h].items():
                        if metric_key.endswith('metrics_info'):
                            continue
                        
                        fm_key = f"{h}:{metric_key}"
                        # state_reward from state_reward:score
                        base_metric = metric_key.split(':')[0]  
                        weight_key = f"{h}:{base_metric}"
                        weight = self.field_metric_weights.get(weight_key, 1.0)
                        
                        weighted_value = value * weight
                        sample_score += weighted_value
                        
                        # Track for logging
                        scores_by_fm.setdefault(fm_key, torch.zeros(batch_size))
                        scores_by_fm[fm_key][i] = weighted_value
                    
                    scores[i] = sample_score

                weighted_scores_by_fm = scores_by_fm
            else:
                scores_by_fm = {}
                try:
                    for field in self.field_to_metrics:
                        for metric in self.field_to_metrics[field]:
                            sub_metrics = set()
                            for reward_dict in reward_dicts:
                                field_data = reward_dict.get(field, {})
                                for key in field_data.keys():
                                    if key.startswith(metric + ":") and not key.endswith('metrics_info'):
                                        sub_metrics.add(key)

                            for sub_metric in sorted(sub_metrics):
                                fm_key = f"{field}:{sub_metric}"
                                # ERROR field may not be in reward_dicts
                                scores_by_fm[fm_key] = torch.tensor([
                                    reward_dicts[i][field][sub_metric]
                                    for i in range(batch_size)
                                ])
                            self.field_metric_weights.update({
                                f"{field}:{sub_metric}": self.field_metric_weights[f"{field}:{metric}"]
                                for sub_metric in sub_metrics
                            })
                except: 
                    raise ValueError("[Error] field not found in reward output")

                # Apply field-metric specific weights
                weighted_scores_by_fm = {
                    fm: scores_by_fm[fm] * self.field_metric_weights[fm]
                    for fm in scores_by_fm
                }

            # Combine weighted scores from all field and metrics into a single tensor
            if scores_by_fm:
                scores = torch.stack(
                    [weighted_scores_by_fm[fm] for fm in scores_by_fm]
                ).sum(dim=0)
            else:
                scores = torch.zeros(batch_size)

            # Compute mean of weighted scores for each metric
            if scores_by_fm:
                log_weighted_scores_by_field_metric = {
                    f"{self.split}/{fm}": weighted_scores_by_fm[fm][should_have_field_masks[fm.split(":")[0]]].mean(dim=0).item()
                    for fm in scores_by_fm
                }
            else:
                log_weighted_scores_by_field_metric = {}

            # constrain the minimum score to be a very small number, so we can use it for tracking in grpo prm 
            reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
            for i in range(len(data)):
                reward_tensor[i, valid_response_length[i].item() - 1] = scores[i]
            
            log_weighted_scores_by_field_metric.update(
                {
                    **{
                        f"{self.split}/valid_rate/{h if self.separate_generation else 'all'}": torch.tensor(valid_rate).mean().item() for h, valid_rate in valid_mask_by_state.items()
                    },
                    **{
                        f"{self.split}/sample_size_in_batch/{h}": counts[h] for h in valid_mask_by_state.keys()
                    }

                }
            )

            try:
                import wandb
                if wandb.run is None:
                    # Try to resume the existing run
                    run_id = os.environ.get("WANDB_RUN_ID")
                    project = os.environ.get("WANDB_PROJECT", "humanlm")
                    if run_id:
                        wandb.init(project=project, id=run_id, resume="allow", reinit=True)
                
                if wandb.run is not None:
                    global_step = data.meta_info.get("global_steps", 0)
                    wandb.log(log_weighted_scores_by_field_metric, step=global_step, commit=False)
            except Exception as e:
                print(f"[wandb log failed] {e}", flush=True)
            
            print(f"[DEBUG METRICS] log_weighted_scores_by_field_metric: {log_weighted_scores_by_field_metric}", flush=True)

            if return_dict:
                return {
                    "reward_tensor": reward_tensor, 
                    "reward_extra_info": {**log_weighted_scores_by_field_metric},
                    "reward_dicts": reward_dicts
                }
            else:
                return reward_tensor, {**log_weighted_scores_by_field_metric}
else:
    HumanLMRewardManager = REWARD_MANAGER_REGISTRY["humanlm"]