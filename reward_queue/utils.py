# Copyright 2026 Huawei Technologies Co., Ltd. and/or its affiliates
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
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from verl import DataProto
from verl.experimental.fully_async_policy.detach_utils import RolloutSample
from verl.trainer.ppo.ray_trainer import compute_response_mask


@dataclass
class SubRewardDataItem:
    reward_input: DataProto
    padded_output: DataProto
    sample_id: str
    epoch: int
    sub_index: int
    total_count: int
    inference_start_timestamp: float
    inference_end_timestamp: float
    generate_data: Any = None
    enqueue_data: Any = None


@dataclass
class _ScoredSubItem:
    sub_index: int
    padded_output: DataProto
    score: float
    reward_extra_info: dict
    reward_start: float
    reward_end: float
    reward_compute_time: float
    inference_start: float
    inference_end: float


@dataclass
class _AggregationGroup:
    total_count: int
    epoch: int
    items: dict[int, _ScoredSubItem] = field(default_factory=dict)

    def add(self, item: _ScoredSubItem):
        self.items[item.sub_index] = item

    @property
    def is_complete(self) -> bool:
        return len(self.items) == self.total_count


class SampleAggregator:
    def __init__(self):
        self._groups: dict[str, _AggregationGroup] = {}

    def add_scored_item(
        self,
        sample_id: str,
        total_count: int,
        epoch: int,
        scored_item: _ScoredSubItem,
    ) -> bool:
        if sample_id not in self._groups:
            self._groups[sample_id] = _AggregationGroup(total_count=total_count, epoch=epoch)
        group = self._groups[sample_id]
        group.add(scored_item)
        return group.is_complete

    def get_and_remove(self, sample_id: str) -> _AggregationGroup:
        return self._groups.pop(sample_id)

    @property
    def total_pending(self) -> int:
        return sum(len(g.items) for g in self._groups.values())

    @property
    def pending_sample_ids(self) -> list[str]:
        return list(self._groups.keys())

    @property
    def pending_groups_count(self) -> int:
        return len(self._groups)


def addition_process(output: DataProto, enable_reward_queue: bool = False):
    """collect metirics"""
    metrics = output.meta_info.pop("metrics")  # List[Dict[str, str]]
    if enable_reward_queue:
        return _addition_process_with_reward_queue(output, metrics)

    processing_times_list = [item["generate_sequences"] for item in metrics]
    tool_calls_times_list = [item["tool_calls"] for item in metrics]

    # Collect reward_compute_time if available
    reward_compute_times_list = []
    for item in metrics:
        if "reward_compute_time" in item:
            reward_compute_times_list.append(item["reward_compute_time"])

    output.non_tensor_batch["processing_times"] = processing_times_list
    output.non_tensor_batch["tool_calls_times"] = tool_calls_times_list

    # Store reward_compute_time if any sample has it
    if reward_compute_times_list:
        output.non_tensor_batch["reward_compute_times"] = reward_compute_times_list

    return output


def _addition_process_with_reward_queue(output, metrics):
    metrics = _normalize_metrics(metrics)
    processing_times_list = [
        item.generate_sequences if hasattr(item, "generate_sequences") else item.get("generate_sequences", 0.0)
        for item in metrics
    ]
    tool_calls_times_list = [
        item.tool_calls if hasattr(item, "tool_calls") else item.get("tool_calls", 0.0) for item in metrics
    ]
    reward_compute_times_list = []
    for item in metrics:
        rct = item.reward_compute_time if hasattr(item, "reward_compute_time") else item.get("reward_compute_time")
        if rct is not None:
            reward_compute_times_list.append(rct)
    output.non_tensor_batch["processing_times"] = np.array(processing_times_list, dtype=np.float64)
    output.non_tensor_batch["tool_calls_times"] = np.array(tool_calls_times_list, dtype=np.float64)
    if reward_compute_times_list:
        output.non_tensor_batch["reward_compute_times"] = reward_compute_times_list

    _flatten_non_tensor_batch(output)
    return output


def _normalize_metrics(metrics):
    if not metrics:
        return []
    if isinstance(metrics, dict):
        keys = list(metrics.keys())
        n = len(metrics[keys[0]]) if keys else 0
        return [{k: metrics[k][i] for k in keys} for i in range(n)]
    if isinstance(metrics, list):
        flat = []
        for item in metrics:
            if isinstance(item, list):
                flat.extend(item)
            else:
                flat.append(item)
        return flat
    return [metrics]


def _flatten_non_tensor_batch(data_proto: DataProto) -> None:
    batch_size = len(data_proto)
    for key in list(data_proto.non_tensor_batch.keys()):
        arr = data_proto.non_tensor_batch[key]
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr, dtype=object)
        if arr.dtype == np.object_ and arr.ndim == 1:
            continue
        new_arr = np.empty(batch_size, dtype=object)
        if arr.ndim == 1:
            new_arr[:] = list(arr)
        else:
            for i in range(batch_size):
                new_arr[i] = arr[i]
        data_proto.non_tensor_batch[key] = new_arr


def assemble_batch_from_rollout_samples(
    rollout_samples: list[RolloutSample], tokenizer, config, balance_batch=None, enable_reward_queue: bool = False
) -> DataProto:
    """
    Assemble gen_batch_output from RolloutSample objects
    Assembles batches from RolloutSample objects, similar to the _post_generate_batch logic in ray_trainer.

    Args:
        rollout_samples: List of RolloutSample objects
        tokenizer: Tokenizer instance
        config: Configuration object containing trainer settings
        balance_batch: Whether to balance the batch (simplified version)

    Returns:
        DataProto: Assembled gen_batch_output

    Raises:
        ValueError: If rollout_samples is empty
    """
    start_time = time.time()

    if not rollout_samples:
        raise ValueError("Empty rollout_samples provided for batch assembly")

    print(f"[BatchUtils] Assembling batch from {len(rollout_samples)} RolloutSample objects")

    rollout_samples_batch = []
    rollout_status = rollout_samples[0].rollout_status
    # Add a prefix to all rollout_status keys
    rollout_status = {f"fully_async/{key}": value for key, value in rollout_status.items()}

    for rs in rollout_samples:
        batch = addition_process(rs.full_batch, enable_reward_queue=enable_reward_queue)
        rollout_samples_batch.append(batch)
    final_batch = DataProto.concat(rollout_samples_batch)

    # Calculate response_mask (if not present)
    if "response_mask" not in final_batch.batch.keys():
        final_batch.batch["response_mask"] = compute_response_mask(final_batch)

    if balance_batch:
        balance_batch(final_batch, metrics={})

    # Calculate the global valid token number
    if "attention_mask" in final_batch.batch:
        final_batch.meta_info["global_token_num"] = torch.sum(final_batch.batch["attention_mask"], dim=-1).tolist()

    processing_times = final_batch.non_tensor_batch["processing_times"]
    tool_calls = final_batch.non_tensor_batch["tool_calls_times"]
    # Collect statistics
    processing_time_stats = {
        "processing_time/avg": np.mean(processing_times),
        "processing_time/max": np.max(processing_times),
        "processing_time/min": np.min(processing_times),
        "processing_time/tp50": np.percentile(processing_times, 50),
        "processing_time/tp99": np.percentile(processing_times, 99),
        "processing_time/tp95": np.percentile(processing_times, 95),
    }
    tool_calls_stats = {}
    if len(tool_calls) > 0:
        tool_calls_stats = {
            "timing_s/agent_loop/tool_calls/max": np.max(tool_calls),
            "timing_s/agent_loop/tool_calls/min": np.min(tool_calls),
            "timing_s/agent_loop/tool_calls/mean": np.mean(tool_calls),
        }

    # Collect reward_compute_time statistics if available
    reward_compute_stats = {}
    if "reward_compute_times" in final_batch.non_tensor_batch:
        reward_compute_times = final_batch.non_tensor_batch["reward_compute_times"]
        if len(reward_compute_times) > 0:
            reward_compute_stats = {
                "timing_s/reward_compute/max": np.max(reward_compute_times),
                "timing_s/reward_compute/min": np.min(reward_compute_times),
                "timing_s/reward_compute/mean": np.mean(reward_compute_times),
                "timing_s/reward_compute/tp50": np.percentile(reward_compute_times, 50),
                "timing_s/reward_compute/tp95": np.percentile(reward_compute_times, 95),
            }
    processing_time_stats = {f"fully_async/{key}": value for key, value in processing_time_stats.items()}

    param_version_start = final_batch.non_tensor_batch["min_global_steps"]
    param_version_end = final_batch.non_tensor_batch["max_global_steps"]
    param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start, strict=False)]
    num_diff0 = param_version_diff.count(0)
    partial_stats = {
        "fully_async/partial/total_partial_num": len(param_version_diff) - num_diff0,
        "fully_async/partial/partial_ratio": (len(param_version_diff) - num_diff0) / len(param_version_diff),
        "fully_async/partial/max_partial_span": max(param_version_diff),
    }
    # add meta_info
    trajectory_param_versions = final_batch.non_tensor_batch["max_global_steps"]

    final_batch.meta_info.update(
        {
            "param_version_diversity": len(set(trajectory_param_versions)),
            "trajectory_param_versions": trajectory_param_versions,
            **processing_time_stats,
            **rollout_status,
            **partial_stats,
            **tool_calls_stats,
            **reward_compute_stats,
        }
    )

    print(f"[BatchUtils] Batch assembly completed in {time.time() - start_time:.2f}s")

    return final_batch
