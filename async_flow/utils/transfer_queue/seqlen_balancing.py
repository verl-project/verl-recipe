#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#


import copy
import heapq
import logging

import torch
from tensordict import TensorDict
from torch import distributed as dist

logger = logging.getLogger(__name__)


def heapq_partition(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    equal_part_num = len(seqlen_list) // k_partitions

    sorted_seqlen = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)], reverse=True)

    # Initialize the heap: each group maintains [current sum, number of elements, group index, elements in the group]
    groups = [[0, 0, i, []] for i in range(k_partitions)]
    heapq.heapify(groups)

    partitions = []
    for seqlen, i in sorted_seqlen:
        current_group = heapq.heappop(groups)
        current_group[3].append(i)
        current_group[0] += seqlen
        current_group[1] += 1
        if equal_size:
            if current_group[1] < equal_part_num:
                heapq.heappush(groups, current_group)
            else:
                partitions.append(current_group[3])
        else:
            heapq.heappush(groups, current_group)

    partitions.extend([group[3] for group in groups])

    if equal_size:
        for i, partition in enumerate(partitions):
            if len(partition) * k_partitions != len(seqlen_list):
                raise ValueError(
                    f"Partition {i} has {len(partition)} items, expected {len(seqlen_list) // k_partitions}"
                )
    return partitions


def get_seqlen_balanced_partitions(seqlen_list: list[int], k_partitions: int, equal_size: bool):
    """get order of seq lengths to make partitions balanced, this is
        used in balancing sum of seq length across dp ranks and micro batches
    Parameters:
        seqlen_list (List[int]):
            seq lengths of each items
        k_partitions (int):
            resulting number of partitions
        equal_size (bool):
            if True, number of items in each partitions must be equal.
            if False, only consider balancing the sum, each partition can have
            variable number of items
    Returns:
        partitions (List[List[int]]):
            return k_partitions list containing the index of items.
    """
    if k_partitions > len(seqlen_list):
        raise ValueError(f"number of items:[{len(seqlen_list)}] < k_partitions:[{k_partitions}]")

    def _check_and_sort_partitions(partitions):
        seen_idx = set()
        sorted_partitions = [None] * k_partitions
        for i, partition in enumerate(partitions):
            for idx in partition:
                seen_idx.add(idx)
            sorted_partitions[i] = sorted(partition)
        return sorted_partitions

    partitions = heapq_partition(seqlen_list=seqlen_list, k_partitions=k_partitions, equal_size=equal_size)
    return _check_and_sort_partitions(partitions)


def heapq_partition_with_max(seqlen_list: list[int], k_partitions: int, max_token_len: int):
    # 初始化
    sorted_seqlen = sorted([(seqlen, i) for i, seqlen in enumerate(seqlen_list)], reverse=True)

    # 初始化堆：每个组维护 [当前和, 元素数量, 组编号, 组内元素]
    groups = [[0, 0, i, []] for i in range(k_partitions)]
    group_num = len(groups)
    heapq.heapify(groups)

    partitions = []
    for seqlen, i in sorted_seqlen:
        current_group = heapq.heappop(groups)

        if current_group[0] + seqlen > max_token_len:
            partitions.append(current_group[3])
            new_group = [seqlen, 1, group_num, [i]]
            group_num = group_num + 1
            heapq.heappush(groups, new_group)
        else:
            # 将元素加入该组
            current_group[0] += seqlen  # 当前组总和增加
            current_group[1] += 1  # 当前组元素数量加1
            current_group[3].append(i)  # 当前组加入元素
            # 如果未满员，重新放回堆中
            heapq.heappush(groups, current_group)

    partitions.extend([group[3] for group in groups])
    return partitions


def rearrange_micro_batches(batch: TensorDict, padding, max_token_len, dynamic_max_batch_size=None, dp_group=None):
    """Split the batch into a list of micro_batches, where the max_token_len is smaller than max_token_len
    and the number of valid tokens in each micro batch is well balanced.
    """
    # if type of batch is dict,convert TensorDict
    if isinstance(batch, dict):
        batch_size = batch["input_ids"].shape[0]
        batch = TensorDict(batch, batch_size=[batch_size])
    # this is per local micro_bsz
    tensor_with_pad = batch["input_ids"]
    mask = tensor_with_pad != padding.to(tensor_with_pad.device)
    seq_len_effective: torch.Tensor = mask.sum(dim=1)
    max_seq_len = max(seq_len_effective)
    assert max_token_len >= max_seq_len, (
        f"max_token_len must be greater than the sequence length. Got {max_token_len=} and {max_seq_len=}"
    )

    total_seqlen = seq_len_effective.sum().item()
    num_micro_batches = (total_seqlen + max_token_len - 1) // max_token_len  # ceildiv(total_seqlen, max_token_len)
    if dynamic_max_batch_size is not None:
        num_micro_batches = max(
            num_micro_batches, (len(seq_len_effective) + dynamic_max_batch_size - 1) // dynamic_max_batch_size
        )
    if dist.is_initialized():
        num_micro_batches = torch.tensor([num_micro_batches], device="cuda")
        dist.all_reduce(num_micro_batches, op=dist.ReduceOp.MAX, group=dp_group)
        num_micro_batches = num_micro_batches.cpu().item()

    seqlen_list = seq_len_effective.tolist()
    assert num_micro_batches <= len(seqlen_list), "number of partitioning exceeding the len(seqlen_list)"

    def _check_partitions(partitions):
        for partition in partitions:
            partition_seqlen = [seq_len_effective[idx] for idx in partition]
            if sum(partition_seqlen) > max_token_len:
                return False
        return True

    partitions = heapq_partition_with_max(
        seqlen_list=seqlen_list, k_partitions=num_micro_batches, max_token_len=max_token_len
    )
    if not _check_partitions(partitions):
        logger.warning("Could not find a valid partitioning,sel length of partitioning exceeding the max_token_len")

    micro_batches = []
    for partition in partitions:
        curr_micro_batch = []
        for idx in partition:
            curr_micro_batch.append(batch[idx : idx + 1])
        curr_micro_batch = torch.cat(curr_micro_batch)
        micro_batches.append(curr_micro_batch)
    return micro_batches, partitions


def get_reverse_idx(idx_map):
    reverse_idx_map = copy.deepcopy(idx_map)

    for i, idx in enumerate(idx_map):
        reverse_idx_map[idx] = i

    return reverse_idx_map
