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
import pickle
import socket

import pytest
import ray
import torch
from recipe.async_flow.utils.transfer_queue.tq_client import get_transferqueue_client
from recipe.async_flow.utils.transfer_queue.tq_mgr import TransferQueueManager
from recipe.async_flow.utils.transfer_queue.tq_sampler import (
    RandomSampler,
    VersionSampler,
)
from torch.nn.utils.rnn import pad_sequence

NUMS_TQ_DATA = 1

# --------------------- Helpers ---------------------


def _find_free_port() -> int:
    """Find a free TCP port on localhost for ZMQ binding."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _make_padded_prompts(lengths, pad_id=0) -> torch.Tensor:
    """Create a right-padded 2D tensor of token ids with the given lengths."""
    seqs = [torch.arange(1, L + 1, dtype=torch.long) for L in lengths]
    return pad_sequence(seqs, batch_first=True, padding_value=pad_id)


class CustomObject:
    """Custom class for testing pickle encoding"""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


# --------------------- Pytest fixtures ---------------------


@pytest.fixture(scope="module", autouse=True)
def setup_transferqueue_manager():
    """
    Start TransferQueueManager with 1 shard and a free base port.
    """
    base_port = _find_free_port()

    mgr = TransferQueueManager.options(num_cpus=6).remote(
        nums_tq_data=NUMS_TQ_DATA,
        base_port=base_port,
    )
    ready = ray.get(mgr.init_ready.remote())
    assert ready is True
    yield mgr

    try:
        ray.get(mgr.shutdown.remote())
    except Exception:
        pass
    try:
        ray.kill(mgr)
    except Exception:
        pass
    if ray.is_initialized():
        ray.shutdown()


@pytest.fixture()
def topic_basic():
    """Create a basic topic for testing"""
    tq_client = get_transferqueue_client()

    topic = "NEW_TEST_TOPIC"
    prompts_num = 3
    n_samples = 2
    experience_columns = ["prompt", "prompt_length", "reward", "response", "response_length"]
    consumers = ["learner", "evaluator"]

    tq_client.add_topic(
        prompts_num=prompts_num,
        n_samples_per_prompt=n_samples,
        experience_columns=experience_columns,
        experience_consumers=consumers,
        topic=topic,
    )

    yield {
        "topic": topic,
        "prompts_num": prompts_num,
        "n_samples": n_samples,
        "columns": experience_columns,
        "consumers": consumers,
    }

    tq_client.delete_topic(topic)


# --------------------- Tests ---------------------


@pytest.mark.xdist_group(name="new_test")
def test_put_experience_with_version(topic_basic):
    """Test put_experience with version parameter"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]
    prompts_num = topic_basic["prompts_num"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    max_len = prompts_num * n_samples
    allocated_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        version=1,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=6,
        indexes=None,
        get_n_samples=False,
        topic=topic,
    )

    assert allocated_indexes == list(range(max_len))
    assert data is not None and idxs is not None
    assert len(idxs) == max_len


@pytest.mark.xdist_group(name="new_test")
def test_put_experience_auto_allocate_indexes(topic_basic):
    """Test put_experience with automatic index allocation (indexes=None)"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # Auto-allocate indexes (indexes=None)
    allocated_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        indexes=None,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    assert isinstance(allocated_indexes, list)
    assert len(allocated_indexes) == 6


@pytest.mark.xdist_group(name="new_test")
def test_get_experience_with_copy_parameter(topic_basic):
    """Test get_experience with copy parameter (zero-copy vs deep copy)"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]
    prompts_num = topic_basic["prompts_num"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    max_len = prompts_num * n_samples
    allocated_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Test with copy=False (zero-copy)
    data_no_copy, idxs_no_copy = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=n_samples,
        indexes=None,
        get_n_samples=True,
        topic=topic,
        copy=False,
    )

    # Test with copy=True (deep copy)
    data_with_copy, idxs_with_copy = tq_client.get_experience(
        consumer="evaluator",
        experience_columns=["prompt"],
        experience_count=n_samples,
        indexes=None,
        get_n_samples=True,
        topic=topic,
        copy=True,
    )

    assert allocated_indexes == list(range(max_len))
    assert data_no_copy is not None
    assert data_with_copy is not None
    assert len(idxs_no_copy) == n_samples
    assert len(idxs_with_copy) == n_samples


@pytest.mark.xdist_group(name="new_test")
def test_get_data_ready_set(topic_basic):
    """Test get_data_ready_set interface"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # Before putting data
    count, idxs = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt"],
    )
    assert count == 0

    # After putting prompts
    tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        # indexes=list(range(prompts_num * n_samples)),
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    count, idxs = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt"],
    )
    assert count == 6  # 3 prompts * 2 samples
    assert len(idxs) == 6


@pytest.mark.xdist_group(name="new_test")
def test_get_data_ready_set_with_get_n_samples(topic_basic):
    """Test get_data_ready_set with get_n_samples=True"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]
    prompts_num = topic_basic["prompts_num"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        # indexes=list(range(prompts_num * n_samples)),
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # get_n_samples=False - return all ready indexes
    count, idxs = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt"],
        get_n_samples=False,
    )
    assert count == 6
    assert len(idxs) == 6

    # get_n_samples=True - return complete groups only
    count, idxs = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt"],
        get_n_samples=True,
    )
    assert count == 6
    # All groups should be complete
    for i in range(prompts_num):
        for j in range(n_samples):
            assert (i * n_samples + j) in idxs


@pytest.mark.xdist_group(name="new_test")
def test_get_data_consumed_set(topic_basic):
    """Test get_data_consumed_set interface"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        # indexes=list(range(prompts_num * n_samples)),
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Before consuming
    count, idxs = tq_client.get_data_consumed_set(
        topic=topic,
        consumer="learner",
    )
    assert count == 0

    # Consume some data
    data, consumed_idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=n_samples,
        indexes=None,
        get_n_samples=True,
        topic=topic,
    )

    count, idxs = tq_client.get_data_consumed_set(
        topic=topic,
        consumer="learner",
    )
    assert count == n_samples
    assert set(idxs) == set(consumed_idxs)


@pytest.mark.xdist_group(name="new_test")
def test_get_data_usable_set(topic_basic):
    """Test get_data_usable_set interface"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Get usable set
    count, idxs = tq_client.get_data_usable_set(
        topic=topic,
        consumer="learner",
        experience_columns=["prompt"],
    )
    assert count == 6

    # After consuming, usable set should decrease
    data, _ = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=n_samples,
        indexes=None,
        get_n_samples=True,
        topic=topic,
    )

    count, idxs = tq_client.get_data_usable_set(
        topic=topic,
        consumer="learner",
        experience_columns=["prompt"],
    )
    assert count == 6 - n_samples


@pytest.mark.xdist_group(name="new_test")
def test_delete_experience_by_indexes(topic_basic):
    """Test delete_experience with indexes parameter"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    allocated_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Delete first 3 allocated indexes
    delete_indexes = allocated_indexes[:3]
    tq_client.delete_experience(
        indexes=delete_indexes,
        topic=topic,
    )

    # Check that deleted indexes are no longer available
    count, idxs = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt"],
    )
    assert delete_indexes[0] not in idxs
    assert delete_indexes[1] not in idxs
    assert delete_indexes[2] not in idxs
    assert count == 3  # Should have 3 remaining


@pytest.mark.xdist_group(name="new_test")
def test_override_indexes_and_delete_experience_by_versions(topic_basic):
    """Test delete_experience with versions parameter"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    # 使用 response 和 response_length（非共享列）
    responses = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_responses = responses.repeat_interleave(n_samples, dim=0)
    response_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # First put with version 1 (auto-allocate indexes)
    allocated_indexes = tq_client.put_experience(
        data_dict={"response": padded_responses, "response_length": response_lengths},
        version=1,
        unpad_pairs=[("response", "response_length", "right_pad")],
        topic=topic,
    )

    # Override first 2 indexes with version 2
    override_indexes = allocated_indexes[:2]
    tq_client.put_experience(
        data_dict={"response": padded_responses[:2], "response_length": response_lengths[:2]},
        indexes=override_indexes,
        version=2,
        unpad_pairs=[("response", "response_length", "right_pad")],
        topic=topic,
    )

    # Delete by version 2
    tq_client.delete_experience(
        versions=[2],
        topic=topic,
    )

    # Indexes deleted by version should be unavailable (indexes 0,1 should be gone)
    # Verify that indexes 0,1 are deleted and indexes 2-5 still exist
    remaining_count, remaining_indexes = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["response", "response_length"],
    )
    assert remaining_count == 4  # 应该剩下 4 个索引 (2,3,4,5)
    assert remaining_indexes == [2, 3, 4, 5]


@pytest.mark.xdist_group(name="new_test")
def test_delete_experience_by_staleness(topic_basic):
    """Test delete_experience with latest_version and allowed_staleness"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # Put all data at once (auto-allocate indexes)
    allocated_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Set different versions for different groups via manager
    tq_mgr = ray.get_actor("TransferQueueManager")
    ray.get(tq_mgr.record_versions.remote(topic, 0, allocated_indexes[0:n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 1, allocated_indexes[n_samples : 2 * n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 2, allocated_indexes[2 * n_samples :]))

    # Delete data older than allowed_staleness
    tq_client.delete_experience(
        latest_version=2,
        allowed_staleness=1,
        topic=topic,
    )


@pytest.mark.xdist_group(name="new_test")
def test_delete_experience_delete_all(topic_basic):
    """Test delete_experience with delete_all=True"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Verify data exists
    count, _ = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt"],
    )
    assert count > 0

    # Delete all data
    tq_client.delete_experience(
        delete_all=True,
        topic=topic,
    )

    # Verify all data is cleared
    count, _ = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt"],
    )
    assert count == 0


@pytest.mark.xdist_group(name="new_test")
def test_put_experience_with_needs_expansion(topic_basic):
    """Test put_experience with needs_expansion=True"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)

    # Put with needs_expansion=True (data not replicated, will allocate expanded indexes)
    allocated = tq_client.put_experience(
        data_dict={"prompt": prompts},
        indexes=None,
        needs_expansion=True,
        topic=topic,
    )

    # Should have allocated expanded indexes (3 prompts * 2 samples = 6 indexes)
    assert isinstance(allocated, list)
    assert len(allocated) == 6


@pytest.mark.xdist_group(name="new_test")
def test_get_allocation_for_new_groups(topic_basic):
    """Test get_allocation_for_new_groups interface"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]

    # Allocate groups before putting data
    allocation = tq_client.get_allocation_for_new_groups(topic=topic, num_new_groups=2)

    assert isinstance(allocation, list)
    assert len(allocation) == NUMS_TQ_DATA

    # Check that we got group IDs allocated
    total_groups = sum(len(groups) for groups in allocation)
    assert total_groups == 2


@pytest.mark.xdist_group(name="new_test")
def test_get_experience_without_group_constraint(topic_basic):
    """Test get_experience with get_n_samples=False (sequential sampling, no group constraint)"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Sequential sampling without group constraint
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt", "prompt_length"],
        experience_count=3,
        indexes=None,
        get_n_samples=False,
        topic=topic,
    )

    assert data is not None
    assert len(idxs) == 3
    assert len(data["prompt"]) == 3


@pytest.mark.xdist_group(name="new_test")
def test_get_experience_with_allowed_staleness(topic_basic):
    """Test get_experience with staleness filtering"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    allocated_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Set versions via manager using allocated indexes
    tq_mgr = ray.get_actor("TransferQueueManager")
    ray.get(tq_mgr.record_versions.remote(topic, 0, allocated_indexes[0:n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 1, allocated_indexes[n_samples : 2 * n_samples]))
    ray.get(tq_mgr.record_versions.remote(topic, 2, allocated_indexes[2 * n_samples :]))

    # Get only recent data
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=2,
        indexes=None,
        get_n_samples=False,
        allowed_staleness=1,
        latest_version=2,
        topic=topic,
    )

    assert data is not None


@pytest.mark.xdist_group(name="new_test")
def test_random_sampler(topic_basic):
    """Test get_experience with RandomSampler"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    allocated = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    assert allocated == list(range(len(padded_prompts)))

    sampler = RandomSampler()
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=2,
        indexes=None,
        get_n_samples=False,
        topic=topic,
        sampler_func=sampler,
    )

    assert data is not None
    assert len(idxs) == 2


@pytest.mark.xdist_group(name="new_test")
def test_version_sampler_newest_mode(topic_basic):
    """Test get_experience with VersionSampler (newest mode)"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # 第一批：indexes[0:n_samples] 使用 version=0
    batch1_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts[:n_samples], "prompt_length": prompt_lengths[:n_samples]},
        version=0,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    assert batch1_indexes == list(range(n_samples))

    # 第二批：indexes[n_samples:2*n_samples] 使用 version=1
    batch2_indexes = tq_client.put_experience(
        data_dict={
            "prompt": padded_prompts[n_samples : 2 * n_samples],
            "prompt_length": prompt_lengths[n_samples : 2 * n_samples],
        },
        version=1,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    assert batch2_indexes == list(range(n_samples, 2 * n_samples))

    # 第三批：indexes[2*n_samples:] 使用 version=2
    batch3_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts[2 * n_samples :], "prompt_length": prompt_lengths[2 * n_samples :]},
        version=2,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    assert batch3_indexes == list(range(2 * n_samples, 3 * n_samples))

    sampler = VersionSampler(n_samples=2, by_group=False, selection_mode="newest")
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=2,
        indexes=None,
        get_n_samples=False,
        topic=topic,
        sampler_func=sampler,
    )

    assert data is not None
    assert len(idxs) == 2
    assert idxs == [4, 5]


@pytest.mark.xdist_group(name="new_test")
def test_version_sampler_oldest_mode(topic_basic):
    """Test get_experience with VersionSampler (oldest mode)"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # 第一批：indexes[0:n_samples] 使用 version=0
    batch1_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts[:n_samples], "prompt_length": prompt_lengths[:n_samples]},
        version=0,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    assert batch1_indexes == list(range(n_samples))

    # 第二批：indexes[n_samples:2*n_samples] 使用 version=1
    batch2_indexes = tq_client.put_experience(
        data_dict={
            "prompt": padded_prompts[n_samples : 2 * n_samples],
            "prompt_length": prompt_lengths[n_samples : 2 * n_samples],
        },
        version=1,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    assert batch2_indexes == list(range(n_samples, 2 * n_samples))

    # 第三批：indexes[2*n_samples:] 使用 version=2
    batch3_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts[2 * n_samples :], "prompt_length": prompt_lengths[2 * n_samples :]},
        version=2,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    assert batch3_indexes == list(range(2 * n_samples, 3 * n_samples))

    sampler = VersionSampler(n_samples=2, by_group=False, selection_mode="oldest")
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=2,
        indexes=None,
        get_n_samples=False,
        topic=topic,
        sampler_func=sampler,
    )

    assert data is not None
    assert len(idxs) == 2
    assert idxs == [0, 1]


@pytest.mark.xdist_group(name="new_test")
def test_get_experience_with_specific_indexes(topic_basic):
    """Test get_experience with specific indexes provided"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    allocated_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Get specific indexes from allocated ones
    specific_indexes = [allocated_indexes[1], allocated_indexes[3], allocated_indexes[5]]
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=3,
        indexes=specific_indexes,
        get_n_samples=False,
        topic=topic,
    )

    assert data is not None
    assert set(idxs) == set(specific_indexes)
    assert len(data["prompt"]) == 3


@pytest.mark.xdist_group(name="new_test")
def test_put_experience_with_save_dtype(topic_basic):
    """Test put_experience with save_dtype parameter"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # Put with specific save_dtype
    tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        save_dtype="float16",
        topic=topic,
    )

    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=6,
        indexes=None,
        get_n_samples=False,
        topic=topic,
    )

    assert data is not None
    assert len(idxs) == 6


@pytest.mark.xdist_group(name="new_test")
def test_multiple_consumers_independent_consumption(topic_basic):
    """Test that multiple consumers consume data independently"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Learner consumes some data
    data1, idxs1 = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=n_samples,
        indexes=None,
        get_n_samples=True,
        topic=topic,
    )

    # Evaluator should still be able to consume the same data
    data2, idxs2 = tq_client.get_experience(
        consumer="evaluator",
        experience_columns=["prompt"],
        experience_count=n_samples,
        indexes=None,
        get_n_samples=True,
        topic=topic,
    )

    assert data1 is not None
    assert data2 is not None
    assert len(idxs1) == n_samples
    assert len(idxs2) == n_samples


@pytest.mark.xdist_group(name="new_test")
def test_reset_all(topic_basic):
    """Test reset_all functionality"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # Put data
    allocated_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    assert allocated_indexes == list(range(6))

    # Verify data exists
    count, _ = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt"],
    )
    assert count > 0

    # Reset all
    tq_client.reset_all()

    # Topic should no longer exist
    with pytest.raises(Exception, match="Unknown topic"):
        tq_client.get_data_ready_set(
            topic=topic,
            experience_columns=["prompt"],
        )


@pytest.mark.xdist_group(name="new_test")
def test_multiple_topics_independent(topic_basic):
    """Test that multiple topics operate independently"""
    tq_client = get_transferqueue_client()

    topic1 = "TOPIC_1"
    topic2 = "TOPIC_2"

    prompts_num = 2
    n_samples = 2
    experience_columns = ["prompt", "prompt_length"]
    consumers = ["learner"]

    # Create two topics
    tq_client.add_topic(
        prompts_num=prompts_num,
        n_samples_per_prompt=n_samples,
        experience_columns=experience_columns,
        experience_consumers=consumers,
        topic=topic1,
    )

    tq_client.add_topic(
        prompts_num=prompts_num,
        n_samples_per_prompt=n_samples,
        experience_columns=experience_columns,
        experience_consumers=consumers,
        topic=topic2,
    )

    # Put different data to each topic
    lengths1 = [3, 5]
    prompts1 = _make_padded_prompts(lengths1, pad_id=0)
    padded1 = prompts1.repeat_interleave(n_samples, dim=0)
    lens1 = torch.tensor(lengths1, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    lengths2 = [2, 4]
    prompts2 = _make_padded_prompts(lengths2, pad_id=0)
    padded2 = prompts2.repeat_interleave(n_samples, dim=0)
    lens2 = torch.tensor(lengths2, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    tq_client.put_experience(
        data_dict={"prompt": padded1, "prompt_length": lens1},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic1,
    )

    tq_client.put_experience(
        data_dict={"prompt": padded2, "prompt_length": lens2},
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic2,
    )

    # Get data from each topic independently
    data1, idxs1 = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt", "prompt_length"],
        experience_count=2,
        indexes=None,
        get_n_samples=True,
        topic=topic1,
    )

    data2, idxs2 = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt", "prompt_length"],
        experience_count=2,
        indexes=None,
        get_n_samples=True,
        topic=topic2,
    )

    assert data1 is not None
    assert data2 is not None

    # Cleanup
    tq_client.delete_topic(topic1)
    tq_client.delete_topic(topic2)


@pytest.fixture()
def topic_with_uuid():
    """Create a topic for testing UUID-related scenarios"""
    tq_client = get_transferqueue_client()

    topic = "UUID_TOPIC"
    prompts_num = 2
    n_samples = 4
    experience_columns = ["prompt", "prompt_length", "response", "prompt_uuid"]
    consumers = ["learner"]

    tq_client.add_topic(
        prompts_num=prompts_num,
        n_samples_per_prompt=n_samples,
        experience_columns=experience_columns,
        experience_consumers=consumers,
        topic=topic,
    )

    yield {
        "topic": topic,
        "prompts_num": prompts_num,
        "n_samples": n_samples,
        "columns": experience_columns,
        "consumers": consumers,
    }

    tq_client.delete_topic(topic)


@pytest.mark.xdist_group(name="new_test")
def test_scenario_a_prompt_first_then_batched_responses(topic_basic):
    """
    Scenario a: 提前写入prompts + 分批responses, nspp = 4
    Steps:
    1. Write prompts with needs_expansion=True → allocate expanded indexes [0-11]
    2. Batch write responses to specific indexes → responses write normally
    """
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    # Create test prompts (3 prompts)
    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).unsqueeze(1)

    # Step 1: Write prompts with needs_expansion=True
    # This should allocate 3 groups * 2 samples = 6 indexes (not 12 since n_samples=2, not 4)
    # But for this test, we use the topic's actual n_samples
    allocated_indexes = tq_client.put_experience(
        data_dict={"prompt": prompts, "prompt_length": prompt_lengths},
        indexes=None,
        needs_expansion=True,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Verify expanded allocation
    assert isinstance(allocated_indexes, list)
    expected_count = len(prompts) * n_samples  # 3 * 2 = 6
    assert len(allocated_indexes) == expected_count, f"Expected {expected_count} indexes, got {len(allocated_indexes)}"

    # Step 2: Batch write responses to first 2 indexes
    responses = torch.tensor([[1.0], [2.0]])
    response_indexes = allocated_indexes[:2]

    returned_indexes = tq_client.put_experience(
        data_dict={"response": responses},
        indexes=response_indexes,
        topic=topic,
    )

    # Should return the same indexes we provided
    assert returned_indexes == response_indexes

    # Verify both prompts and responses are ready for those indexes
    count, idxs = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt", "response"],
    )
    assert count >= 2


@pytest.mark.xdist_group(name="new_test")
def test_scenario_b_multiple_complete_batches(topic_basic):
    """
    Scenario b: 多批次完整写入
    First batch: complete write → [0-7]
    Second batch: complete write → [8-15]
    """
    tq_client = get_transferqueue_client()
    topic = "MULTI_BATCH_TOPIC"

    # Create topic with n_samples=4
    prompts_num = 2
    n_samples = 4
    experience_columns = ["prompt", "response"]
    consumers = ["learner"]

    tq_client.add_topic(
        prompts_num=prompts_num,
        n_samples_per_prompt=n_samples,
        experience_columns=experience_columns,
        experience_consumers=consumers,
        topic=topic,
    )

    # First batch: 2 prompts, each with 4 responses
    pad_id = 0
    padded_prompts = _make_padded_prompts([3, 5], pad_id=pad_id)
    prompts1 = padded_prompts.repeat_interleave(n_samples, dim=0)
    responses1 = _make_padded_prompts([2, 3, 4, 5, 1, 2, 3, 4], pad_id=pad_id)
    batch1_indexes = tq_client.put_experience(
        data_dict={"prompt": prompts1, "response": responses1},
        indexes=None,
        topic=topic,
    )
    assert len(batch1_indexes) == 8  # 2 * 4
    assert batch1_indexes == list(range(8))

    # Second batch: 2 more prompts, each with 4 responses
    padded_prompts = _make_padded_prompts([2, 4], pad_id=pad_id)
    prompts2 = padded_prompts.repeat_interleave(n_samples, dim=0)
    responses2 = _make_padded_prompts([1, 2, 3, 4, 2, 3, 4, 5], pad_id=pad_id)
    batch2_indexes = tq_client.put_experience(
        data_dict={"prompt": prompts2, "response": responses2},
        indexes=None,
        topic=topic,
    )
    assert len(batch2_indexes) == 8  # 2 * 4
    assert batch2_indexes == list(range(8, 16))

    # Verify all data is ready
    count, idxs = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt", "response"],
    )
    assert count == 16
    assert len(idxs) == 16

    # Cleanup
    tq_client.delete_topic(topic)


@pytest.mark.xdist_group(name="new_test")
def test_scenario_c_uuid_out_of_order_writes(topic_with_uuid):
    """
    Scenario c: UUID乱序写入
    - First write with uuid1, uuid2 → allocate to groups 0, 1
    - Second write with uuid1 → allocate to same group (index 1)
    - Third write with uuid1 → allocate to same group (index 2)
    """
    tq_client = get_transferqueue_client()
    topic = topic_with_uuid["topic"]

    # First batch: 2 prompts with different UUIDs
    pad_id = 0
    prompts_batch1 = _make_padded_prompts([3, 5], pad_id=pad_id)
    responses_batch1 = _make_padded_prompts([2, 3], pad_id=pad_id)
    prompt_uuids_batch1 = ["uuid1", "uuid2"]

    # Write with UUIDs - should allocate [0, 2] (group 0 index 0, group 1 index 0)
    batch1_indexes = tq_client.put_experience(
        data_dict={
            "prompt": prompts_batch1,
            "response": responses_batch1,
            "prompt_uuid": prompt_uuids_batch1,
        },
        indexes=None,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Should have allocated first index of each group
    assert len(batch1_indexes) == 2
    assert batch1_indexes == [0, 4]

    # Second batch: 1 prompt with uuid1 (should go to same group, index 1)
    prompts_batch2 = _make_padded_prompts([3], pad_id=pad_id)
    responses_batch2 = _make_padded_prompts([2], pad_id=pad_id)
    prompt_uuids_batch2 = ["uuid1"]

    batch2_indexes = tq_client.put_experience(
        data_dict={
            "prompt": prompts_batch2,
            "response": responses_batch2,
            "prompt_uuid": prompt_uuids_batch2,
        },
        indexes=None,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    assert len(batch2_indexes) == 1
    assert batch2_indexes[0] == 1

    # Third batch: another uuid1 prompt (index 2)
    prompts_batch3 = _make_padded_prompts([3], pad_id=pad_id)
    responses_batch3 = _make_padded_prompts([2], pad_id=pad_id)
    prompt_uuids_batch3 = ["uuid1"]

    batch3_indexes = tq_client.put_experience(
        data_dict={
            "prompt": prompts_batch3,
            "response": responses_batch3,
            "prompt_uuid": prompt_uuids_batch3,
        },
        indexes=None,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    assert len(batch3_indexes) == 1
    assert batch3_indexes[0] == 2

    # Verify all written data is ready
    count, idxs = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt", "response"],
    )
    assert count == 4  # 2 + 1 + 1
    assert len(idxs) == 4


@pytest.mark.xdist_group(name="new_test")
def test_scenario_d_incomplete_sequential_batches(topic_basic):
    """
    Scenario d: 不完整批次顺序写入
    - First batch: write 2 prompts → [0, 1] (group 0, indexes 0, 1)
    - Second batch: write 2 prompts with same content → [2, 3] (group 0, indexes 2, 3)
    - Should reuse existing prompt data, only allocate new indexes
    """
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]

    # Create repeated prompts
    pad_id = 0
    prompt_length = 3
    prompts = _make_padded_prompts([prompt_length, prompt_length], pad_id=pad_id)
    prompt_lengths = torch.tensor([prompt_length, prompt_length], dtype=torch.int32).unsqueeze(1)

    # First batch: write to indexes None (auto-allocate)
    batch1_indexes = tq_client.put_experience(
        data_dict={"prompt": prompts, "prompt_length": prompt_lengths},
        indexes=None,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Should allocate [0, 1]
    assert len(batch1_indexes) == 2
    assert batch1_indexes[0] == 0
    assert batch1_indexes[1] == 1

    # Second batch: write same prompts again (should allocate [2, 3])
    batch2_indexes = tq_client.put_experience(
        data_dict={"prompt": prompts, "prompt_length": prompt_lengths},
        indexes=None,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    # Should allocate next available indexes [2, 3]
    assert len(batch2_indexes) == 2
    assert batch2_indexes[0] == 2
    assert batch2_indexes[1] == 3

    # Verify all data is ready
    count, idxs = tq_client.get_data_ready_set(
        topic=topic,
        experience_columns=["prompt"],
    )
    assert count == 4
    assert len(idxs) == 4
    assert set(idxs) == {0, 1, 2, 3}


# --------------------- Pickle Encoding Tests ---------------------


@pytest.fixture()
def topic_pickle_encoding():
    """Create a topic for testing pickle encoding"""
    tq_client = get_transferqueue_client()

    topic = "PICKLE_TOPIC"
    prompts_num = 3
    n_samples = 2
    experience_columns = ["prompt", "prompt_length", "metadata", "tags", "custom_data"]
    consumers = ["learner"]

    tq_client.add_topic(
        prompts_num=prompts_num,
        n_samples_per_prompt=n_samples,
        experience_columns=experience_columns,
        experience_consumers=consumers,
        topic=topic,
    )

    yield {
        "topic": topic,
        "prompts_num": prompts_num,
        "n_samples": n_samples,
        "columns": experience_columns,
        "consumers": consumers,
    }

    tq_client.delete_topic(topic)


@pytest.mark.xdist_group(name="new_test")
def test_pickle_encoding_list_str(topic_pickle_encoding):
    """Test pickle encoding for list[str]"""
    tq_client = get_transferqueue_client()
    topic = topic_pickle_encoding["topic"]
    n_samples = topic_pickle_encoding["n_samples"]

    # Prepare prompt tensors (raw encoding)
    pad_id = 0
    lengths = [3, 5, 4]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # Prepare list[str] data (pickle encoding)
    metadata = [
        {"user_id": 123, "timestamp": 1700000000},
        {"user_id": 456, "timestamp": 1700000001},
        {"user_id": 789, "timestamp": 1700000002},
    ]
    # Replicate to match n_samples
    metadata_replicated = []
    for m in metadata:
        metadata_replicated.extend([m] * n_samples)

    # Mixed encoding: prompts (raw) + metadata (pickle)
    allocated_indexes = tq_client.put_experience(
        data_dict={
            "prompt": padded_prompts,
            "prompt_length": prompt_lengths,
            "metadata": metadata_replicated,
        },
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    assert len(allocated_indexes) == 6

    # Get and verify mixed encoding data
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt", "metadata"],
        experience_count=6,
        indexes=None,
        get_n_samples=False,
        topic=topic,
    )

    assert data is not None
    assert len(idxs) == 6
    assert len(data["prompt"]) == 6
    assert len(data["metadata"]) == 6

    # Verify metadata is list of dicts (not tensors)
    assert all(isinstance(m, dict) for m in data["metadata"])
    assert data["metadata"][0] == {"user_id": 123, "timestamp": 1700000000}
    assert data["metadata"][1] == {"user_id": 123, "timestamp": 1700000000}
    assert data["metadata"][2] == {"user_id": 456, "timestamp": 1700000001}


@pytest.mark.xdist_group(name="new_test")
def test_pickle_encoding_list_any_nested(topic_pickle_encoding):
    """Test pickle encoding for nested complex objects"""
    tq_client = get_transferqueue_client()
    topic = topic_pickle_encoding["topic"]
    n_samples = topic_pickle_encoding["n_samples"]

    # Prepare complex nested data
    tags = [
        ["ai", "ml", "training"],
        ["rlhf", "language"],
        ["generation", "gpt"],
    ]
    # Replicate to match n_samples
    tags_replicated = []
    for t in tags:
        tags_replicated.extend([t] * n_samples)

    allocated_indexes = tq_client.put_experience(
        data_dict={"tags": tags_replicated},
        topic=topic,
    )

    assert len(allocated_indexes) == 6

    # Get and verify
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["tags"],
        experience_count=6,
        indexes=None,
        get_n_samples=False,
        topic=topic,
    )

    assert data is not None
    assert len(data["tags"]) == 6
    assert all(isinstance(t, list) for t in data["tags"])
    assert data["tags"][0] == ["ai", "ml", "training"]
    assert data["tags"][1] == ["ai", "ml", "training"]


@pytest.mark.xdist_group(name="new_test")
def test_pickle_encoding_backward_compatibility(topic_pickle_encoding):
    """Test that raw encoding (tensors) still works alongside pickle encoding"""
    tq_client = get_transferqueue_client()
    topic = topic_pickle_encoding["topic"]

    # Put only raw encoding data (old behavior)
    pad_id = 0
    lengths = [3, 5]
    prompts = _make_padded_prompts(lengths, pad_id=pad_id)
    padded_prompts = prompts.repeat_interleave(2, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(2, dim=0).unsqueeze(1)

    allocated_indexes = tq_client.put_experience(
        data_dict={
            "prompt": padded_prompts,
            "prompt_length": prompt_lengths,
        },
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )

    assert len(allocated_indexes) == 4

    # Get and verify - should still return tensors
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt", "prompt_length"],
        experience_count=4,
        indexes=None,
        get_n_samples=False,
        topic=topic,
    )

    assert data is not None
    assert all(isinstance(t, torch.Tensor) for t in data["prompt"])
    assert all(isinstance(t, torch.Tensor) for t in data["prompt_length"])


@pytest.mark.xdist_group(name="new_test")
def test_pickle_encoding_custom_objects(topic_pickle_encoding):
    """Test pickle encoding with custom class instances"""
    tq_client = get_transferqueue_client()
    topic = topic_pickle_encoding["topic"]

    # Create custom objects
    custom_objs = [
        CustomObject(1, "a"),
        CustomObject(2, "b"),
    ]
    custom_objs_replicated = custom_objs * 2  # Replicate for n_samples

    allocated_indexes = tq_client.put_experience(
        data_dict={"custom_data": custom_objs_replicated},
        topic=topic,
    )

    assert len(allocated_indexes) == 4

    # Get and verify
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["custom_data"],
        experience_count=4,
        indexes=None,
        get_n_samples=False,
        topic=topic,
    )

    assert data is not None
    assert len(data["custom_data"]) == 4
    assert all(isinstance(obj, CustomObject) for obj in data["custom_data"])
    assert data["custom_data"][0].x == 1
    assert data["custom_data"][0].y == "a"
    assert data["custom_data"][1].x == 2
    assert data["custom_data"][1].y == "b"


@pytest.mark.xdist_group(name="new_test")
def test_shared_column_storage_efficiency(topic_with_uuid):
    """Shard端测试：验证共享列通过标记避免重复存储，减少实际存储字节数。"""
    pad_id = 0
    topic = topic_with_uuid["topic"]
    tq_client = get_transferqueue_client()
    tq_mgr = ray.get_actor("TransferQueueManager")
    n_samples = ray.get(tq_mgr.get_n_samples_per_prompt.remote(topic))
    assert n_samples == 4

    # ========== 批次1：uuid1, uuid2 ==========
    prompts_batch1 = _make_padded_prompts([3, 5], pad_id=pad_id)
    responses_batch1 = _make_padded_prompts([2, 3], pad_id=pad_id)
    prompt_lengths1 = torch.tensor([3, 5], dtype=torch.int32).unsqueeze(1)
    prompt_uuids_batch1 = ["uuid1", "uuid2"]

    batch1_indexes = tq_client.put_experience(
        data_dict={
            "prompt": prompts_batch1,
            "response": responses_batch1,
            "prompt_length": prompt_lengths1,
            "prompt_uuid": prompt_uuids_batch1,
        },
        indexes=None,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    assert batch1_indexes == [0, 4]

    # ========== 批次2：uuid1, uuid1, uuid2 ==========
    prompts_batch2 = _make_padded_prompts([3, 5, 7], pad_id=pad_id)
    responses_batch2 = _make_padded_prompts([2, 3, 4], pad_id=pad_id)
    prompt_lengths2 = torch.tensor([3, 5, 7], dtype=torch.int32).unsqueeze(1)
    prompt_uuids_batch2 = ["uuid1", "uuid1", "uuid2"]

    batch2_indexes = tq_client.put_experience(
        data_dict={
            "prompt": prompts_batch2,
            "response": responses_batch2,
            "prompt_length": prompt_lengths2,
            "prompt_uuid": prompt_uuids_batch2,
        },
        indexes=None,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    assert batch2_indexes == [1, 2, 5]

    # ========== 批次3：uuid1, uuid2, uuid3 ==========
    prompts_batch3 = _make_padded_prompts([3, 9, 8], pad_id=pad_id)
    responses_batch3 = _make_padded_prompts([2, 6, 5], pad_id=pad_id)
    prompt_lengths3 = torch.tensor([3, 9, 8], dtype=torch.int32).unsqueeze(1)
    prompt_uuids_batch3 = ["uuid1", "uuid2", "uuid3"]

    batch3_indexes = tq_client.put_experience(
        data_dict={
            "prompt": prompts_batch3,
            "response": responses_batch3,
            "prompt_length": prompt_lengths3,
            "prompt_uuid": prompt_uuids_batch3,
        },
        indexes=None,
        unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    assert batch3_indexes == [3, 6, 8]

    # ========== 获取存储字节数 ==========
    # 获取实际的存储字节数
    storage_bytes = ray.get(tq_mgr.get_columns_storage_bytes.remote(topic))
    shared_bytes = sum(storage_bytes.get(col, 0) for col in ["prompt", "prompt_length", "prompt_uuid"])

    # 验证唯一group数是 {0, 1, 2}
    # 实际group分布：index 0-3 → group 0, index 4-5 → group 1, index 6 → group 1, index 8 → group 2
    all_indexes = batch1_indexes + batch2_indexes + batch3_indexes  # [0, 4, 1, 2, 5, 3, 6, 8]
    unique_groups = set(i // n_samples for i in all_indexes)
    assert unique_groups == set({0, 1, 2}), f"应该有3个唯一group，实际有{unique_groups}"

    # ========== 验证共享列存储效率 ==========
    # 理论无共享：如果每个索引都独立存储完整数据，需要存储所有批次的数据总和
    # 即：8个索引的完整数据 = 批次1+批次2+批次3的所有共享列数据
    expected_without_sharing = (
        # prompts: 所有8个样本的prompt数据
        _calculate_tensor_bytes(prompts_batch1)
        + _calculate_tensor_bytes(prompts_batch2)
        + _calculate_tensor_bytes(prompts_batch3)
        +
        # prompt_lengths: 所有8个样本的prompt_length数据
        _calculate_tensor_bytes(prompt_lengths1)
        + _calculate_tensor_bytes(prompt_lengths2)
        + _calculate_tensor_bytes(prompt_lengths3)
        +
        # prompt_uuid: 所有8个样本的uuid数据
        _calculate_pickle_bytes(prompt_uuids_batch1)
        + _calculate_pickle_bytes(prompt_uuids_batch2)
        + _calculate_pickle_bytes(prompt_uuids_batch3)
    )

    # 验证共享列存储效率：压缩比 = 实际存储 / 理论无共享，应 < 1
    compression_ratio = shared_bytes / expected_without_sharing
    assert 0 < compression_ratio < 1, f"压缩比{compression_ratio:.2%}应该在(0, 1)之间，表示共享机制有效减少了存储"

    # 验证各共享列都有数据
    assert storage_bytes.get("prompt", 0) > 0, "prompt列应该有存储数据"
    assert storage_bytes.get("prompt_length", 0) > 0, "prompt_length列应该有存储数据"
    assert storage_bytes.get("prompt_uuid", 0) > 0, "prompt_uuid列应该有存储数据"


def _calculate_tensor_bytes(tensor: torch.Tensor) -> int:
    """计算 tensor 的字节数"""
    return tensor.element_size() * tensor.numel()


def _calculate_pickle_bytes(data_list: list) -> int:
    """计算 pickle 编码数据的字节数"""
    total_bytes = 0
    for item in data_list:
        total_bytes += len(pickle.dumps(item))
    return total_bytes


@pytest.mark.xdist_group(name="bf16_test")
def test_bf16_serialization(topic_basic):
    """测试 bf16 数据类型的序列化和反序列化"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    # 创建 bf16 类型的 tensor
    pad_id = 0
    lengths = [3, 5, 4]
    prompts_bf16 = _make_padded_prompts(lengths, pad_id=pad_id).to(torch.bfloat16)
    padded_prompts = prompts_bf16.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # 写入 bf16 数据
    allocated_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        # unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    ready_count, ready_idx = tq_client.get_data_ready_set(topic=topic, experience_columns=["prompt", "prompt_length"])
    assert allocated_indexes == list(range(6))
    assert ready_idx == list(range(6))

    # 读取 bf16 数据
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=6,
        get_n_samples=False,
        topic=topic,
    )

    # 验证返回的 tensor 类型是 bf16
    assert data is not None
    assert len(idxs) == 6
    assert all(t.dtype == torch.bfloat16 for t in data["prompt"])

    # 验证数据完整性
    assert len(data["prompt"]) == 6


@pytest.mark.xdist_group(name="float8_test")
def test_float8_serialization(topic_basic):
    """测试 float8 数据类型的序列化和反序列化"""
    tq_client = get_transferqueue_client()
    topic = topic_basic["topic"]
    n_samples = topic_basic["n_samples"]

    # 创建 float8 类型的 tensor
    pad_id = 0
    lengths = [3, 5, 4]
    prompts_float8 = _make_padded_prompts(lengths, pad_id=pad_id).to(torch.float8_e4m3fn)
    padded_prompts = prompts_float8.repeat_interleave(n_samples, dim=0)
    prompt_lengths = torch.tensor(lengths, dtype=torch.int32).repeat_interleave(n_samples, dim=0).unsqueeze(1)

    # 写入 float8 数据
    allocated_indexes = tq_client.put_experience(
        data_dict={"prompt": padded_prompts, "prompt_length": prompt_lengths},
        # unpad_pairs=[("prompt", "prompt_length", "right_pad")],
        topic=topic,
    )
    ready_count, ready_idx = tq_client.get_data_ready_set(topic=topic, experience_columns=["prompt", "prompt_length"])
    assert allocated_indexes == list(range(6))
    assert ready_idx == list(range(6))

    # 读取 float8 数据
    data, idxs = tq_client.get_experience(
        consumer="learner",
        experience_columns=["prompt"],
        experience_count=6,
        get_n_samples=False,
        topic=topic,
    )

    # 验证返回的 tensor 类型是 float8
    assert data is not None
    assert len(idxs) == 6
    assert all(t.dtype == torch.float8_e4m3fn for t in data["prompt"])

    # 验证数据完整性
    assert len(data["prompt"]) == 6


# @pytest.mark.xdist_group(name="new_test")
# def test_pickle_encoding_empty_list(topic_pickle_encoding):
#     """Test pickle encoding with empty lists"""
#     tq_client = get_transferqueue_client()
#     topic = topic_pickle_encoding["topic"]

#     # Put empty list
#     allocated_indexes = tq_client.put_experience(
#         data_dict={"tags": []},
#         topic=topic,
#     )

#     assert allocated_indexes == []

#     # Get with empty list should work
#     data, idxs = tq_client.get_experience(
#         consumer="learner",
#         experience_columns=["tags"],
#         experience_count=0,
#         indexes=None,
#         get_n_samples=False,
#         topic=topic,
#     )

#     assert data is not None
#     assert len(data["tags"]) == 0
