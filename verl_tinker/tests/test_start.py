from unittest.mock import MagicMock, patch

import pytest

from verl_tinker.start import BYTES_PER_GB, check_cluster_disk_space


def _node(node_id, name, address, *, gpu=8):
    return {
        "Alive": True,
        "NodeID": node_id,
        "NodeName": name,
        "NodeManagerAddress": address,
        "Resources": {"GPU": gpu} if gpu else {"CPU": 16},
    }


def _usage(*, free_gb, total_gb=1000):
    return {
        "hostname": "host",
        "path": "/tmp",
        "total_bytes": total_gb * BYTES_PER_GB,
        "used_bytes": (total_gb - free_gb) * BYTES_PER_GB,
        "free_bytes": free_gb * BYTES_PER_GB,
    }


def test_disk_preflight_probes_each_accelerator_worker_and_skips_cpu_head():
    nodes = [
        _node("0" * 55 + "1", "head", "10.0.0.1", gpu=0),
        _node("0" * 55 + "2", "worker1", "10.0.0.2"),
        _node("0" * 55 + "3", "worker2", "10.0.0.3"),
    ]
    remote_probe = MagicMock()
    remote_probe.options.return_value.remote.side_effect = ["ref-1", "ref-2"]

    with (
        patch("verl_tinker.start.ray.nodes", return_value=nodes),
        patch("verl_tinker.start.ray.remote", return_value=remote_probe),
        patch("verl_tinker.start.ray.get", return_value=[_usage(free_gb=400), _usage(free_gb=500)]),
    ):
        results = check_cluster_disk_space("/tmp", min_free_gb=350)

    assert len(results) == 2
    assert remote_probe.options.return_value.remote.call_count == 2


def test_disk_preflight_fails_with_all_underprovisioned_workers_in_message():
    nodes = [
        _node("0" * 55 + "2", "worker1", "10.0.0.2"),
        _node("0" * 55 + "3", "worker2", "10.0.0.3"),
    ]
    remote_probe = MagicMock()
    remote_probe.options.return_value.remote.side_effect = ["ref-1", "ref-2"]

    with (
        patch("verl_tinker.start.ray.nodes", return_value=nodes),
        patch("verl_tinker.start.ray.remote", return_value=remote_probe),
        patch("verl_tinker.start.ray.get", return_value=[_usage(free_gb=19), _usage(free_gb=340)]),
        pytest.raises(RuntimeError, match=r"worker1.*19.0 GB free.*worker2.*340.0 GB free"),
    ):
        check_cluster_disk_space("/tmp", min_free_gb=350)
