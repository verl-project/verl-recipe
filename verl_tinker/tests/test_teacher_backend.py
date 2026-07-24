from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from verl_tinker.backends.teacher import TeacherClient, TeacherInferenceBackend


def _teacher(key, path, name=None, world_size=2, engine="vllm"):
    return SimpleNamespace(
        key=key,
        model_name=name or path,
        model_path=path,
        world_size=world_size,
        inference=SimpleNamespace(
            name=engine,
            engine_kwargs={"vllm": {"max_logprobs": 64}},
        ),
    )


def test_teacher_backend_partitions_pool_and_builds_one_manager_per_teacher():
    teachers = {
        "small": _teacher("small", "/models/qwen3-1.7b", "Qwen/Qwen3-1.7B"),
        "large": _teacher("large", "/models/qwen3-30b", "Qwen/Qwen3-30B-A3B"),
    }
    distillation = SimpleNamespace(nnodes=1, n_gpus_per_node=4, teacher_models=teachers)
    config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "teacher_models": {
                    "small": {"key": "small", "model_name": "Qwen/Qwen3-1.7B", "model_path": "/models/qwen3-1.7b"},
                    "large": {"key": "large", "model_name": "Qwen/Qwen3-30B-A3B", "model_path": "/models/qwen3-30b"},
                },
            }
        }
    )
    pool = SimpleNamespace(pgs=None)
    manager_calls = []

    class FakeManager:
        def __init__(self, distillation_config, teacher_config, sub_pool):
            manager_calls.append((distillation_config, teacher_config, sub_pool))
            self.load_balancer_handle = MagicMock()
            self.rollout_replicas = []

    with (
        patch("verl_tinker.config_utils.omega_conf_to_dataclass", return_value=distillation),
        patch("verl_tinker.backends.teacher.RayResourcePool", return_value=pool) as mock_pool,
        patch("verl_tinker.backends.teacher.split_resource_pool", return_value=["pool-a", "pool-b"]) as mock_split,
        patch("verl_tinker.backends.teacher.TeacherModelManager", FakeManager),
        patch("verl_tinker.backends.teacher.LLMServerClient"),
        patch("verl_tinker.backends.teacher.kill_ray_actors_and_wait"),
        patch("verl_tinker.backends.teacher.remove_placement_groups_and_wait"),
    ):
        backend = TeacherInferenceBackend(config)

    mock_pool.assert_called_once_with(
        process_on_nodes=[4],
        use_gpu=True,
        max_colocate_count=3,
        name_prefix="teacher_pool",
    )
    mock_split.assert_called_once_with(pool, [2, 2])
    assert [call[1].model_path for call in manager_calls] == ["/models/qwen3-1.7b", "/models/qwen3-30b"]
    assert ("Qwen/Qwen3-30B-A3B", "/models/qwen3-30b") in backend.sampling_targets
    assert ("/models/qwen3-30b", "/models/qwen3-30b") in backend.sampling_targets
    assert backend.get_client("/models/qwen3-30b").model_path == "/models/qwen3-30b"


def test_teacher_backend_dedicated_pools_allocate_largest_teacher_first():
    teachers = {
        "deepmath": _teacher("deepmath", "/models/qwen3-32b", world_size=4),
        "tulu3": _teacher("tulu3", "/models/qwen3-235b", world_size=8),
    }
    distillation = SimpleNamespace(nnodes=2, n_gpus_per_node=6, teacher_models=teachers)
    config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "dedicated_resource_pools": True,
                "teacher_models": {
                    "deepmath_teacher": {
                        "key": "deepmath",
                        "model_name": "Qwen/Qwen3-32B",
                        "model_path": "/models/qwen3-32b",
                    },
                    "tulu3_teacher": {
                        "key": "tulu3",
                        "model_name": "Qwen/Qwen3-235B-A22B-Instruct-2507",
                        "model_path": "/models/qwen3-235b",
                    },
                },
            }
        }
    )
    pools = [SimpleNamespace(pgs=None), SimpleNamespace(pgs=None)]
    manager_calls = []

    class FakeManager:
        def __init__(self, distillation_config, teacher_config, resource_pool):
            manager_calls.append((distillation_config, teacher_config, resource_pool))
            self.load_balancer_handle = MagicMock()
            self.rollout_replicas = []

    with (
        patch("verl_tinker.config_utils.omega_conf_to_dataclass", return_value=distillation),
        patch("verl_tinker.backends.teacher.RayResourcePool", side_effect=pools) as mock_pool,
        patch("verl_tinker.backends.teacher.split_resource_pool") as mock_split,
        patch("verl_tinker.backends.teacher.TeacherModelManager", FakeManager),
        patch("verl_tinker.backends.teacher.LLMServerClient"),
        patch("verl_tinker.backends.teacher.kill_ray_actors_and_wait"),
        patch("verl_tinker.backends.teacher.remove_placement_groups_and_wait"),
    ):
        backend = TeacherInferenceBackend(config)

    assert [call.kwargs["process_on_nodes"] for call in mock_pool.call_args_list] == [[8], [4]]
    assert [call.kwargs["name_prefix"] for call in mock_pool.call_args_list] == [
        "teacher_pool_tulu3",
        "teacher_pool_deepmath",
    ]
    assert [call[1].model_path for call in manager_calls] == [
        "/models/qwen3-235b",
        "/models/qwen3-32b",
    ]
    assert [call[2] for call in manager_calls] == pools
    assert [(call[0].nnodes, call[0].n_gpus_per_node) for call in manager_calls] == [
        (1, 8),
        (1, 4),
    ]
    mock_split.assert_not_called()
    assert ("Qwen/Qwen3-32B", "/models/qwen3-32b") in backend.sampling_targets
    assert (
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "/models/qwen3-235b",
    ) in backend.sampling_targets


def test_dedicated_pool_startup_failure_cleans_all_created_pools():
    teachers = {
        "small": _teacher("small", "/models/small", world_size=4),
        "large": _teacher("large", "/models/large", world_size=8),
    }
    distillation = SimpleNamespace(nnodes=2, n_gpus_per_node=6, teacher_models=teachers)
    config = OmegaConf.create(
        {
            "distillation": {
                "enabled": True,
                "dedicated_resource_pools": True,
                "teacher_models": {
                    "small": {"key": "small", "model_name": "small", "model_path": "/models/small"},
                    "large": {"key": "large", "model_name": "large", "model_path": "/models/large"},
                },
            }
        }
    )
    placement_group = MagicMock()
    pools = [SimpleNamespace(pgs=[placement_group]), SimpleNamespace(pgs=None)]

    class FailingManager:
        def __init__(self, *_args):
            self.load_balancer_handle = MagicMock()
            self.rollout_replicas = []
            raise RuntimeError("teacher failed to start")

    with (
        patch("verl_tinker.config_utils.omega_conf_to_dataclass", return_value=distillation),
        patch("verl_tinker.backends.teacher.RayResourcePool", side_effect=pools),
        patch("verl_tinker.backends.teacher.TeacherModelManager", FailingManager),
        patch("verl_tinker.backends.teacher.kill_ray_actors_and_wait") as mock_kill,
        patch("verl_tinker.backends.teacher.remove_placement_groups_and_wait") as mock_remove,
    ):
        with pytest.raises(RuntimeError, match="teacher failed to start"):
            TeacherInferenceBackend(config)

    mock_kill.assert_called_once()
    assert mock_remove.call_args.args[0] == [placement_group]


@pytest.mark.asyncio
async def test_teacher_client_rejects_topk_above_vllm_boot_limit():
    client = TeacherClient(MagicMock(), model_path="teacher", max_prompt_logprobs=8)

    with pytest.raises(ValueError, match="supports at most 8"):
        await client.generate(
            "request",
            prompt_ids=[1, 2],
            sampling_params={"max_tokens": 1, "prompt_logprobs": 9},
        )


def test_teacher_backend_shutdown_releases_server_workers_and_pools():
    backend = object.__new__(TeacherInferenceBackend)
    server = MagicMock()
    worker = MagicMock()
    load_balancer = MagicMock()
    replica = SimpleNamespace(servers=[server], _server_handle=server, workers=[worker])
    backend._managers = {"teacher": SimpleNamespace(load_balancer_handle=load_balancer, rollout_replicas=[replica])}
    backend._clients = {"teacher": MagicMock()}
    placement_groups = [MagicMock(), MagicMock()]
    backend._resource_pools = [
        SimpleNamespace(pgs=[placement_groups[0]]),
        SimpleNamespace(pgs=[placement_groups[1]]),
    ]

    with (
        patch("verl_tinker.backends.teacher.kill_ray_actors_and_wait") as mock_kill,
        patch("verl_tinker.backends.teacher.remove_placement_groups_and_wait") as mock_remove,
    ):
        backend.shutdown()

    killed = mock_kill.call_args.args[0]
    assert load_balancer in killed
    assert server in killed
    assert worker in killed
    mock_remove.assert_called_once()
    assert mock_remove.call_args.args[0] == placement_groups
    assert backend._resource_pools == []
