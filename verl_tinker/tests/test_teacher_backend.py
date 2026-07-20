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
    assert backend.resolve("Qwen/Qwen3-30B-A3B") == "large"
    assert backend.resolve("/models/qwen3-30b") == "large"
    assert backend.resolve("small") == "small"


@pytest.mark.asyncio
async def test_teacher_client_rejects_topk_above_vllm_boot_limit():
    client = TeacherClient(MagicMock(), model_path="teacher", max_prompt_logprobs=8)

    with pytest.raises(ValueError, match="supports at most 8"):
        await client.generate(
            "request",
            prompt_ids=[1, 2],
            sampling_params={"max_tokens": 1, "prompt_logprobs": 9},
        )


def test_teacher_backend_shutdown_releases_server_workers_and_pool():
    backend = object.__new__(TeacherInferenceBackend)
    server = MagicMock()
    worker = MagicMock()
    load_balancer = MagicMock()
    replica = SimpleNamespace(servers=[server], _server_handle=server, workers=[worker])
    backend._managers = {"teacher": SimpleNamespace(load_balancer_handle=load_balancer, rollout_replicas=[replica])}
    backend._clients = {"teacher": MagicMock()}
    backend._aliases = {"teacher": "teacher"}
    placement_group = MagicMock()
    backend._resource_pool = SimpleNamespace(pgs=[placement_group])

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
    assert backend._resource_pool is None
