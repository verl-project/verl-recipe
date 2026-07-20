# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Dedicated, inference-only teacher models backed by VeRL rollout servers."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import ray
from omegaconf import DictConfig

from verl.experimental.teacher_loop.teacher_model import TeacherModelManager
from verl.single_controller.ray import RayResourcePool
from verl.single_controller.ray.base import split_resource_pool
from verl.workers.config import DistillationConfig
from verl.workers.rollout.llm_server import LLMServerClient

from ..config_utils import _normalize_teacher_model_identifiers, _to_verl_distillation_config
from .backend_utils import kill_ray_actors_and_wait, remove_placement_groups_and_wait

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TeacherDescriptor:
    key: str
    model_name: str
    model_path: str
    max_context_length: int | None


class TeacherClient:
    """Small guard around VeRL's client for per-engine request limits."""

    def __init__(self, client: LLMServerClient, *, model_path: str, max_prompt_logprobs: int | None):
        self._client = client
        self.model_path = model_path
        self.max_prompt_logprobs = max_prompt_logprobs

    async def generate(self, request_id, *, prompt_ids, sampling_params, **kwargs):
        requested = sampling_params.get("prompt_logprobs")
        if requested is not None and self.max_prompt_logprobs is not None and int(requested) > self.max_prompt_logprobs:
            raise ValueError(
                f"Teacher {self.model_path!r} supports at most {self.max_prompt_logprobs} prompt logprobs, "
                f"but the request asked for {requested}. Increase inference.engine_kwargs.vllm.max_logprobs."
            )
        return await self._client.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            **kwargs,
        )


class TeacherInferenceBackend:
    """Own one VeRL ``TeacherModelManager`` per configured frozen teacher."""

    def __init__(self, config: DictConfig):
        self.config = config
        identifier_errors = _normalize_teacher_model_identifiers(config)
        if identifier_errors:
            raise ValueError("; ".join(identifier_errors))
        self._model_names = self._get_model_names(config.distillation.teacher_models)
        self.distillation_config: DistillationConfig = _to_verl_distillation_config(config.distillation)
        self._resource_pool: RayResourcePool | None = None
        self._managers: dict[str, TeacherModelManager] = {}
        self._clients: dict[str, TeacherClient] = {}
        self._aliases: dict[str, str] = {}

        try:
            self._initialize()
        except BaseException:
            self.shutdown()
            raise

    @staticmethod
    def is_enabled(config: DictConfig) -> bool:
        return bool(config.get("distillation", {}).get("enabled", False))

    @property
    def descriptors(self) -> list[TeacherDescriptor]:
        return [
            TeacherDescriptor(
                key=key,
                model_name=self._model_names[key],
                model_path=teacher.model_path,
                max_context_length=teacher.inference.max_model_len,
            )
            for key, teacher in self.distillation_config.teacher_models.items()
        ]

    def _initialize(self) -> None:
        cfg = self.distillation_config
        self._resource_pool = RayResourcePool(
            process_on_nodes=[cfg.n_gpus_per_node] * cfg.nnodes,
            use_gpu=True,
            max_colocate_count=3,
            name_prefix="teacher_pool",
        )
        teachers = list(cfg.teacher_models.items())
        sub_pools = split_resource_pool(self._resource_pool, [teacher.world_size for _, teacher in teachers])

        for (key, teacher_config), sub_pool in zip(teachers, sub_pools, strict=True):
            # Keep the partially initialized object reachable so shutdown can
            # clean actors created before a constructor failure.
            manager = TeacherModelManager.__new__(TeacherModelManager)
            self._managers[key] = manager
            manager.__init__(cfg, teacher_config, sub_pool)

            client = LLMServerClient(config=self.config, load_balancer_handle=manager.load_balancer_handle)
            max_prompt_logprobs = self._max_prompt_logprobs(teacher_config)
            self._clients[key] = TeacherClient(
                client,
                model_path=teacher_config.model_path,
                max_prompt_logprobs=max_prompt_logprobs,
            )
            self._register_alias(key, key)
            self._register_alias(self._model_names[key], key)
            self._register_alias(teacher_config.model_path, key)

    def _get_model_names(self, teacher_models: DictConfig) -> dict[str, str]:
        if len(teacher_models) == 1:
            teacher = next(iter(teacher_models.values()))
            return {"default": str(teacher.model_name)}
        return {
            str(teacher.key): str(teacher.model_name)
            for entry_name, teacher in teacher_models.items()
            if entry_name != "teacher_model"
        }

    def _register_alias(self, alias: str | None, key: str) -> None:
        if not alias:
            return
        previous = self._aliases.get(alias)
        if previous is not None and previous != key:
            raise ValueError(f"Teacher identifier {alias!r} is ambiguous between {previous!r} and {key!r}")
        self._aliases[alias] = key

    @staticmethod
    def _max_prompt_logprobs(teacher_config) -> int | None:
        if teacher_config.inference.name != "vllm":
            return None
        vllm_kwargs = teacher_config.inference.engine_kwargs.get("vllm", {})
        return int(vllm_kwargs.get("max_logprobs", 20))

    def resolve(self, *identifiers: str | None) -> str | None:
        matches = {self._aliases[value] for value in identifiers if value in self._aliases}
        if len(matches) > 1:
            raise ValueError(f"Sampling request identifies multiple teachers: {sorted(matches)}")
        return next(iter(matches), None)

    def get_client(self, key: str) -> TeacherClient:
        return self._clients[key]

    def get_model_path(self, key: str) -> str:
        return self.distillation_config.teacher_models[key].model_path

    def shutdown(self) -> None:
        actors = []
        for manager in self._managers.values():
            actors.append(getattr(manager, "load_balancer_handle", None))
            for replica in getattr(manager, "rollout_replicas", []) or []:
                actors.extend(getattr(replica, "servers", []) or [])
                actors.append(getattr(replica, "_server_handle", None))
                workers = getattr(replica, "workers", None)
                if workers is not None:
                    actors.extend(workers)

        placement_groups = list(getattr(self._resource_pool, "pgs", None) or [])
        try:
            kill_ray_actors_and_wait(actors, logger=logger, description="teacher backend", ray_module=ray)
        finally:
            try:
                remove_placement_groups_and_wait(
                    placement_groups,
                    logger=logger,
                    description="teacher backend",
                    ray_module=ray,
                )
            finally:
                self._clients.clear()
                self._managers.clear()
                self._aliases.clear()
                self._resource_pool = None
