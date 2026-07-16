"""Recipe-side Dynamo rollout registration for VERL_USE_EXTERNAL_MODULES."""

from __future__ import annotations

import sys

from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.rollout.replica import RolloutReplicaRegistry


def _load_dynamo():
    from recipe.dynamo.dynamo_async_server import DynamoReplica

    return DynamoReplica


RolloutReplicaRegistry.register("dynamo", _load_dynamo)
_ROLLOUT_REGISTRY[("dynamo", "async")] = "recipe.dynamo.dynamo_rollout.ServerAdapter"


def _patch_dynamo_llm_server_manager():
    partial = sys.modules.get("recipe.dynamo.dynamo_agent_loop")
    if partial is not None and not hasattr(partial, "DynamoLLMServerManager"):
        return

    from recipe.dynamo.dynamo_agent_loop import DynamoLLMServerManager

    from verl.workers.rollout import llm_server

    llm_server.LLMServerManager = DynamoLLMServerManager
    try:
        from verl.trainer.ppo import ray_trainer

        ray_trainer.LLMServerManager = DynamoLLMServerManager
    except Exception:
        pass


_patch_dynamo_llm_server_manager()
