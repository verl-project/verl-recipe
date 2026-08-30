"""Recipe-side Dynamo rollout registration for VERL_USE_EXTERNAL_MODULES."""

from __future__ import annotations

import sys

from verl.workers.rollout.base import _ROLLOUT_REGISTRY
from verl.workers.rollout.replica import RolloutReplicaRegistry


def _load_dynamo():
    from recipe.dynamo.dynamo_thunderagent import DynamoThunderAgentReplica

    return DynamoThunderAgentReplica


RolloutReplicaRegistry.register("dynamo", _load_dynamo)
_ROLLOUT_REGISTRY[("dynamo", "async")] = "recipe.dynamo.dynamo_rollout.ServerAdapter"


def _patch_dynamo_llm_server_manager():
    partial = sys.modules.get("recipe.dynamo.dynamo_agent_loop")
    if partial is not None and not hasattr(partial, "DynamoLLMServerManager"):
        # dynamo_agent_loop is mid-import: this is the NORMAL path inside Ray
        # worker processes — ray.remote(DynamoAgentLoopWorker) deserialization
        # imports dynamo_agent_loop, whose own `import verl` triggers this
        # register module re-entrantly (VERL_USE_EXTERNAL_MODULES). Worker
        # processes never instantiate LLMServerManager, so skipping the patch
        # here is harmless; raising would kill every AgentLoopWorker actor.
        # The driver process always imports verl first (via main_dynamo /
        # main_ppo), where the patch applies cleanly.
        return

    from recipe.dynamo.dynamo_agent_loop import DynamoLLMServerManager

    from verl.workers.rollout import llm_server

    llm_server.LLMServerManager = DynamoLLMServerManager
    try:
        from verl.trainer.ppo import ray_trainer

        ray_trainer.LLMServerManager = DynamoLLMServerManager
    except Exception:
        pass
    # The V1 trainer binds LLMServerManager by name at import time. Under
    # VERL_USE_EXTERNAL_MODULES this module runs during `import verl`, i.e.
    # before verl.trainer.ppo.v1.trainer_base is imported, so the by-name
    # import picks up the patched class. If some other path imported the V1
    # trainer first, rebind it explicitly instead of leaving a stale class.
    # (Do NOT import trainer_base eagerly here: its import chain requires the
    # transfer_queue package, which pure-V0 environments may not have.)
    v1_trainer_base = sys.modules.get("verl.trainer.ppo.v1.trainer_base")
    if v1_trainer_base is not None:
        v1_trainer_base.LLMServerManager = DynamoLLMServerManager


_patch_dynamo_llm_server_manager()
