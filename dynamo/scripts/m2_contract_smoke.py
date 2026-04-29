#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""m2 verification: the recipe-side dynamo classes inherit from verl's vLLM
async server stack and expose the RPC contract ServerAdapter expects.

CPU-only, no Ray cluster needed. We do NOT instantiate the Ray actor (that
needs the GPU + a full launch); we only check class structure, MRO, and
the methods ServerAdapter._execute_method / update_weights / resume / release
will call at runtime.
"""

from __future__ import annotations

import inspect
import sys
import traceback


def assert_(cond: bool, msg: str) -> None:
    if not cond:
        print(f"[m2] FAIL: {msg}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[m2] ok: {msg}")


def main() -> None:
    import recipe.dynamo  # noqa: F401  side-effect: registration

    from verl.workers.rollout.replica import RolloutReplicaRegistry
    from verl.workers.rollout.vllm_rollout.vllm_async_server import (
        vLLMHttpServer,
        vLLMReplica,
    )
    from verl.workers.rollout.vllm_rollout.vllm_rollout import (
        ServerAdapter as VllmServerAdapter,
    )

    from recipe.dynamo.dynamo_async_server import DynamoHttpServer, DynamoReplica
    from recipe.dynamo.dynamo_rollout import ServerAdapter

    # 1. Inheritance: the three classes are vLLM subclasses (not duplicates).
    assert_(issubclass(ServerAdapter, VllmServerAdapter), "ServerAdapter <: vLLM ServerAdapter")
    assert_(issubclass(DynamoHttpServer, vLLMHttpServer), "DynamoHttpServer <: vLLMHttpServer")
    assert_(issubclass(DynamoReplica, vLLMReplica), "DynamoReplica <: vLLMReplica")

    # 2. RolloutReplicaRegistry.get('dynamo') returns DynamoReplica (not vLLMReplica).
    assert_(
        RolloutReplicaRegistry.get("dynamo") is DynamoReplica,
        "RolloutReplicaRegistry.get('dynamo') is DynamoReplica",
    )

    # 3. RPC surface: every method ServerAdapter._execute_method / update_weights
    #    routes to via .remote() must exist on DynamoHttpServer.
    REQUIRED_RPC_METHODS = [
        # called via collective_rpc (server-side .collective_rpc.remote(method,...))
        "collective_rpc",
        # called directly via ray actor handle by ServerAdapter
        "clear_kv_cache",
        "set_global_steps",
        # called by RolloutReplica orchestrators (sleep/wake_up/abort etc.)
        "wake_up",
        "sleep",
        # bring-up
        "launch_server",
        "get_server_address",
        "get_master_address",
    ]
    for name in REQUIRED_RPC_METHODS:
        method = getattr(DynamoHttpServer, name, None)
        assert_(callable(method), f"DynamoHttpServer.{name} exists and is callable")

    # 4. Actor name prefix is dynamo_ (so ServerAdapter and DynamoReplica meet
    #    on the same Ray actor name). DynamoReplica overrides the hardcoded
    #    "vllm_" from vLLMReplica.
    src = inspect.getsource(DynamoReplica._get_server_name_prefix)
    assert_('"dynamo_"' in src or "'dynamo_'" in src,
            "DynamoReplica._get_server_name_prefix returns 'dynamo_'")

    # 5. Replica.__init__ rebinds server_class to recipe-side actor class.
    #    We don't actually call __init__ (would need a full RolloutConfig);
    #    instead we read the source and assert the rebinding statement is present.
    init_src = inspect.getsource(DynamoReplica.__init__)
    assert_("DynamoHttpServer" in init_src,
            "DynamoReplica.__init__ rebinds server_class to DynamoHttpServer")

    # 6. ServerAdapter side: _get_server_name_prefix reads config.name (so when
    #    config.name == 'dynamo' it produces 'dynamo_' that matches Replica).
    sa_prefix_src = inspect.getsource(VllmServerAdapter._get_server_name_prefix)
    assert_("config" in sa_prefix_src and "name" in sa_prefix_src,
            "vllm ServerAdapter._get_server_name_prefix reads config.name (inherited)")

    print("[m2] PASS")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        sys.exit(2)
