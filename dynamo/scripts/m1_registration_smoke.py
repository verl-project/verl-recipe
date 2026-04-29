#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""m1 verification: importing recipe.dynamo wires the backend into both verl
registries, and the resolved classes are subclasses of the base abstractions.

This is a CPU-only, non-Ray check — it only validates the main-process side
of the registration contract. Ray-actor-subprocess registration is m4 work.

Run from inside a verl-equipped Python env (the dynamo container with
`pip install --no-deps -e <verl repo>` is the canonical case).

Exit code 0 = all assertions pass; non-zero = a specific check failed.
"""

from __future__ import annotations

import sys
import traceback


def assert_(cond: bool, msg: str) -> None:
    if not cond:
        print(f"[m1] FAIL: {msg}", file=sys.stderr)
        raise SystemExit(1)
    print(f"[m1] ok: {msg}")


def main() -> None:
    # 1. Importing the recipe must succeed and run the registration side-effect.
    import recipe.dynamo  # noqa: F401

    from verl.workers.rollout.base import _ROLLOUT_REGISTRY, BaseRollout, get_rollout_class
    from verl.workers.rollout.replica import RolloutReplica, RolloutReplicaRegistry

    # 2. ServerAdapter registry entry is present and points where __init__.py says.
    expected_fqdn = "recipe.dynamo.dynamo_rollout.ServerAdapter"
    assert_(
        ("dynamo", "async") in _ROLLOUT_REGISTRY,
        "_ROLLOUT_REGISTRY contains ('dynamo', 'async')",
    )
    assert_(
        _ROLLOUT_REGISTRY[("dynamo", "async")] == expected_fqdn,
        f"_ROLLOUT_REGISTRY[('dynamo','async')] == {expected_fqdn}",
    )

    # 3. get_rollout_class resolves the FQDN string and the class is a BaseRollout.
    rollout_cls = get_rollout_class("dynamo", "async")
    assert_(
        issubclass(rollout_cls, BaseRollout),
        f"get_rollout_class('dynamo', 'async') -> {rollout_cls.__name__} is BaseRollout subclass",
    )

    # 4. RolloutReplicaRegistry has a 'dynamo' loader and it returns DynamoReplica.
    replica_cls = RolloutReplicaRegistry.get("dynamo")
    assert_(
        issubclass(replica_cls, RolloutReplica),
        f"RolloutReplicaRegistry.get('dynamo') -> {replica_cls.__name__} is RolloutReplica subclass",
    )
    assert_(
        replica_cls.__name__ == "DynamoReplica",
        "replica class is named DynamoReplica",
    )

    # 5. The other backends are still registered (we did not clobber the dict).
    for built_in in ("vllm", "sglang", "trtllm"):
        assert_(
            (built_in, "async") in _ROLLOUT_REGISTRY,
            f"existing backend ({built_in}, 'async') still registered",
        )

    print("[m1] PASS")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001  -- top-level reporter
        traceback.print_exc()
        sys.exit(2)
