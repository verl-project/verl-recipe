# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""m3 prerequisite smoke (recipe/dynamo step4_feasibility.md §7).

Validates the load-bearing assumption for route A (actor-as-worker):

    Can dynamo's `create_runtime(discovery_backend="mem", ...)`
    coexist with a Ray actor's asyncio event loop?

Background:
    - dynamo.common.utils.runtime.create_runtime calls
      asyncio.get_running_loop() and hands that loop to the Rust
      DistributedRuntime constructor (PyO3 binding). It does NOT spin
      its own event loop.
    - dynamo.vllm.main uses `uvloop.run(worker())` only because it
      runs as a standalone process; the dynamo runtime itself does not
      require uvloop or loop ownership.
    - Therefore the runtime should be reentrant on whatever loop the
      caller already owns — including a Ray actor's loop. This script
      verifies that empirically before committing to route A.

Pass criteria:
    1. DistributedRuntime constructs without error inside a Ray actor.
    2. runtime.endpoint(...) returns a usable Endpoint object.
    3. runtime.shutdown() returns cleanly; the actor stays alive.

Run:
    python recipe/dynamo/scripts/smoke_runtime_in_ray_actor.py
"""

from __future__ import annotations

import asyncio
import sys

import ray


@ray.remote(num_cpus=1)
class DynamoRuntimeProbe:
    """Minimal Ray actor that tries to host a dynamo DistributedRuntime."""

    async def probe(self) -> dict:
        result: dict = {
            "loop_id_before": id(asyncio.get_running_loop()),
            "create_runtime_ok": False,
            "endpoint_ok": False,
            "shutdown_ok": False,
            "loop_id_after": None,
            "error": None,
        }
        try:
            from dynamo.common.utils.runtime import create_runtime

            runtime, loop = create_runtime(
                discovery_backend="mem",
                request_plane="tcp",
                event_plane="zmq",
                use_kv_events=False,
            )
            result["create_runtime_ok"] = True
            result["loop_id_from_runtime"] = id(loop)

            endpoint = runtime.endpoint("verl_smoke.probe.generate")
            result["endpoint_ok"] = endpoint is not None
            result["endpoint_repr"] = repr(endpoint)

            runtime.shutdown()
            result["shutdown_ok"] = True
        except Exception as e:  # noqa: BLE001 — we want everything for triage
            import traceback

            result["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        result["loop_id_after"] = id(asyncio.get_running_loop())
        return result


def main() -> int:
    ray.init(num_cpus=2, log_to_driver=True, ignore_reinit_error=True)
    try:
        probe = DynamoRuntimeProbe.remote()
        result = ray.get(probe.probe.remote(), timeout=60)
    finally:
        ray.shutdown()

    print("=" * 60)
    print("dynamo runtime in Ray actor — smoke result")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    ok = result["create_runtime_ok"] and result["endpoint_ok"] and result["shutdown_ok"]
    if ok:
        print("PASS — route A m3 unblocked")
        return 0
    print("FAIL — route A blocked; consider route C (headless) or B (subprocess)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
