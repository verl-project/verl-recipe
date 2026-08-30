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
"""ServerAdapter for the dynamo backend.

Inherits the vLLM ServerAdapter (HTTP path is identical: trainer rank reads
``replica.server_address`` and POSTs chat completions to it) and only
overrides the Ray actor name prefix used for sleep/wake/update_weights RPC,
so it lands on ``dynamo_server_*`` (created by DynamoReplica.launch_servers)
rather than ``vllm_server_*``.
"""

import os
from typing import Any, Optional

import ray
import torch

from verl.workers.rollout.vllm_rollout.vllm_rollout import (
    ServerAdapter as _VllmServerAdapter,
)


class ServerAdapter(_VllmServerAdapter):
    """Per-rank dynamo client.

    All HTTP-based generation goes through the frontend URL stored in
    ``RolloutReplica.server_address``; weight-update / wake-up / sleep
    requests go to the per-replica Ray actor named ``dynamo_server_{r}_{n}``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        rank = int(os.environ["RANK"])
        local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        self._dynamo_node_rank = rank // local_world_size
        self._dynamo_node_local_rank = rank % local_world_size
        # Standalone CheckpointEngineWorkers pass an explicit POOL-level
        # replica_rank (hybrid engine workers leave the default -1). The base
        # class then skips its per-logical-replica derivation, so the ZMQ
        # pairing socket must be rebuilt on the shared contract both sides
        # use: replica component = pool offset + pool-local logical replica
        # index (matches the engine-side VERL_REPLICA_RANK injected in
        # _launch_shared_worker_pool), rank component = the base formula
        # (within-replica rank modulo node size, matching the engine-side
        # VERL_DYNAMO_RANK_OFFSET + tp-local rank).
        self._dynamo_standalone = kwargs.get("replica_rank", -1) >= 0
        if self._dynamo_standalone:
            rollout_world_size = (
                self.config.tensor_model_parallel_size
                * self.config.data_parallel_size
                * self.config.pipeline_model_parallel_size
            )
            self._dynamo_pool_offset = self.replica_rank
            logical_replica_rank = rank // rollout_world_size
            self._dynamo_global_replica_rank = self._dynamo_pool_offset + logical_replica_rank
            within_replica_local_rank = (rank % rollout_world_size) % local_world_size
            job_id = ray.get_runtime_context().get_job_id()
            self.zmq_handle = (
                f"ipc:///tmp/rl-colocate-zmq-{job_id}-replica-{self._dynamo_global_replica_rank}"
                f"-rank-{within_replica_local_rank}.sock"
            )

    def _get_server_name_prefix(self) -> str:
        return "dynamo_"

    def _get_control_actor_name(self) -> str:
        """Return the shared Dynamo server actor name for control RPCs."""
        if self._dynamo_standalone:
            # Standalone pool servers are named by THIS pool's replica_rank
            # (offset past the hybrid pool by LLMServerManager.start_rank).
            return f"{self._get_server_name_prefix()}server_{self.replica_rank}_{self._dynamo_node_rank}"
        dynamo_cfg = (self.config.engine_kwargs or {}).get("dynamo", {}) or {}
        shared_replica_rank = int(dynamo_cfg.get("shared_pool_replica_rank", 0))
        return f"{self._get_server_name_prefix()}server_{shared_replica_rank}_{self._dynamo_node_rank}"

    def _is_node_control_rank(self) -> bool:
        """True for the one trainer rank that controls Dynamo on this node."""
        return self._dynamo_node_local_rank == 0

    def _ensure_server_handle(self) -> bool:
        """Lazy-init the shared Dynamo control actor handle."""
        if not self._is_node_control_rank():
            return False
        if self.server_handle is None:
            self.server_handle = ray.get_actor(self._get_control_actor_name())
        return True

    async def _execute_method(
        self,
        method: str,
        non_block: bool = False,
        timeout: Optional[float] = None,
        args: tuple = (),
        kwargs: Optional[dict] = None,
    ) -> Any:
        """Execute control RPCs against the shared Dynamo pool actor.

        Native vLLM has one named server actor per rollout replica. All logical rollout
        replicas on a node share ``dynamo_server_0_<node_rank>``.

        Gate on one trainer rank per physical node. Each node has one shared
        DynamoHttpServer actor whose collective_rpc broadcasts to the node-local
        sidecars. Firing once per logical replica would duplicate sidecar RPCs;
        firing only on global rank 0 would miss non-master nodes.
        """
        if not self._is_node_control_rank():
            return None

        if self.server_handle is None:
            self.server_handle = ray.get_actor(self._get_control_actor_name())

        future = self.server_handle.collective_rpc.remote(
            method,
            timeout=timeout,
            args=args,
            kwargs=kwargs,
        )
        return future if non_block else await future

    @torch.no_grad()
    async def update_weights(self, weights, global_steps=None, wire_format: str = "named_tensors", **kwargs):
        # Consumed here (checkpoint-engine workers pass it unconditionally);
        # never forward it into the update_weights_from_ipc RPC kwargs — the
        # engine extension's signature would reject it.
        assert wire_format == "named_tensors", (
            f"dynamo rollout only consumes full named tensors; got wire_format={wire_format!r}"
        )
        import asyncio
        import time as _time

        from verl.workers.rollout.vllm_rollout.bucketed_weight_transfer import (
            BucketedWeightSender,
        )

        t_enter = _time.time()
        tag = f"[v4a-4][rank={self.rollout_rank}]"
        print(f"{tag} ENTER update_weights", flush=True)

        # Fire RPC (only rank 0 actually fires per parent _execute_method gating).
        future = await self._execute_method(
            "update_weights_from_ipc",
            non_block=True,
            kwargs={**kwargs, "use_shm": self.use_shm},
        )
        print(
            f"{tag} RPC fired +{_time.time() - t_enter:.2f}s "
            f"future={'present' if future is not None else 'None (non-rank-0)'}",
            flush=True,
        )

        # Build sender (every rank has its own zmq_handle to its paired
        # engine worker; receiver setup on engine side is triggered by
        # rank 0's RPC, but all ranks then send via their own pair).
        bucket_size_mb = self.config.checkpoint_engine.update_weights_bucket_megabytes
        sender = BucketedWeightSender(
            zmq_handle=self.zmq_handle,
            bucket_size_mb=bucket_size_mb,
            use_shm=self.use_shm,
        )
        print(f"{tag} sender ready zmq_handle={self.zmq_handle}", flush=True)

        sender_task = asyncio.create_task(sender.async_send_weights(weights))

        # Race future vs sender for 60s. If future errors fast, we catch it.
        if future is not None:
            future_task = asyncio.ensure_future(future)
            done, pending = await asyncio.wait(
                {sender_task, future_task},
                timeout=60,
                return_when=asyncio.FIRST_COMPLETED,
            )
            elapsed = _time.time() - t_enter
            print(
                f"{tag} race done={len(done)} pending={len(pending)} +{elapsed:.2f}s",
                flush=True,
            )
            for t in done:
                which = "future" if t is future_task else "sender"
                if t.exception():
                    err = t.exception()
                    print(
                        f"{tag} {which} ERROR: {type(err).__name__}: {err}",
                        flush=True,
                    )
                    for p in pending:
                        p.cancel()
                    raise err
                print(f"{tag} {which} completed OK", flush=True)

            # Continue waiting for whatever is still pending
            if pending:
                print(f"{tag} waiting for {len(pending)} pending task(s)...", flush=True)
                more_done, more_pending = await asyncio.wait(
                    pending,
                    timeout=600,
                    return_when=asyncio.ALL_COMPLETED,
                )
                if more_pending:
                    print(
                        f"{tag} TIMEOUT: {len(more_pending)} task(s) still pending "
                        f"after 600s +{_time.time() - t_enter:.2f}s",
                        flush=True,
                    )
                    for p in more_pending:
                        p.cancel()
                    raise RuntimeError("v4a-4 hung: tasks still pending after 660s total")
                for t in more_done:
                    which = "future" if t is future_task else "sender"
                    if t.exception():
                        err = t.exception()
                        print(
                            f"{tag} {which} ERROR (late): {type(err).__name__}: {err}",
                            flush=True,
                        )
                        raise err
                    print(f"{tag} {which} completed OK (late)", flush=True)
        else:
            # Non-rank-0: only sender (no future to race against).
            await sender_task
            print(f"{tag} sender DONE (non-rank-0) +{_time.time() - t_enter:.2f}s", flush=True)

        # Cleanup once per node: every shared Dynamo actor owns node-local
        # sidecars and caches.
        if self._is_node_control_rank():
            try:
                await asyncio.wait_for(
                    self.server_handle.clear_kv_cache.remote(),
                    timeout=30,
                )
                print(f"{tag} kv cache cleared", flush=True)
            except asyncio.TimeoutError:
                print(
                    f"{tag} clear_kv_cache TIMEOUT 30s (continuing; prefix cache may be stale)",
                    flush=True,
                )
            if global_steps is not None:
                try:
                    await asyncio.wait_for(
                        self.server_handle.set_global_steps.remote(global_steps),
                        timeout=10,
                    )
                except asyncio.TimeoutError:
                    print(f"{tag} set_global_steps TIMEOUT 10s", flush=True)

        print(f"{tag} EXIT +{_time.time() - t_enter:.2f}s", flush=True)


__all__ = ["ServerAdapter"]
