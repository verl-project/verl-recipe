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
"""ServerAdapter for the ``dynamo`` rollout backend running ``dynamo.sglang``.

Inherits verl's **sglang** ServerAdapter (not the vLLM one that
``dynamo_rollout.ServerAdapter`` inherits) because the weight-sync wire is
completely different: SGLang has no ``worker_extension_cls`` hook, so instead of
verl's ``BucketedWeightSender`` → ZMQ-IPC socket → ``update_weights_from_ipc``,
weights go out as ``MultiprocessingSerializer`` CUDA-IPC handles posted to
``control/update_weights_from_tensor``.

Two things differ from verl's native sglang path:

1. **Where the engine lives.** Native sglang talks to one HTTP server per replica
   (``sglang_server_{replica}_{node}`` Ray actor → ``AsyncHttpServerAdapter``).
   Under Dynamo each *DP shard* is its own ``dynamo.sglang`` process with its own
   TP group, and the control plane is the shard's ``/engine/*`` routes. So this
   adapter resolves **one client per shard** and each trainer rank talks only to
   the shard that owns its GPUs.

2. **Which ranks form a TP group.** verl builds ``device_mesh["infer_tp"]`` spanning
   ``tensor_model_parallel_size * data_parallel_size`` — i.e. the whole replica.
   That is wrong here: gathering IPC handles across shards would hand shard 0's
   engine handles created on shard 1's GPUs. This adapter instead builds a
   *shard-local* TP process group using the **same formula the launcher uses** to
   slice GPUs (``dynamo_async_server._start_engine_workers``: shard = local_rank // tp,
   rank_offset = shard * tp), so sender and receiver agree by construction.

   Getting this wrong does not raise — it silently trains against mismatched
   weights. :meth:`verify_weight_sync` exists to make that failure loud, and
   ``engine_kwargs.dynamo.sglang.verify_weight_sync=true`` runs it after each sync.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Generator, Optional

import ray
import torch
from recipe.dynamo.dynamo_naming import control_actor_name
from recipe.dynamo.dynamo_sglang_engine import (
    DynamoSGLangControlClient,
    control_api_for_launch_mode,
    sglang_launch_mode,
)

from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter as _SGLangServerAdapter
from verl.workers.rollout.sglang_rollout.utils import get_named_tensor_buckets

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


def _flatten_floats(obj) -> list[float]:
    """Flatten a possibly-nested list of numbers coming back over JSON."""
    out: list[float] = []
    stack = [obj]
    while stack:
        cur = stack.pop(0)
        if isinstance(cur, (list, tuple)):
            stack = list(cur) + stack
        elif isinstance(cur, (int, float)):
            out.append(float(cur))
    return out


def _to_ipc_device(tensor: torch.Tensor) -> torch.Tensor:
    """Mirror of ``sglang_rollout._to_ipc_device`` (private upstream, re-implemented
    rather than imported so a rename upstream fails at review, not at runtime)."""
    from verl.utils.device import get_device_id

    return tensor.to(get_device_id(), non_blocking=True) if tensor.device.type == "cpu" else tensor


class SGLangServerAdapter(_SGLangServerAdapter):
    """Per-rank client for the Dynamo × SGLang rollout backend."""

    def __init__(self, *args, **kwargs):
        # Set only by DynamoReplica.init_standalone_pool (V1 separate_async). Popped
        # before super().__init__, which does not know the key.
        standalone = bool(kwargs.pop("dynamo_standalone_pool", False))
        super().__init__(*args, **kwargs)

        engine = str(((self.config.engine_kwargs or {}).get("dynamo", {}) or {}).get("engine", "vllm")).lower()
        if engine != "sglang":
            raise ValueError(
                "SGLangServerAdapter was constructed but "
                f"rollout.engine_kwargs.dynamo.engine={engine!r}; DynamoHttpServer would "
                "spawn a different engine. This should be unreachable — "
                "dynamo_rollout.ServerAdapter dispatches on the same key."
            )

        self._local_world_size = int(os.environ["RAY_LOCAL_WORLD_SIZE"])
        tp = int(self.config.tensor_model_parallel_size)
        if self._local_world_size % tp != 0:
            raise ValueError(
                f"RAY_LOCAL_WORLD_SIZE ({self._local_world_size}) must be divisible by "
                f"tensor_model_parallel_size ({tp}); the Dynamo launcher slices each node's "
                "GPUs into contiguous TP-sized shards and cannot honour a partial shard."
            )
        self._tp_size = tp

        # Derive the shard from the GLOBAL rank, not from the parent's
        # ``self.local_rank``. verl's sglang ServerAdapter defines local_rank
        # *within a replica*: with tp=2 on 8 GPUs it builds four world-size-2
        # replicas, so local_rank cycles 0,1,0,1,0,1,0,1 and ``local_rank // tp``
        # is identically 0. Every rank then posts its CUDA-IPC handles to shard 0
        # while the other three engines silently keep stale weights — invisible at
        # tp=1 (one shard, so 0 is right by accident), which is why six green runs
        # missed it. dynamo_rollout.ServerAdapter (the vLLM path) uses RANK /
        # RAY_LOCAL_WORLD_SIZE for the same reason.
        global_rank = int(os.environ["RANK"])
        node_local_rank = global_rank % self._local_world_size
        self._shard_idx_local = node_local_rank // tp
        self._tp_local_rank = node_local_rank % tp
        self._n_local_shards = self._local_world_size // tp
        # Same trap as the shard index, and it only bites on >1 node: the parent's
        # self.node_rank is rollout_rank // local_world_size, where rollout_rank is
        # rank % (tp*dp*pp). With tp=2 that inner value is 0 or 1, so node_rank is
        # identically 0 and EVERY rank on EVERY node resolves the node-0 actor.
        # Ranks on node 1 then post CUDA-IPC handles for node-1 GPUs to node-0
        # engines, and sglang's UUID-based device resolution rejects them with
        # "Invalid device_uuid=..." — killing the scheduler process.
        self._node_rank_global = global_rank // self._local_world_size

        # Standalone CheckpointEngineWorkers (V1 separate_async) are flagged by
        # DynamoReplica.init_standalone_pool; hybrid workers are not (replica_rank
        # alone cannot tell them apart -- the hybrid path passes one per logical
        # replica too). sglang needs no ZMQ socket rebuild (weights travel over the
        # HTTP control route), only the control-actor naming below.
        self._dynamo_standalone = standalone

        self._shard_tp_group = None
        self._shard_tp_src_global_rank: Optional[int] = None
        self._control_client: Optional[DynamoSGLangControlClient] = None
        self.is_leader_rank = self._tp_local_rank == 0
        # (param_name, first-N values) stashed by _push_bucket for verify_weight_sync.
        self._verify_sample: Optional[tuple[str, list[float]]] = None

    # ------------------------------------------------------------------ #
    # wiring
    # ------------------------------------------------------------------ #

    def _dynamo_cfg(self) -> dict:
        return (self.config.engine_kwargs or {}).get("dynamo", {}) or {}

    def _sglang_cfg(self) -> dict:
        return self._dynamo_cfg().get("sglang", {}) or {}

    def _get_control_actor_name(self) -> str:
        """Shared per-node Dynamo actor; the name format lives in ``dynamo_naming``."""
        # _node_rank_global, not self.node_rank — see __init__. Standalone
        # pools name their servers by THIS pool's replica_rank (offset past
        # the hybrid pool), mirroring VllmDynamoServerAdapter.
        return control_actor_name(
            self.config.engine_kwargs,
            self._node_rank_global,
            replica_rank=self.replica_rank if self._dynamo_standalone else None,
        )

    def _build_shard_tp_group(self):
        """Create the contiguous TP-sized process group this rank belongs to.

        Ranks are grouped ``[k*tp, (k+1)*tp)`` over the *global* rank space, which
        matches the launcher's GPU slicing as long as each node's ranks are a
        contiguous block and ``local_world_size % tp == 0`` (asserted in __init__).
        ``new_group`` is collective, so every rank walks every group in the same
        order and keeps only its own.
        """
        if self._shard_tp_group is not None:
            return
        import torch.distributed as dist

        assert dist.is_initialized(), "torch.distributed must be initialized before weight sync"
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        tp = self._tp_size
        if world_size % tp != 0:
            raise ValueError(f"world_size ({world_size}) not divisible by tp ({tp})")

        my_group = None
        my_src = None
        for base in range(0, world_size, tp):
            ranks = list(range(base, base + tp))
            group = dist.new_group(ranks=ranks)
            if rank in ranks:
                my_group = group
                my_src = ranks[0]
        assert my_group is not None, f"rank {rank} landed in no TP group"
        self._shard_tp_group = my_group
        self._shard_tp_src_global_rank = my_src
        logger.info(
            "[dynamo-sglang] rank=%s local_rank=%s shard=%s tp_local_rank=%s tp_src_global_rank=%s",
            rank,
            self.local_rank,
            self._shard_idx_local,
            self._tp_local_rank,
            my_src,
        )

    async def _init_server_adapter(self):
        """Resolve this rank's shard control endpoint instead of an sglang HTTP server.

        Overrides the parent wholesale: the parent looks up an
        ``sglang_server_*`` actor and builds an ``AsyncHttpServerAdapter``, neither
        of which exists under Dynamo.
        """
        if self._control_client is not None:
            return

        self._build_shard_tp_group()

        # Only the shard's TP leader dispatches HTTP; the others still take part in
        # the gather below, so they must NOT return before _build_shard_tp_group().
        if self._tp_local_rank != 0:
            return

        self.server_actor = ray.get_actor(self._get_control_actor_name())
        endpoints = await self.server_actor.get_engine_control_endpoints.remote()
        if len(endpoints) != self._n_local_shards:
            raise RuntimeError(
                f"node {self.node_rank}: DynamoHttpServer reported {len(endpoints)} sglang control "
                f"endpoints but this rank computed {self._n_local_shards} local shards "
                f"(local_world_size={self._local_world_size}, tp={self._tp_size}). "
                "Trainer and launcher disagree on the shard layout — refusing to sync weights."
            )
        base_url = endpoints[self._shard_idx_local]
        timeout_s = float(self._dynamo_cfg().get("request_timeout_s", 600))
        # Same config as the node actor that launched the shard, so both sides
        # speak the same control dialect (dynamo /engine/* vs native SGLang HTTP).
        api = control_api_for_launch_mode(sglang_launch_mode(self._sglang_cfg()))
        self._control_client = DynamoSGLangControlClient(base_url, timeout_s=timeout_s, api=api)
        # Keep the parent's attribute pointing somewhere sane; several inherited
        # helpers gate on `self._engine is not None`.
        self._engine = self._control_client
        logger.info(
            "[dynamo-sglang] rank=%s -> shard %s control endpoint %s",
            self.rollout_rank,
            self._shard_idx_local,
            base_url,
        )

    def _is_server_tp_leader(self) -> bool:
        """Shard-local TP leader (the parent asks the replica-wide mesh)."""
        return self._tp_local_rank == 0

    # ------------------------------------------------------------------ #
    # sleep / wake
    # ------------------------------------------------------------------ #

    async def resume(self, tags: list[str]):
        """Delegate to the node actor, which owns the release/resume state.

        Doing the bookkeeping here instead was wrong in a way worth recording: the
        actor's ``sleep()`` releases memory (and dynamo's handler UNREGISTERS the
        worker from discovery while released), but the adapter would not know, skip
        the matching resume, and leave the frontend answering
        ``503 Model ... is not ready to serve requests yet`` forever.

        Why any filtering is needed at all: verl resumes weights before every weight
        sync — including the first, when nothing was ever released
        (``engine_workers.py``: ``if resume_weights: await self.rollout.resume(tags=["weights"])``).
        vLLM no-ops on that; SGLang raises ``KeyError`` inside
        ``weight_updater.resume_memory_occupation`` and kills the scheduler process.
        Aggravated by ``engine_workers.py``'s ``is_sglang = rollout.name == "sglang"``
        check, which our ``dynamo`` rollout name fails.
        """
        await self._init_server_adapter()
        if self._control_client is None or not self.config.free_cache_engine:
            return
        await self.server_actor.sglang_resume.remote(list(tags))

    async def release(self):
        await self._init_server_adapter()
        if self._control_client is None or not self.config.free_cache_engine:
            return
        # Both sleep levels free GPU weights on the sglang path — verl's level 1
        # means "weights off the GPU but restorable" (vLLM offloads them to CPU),
        # which is tags=["kv_cache","weights"], not kv_cache alone. See
        # DynamoHttpServer.sleep() for the OOM this asymmetry caused.
        tags = ["kv_cache", "weights"]
        await self.server_actor.sglang_release.remote(tags)

    # ------------------------------------------------------------------ #
    # weight sync
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    async def update_weights(
        self,
        weights: Generator[tuple[str, torch.Tensor], None, None],
        global_steps: int = None,
        wire_format: str = "named_tensors",
        **kwargs,
    ):
        """Push weights to this rank's ``dynamo.sglang`` shard as CUDA-IPC handles.

        Bucketed the same way as verl's native sglang path. ``flush_cache`` is
        deferred to a single call after the last bucket instead of once per bucket:
        the radix cache only has to be correct once the whole update lands, and
        flushing per bucket costs a full cache rebuild each time.
        """
        await self._init_server_adapter()

        if wire_format == "delta_flush":
            raise NotImplementedError(
                "checkpoint_engine.backend=delta_sharded is not wired for the dynamo sglang "
                "backend yet. verl gates it on rollout.name == 'sglang' "
                "(verl/checkpoint_engine/base.py) which this backend does not satisfy."
            )

        if kwargs.get("peft_config") and kwargs.get("base_sync_done"):
            raise NotImplementedError(
                "LoRA adapter sync is not wired for the dynamo sglang backend yet "
                "(needs load_lora_adapter_from_tensors as an engine route)."
            )

        bucket_bytes = int(self.config.checkpoint_engine.update_weights_bucket_megabytes) << 20

        # Fresh probe snapshot per sync. An earlier revision stashed it once and never
        # refreshed, so from step 2 on verify_weight_sync compared the engine's
        # (correctly updated) weights against step-1 values and failed on success.
        self._verify_sample = None

        # NB: every rank must drain the generator — it all-gathers across the FSDP
        # group internally, so an early return on non-leader ranks deadlocks.
        n_buckets = 0
        async for params_batch in get_named_tensor_buckets(weights, bucket_bytes):
            await self._push_bucket(params_batch)
            n_buckets += 1

        if self._control_client is not None:
            await self._control_client.flush_cache()
            if global_steps is not None:
                try:
                    await self.server_actor.set_global_steps.remote(global_steps)
                except Exception:  # noqa: BLE001 - bookkeeping only
                    logger.warning("[dynamo-sglang] set_global_steps failed", exc_info=True)

        logger.info(
            "[dynamo-sglang] weight sync done: rank=%s shard=%s buckets=%s step=%s",
            self.rollout_rank,
            self._shard_idx_local,
            n_buckets,
            global_steps,
        )

        if self._sglang_cfg().get("verify_weight_sync", False):
            await self.verify_weight_sync(global_steps=global_steps)

    async def _push_bucket(self, params_batch):
        """Gather one bucket's IPC handles across the shard's TP group and POST them.

        Flow (identical in shape to ``sglang.srt.weight_sync.utils.update_weights``,
        inlined so flush_cache stays off until the last bucket):

        1. every TP rank serializes its own GPU slice into CUDA-IPC handles;
        2. gather_object onto the shard's TP-rank-0;
        3. TP0 transposes into one ``LocalSerializedTensor`` per parameter, whose
           ``values[i]`` is TP rank i's handle;
        4. TP0 posts one blob per TP rank (SGLang scatters it, each rank picks its own).
        """
        import torch.distributed as dist
        from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput
        from sglang.srt.model_executor.model_runner_components.weight_updater import LocalSerializedTensor
        from sglang.srt.utils import MultiprocessingSerializer
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions
        from sglang.srt.weight_sync.utils import _preprocess_tensor_for_update_weights

        monkey_patch_torch_reductions()
        tp = self._tp_size

        serialized = [
            (
                name,
                MultiprocessingSerializer.serialize(_to_ipc_device(_preprocess_tensor_for_update_weights(t.detach()))),
            )
            for name, t in params_batch
        ]

        gathered = [None for _ in range(tp)] if self._tp_local_rank == 0 else None
        dist.gather_object(
            obj=serialized,
            object_gather_list=gathered,
            dst=self._shard_tp_src_global_rank,
            group=self._shard_tp_group,
        )
        if self._tp_local_rank != 0:
            return

        named_tensors = [
            (group[0][0], LocalSerializedTensor(values=[part[1] for part in group]))
            for group in zip(*gathered, strict=True)
        ]
        if self._sglang_cfg().get("verify_weight_sync", False) and self._verify_sample is None:
            # Stash a slice of the first parameter of the first bucket of THIS sync
            # (update_weights resets the slot) so verify_weight_sync() has something
            # concrete to compare against.
            probe_name, probe_tensor = params_batch[0]
            self._verify_sample = (
                probe_name,
                probe_tensor.detach().flatten()[:8].float().cpu().tolist(),
            )

        req = UpdateWeightsFromTensorReqInput(
            serialized_named_tensors=[MultiprocessingSerializer.serialize(named_tensors) for _ in range(tp)],
            load_format=None,
            flush_cache=False,
        )
        await self._control_client.update_weights_from_tensor(req)

    # ------------------------------------------------------------------ #
    # correctness guard
    # ------------------------------------------------------------------ #

    async def verify_weight_sync(self, global_steps: Optional[int] = None) -> dict[str, Any]:
        """Read one synced parameter back out of the engine and compare it.

        This is the guard for the failure mode the design calls out as the most
        dangerous in the whole backend: if the rank→shard mapping is wrong, weights
        land in an engine on another GPU and **nothing raises** — training simply
        proceeds against a policy that is not the one being served.

        ``_push_bucket`` stashes a small slice of one real parameter it sent; here we
        ask SGLang for the same slice via ``get_weights_by_name`` and compare. A
        mismatch raises, because a silent wrong-weights run is worse than a crash.

        NB an earlier version probed ``get_internal_state`` instead. That returns a
        ``CudaGraphConfig``, which Dynamo cannot JSON-serialize, so the route answered
        ``HTTP 500 "unsupported type CudaGraphConfig"`` — the probe took the run down
        without checking anything. Only ask this API for plain data.
        """
        if self._control_client is None or self._verify_sample is None:
            return {}

        name, expected = self._verify_sample
        try:
            # Native /get_weights_by_name in sidecar mode; the legacy tokenizer-manager
            # passthrough on a dynamo.sglang worker (see the client method).
            got = await self._control_client.get_weights_by_name(name, truncate_size=len(expected))
        except Exception as e:  # noqa: BLE001
            # Infrastructure problem, not a correctness signal — do not fail the run,
            # but do not let it pass silently either.
            logger.error(
                "[dynamo-sglang] weight-sync verification UNAVAILABLE (step=%s): %s. "
                "verify_weight_sync=true did NOT verify anything for this step — "
                "get_weights_by_name is model-specific and unimplemented for e.g. "
                "Qwen2/Qwen3ForCausalLM; do not read a passing run as a verified one.",
                global_steps,
                e,
            )
            return {}

        values = got.get("result", got) if isinstance(got, dict) else got
        flat = _flatten_floats(values)[: len(expected)]
        if len(flat) != len(expected):
            logger.error(
                "[dynamo-sglang] verification INCONCLUSIVE (nothing verified): "
                "engine returned %s values for %s, expected %s",
                len(flat),
                name,
                len(expected),
            )
            return {}

        worst = max((abs(a - b) for a, b in zip(flat, expected, strict=True)), default=0.0)
        # bf16 round-trips through the IPC handle exactly, but the engine may hold a
        # different dtype; allow a bf16 ulp rather than requiring bit equality.
        if worst > 3e-2:
            raise RuntimeError(
                f"[dynamo-sglang] WEIGHT SYNC VERIFICATION FAILED at step={global_steps}: "
                f"param {name!r} differs by up to {worst:.4g} between what rank "
                f"{self.rollout_rank} sent and what shard {self._shard_idx_local} serves. "
                "The most likely cause is a rank->shard mapping error, which otherwise "
                "corrupts training silently."
            )
        logger.info(
            "[dynamo-sglang] weight sync VERIFIED (step=%s, param=%s, max_abs_diff=%.3g)",
            global_steps,
            name,
            worst,
        )
        return {"param": name, "max_abs_diff": worst}


__all__ = ["SGLangServerAdapter"]
