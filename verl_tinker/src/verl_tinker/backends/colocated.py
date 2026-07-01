# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Tinker server engine: FSDP actor + optional rollout on the same Ray placement group.

Mirrors the worker-init logic in VeRL's ``RayPPOTrainer.init_workers()``
(verl/trainer/ppo/ray_trainer.py), but runs independently on the Tinker server
side. Creates Ray worker groups for actor/rollout, initializes vLLM rollout
replicas directly on the SAME GPU pool, and manages a
sleep/wake state machine so FSDP and vLLM can share the GPUs.

This is the only supported Tinker server engine. It owns the cross-request
serialization lock for synchronous training and checkpoint operations.
"""

import asyncio
import logging
import threading
from typing import Any, Optional

import ray
from omegaconf import DictConfig

from verl.checkpoint_engine.base import CheckpointEngineManager
from verl.protocol import DataProtoFuture
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.utils import Role, need_reference_policy
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.import_utils import import_external_libs
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker
from verl.workers.rollout.llm_server import LLMServerClient
from verl.workers.rollout.replica import TokenOutput, get_rollout_replica_class
from verl.workers.utils.padding import no_padding_2_padding

from ..config_utils import is_no_rollout_deployment
from ..schemas import ServerCapabilities
from ._loss import is_ref_in_actor, make_branching_loss
from .backend_utils import kill_ray_actors_and_wait, remove_placement_groups_and_wait


logger = logging.getLogger("ray")


class NoRolloutWorker(TinkerActorRolloutRefWorker):
    """TinkerActorRolloutRefWorker that skips rollout (vLLM) initialization."""

    def _build_rollout(self, **kwargs):
        pass


class ColocatedBackend:
    """Creates and manages VeRL RayWorkerGroups on a single GPU pool.

    Initialization order:
    1. _build_role_cls()          — build role -> worker class mapping
    2. _spawn_worker_groups()     — create resource pool, colocate, spawn
    3. _init_worker_groups()      — init actor/rollout, ref
    4. _init_rollout_replicas()   — vLLM replicas, checkpoint manager

    After initialization, provides operation methods (generate,
    compute_log_prob, forward_backward, optim_step, etc.) that route to the appropriate
    worker group or replica.

    Synchronous training/checkpoint operations are serialized by
    ``_engine_lock``. Tinker router handlers run request work in threads,
    and two concurrent ``/forward_backward`` + ``/save_weights_for_sampler``
    calls can otherwise race on shared replica sleep/wake state.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self._engine_lock = threading.Lock()
        self._no_rollout_deployment: bool = is_no_rollout_deployment(config)
        self.use_kl_loss: bool = config.actor_rollout_ref.actor.get("use_kl_loss", False)
        self.use_kl_in_reward: bool = config.algorithm.get("use_kl_in_reward", False)
        self.use_reference_policy: bool = need_reference_policy(config)
        self._ref_in_actor: bool = is_ref_in_actor(config)
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        engine_config = actor_config.engine
        self._actor_param_offload = bool(engine_config.param_offload)
        self._actor_optimizer_offload = bool(engine_config.optimizer_offload)

        self.actor_rollout_wg: Optional[RayWorkerGroup] = None
        self.ref_policy_wg: Optional[RayWorkerGroup] = None
        self.checkpoint_manager = None
        self.rollout_replicas = []
        self._server_manager = None
        self._resource_pool = None
        self._replicas_awake = True
        self._replica_wake_lock = asyncio.Lock()

        # Ray workers import model.external_lib through HFModelConfig, but rollout
        # registries are process-local, so the backend must import them before
        # resolving custom rollout replica classes.
        external_libs = config.get("external_libs", None)
        if external_libs is not None:
            import_external_libs(list(external_libs))

        role_cls, actor_role = self._build_role_cls()
        all_wg = self._spawn_worker_groups(role_cls)
        self._init_worker_groups(all_wg, actor_role)
        if not self._no_rollout_deployment:
            self._offload_actor_if_enabled("before rollout init")
            self._init_rollout_replicas()
        else:
            logger.info("No-rollout deployment: skipping rollout replica initialization")

    # ==================== Backend protocol — properties ====================

    @property
    def capabilities(self) -> ServerCapabilities:
        return ServerCapabilities(
            use_kl_loss=self.use_kl_loss,
            use_kl_in_reward=self.use_kl_in_reward,
            use_critic=False,
            use_reference_policy=self.use_reference_policy,
            no_rollout_deployment=self._no_rollout_deployment,
        )

    @property
    def world_size(self) -> int:
        return self.actor_rollout_wg.world_size

    @property
    def no_rollout_deployment(self) -> bool:
        return self._no_rollout_deployment

    @property
    def server_manager(self):
        return self._server_manager

    # ==================== Worker init ====================

    def _build_role_cls(self) -> tuple[dict, Role]:
        """Build role -> RayClassWithInitArgs mapping."""
        config = self.config

        # LoRA: ref reuses actor weights with adapters disabled (no extra memory) → ActorRollout.
        # Full fine-tune: ref is a separate frozen copy inside the same worker (2x memory) → ActorRolloutRef.
        # Ref always shares actor_rollout_wg.
        ref_in_actor = is_ref_in_actor(config)
        if self._no_rollout_deployment:
            if need_reference_policy(config):
                raise ValueError("no_rollout_deployment does not support reference policy")
            actor_role = Role.Actor
        else:
            actor_role = (
                Role.ActorRollout if ref_in_actor or not need_reference_policy(config) else Role.ActorRolloutRef
            )

        worker_cls = NoRolloutWorker if self._no_rollout_deployment else TinkerActorRolloutRefWorker
        role_cls = {
            str(actor_role): RayClassWithInitArgs(
                cls=ray.remote(worker_cls),
                config=config.actor_rollout_ref,
                role=str(actor_role),
            ),
        }

        return role_cls, actor_role

    def _spawn_worker_groups(self, role_cls: dict) -> dict[str, RayWorkerGroup]:
        """Create GPU resource pool, colocate workers, and spawn per-role groups.

        Uses a single resource pool instead of RayPPOTrainer's ResourcePoolManager.
        """
        config = self.config

        self._resource_pool = RayResourcePool(
            process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes,
            use_gpu=True,
            max_colocate_count=3,
            name_prefix="gpu_serve",
        )
        worker_dict_cls = create_colocated_worker_cls(class_dict=role_cls)
        wg = RayWorkerGroup(resource_pool=self._resource_pool, ray_cls_with_init=worker_dict_cls)
        return wg.spawn(prefix_set=role_cls.keys())

    def _init_worker_groups(self, all_wg: dict, actor_role: Role):
        """Initialize worker groups in the correct order."""
        config = self.config

        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        # Replace the actor's loss_fn (which init_model just set to
        # ppo_loss at engine_workers.py:579-582) with a wrapper that
        # branches on a TD sentinel ``__loss_mode__``. Runtime hot-swap
        # via ``set_loss_fn`` per-step proved unreliable in this codepath
        # — the actor kept running ppo_loss despite a successful swap
        # call. Setting the wrapper ONCE at init avoids any race between
        # ``set_loss_fn`` and ``forward_backward`` dispatch and any decorator
        # quirks. ``_datums_to_sft_td`` tags its TDs with
        # ``__loss_mode__="sft"``; RL TDs leave it unset and the wrapper
        # falls back to ppo_loss.
        self.actor_rollout_wg.set_loss_fn(make_branching_loss(config))

        # Ref policy always shares actor_rollout_wg:
        # - LoRA: same weights, adapters disabled for ref forward
        # - Full fine-tune: ActorRolloutRef worker holds both actor and ref model state
        if need_reference_policy(config):
            self.ref_policy_wg = self.actor_rollout_wg

    def _offload_actor_if_enabled(self, reason: str):
        """Offload actor state when the resolved actor engine config enables it."""
        if self.actor_rollout_wg is None or not self._actor_param_offload:
            return
        logger.info("[engine] offloading actor %s", reason)
        self.actor_rollout_wg.to(
            device="cpu",
            model=True,
            optimizer=self._actor_optimizer_offload,
            grad=True,
        )

    def _init_rollout_replicas(self):
        """Initialize vLLM rollout replicas and checkpoint manager.

        Creates replicas directly using the same formula as AgentLoopManager._initialize_llm_servers,
        but without creating AgentLoopManager (which lives on the client side).
        """
        config = self.config
        rollout_config = config.actor_rollout_ref.rollout
        model_config = config.actor_rollout_ref.model

        rollout_world_size = (
            rollout_config.tensor_model_parallel_size
            * rollout_config.data_parallel_size
            * rollout_config.pipeline_model_parallel_size
        )
        world_size = self.actor_rollout_wg.world_size
        num_replicas = world_size // rollout_world_size

        replica_class = get_rollout_replica_class(rollout_config.name)

        self.rollout_replicas = [
            replica_class(
                replica_rank=replica_rank,
                config=rollout_config,
                model_config=model_config,
                gpus_per_node=config.trainer.n_gpus_per_node,
            )
            for replica_rank in range(num_replicas)
        ]

        self._run_all([r.init_hybrid(self.actor_rollout_wg) for r in self.rollout_replicas])

        from verl.workers.rollout.llm_server import GlobalRequestLoadBalancer

        servers = {r.server_address: r._server_handle for r in self.rollout_replicas}
        load_balancer = GlobalRequestLoadBalancer.remote(servers=servers)
        self._server_manager = LLMServerClient(config, load_balancer)

        ckpt_engine_config = omega_conf_to_dataclass(rollout_config.checkpoint_engine)
        self.checkpoint_manager = CheckpointEngineManager(
            config=ckpt_engine_config,
            actor_wg=self.actor_rollout_wg,
            replicas=self.rollout_replicas,
        )

        # Bring the engine into a deterministic fully-slept post-init state.
        #
        # The bug being avoided: verl's ``rollout.update_weights`` only resumes
        # the ``weights`` tag, leaving vLLM's KV-cache pool unmapped. A
        # subsequent ``sleep(level=2)`` then crashes with
        # ``cumem_allocator.cpp:209 CUDA Error: invalid argument`` because
        # ``cuMemUnmap`` is called on the never-mapped KV-cache. RL flows
        # avoid this naturally — /asample re-maps KV-cache before the next
        # sleep — but SFT clients (cookbook sl_basic, sdft, ...) call
        # /forward_backward as the very first request and hit the unmapped
        # KV-cache case immediately, plus the same trap exists for any
        # client doing two update_weights cycles back to back.
        #
        # Workaround: at init we (1) warm each replica with a one-token
        # generate so both ``weights`` AND ``kv_cache`` are mapped, then
        # (2) drive ``sleep_replicas`` once. Step (2) succeeds because step
        # (1) ensured everything is mapped, and the engine ends init in
        # ``_replicas_awake=False``. Subsequent ``_ensure_replicas_slept``
        # calls early-return; the first /asample (or full ``update_weights``)
        # re-wakes vLLM and rebuilds the invariant ``sleep is preceded by
        # full wake``.
        self._warmup_replicas()
        self.checkpoint_manager.sleep_replicas()
        self._replicas_awake = False
        logger.info("[engine] post-init transition: warmup + sleep done")

    def _is_vexact_rollout(self) -> bool:
        """vexact's VeXactServer.generate hard-rejects ``max_tokens`` /
        ``max_new_tokens`` in ``sampling_params`` (see open-vexact
        async_server.py:220) and derives its own bound from
        ``rollout.{response_length, prompt_length}``. Used by the
        generate / warmup paths to strip those keys before dispatch.
        """
        return self.config.actor_rollout_ref.rollout.get("name") == "vexact"

    @staticmethod
    def _sanitize_sampling_params_for_vexact(sampling_params: dict) -> dict:
        """Drop the two keys vexact's assertion rejects. Returns a new dict
        so we never mutate the caller's params (the tinker SDK reuses the
        same dict across multiple sample_async calls)."""
        if not sampling_params:
            return sampling_params
        return {
            k: v
            for k, v in sampling_params.items()
            if k not in ("max_tokens", "max_new_tokens", "include_stop_str_in_output")
        }

    def _warmup_replicas(self):
        """Issue one tiny generate per rollout replica to populate vLLM's
        KV-cache pool before any sleep transition can be requested."""
        n = len(self.rollout_replicas)
        if n == 0:
            return
        warmup_params: dict[str, Any] = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "n": 1,
        }
        if not self._is_vexact_rollout():
            # vLLM honours ``max_tokens=1`` to keep warmup cheap; vexact
            # derives its own response budget from rollout config and
            # raises if we pass ``max_tokens`` here.
            warmup_params["max_tokens"] = 1
        # Concurrent dispatch: GlobalRequestLoadBalancer's least-requests
        # policy fans the N concurrent generates out across the N replicas
        # (each acquire bumps the in-flight count before the next picks).
        self._run_all(
            [
                self._server_manager.generate(
                    request_id=f"engine-warmup-{i}",
                    prompt_ids=[1, 2, 3, 4],
                    sampling_params=warmup_params,
                )
                for i in range(n)
            ]
        )
        logger.info(f"[engine] warmed {n} rollout replica(s) for CuMemAllocator")

    def _run_all(self, tasks: list):
        """Run a list of async tasks synchronously."""

        async def run_all():
            await asyncio.gather(*tasks)

        asyncio.run(run_all())

    @staticmethod
    def _wait_for_nonblocking_result(result):
        """Materialize non-blocking VeRL worker-group results."""
        if isinstance(result, DataProtoFuture):
            return result.get()
        if isinstance(result, ray.ObjectRef):
            return ray.get(result)
        if isinstance(result, list) and result and all(isinstance(item, ray.ObjectRef) for item in result):
            return ray.get(result)
        return result

    # ==================== Replica lifecycle ====================

    def _ensure_replicas_slept(self):
        """Sleep vLLM replicas if they are currently awake.

        In the standard VeRL trainer, sleep_replicas() is called after
        generate_sequences() and before training ops (ray_trainer.py:1310).
        In the remote architecture the client's GPUServerReplica.sleep() is a
        no-op, so the server must manage the sleep/wake lifecycle itself.

        Idempotent: skips if replicas are already slept.  This prevents the
        CUDA error that occurs when CuMemAllocator.sleep() is called on
        already-unmapped memory. The complementary case — the first sleep
        firing before any generate has populated the KV-cache pool — is
        handled at engine init by ``_warmup_replicas``.

        No-op in no_rollout_deployment mode (no checkpoint_manager).
        """
        if self._no_rollout_deployment:
            return
        if not self._replicas_awake:
            return
        self.checkpoint_manager.sleep_replicas()
        self._replicas_awake = False

    def _mark_replicas_awake(self):
        """Mark replicas as awake after update_weights() wakes them."""
        self._replicas_awake = True

    async def _ensure_replicas_awake_for_generation(self):
        """Wake rollout replicas once before allowing concurrent generation.

        ``generate`` can be called by many /asample requests at the same time.
        Only the slept -> awake transition needs serialization; once vLLM is
        resident, the requests should flow through the async server concurrently.
        """
        if self._no_rollout_deployment:
            return
        if self._replicas_awake:
            return

        async with self._replica_wake_lock:
            if self._replicas_awake:
                return
            self._offload_actor_if_enabled("before rollout wake")
            result = self.checkpoint_manager.wake_up_replicas()
            if asyncio.iscoroutine(result):
                await result
            self._replicas_awake = True

    # ==================== Operations ====================

    async def generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        """Dispatch a single generation request to a rollout replica (least-requests)."""
        if self._server_manager is None:
            raise RuntimeError("No rollout replicas available")

        await self._ensure_replicas_awake_for_generation()

        if self._is_vexact_rollout():
            sampling_params = self._sanitize_sampling_params_for_vexact(sampling_params)

        return await self._server_manager.generate(
            request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            image_data=image_data,
            video_data=video_data,
        )

    def compute_log_prob(self, data):
        with self._engine_lock:
            self._ensure_replicas_slept()
            return self.actor_rollout_wg.compute_log_prob(data)

    def compute_ref_log_prob(self, data):
        with self._engine_lock:
            if self.ref_policy_wg is None:
                raise RuntimeError("Reference policy not initialized (need_reference_policy is False)")
            self._ensure_replicas_slept()
            return self.ref_policy_wg.compute_ref_log_prob(data)

    def _ensure_ref_log_prob_for_kl_loss(self, data):
        """In case KL is enabled and we do not have ref_log_prob in our input,
        we will have to compute it here"""

        if not self.use_kl_loss:
            return
        loss_mode = tu.get_non_tensor_data(data, "__loss_mode__", "ppo")
        if loss_mode != "ppo" or "ref_log_prob" in data:
            return
        if self.ref_policy_wg is None and not self._ref_in_actor:
            logger.error(
                "[kl_loss] ensure ref_log_prob exit: reference policy is not initialized"
                " and reference also not live in actor"
            )
            raise RuntimeError(
                "KL loss is enabled but reference policy is not initialized; "
                "cannot populate missing ref_log_prob before forward_backward"
            )

        ref_input = data.clone()
        tu.assign_non_tensor_data(ref_input, "compute_loss", False)
        tu.assign_non_tensor_data(ref_input, "calculate_entropy", False)

        # We currently don't support lora, but leave here for potential future expansion
        # if self._ref_in_actor:
        #     tu.assign_non_tensor_data(ref_input, "no_lora_adapter", True)
        #     ref_output = self.actor_rollout_wg.compute_log_prob(ref_input)
        ref_output = self.ref_policy_wg.compute_ref_log_prob(ref_input)
        ref_output = self._wait_for_nonblocking_result(ref_output)
        if ref_output is None:
            logger.error("[kl_loss] ensure ref_log_prob exit: reference policy returned no output")
            raise RuntimeError("Reference policy returned no output while computing ref_log_prob")

        ref_log_prob = None
        for key in ("ref_log_prob", "log_probs"):
            if key in ref_output:
                ref_log_prob = ref_output[key]
                break

        if ref_log_prob is None:
            keys = list(ref_output.keys()) if hasattr(ref_output, "keys") else type(ref_output).__name__
            logger.error("[kl_loss] ensure ref_log_prob exit: missing ref log prob key in output keys=%s", keys)
            raise RuntimeError(
                "Reference policy output must contain 'log_probs' or 'ref_log_prob' "
                f"to populate KL loss input; got {keys}"
            )

        old_log_probs = data["old_log_probs"]
        if ref_log_prob.shape != old_log_probs.shape:
            try:
                ref_log_prob = no_padding_2_padding(ref_log_prob, data)
            except Exception as exc:
                logger.exception("[kl_loss] ensure ref_log_prob exit: failed to convert ref_log_prob shape")
                raise RuntimeError(
                    "Reference policy returned ref_log_prob/log_probs that cannot be converted "
                    f"to response-padded shape {tuple(old_log_probs.shape)}; got {tuple(ref_log_prob.shape)}"
                ) from exc
        if ref_log_prob.shape != old_log_probs.shape:
            logger.error(
                "[kl_loss] ensure ref_log_prob exit: invalid shape after conversion expected=%s got=%s",
                tuple(old_log_probs.shape),
                tuple(ref_log_prob.shape),
            )
            raise RuntimeError(
                "Reference policy returned ref_log_prob/log_probs with invalid shape after conversion: "
                f"expected {tuple(old_log_probs.shape)}, got {tuple(ref_log_prob.shape)}"
            )

        data["ref_log_prob"] = ref_log_prob

    def forward_backward(self, data):
        with self._engine_lock:
            self._ensure_replicas_slept()
            self._ensure_ref_log_prob_for_kl_loss(data)
            result = self.actor_rollout_wg.forward_backward(data)
            return self._wait_for_nonblocking_result(result)

    def optim_step(self, optim_step_params: dict[str, Any] | None = None):
        with self._engine_lock:
            self._ensure_replicas_slept()
            if optim_step_params is None:
                return self.actor_rollout_wg.optimizer_step()
            return self.actor_rollout_wg.optimizer_step(optim_step_params=optim_step_params)

    def update_weights(self):
        """Sleep replicas and sync FSDP weights (which wakes them).

        For the naive backend, checkpoint_manager.update_weights() internally
        does: resume(weights) → push FSDP params → resume(kv_cache), which
        requires replicas to be in the slept state and leaves them awake.

        Raises RuntimeError in no_rollout_deployment mode (no rollout replicas to sync).
        """
        with self._engine_lock:
            if self._no_rollout_deployment:
                raise RuntimeError("update_weights is not available in no_rollout_deployment mode")
            self._ensure_replicas_slept()
            self.checkpoint_manager.update_weights()
            self._mark_replicas_awake()

    def save_checkpoint(self, local_path: str, global_step: int = 0, max_ckpt_to_keep=None, **kwargs):
        with self._engine_lock:
            self.actor_rollout_wg.save_checkpoint(
                local_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep
            )

    def load_checkpoint(self, checkpoint_path: str, zero_optimizer_grad: bool = False, **kwargs):
        with self._engine_lock:
            self.actor_rollout_wg.load_checkpoint(checkpoint_path)
            if zero_optimizer_grad:
                self.actor_rollout_wg.optimizer_zero_grad()

    # ==================== Lifecycle ====================

    def shutdown(self):
        """Kill all Ray worker actors and remove placement groups to release GPU resources.

        Order matters: rollout server actors come down first so vLLM
        releases its GPUs before we drop the placement group, and so
        ``/v1/reset`` can actually re-init in the same Ray cluster.
        Without this the colocated path leaks ``replica._server_handle``
        + the LLMServerManager's load-balancer actor.
        """
        actors_to_kill = []

        # 1. Rollout-side: vLLM replica server actors + the optional load
        #    balancer the LLMServerManager spawns in front of them.
        for replica in self.rollout_replicas:
            actors_to_kill.extend(getattr(replica, "servers", []) or [])
            actors_to_kill.append(getattr(replica, "_server_handle", None))
        if self._server_manager is not None:
            load_balancer = getattr(self._server_manager, "_load_balancer", None)
            if load_balancer is not None:
                actors_to_kill.append(load_balancer)

        # 2. Training worker groups.
        for wg in [self.actor_rollout_wg]:
            if wg is not None:
                actors_to_kill.extend(wg.workers)

        kill_ray_actors_and_wait(actors_to_kill, logger=logger, description="colocated backend", ray_module=ray)

        placement_groups = list(getattr(self._resource_pool, "pgs", None) or [])
        self.actor_rollout_wg = None
        self.ref_policy_wg = None
        self.checkpoint_manager = None
        self.rollout_replicas = []
        self._server_manager = None

        # Remove placement groups so the name can be reused on reinit.
        remove_placement_groups_and_wait(
            placement_groups,
            logger=logger,
            description="colocated backend",
            ray_module=ray,
        )
        self._resource_pool = None

        logger.info("All worker actors killed and placement groups removed")
