import logging
import os

import psutil
from codetiming import Timer
from omegaconf import OmegaConf, open_dict

from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.device import get_device_id, get_torch_device
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import DistProfiler, log_gpu_memory_usage
from verl.workers.config import FSDPEngineConfig
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    AsyncActorRolloutRefWorker,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class OPKDWorker(ActorRolloutRefWorker):
    """
    Synchronous OPKD worker.

    Extends ActorRolloutRefWorker:
      - Replace actor wrapper with DataParallelOPKDActor.
      - Replace ref wrapper with DataParallelOPKDActor using teacher_path.
      - Add compute_teacher_log_prob and compute_student_topk_index RPCs.
      - Override update_actor to keep OPKD-specific behavior.
      - Inline and modify init_model to avoid omega_conf_to_dataclass.
    """

    def __init__(self, config, role: str, **kwargs):
        super().__init__(config, role, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Initialization order:
          1. Import external libraries for HF models.
          2. Build actor model (optionally optimizer) and optionally rollout model.
          3. Build ref model using teacher_path when ref role is enabled.
          4. Wrap actor/ref with DataParallelOPKDActor.
          5. Create FlopsCounter and FSDPCheckpointManager.

        Differences from the base class:
          - No omega_conf_to_dataclass calls.
          - Actor/ref wrappers are DataParallelOPKDActor instead of DataParallelPPOActor.
          - Ref model path is taken from config.ref.teacher_path by default.
        """
        from dp_actor import DataParallelOPKDActor

        # Import external libs into the HF ecosystem
        import_external_libs(self.config.model.get("external_lib", None))

        override_model_config = OmegaConf.to_container(OmegaConf.create(self.config.model.get("override_config", {})))
        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        # Build actor (and rollout) model
        if self._is_actor or self._is_rollout:
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config  # no dataclass conversion
            else:
                optim_config = None
                fsdp_config = FSDPEngineConfig()  # default config for rollout-only mode

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get("enable_gradient_checkpointing", False),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get("enable_activation_offload", False),
            )

            # Get original unwrapped module for FSDP v1
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            # Optional offload after init
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage("After offload actor model during init", logger=logger)

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage("After offload actor optimizer during init", logger=logger)

        # Wrap actor with OPKD actor
        if self._is_actor:
            actor_cfg = self.config.actor  # keep as DictConfig
            self.actor = DataParallelOPKDActor(
                config=actor_cfg,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )

        # Build rollout engine
        if self._is_rollout:
            self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))

        # Build reference (teacher) model
        if self._is_ref:
            # Prefer teacher_path as the ref model path if provided
            if hasattr(self.config.ref, "teacher_path"):
                ref_model_path = self.config.ref.teacher_path
            else:
                ref_model_path = self.config.model.path

            ref_model = self.config.ref.get("model", None)
            if ref_model is not None:
                ref_model_path = ref_model.get("path", self.config.model.path)

            if self.rank == 0:
                print("reference model:", ref_model_path)

            local_path = copy_to_local(ref_model_path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,  # no dataclass conversion
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
            )[0]

            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels

            self.ref_policy = DataParallelOPKDActor(
                config=self.config.ref,
                actor_module=self.ref_module_fsdp,
                actor_optimizer=None,
            )

        # Create FlopsCounter and checkpoint manager for actor
        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=self.config.actor.checkpoint,
            )

        # Standalone rollout checkpoint manager
        if not self._is_actor and self._is_rollout:
            checkpoint_contents = OmegaConf.create({"load_contents": ["model"], "save_contents": []})
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=None,
                lr_scheduler=None,
                processing_class=self.processor if self.processor is not None else self.tokenizer,
                checkpoint_config=checkpoint_contents,
            )

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_teacher_log_prob(self, data: DataProto):
        """
        Compute teacher (ref) log probabilities and union logits.

        Requires DataParallelOPKDActor to implement compute_union_logits.

        Returns a DataProto with:
          - merged_indices
          - merged_logits
          - teacher_token_logp
        """
        if self._is_lora:
            data.meta_info["is_lora"] = True
            out = self.compute_log_prob(data)
            return DataProto.from_dict(tensors={"ref_log_prob": out.batch["old_log_probs"]})

        assert self._is_ref, "compute_teacher_log_prob requires ref role"

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz

        with self.ulysses_sharding_manager:
            data = data.to("cpu")
            with Timer(name="compute_teacher_log_prob", logger=None):
                (
                    merged_indices,
                    merged_logits,
                    teacher_token_logp,
                ) = self.ref_policy.compute_union_logits(
                    data=data,
                    calculate_entropy=False,
                )

            output = DataProto.from_dict(
                tensors={
                    "merged_indices": merged_indices,
                    "merged_logits": merged_logits,
                    "teacher_token_logp": teacher_token_logp,
                }
            )

        output = output.to("cpu")

        if self.world_size > 1:
            if fsdp_version(self.ref_policy.actor_module) == 1:
                self.ref_policy.actor_module._handle.reshard(True)
            elif fsdp_version(self.ref_policy.actor_module) == 2:
                self.ref_policy.actor_module.reshard()

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_student_topk_index(self, data: DataProto):
        """
        Compute student (actor) top-k indices.

        Requires DataParallelOPKDActor to implement compute_student_index.

        Returns a DataProto with:
          - student_topk_index
        """
        assert self._is_actor, "compute_student_topk_index requires actor role"

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = self.actor.actor_module.disable_adapter() if is_lora else nullcontext()

        data.meta_info["micro_batch_size"] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info["max_token_len"] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature

        with self.ulysses_sharding_manager, adapter_ctx:
            student_index = self.actor.compute_student_index(
                data=data,
                calculate_entropy=False,
            )
            output = DataProto.from_dict(
                tensors={"student_topk_index": student_index},
                meta_info={"temperature": self.config.rollout.temperature},
            )

        output = output.to("cpu")

        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        return output

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        """
        Update actor parameters using OPKD actor implementation.

        Logic is aligned with the base class update_actor, but uses the custom actor wrapper.
        """
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=get_device_id())

        with self.ulysses_sharding_manager:
            data = data.to("cpu")

            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)

            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics["perf/mfu/actor"] = (
                estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = get_torch_device().max_memory_allocated() / (1024**3)
            metrics["perf/max_memory_reserved_gb"] = get_torch_device().max_memory_reserved() / (1024**3)
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (1024**3)

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.actor_lr_scheduler.step()

            output = DataProto(meta_info={"metrics": metrics}).to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage("After offload actor model during update_actor", logger=logger)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage("After offload actor optimizer during update_actor", logger=logger)

        return output


class AsyncOPKDWorker(AsyncActorRolloutRefWorker):
    """
    Asynchronous OPKD worker.

    Extends AsyncActorRolloutRefWorker:
      - Reuses all async capabilities (wake_up, sleep, chat_completion, generate, etc.).
      - Replaces actor/ref wrappers with DataParallelOPKDActor using the same
        initialization logic as OPKDWorker (without omega_conf_to_dataclass).
      - Exposes the same OPKD-specific RPCs as OPKDWorker:
        * compute_teacher_log_prob
        * compute_student_topk_index
        * update_actor
    """

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Use the same init_model logic as OPKDWorker, but on the async worker.

        This avoids omega_conf_to_dataclass and ensures OPKD-specific actor/ref
        wrappers are used while preserving async worker behavior.
        """
        # Reuse OPKDWorker.init_model implementation on this instance
        OPKDWorker.init_model(self)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_teacher_log_prob(self, data: DataProto):
        """
        Asynchronous worker variant of teacher log-prob RPC.

        Delegates to OPKDWorker.compute_teacher_log_prob.
        """
        return OPKDWorker.compute_teacher_log_prob(self, data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_student_topk_index(self, data: DataProto):
        """
        Asynchronous worker variant of student top-k index RPC.

        Delegates to OPKDWorker.compute_student_topk_index.
        """
        return OPKDWorker.compute_student_topk_index(self, data)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    @DistProfiler.annotate(color="red", role="actor_update")
    def update_actor(self, data: DataProto):
        """
        Asynchronous worker variant of actor update RPC.

        Delegates to OPKDWorker.update_actor.
        """
        return OPKDWorker.update_actor(self, data)
