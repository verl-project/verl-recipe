# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Copyright 2025 Meituan Ltd. and/or its affiliates
# Copyright 2025 Individual Contributor: Brilliant Hanabi, funrunding
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
"""FSDP/FSDP2 worker implementation for the GKD recipe.

This file adds an FSDP backend to GKD, reusing the following two pieces of
infrastructure:

1. ``recipe.one_step_off_policy.fsdp_workers.DetachActorWorker`` /
   ``DetachAsyncRolloutWorker`` -- already implements the disaggregated
   weight-sync link between actor and rollout via ``ray.util.collective``;
2. ``verl.workers.fsdp_workers.ActorRolloutRefWorker`` -- provides generic
   capabilities such as FSDP model construction, checkpoint management,
   profiler and offload.

GKD has a different training objective from PPO (the former does KL
distillation using teacher top-k logps, the latter uses policy gradient),
so ``update_actor`` and ``async_generate_sequences`` are overridden here.
"""

import logging
import os

import numpy as np
import torch
import torch.distributed
from fsdp_kl_loss import topk_kl_divergence
from omegaconf import DictConfig, OmegaConf
from ray.util.collective import collective
from recipe.one_step_off_policy.distributed_util import vllm_stateless_init_process_group
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    fsdp_version,
    load_fsdp_model_to_gpu,
    offload_fsdp_model_to_cpu,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.profiler import (
    DistProfiler,
    GPUMemoryLogger,
    log_gpu_memory_usage,
    simple_timer,
)
from verl.utils.profiler.performance import gather_timing
from verl.utils.py_functional import append_to_dict
from verl.utils.ray_utils import get_event_loop
from verl.utils.seqlen_balancing import prepare_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.workers.actor import DataParallelPPOActor
from verl.workers.config import HFModelConfig, RolloutConfig
from verl.workers.fsdp_workers import AsyncActorRolloutRefWorker
from verl.workers.rollout import get_rollout_class

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()

__all__ = [
    "FSDPOnPolicyDistillActorWorker",
    "FSDPOnPolicyDistillRolloutWorker",
]


def _ensure_required_actor_rollout_fields(config: DictConfig) -> None:
    """Same as megatron_workers: fill in fields expected by the base class.

    GKD does not use PPO's mini-batch / epoch / router_replay fields, but
    ``ActorRolloutRefWorker.__init__`` still accesses them while normalizing
    the config, so safe default values are provided here.
    """
    is_struct = OmegaConf.is_struct(config) or False
    OmegaConf.set_struct(config, False)
    try:
        if OmegaConf.select(config, "actor") is not None:
            actor_cfg = config.actor
            if OmegaConf.select(actor_cfg, "ppo_mini_batch_size") is None:
                actor_cfg.ppo_mini_batch_size = 1
            if OmegaConf.select(actor_cfg, "ppo_epochs") is None:
                actor_cfg.ppo_epochs = 1
            if OmegaConf.select(actor_cfg, "router_replay") is None:
                actor_cfg.router_replay = OmegaConf.create({"mode": "disabled"})
            # The FSDP base class's _forward_micro_batch reads the following
            # fields; providing default values is sufficient.
            if OmegaConf.select(actor_cfg, "use_remove_padding") is None:
                actor_cfg.use_remove_padding = False
            if OmegaConf.select(actor_cfg, "use_fused_kernels") is None:
                actor_cfg.use_fused_kernels = False
            if OmegaConf.select(actor_cfg, "ulysses_sequence_parallel_size") is None:
                actor_cfg.ulysses_sequence_parallel_size = 1
        if OmegaConf.select(config, "rollout") is not None:
            rollout_cfg = config.rollout
            if OmegaConf.select(rollout_cfg, "n") is None:
                rollout_cfg.n = 1
    finally:
        OmegaConf.set_struct(config, is_struct)


def _to_tensor(value, *, dtype, device):
    """Safely convert numpy / list / tensor fields in ``non_tensor_batch`` to a tensor on ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            # Object array: comes from distributed aggregation or chunk; stack element-wise.
            value = np.stack([np.asarray(x) for x in value], axis=0)
        return torch.from_numpy(np.ascontiguousarray(value)).to(device=device, dtype=dtype)
    # list / tuple of tensors
    if isinstance(value, (list, tuple)) and len(value) > 0 and isinstance(value[0], torch.Tensor):
        return torch.stack(list(value), dim=0).to(device=device, dtype=dtype)
    return torch.as_tensor(value, dtype=dtype, device=device)


class DetachSync(AsyncActorRolloutRefWorker):
    def _get_actor_params(self):
        pass

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def create_weight_sync_group(self, master_address, master_port, rank_offset, world_size):
        rank = torch.distributed.get_rank() + rank_offset
        self._weight_sync_group = vllm_stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            get_torch_device().current_device(),
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL, blocking=False)
    def sync_rollout_weights(self):
        assert (self._is_actor or self._is_rollout) and not self.config.hybrid_engine
        assert hasattr(self, "_weights_info") and self._weights_info is not None

        if self._is_actor and self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        params = self._get_actor_params() if self._is_actor else None

        rollout_name = self.config.rollout.name
        if self._is_rollout:
            if rollout_name == "vllm":
                from verl.utils.vllm.patch import patch_vllm_moe_model_weight_loader

                inference_model = self.rollout.inference_engine.worker.model_runner.model
                patch_vllm_moe_model_weight_loader(inference_model)
            elif rollout_name == "sglang":
                inference_model = self.rollout._engine
            else:
                raise NotImplementedError(f"Unknown rollout name: {rollout_name}")
        loop = get_event_loop()
        for key, shape, dtype in self._weights_info:
            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor:
                assert key in params
                origin_data = params[key]
                if hasattr(origin_data, "full_tensor"):
                    origin_data = origin_data.full_tensor()
                if torch.distributed.get_rank() == 0:
                    tensor.copy_(origin_data)

            if device_name == "npu":
                self._weight_sync_group.broadcast(tensor, src=0, stream=get_torch_device().current_stream())
            else:
                collective.broadcast(tensor, src_rank=0, group_name="actor_rollout")

            if self._is_rollout:
                if rollout_name == "vllm":
                    inference_model.load_weights([(key, tensor)])
                elif rollout_name == "sglang":
                    # first_rank_in_node = self._tp_rank % tp_size_per_node == 0，
                    # Only the first rank within each node (i.e., the local rank is 0) initializes the engine;
                    # engines for other ranks are set to None.

                    if inference_model is not None:
                        loop.run_until_complete(self.update_weights(inference_model, [(key, tensor)]))

        if self._is_actor and self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        get_torch_device().empty_cache()

    async def update_weights(self, inference_engine, params):
        from sglang.srt.weight_sync.utils import update_weights as sgl_update_weights

        await sgl_update_weights(
            engine=inference_engine,
            params_batch=params,
            device_mesh_key="infer_tp",
            device_mesh=self.rollout_device_mesh,
        )

        if self.rollout_device_mesh["infer_tp"].get_local_rank() == 0:
            await inference_engine.flush_cache()


class _GKDFSDPDistillActor(DataParallelPPOActor):
    """GKD KL-distillation training step under FSDP.

    This is a lightweight trainer: it directly runs a forward on
    ``actor_module`` to obtain the full logits and computes KL(P||Q) on the
    teacher's top-k indices. It deliberately does not reuse
    :class:`verl.workers.actor.dp_actor.DataParallelPPOActor`, because that
    class hard-codes the PPO advantages / old_log_probs flow.
    """

    def __init__(self, config, actor_module, actor_optimizer):
        super().__init__(config, actor_module, actor_optimizer)
        self.param_dtype = PrecisionType.to_dtype(
            config.fsdp_config.get("dtype", "bfloat16") if config.get("fsdp_config") else "bfloat16"
        )
        self.grad_clip = config.get("grad_clip", 1.0)

    def _forward_logits(self, micro_batch):
        """Run a single forward and return the full-sequence logits (used for alignment with the teacher tensor)."""
        input_ids = micro_batch["input_ids"]
        attention_mask = micro_batch["attention_mask"]
        position_ids = micro_batch["position_ids"]

        with torch.autocast(device_type=device_name, dtype=self.param_dtype):
            output = self.actor_module(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,
            )

        return output.logits  # (bs, seq_len, vocab)

    @staticmethod
    def _build_calc_kl_mask(micro_batch):
        """Semantically identical to ``recipe.gkd.megatron_workers.forward_backward_batch``:

        ``calc_kl_mask = attention_mask.clone()``
        ``calc_kl_mask[:, :(-response_length - 1)] = False``

        Only positions within ``[-response_length - 1 :]`` that are also true
        in ``attention_mask`` are kept. KL is computed pointwise between the
        student logits and the teacher top-k logps at those positions.
        Returns a tensor of shape (B, S).
        """
        responses = micro_batch["responses"]
        response_length = responses.size(-1)
        calc_kl_mask = micro_batch["attention_mask"].clone().to(torch.bool)
        if calc_kl_mask.size(1) > response_length + 1:
            calc_kl_mask[:, : -(response_length + 1)] = False
        return calc_kl_mask

    def update_policy(self, data: DataProto) -> dict:
        """Run KL-distillation forward / backward and an optimizer step on one mini-batch."""
        self.actor_module.train()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_keys = ["teacher_topk_logps", "teacher_topk_indices"]
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_keys)

        use_dynamic_bsz = bool(self.config.get("use_dynamic_bsz", False))
        if use_dynamic_bsz:
            max_token_len = int(self.config.max_token_len)
            micro_batches, _ = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batch_size = int(self.config.get("micro_batch_size", 1))
            micro_batches = data.split(micro_batch_size)

        n_micro_batch = max(len(micro_batches), 1)
        loss_scale = 1.0 / n_micro_batch

        metrics: dict = {}
        self.actor_optimizer.zero_grad()

        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            # (B, S) boolean mask, keeping only positions in the prompt-tail +
            # response range where the attention mask is active.
            calc_kl_mask = self._build_calc_kl_mask(model_inputs)

            logits = self._forward_logits(model_inputs)  # (B, S, V)
            assert logits.shape[:2] == calc_kl_mask.shape, (
                f"logits / mask shape mismatch: {logits.shape} vs {calc_kl_mask.shape}"
            )
            logits_fp32 = logits.float()

            # teacher_topk_* are written into non_tensor_batch by
            # teacher_utils.get_teacher_knowledge, with shape (B, S, topk),
            # aligned positionally with logits.
            teacher_topk_logps = _to_tensor(
                model_inputs["teacher_topk_logps"], dtype=torch.float32, device=logits_fp32.device
            )
            teacher_topk_indices = _to_tensor(
                model_inputs["teacher_topk_indices"], dtype=torch.long, device=logits_fp32.device
            )
            assert teacher_topk_logps.shape[:2] == calc_kl_mask.shape, (
                f"teacher / mask shape mismatch: {teacher_topk_logps.shape} vs {calc_kl_mask.shape}"
            )

            # Same semantics as the megatron version: only compute KL where
            # calc_kl_mask is true.
            masked_logits = logits_fp32[calc_kl_mask]  # (n_valid, V)
            masked_teacher_logps = teacher_topk_logps[calc_kl_mask]  # (n_valid, topk)
            masked_teacher_indices = teacher_topk_indices[calc_kl_mask]  # (n_valid, topk)

            if masked_logits.numel() == 0:
                kl_loss = logits_fp32.sum() * 0.0  # keep the grad graph
            else:
                per_token_kl = topk_kl_divergence(masked_logits, masked_teacher_logps, masked_teacher_indices)
                kl_loss = per_token_kl.mean()

            (kl_loss * loss_scale).backward()

            append_to_dict(metrics, {"actor/kl_loss": kl_loss.detach().item()})

        grad_norm = self._optimizer_step()
        append_to_dict(
            metrics,
            {"actor/grad_norm": grad_norm.detach().item() if torch.is_tensor(grad_norm) else float(grad_norm)},
        )

        get_torch_device().empty_cache()
        return metrics


class FSDPOnPolicyDistillActorWorker(DetachSync):
    """FSDP version of the GKD actor worker.

    Inherits from ``DetachActorWorker`` to reuse its actor<->rollout
    collective weight sync, then overrides the tail of ``init_model`` to
    install GKD's own trainer, and overrides ``update_actor`` to run the
    KL-distillation path.
    """

    def __init__(self, config: DictConfig, role: str = "actor", **kwargs):
        _ensure_required_actor_rollout_fields(config)
        super().__init__(config=config, role=role, **kwargs)
        assert self._is_actor and not self._is_rollout, "FSDPOnPolicyDistillActorWorker must be actor-only."

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # Go through the base class (ultimately verl's ActorRolloutRefWorker)
        # to fully initialize the model / optimizer / checkpoint.
        super().init_model()
        # Replace self.actor with the GKD custom trainer to avoid triggering
        # the PPO flow.

        actor_cfg = omega_conf_to_dataclass(self.config.actor)
        self.actor = _GKDFSDPDistillActor(
            config=actor_cfg,
            actor_module=self.actor_module_fsdp,
            actor_optimizer=self.actor_optimizer,
        )
        log_gpu_memory_usage("After GKD FSDP actor init", logger=logger)

    def _get_actor_params(self):
        assert self._is_actor
        params = self.actor_module_fsdp.state_dict()
        from verl.utils.model import convert_weight_keys

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )
        return params

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info
        if fsdp_version(self.actor_module_fsdp) == 1:
            from torch.distributed.fsdp.api import ShardedStateDictConfig, StateDictType

            FSDP.set_state_dict_type(
                self.actor_module_fsdp,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )
        params = self._get_actor_params()
        ret = []
        for key, tensor in params.items():
            ret.append((key, tensor.size(), tensor.dtype))
        self._weights_info = ret
        return ret


class FSDPOnPolicyDistillRolloutWorker(DetachSync):
    """FSDP version of the GKD rollout worker.

    Inherits from ``DetachAsyncRolloutWorker`` and already provides:
    * ``set_actor_weights_info`` -- receives weight metadata from the actor side;
    * ``sync_rollout_weights`` -- syncs actor weights to vllm/sglang via collective.broadcast;
    * the rollout engine built by ``ActorRolloutRefWorker.__init__``.

    GKD additionally needs a non-blocking ``async_generate_sequences`` and
    attaches ``timing`` info during generation for the trainer to log; the
    implementation of these two APIs is kept consistent with
    ``recipe.gkd.megatron_workers.MegatronOnPolicyDistillRolloutWorker``.
    """

    # NOTE: the parent class ``DetachAsyncRolloutWorker.__init__`` only
    # accepts (config, role); the same signature is strictly preserved here
    # and no extra kwargs are accepted.
    def __init__(self, config: DictConfig, role: str = "rollout"):
        _ensure_required_actor_rollout_fields(config)
        super().__init__(config=config, role=role)
        assert self.role == "rollout"
        assert self._is_rollout and not self._is_actor and not self._is_ref, (
            "FSDPOnPolicyDistillRolloutWorker must be rollout-only."
        )

        from verl.utils.model import get_generation_config

        self.local_path = copy_to_local(self.config.model.path)
        self.generation_config = get_generation_config(self.local_path)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    @GPUMemoryLogger(role="generate_sequences", logger=logger)
    @DistProfiler.annotate(color="red")
    def generate_sequences(self, prompts: DataProto):
        prompts.batch = prompts.batch.to(get_device_name())
        meta_info = {
            "eos_token_id": self.generation_config.eos_token_id
            if self.generation_config is not None
            else self.tokenizer.eos_token_id,
            "pad_token_id": self.generation_config.pad_token_id
            if self.generation_config is not None
            else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        timing_generate: dict = {}
        with simple_timer("generate_sequences", timing_generate):
            output = self.rollout.generate_sequences(prompts=prompts)

        timing_generate = gather_timing(timing_generate)
        output.meta_info["timing"] = timing_generate
        output = output.to("cpu")
        get_torch_device().empty_cache()
        return output

    @register(
        dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"),
        blocking=False,
    )
    def async_generate_sequences(self, prompts: DataProto):
        return self.generate_sequences(prompts)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from torch.distributed.device_mesh import init_device_mesh

        self.param_dtype = torch.bfloat16
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        use_shm = self.config.model.get("use_shm", False)
        local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
        trust_remote_code = self.config.model.get("trust_remote_code", False)

        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        from verl.utils.model import get_generation_config

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, (
            f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        )
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        self.rollout_device_mesh = rollout_device_mesh

        is_collect = rollout_device_mesh["infer_tp"].get_local_rank() == 0
        self._register_dispatch_collect_info(
            "rollout", dp_rank=rollout_device_mesh["dp"].get_local_rank(), is_collect=is_collect
        )

        rollout_name = self.config.rollout.name
        if rollout_name not in ("vllm", "sglang"):
            raise NotImplementedError(f"rollout_name: {rollout_name} is not supported")

        rollout_config: RolloutConfig = omega_conf_to_dataclass(self.config.rollout)
        model_config: HFModelConfig = omega_conf_to_dataclass(self.config.model, dataclass_type=HFModelConfig)
        self.model_config = model_config

        log_gpu_memory_usage(f"Before building {rollout_name} rollout", logger=logger)
        self.rollout = get_rollout_class(rollout_config.name, rollout_config.mode)(
            config=rollout_config, model_config=model_config, device_mesh=rollout_device_mesh
        )
        log_gpu_memory_usage(f"After building {rollout_name} rollout", logger=logger)

        get_torch_device().empty_cache()
