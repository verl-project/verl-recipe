# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from typing import Iterable

import numpy as np
import torch
from distributed_util import vllm_stateless_init_process_group
from megatron.core import parallel_state as mpu
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron_kl_loss import vocab_parallel_kl_divergence
from omegaconf import DictConfig, OmegaConf
from ray.util.collective import collective

from verl import DataProto
from verl.single_controller.base.decorator import (
    Dispatch,
    make_nd_compute_dataproto_dispatch_fn,
    register,
)
from verl.utils.checkpoint.megatron_checkpoint_manager import MegatronCheckpointManager
from verl.utils.device import get_device_id, get_device_name, get_torch_device
from verl.utils.flops_counter import FlopsCounter
from verl.utils.megatron.pipeline_parallel import make_batch_generator
from verl.utils.megatron.router_replay_utils import (
    RouterReplayHelper,
    reorder_and_merge_vpp_layers,
)
from verl.utils.megatron_utils import load_megatron_model_to_gpu, offload_megatron_model_to_cpu
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import append_to_dict
from verl.utils.ray_utils import get_event_loop
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.utils.torch_functional import broadcast_dict_tensor
from verl.workers.actor.megatron_actor import MegatronPPOActor
from verl.workers.megatron_workers import AsyncActorRolloutRefWorker

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def _ensure_required_actor_rollout_fields(config: DictConfig) -> None:
    """Pad GKD configs with dummy fields that the up-to-date verl
    `ActorRolloutRefWorker` expects during ``__init__`` (e.g. PPO batch sizes,
    rollout sample count and router-replay options).

    GKD uses pure on-policy distillation and therefore intentionally omits these
    fields, but the parent class still references them while normalising the
    config. We inject safe defaults instead of failing fast.
    """
    is_struct = OmegaConf.is_struct(config) or False
    OmegaConf.set_struct(config, False)
    try:
        if OmegaConf.select(config, "actor") is not None:
            actor_cfg = config.actor
            if OmegaConf.select(actor_cfg, "ppo_mini_batch_size") is None:
                actor_cfg.ppo_mini_batch_size = 1
            if OmegaConf.select(actor_cfg, "ppo_micro_batch_size") is None:
                # The parent worker only normalises this field when it is set,
                # so leaving it as None keeps GKD's `micro_batch_size` flow.
                pass
            if OmegaConf.select(actor_cfg, "ppo_epochs") is None:
                actor_cfg.ppo_epochs = 1
            if OmegaConf.select(actor_cfg, "router_replay") is None:
                actor_cfg.router_replay = OmegaConf.create({"mode": "disabled"})
        if OmegaConf.select(config, "rollout") is not None:
            rollout_cfg = config.rollout
            if OmegaConf.select(rollout_cfg, "n") is None:
                rollout_cfg.n = 1
            if OmegaConf.select(rollout_cfg, "log_prob_micro_batch_size") is None:
                # Same as above; only normalised when set.
                pass
    finally:
        OmegaConf.set_struct(config, is_struct)


class TensorBuffer:
    def __init__(self, memory_alloc, dtype):
        device = get_device_id()
        dtype_size = torch.tensor([], dtype=dtype).element_size()
        self.capacity = memory_alloc // dtype_size
        self.tensor = torch.empty(self.capacity, dtype=dtype, device=device)
        self.keys = []
        self.shapes = []

    @property
    def size(self):
        return sum(shape.numel() for shape in self.shapes)

    def clear(self):
        self.keys.clear()
        self.shapes.clear()

    def append(self, key, shape, weight=None):
        if weight is not None:
            self.tensor[self.size : self.size + shape.numel()] = weight.view(-1)
        self.keys.append(key)
        self.shapes.append(shape)

    def to_tensors(self):
        tensors = []
        start = 0
        for key_, shape_ in zip(self.keys, self.shapes, strict=False):
            tensors.append((key_, self.tensor[start : start + shape_.numel()].view(shape_)))
            start += shape_.numel()
        return tensors


def record_time(func):
    def wrapper(*args, **kwargs):
        tik = time.time()
        func(*args, **kwargs)
        tok = time.time()
        return tok - tik

    return wrapper


class OnPolicyDistillActor(MegatronPPOActor):
    """
    Responsible purely for the training step (forward-backward + optimizer).
    """

    def make_minibatch_iterator(self, data: DataProto) -> Iterable[DataProto]:
        """Make minibatch iterator for updating the actor

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64, where
                ``sequence_length = prompt_length + response_length``

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64

                ``responses``: tensor of shape [batch_size, response_length]. torch.int64. Note that
                responses = input_ids[:, -response_length:]

                ``old_log_probs``: tensor of shape [batch_size, response_length]. torch.float32. The log probability
                of responses.

                ``advantages``: tensor of shape [batch_size, response_length]. torch.float32. The advantages of
                responses.
                See PPO paper for details. https://arxiv.org/abs/1707.06347

        Returns:

        """
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_keys = ["teacher_topk_logps", "teacher_topk_indices"]

        self.has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        # router replay
        if self.enable_routing_replay:
            select_keys.append("routed_experts")
        if self.has_multi_modal_inputs:
            non_tensor_keys.append("multi_modal_inputs")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_keys)

        return data.make_iterator(
            mini_batch_size=self.config.ppo_mini_batch_size,
            epochs=self.config.ppo_epochs,
            seed=self.config.data_loader_seed,
            dataloader_kwargs={"shuffle": self.config.shuffle},
        )

    def forward_backward_batch(
        self,
        data: DataProto,
        forward_only=False,
        post_process_fn=None,
        calculate_entropy=False,
        use_dynamic_bsz=False,
        micro_batch_size=None,
        max_token_len=None,
        mini_batch_size=None,
    ):
        """
        We assume:
        - The model takes input: (input_ids, attention_mask, position_ids). No rmpad for the input
        - The communication shape is (total_nnz_pad_to_sp // tp_size, 1, hidden_size) if sequence parallel is enabled
        """
        # broadcast from last pp rank to all other pp ranks

        data.to(get_device_id())
        data.batch = data.batch.contiguous()
        mini_batch = data
        broadcast_dict_tensor(
            mini_batch.batch,
            src=mpu.get_pipeline_model_parallel_last_rank(),
            group=mpu.get_pipeline_model_parallel_group(),
        )
        mini_batch.to("cpu")
        # split into micro-batches
        mini_batch.batch["attention_mask"] = mini_batch.batch["attention_mask"].to(bool)

        if mpu.is_pipeline_last_stage():
            for key, dtype in [("teacher_topk_logps", np.float32), ("teacher_topk_indices", np.int32)]:
                if key in mini_batch.non_tensor_batch:
                    arr = mini_batch.non_tensor_batch[key]
                    if isinstance(arr, np.ndarray) and arr.dtype == object:
                        mini_batch.non_tensor_batch[key] = np.stack(arr).astype(dtype)

        indices = None
        if use_dynamic_bsz:
            assert max_token_len is not None, "max_token_len must be set when use_dynamic_bsz is True"
            vpp_size = mpu.get_virtual_pipeline_model_parallel_world_size()
            if vpp_size is not None and vpp_size > 1:
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                micro_batches, indices = rearrange_micro_batches(
                    batch=mini_batch.batch,
                    num_batches_divided_by=microbatch_group_size_per_vp_stage,
                    max_token_len=max_token_len,
                )
                assert len(micro_batches) % self.tf_config.microbatch_group_size_per_vp_stage == 0, (
                    f"micro_batches {len(micro_batches)} must be divisible by microbatch_group_size_per_vp_stage "
                    f"{microbatch_group_size_per_vp_stage} for megatron backend"
                )
            else:
                micro_batches, indices = rearrange_micro_batches(batch=mini_batch.batch, max_token_len=max_token_len)
            # total_seqlen = max_token_len
            if mpu.is_pipeline_last_stage():
                teacher_topk_logps_tensor = torch.tensor(mini_batch.non_tensor_batch["teacher_topk_logps"])
                teacher_topk_indices_tensor = torch.tensor(mini_batch.non_tensor_batch["teacher_topk_indices"])
                teacher_topk_logps, teacher_topk_indices = [], []
                for partition in indices:
                    curr_logp_micro_batch, curr_idx_micro_batch = [], []
                    for idx in partition:
                        curr_logp_micro_batch.append(teacher_topk_logps_tensor[idx : idx + 1])
                        curr_idx_micro_batch.append(teacher_topk_indices_tensor[idx : idx + 1])
                    curr_logp_micro_batch = torch.cat(curr_logp_micro_batch)
                    curr_idx_micro_batch = torch.cat(curr_idx_micro_batch)

                    teacher_topk_logps.append(curr_logp_micro_batch)
                    teacher_topk_indices.append(curr_idx_micro_batch)

                for i, mb in enumerate(micro_batches):
                    responses = mb["responses"]
                    response_length = responses.size(1)
                    calc_kl_mask = mb["attention_mask"].clone()
                    calc_kl_mask[:, : (-response_length - 1)] = False
                    mb["calc_kl_mask"] = calc_kl_mask
                    mb["kl_losses"] = torch.zeros_like(calc_kl_mask, dtype=torch.float32)
                    mb["teacher_topk_logps"] = teacher_topk_logps[i].pin_memory()
                    mb["teacher_topk_indices"] = teacher_topk_indices[i].pin_memory()
        else:
            assert micro_batch_size is not None, (
                "micro_batch_size is needed to be passed in when not using dynamic batch size"
            )

            micro_batches = mini_batch.batch.split(micro_batch_size)
            # seq_len = micro_batches[0]["input_ids"].shape[1]
            # total_seqlen = micro_batch_size * seq_len

            if mpu.is_pipeline_last_stage():
                teacher_topk_logps = np.array_split(
                    mini_batch.non_tensor_batch["teacher_topk_logps"], len(micro_batches)
                )
                teacher_topk_indices = np.array_split(
                    mini_batch.non_tensor_batch["teacher_topk_indices"], len(micro_batches)
                )

                for i, mb in enumerate(micro_batches):
                    responses = mb["responses"]
                    response_length = responses.size(1)
                    calc_kl_mask = mb["attention_mask"].clone()
                    calc_kl_mask[:, : (-response_length - 1)] = False
                    mb["calc_kl_mask"] = calc_kl_mask
                    mb["kl_losses"] = torch.zeros_like(calc_kl_mask, dtype=torch.float32)
                    mb["teacher_topk_logps"] = torch.from_numpy(teacher_topk_logps[i]).pin_memory()
                    mb["teacher_topk_indices"] = torch.from_numpy(teacher_topk_indices[i]).pin_memory()

        # compute input shapes for pp stages
        n_micro_batch = len(micro_batches)

        forward_backward_func = get_forward_backward_func()

        def loss_func(output):
            # For memory efficiency
            # We move calculation of entropy to compute_log_probs, forward_only == True
            metrics = {}

            ret_entropy = None
            stats = {}
            kl_losses = output["kl_losses"]
            calc_kl_mask = output["calc_kl_mask"]
            # inf_cnt = masked_kl_lossed.isinf().sum().item()
            # nan_cnt = masked_kl_lossed.isnan().sum().item()
            # total_cnt = masked_kl_lossed.nelement()
            # print(f"rank: {rank}, kl_loss inf_cnt/nan_cnt/total_cnt: {inf_cnt} / {nan_cnt} /{total_cnt}")
            masked_kl_lossed = kl_losses[calc_kl_mask]
            mean_kl_loss = masked_kl_lossed.mean()
            stats.update({"actor/kl_loss": mean_kl_loss.detach().item()})

            append_to_dict(metrics, stats)
            return mean_kl_loss, [metrics, ret_entropy]

        def forward_step(batch_iter, model):
            batch = next(batch_iter)
            batch = batch.to(get_device_id())
            batch = batch.contiguous()

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]

            def logits_processor(logits, teacher_topk_logps, teacher_topk_indices, calc_kl_mask, kl_losses):
                assert logits.shape[:2] == calc_kl_mask.shape[:2]
                assert logits.shape[:2] == teacher_topk_indices.shape[:2]
                assert logits.shape[:2] == teacher_topk_logps.shape[:2]

                masked_logits = logits[calc_kl_mask]
                masked_teacher_topk_logps = teacher_topk_logps[calc_kl_mask]
                masked_teacher_topk_indices = teacher_topk_indices[calc_kl_mask]

                kl_losses[calc_kl_mask] = vocab_parallel_kl_divergence(
                    masked_logits, masked_teacher_topk_logps, masked_teacher_topk_indices
                )
                return {"kl_losses": kl_losses, "calc_kl_mask": calc_kl_mask}

            if mpu.is_pipeline_last_stage():
                device = get_device_id()
                teacher_topk_logps = batch["teacher_topk_logps"].to(device, non_blocking=True)
                teacher_topk_indices = batch["teacher_topk_indices"].to(device, non_blocking=True)
                logits_processor_args = {
                    "calc_kl_mask": batch["calc_kl_mask"],
                    "kl_losses": batch["kl_losses"],
                    "teacher_topk_logps": teacher_topk_logps,
                    "teacher_topk_indices": teacher_topk_indices,
                }
            else:
                logits_processor_args = None

            multi_modal_inputs = {}
            if "multi_modal_inputs" in batch:
                from verl.utils.model import extract_multi_modal_inputs

                indices = batch.get("multi_modal_inputs_idx", None)
                multi_modal_inputs = extract_multi_modal_inputs(batch["multi_modal_inputs"], indices)

            from verl.models.mcore import get_mcore_forward_fn

            forward_fn = get_mcore_forward_fn(self.hf_config)

            output = forward_fn(
                model,
                input_ids,
                attention_mask,
                position_ids,
                multi_modal_inputs,
                logits_processor=logits_processor,
                logits_processor_args=logits_processor_args,
            )

            return output, loss_func

        # batch should be a list of batches inside micro-batches
        batch_generator = make_batch_generator(micro_batches, vpp_size=len(self.actor_module))

        # TODO: we may use the new schedule instead
        # for flash-attn: (seq_len, batch_size, hidden_size) = (mbs*seq_len, 1, hidden_size)
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=batch_generator,
            model=self.actor_module,
            num_microbatches=n_micro_batch,
            seq_length=-1,  # no use when variable_seq_lengths was set
            micro_batch_size=-1,  # no use when variable_seq_lengths was set
            forward_only=False,
        )

        # loss_reduces contains the stats returned from loss_func

        losses_reduced = {"output": losses_reduced}
        if use_dynamic_bsz:
            losses_reduced["indices"] = indices
        if RouterReplayHelper.is_r2_record_action(self.tf_config):
            if self.tf_config.virtual_pipeline_model_parallel_size is not None:
                # config = self.actor_module[0].module.module.config
                vp_size = len(self.actor_module)
                microbatch_group_size_per_vp_stage = self.tf_config.microbatch_group_size_per_vp_stage
                bs = n_micro_batch
                losses_reduced["mini_layer_topk_idx_tensor"] = reorder_and_merge_vpp_layers(
                    self.mini_layer_topk_idx_list, bs, vp_size, microbatch_group_size_per_vp_stage
                )
            else:
                losses_reduced["mini_layer_topk_idx_tensor"] = torch.cat(self.mini_layer_topk_idx_list, dim=0)
            self.mini_layer_topk_idx_list = []

        return losses_reduced


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

        params_generator = self._get_actor_params_generator() if self._is_actor else None

        if self._is_actor and self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)

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
            if self._is_actor:
                weight_key, weight = next(params_generator)
                assert key == weight_key
                assert shape == weight.size()
                assert dtype == weight.dtype

            tensor = torch.empty(shape, dtype=dtype, device=get_torch_device().current_device())
            if self._is_actor and torch.distributed.get_rank() == 0:
                tensor.copy_(weight)

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
            offload_megatron_model_to_cpu(self.actor_module)

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


class MegatronOnPolicyDistillActorWorker(DetachSync):
    """
    Actor-only worker: owns the trainable Megatron model and optimizer, performs update_actor.
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        # Ensure we run as actor-only worker
        _ensure_required_actor_rollout_fields(config)

        super().__init__(config, role, **kwargs)
        assert self._is_actor and not self._is_rollout, "Actor worker must be actor-only."

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def _get_actor_params_generator(self):
        assert self._is_actor
        if self.bridge is not None:
            generator = self.bridge.export_weights(self.actor.actor_module)
        else:
            # from verl.utils.megatron_utils import per_tensor_generator
            from megatron_utils import per_tensor_generator

            from verl.models.mcore import get_mcore_weight_converter

            layer_name_mapping = {
                "qkv_layer_name": "self_attention.linear_qkv.",
                "gate_proj_layer_name": "linear_fc1.",
            }
            weight_converter = get_mcore_weight_converter(self.actor_model_config, self.dtype)
            generator = per_tensor_generator(
                self.actor.actor_module,
                self.actor_model_config,
                weight_converter,
                self.tf_config,
                layer_name_mapping,
            )
        return generator

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.utils.torch_dtypes import PrecisionType

        override_model_config = OmegaConf.to_container(self.config.model.get("override_config", OmegaConf.create()))
        override_transformer_config = OmegaConf.to_container(
            self.config.actor.megatron.get("override_transformer_config", OmegaConf.create()), resolve=True
        )
        # `override_ddp_config` was added in the up-to-date verl `_build_model_optimizer`
        # signature. Resolve it from config (default to empty dict) for forward-compat.
        override_ddp_config = OmegaConf.to_container(
            OmegaConf.create(self.config.actor.megatron.get("override_ddp_config", {}))
        )

        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init actor model and optimizer", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)
        # we need the model for actor and rollout
        optim_config = self.config.actor.optim
        (
            self.actor_module,
            self.actor_optimizer,
            self.actor_optimizer_scheduler,
            self.actor_model_config,
            self.actor_optim_config,
        ) = self._build_model_optimizer(
            model_path=self.config.model.path,
            optim_config=optim_config,
            override_model_config=override_model_config,
            override_transformer_config=override_transformer_config,
            override_ddp_config=override_ddp_config,
        )

        self.actor = OnPolicyDistillActor(
            config=self.config.actor,
            model_config=self.actor_model_config,
            hf_config=self.hf_config,
            tf_config=self.tf_config,
            actor_module=self.actor_module,
            actor_optimizer=self.actor_optimizer,
        )
        log_gpu_memory_usage("After OnPolicyDistillActor init", logger=logger)

        self.flops_counter = FlopsCounter(self.actor_model_config)
        # NOTE: in the up-to-date verl `MegatronCheckpointManager`, additional
        # optional kwargs (`provider`, `peft_cls`) were introduced. Pass them
        # through if available so that GKD plays nicely with both old and new versions.
        self.checkpoint_mananager = MegatronCheckpointManager(
            config=self.config,
            checkpoint_config=self.config.actor.checkpoint,
            model_config=self.actor_model_config,
            transformer_config=self.tf_config,
            role="actor",
            model=self.actor_module,
            arch=self.architectures[0],
            hf_config=self.hf_config,
            param_dtype=self.param_dtype,
            share_embeddings_and_output_weights=self.share_embeddings_and_output_weights,
            processing_class=self.processor if self.processor is not None else self.tokenizer,
            optimizer=self.actor_optimizer,
            optimizer_scheduler=self.actor_optimizer_scheduler,
            use_distributed_optimizer=self.config.actor.megatron.use_distributed_optimizer,
            use_checkpoint_opt_param_scheduler=self.config.actor.optim.use_checkpoint_opt_param_scheduler,
            bridge=self.bridge,
            provider=self.provider,
            use_dist_checkpointing=self.config.actor.megatron.use_dist_checkpointing,
            peft_cls=self.peft_cls,
        )
        get_torch_device().empty_cache()
        log_gpu_memory_usage("Actor init_model finished", logger=logger)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def get_actor_weights_info(self):
        assert self._is_actor
        if hasattr(self, "_weights_info"):
            return self._weights_info
        if self._is_offload_param:
            load_megatron_model_to_gpu(self.actor_module)
        params_generator = self._get_actor_params_generator()
        ret = []
        for key, tensor in params_generator:
            ret.append((key, tensor.size(), tensor.dtype))

        self._weights_info = ret
        # Here, we only call this function at the beginning,
        # and immediately afterwards we call sync_rollout_weights.
        # So we no longer call offload in this.
        return ret


class MegatronOnPolicyDistillRolloutWorker(DetachSync):
    """
    Rollout-only worker: owns the inference engine (vLLM/SGlang, or Megatron forward) and generates sequences.
    """

    def __init__(self, config: DictConfig, role: str, **kwargs):
        # The up-to-date verl ActorRolloutRefWorker.__init__ already handles all the
        # heavy lifting (process group init, profiler, offload flags, ...). For a
        # rollout-only worker the actor branches in the parent are skipped, so we
        # can simply delegate the construction once mandatory fields are present.
        _ensure_required_actor_rollout_fields(config)

        from verl.utils.fs import copy_to_local
        from verl.utils.model import get_generation_config

        super().__init__(config, role, **kwargs)
        assert self.role == "rollout"
        assert self._is_rollout and not self._is_actor and not self._is_ref, "Rollout worker must be rollout-only."

        # Cached locally because the parent class only sets `self.local_path`
        # inside `_init_hf_config_and_tf_config`, which we still want to call
        # when the rollout engine boots up. Pre-fetching the model files here
        # is a no-op when `_init_hf_config_and_tf_config` runs later.
        self.local_path = copy_to_local(self.config.model.path)
        self.generation_config = get_generation_config(self.local_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        """
        Build the actor module only for inference + rollout engine; no optimizer/updates.
        """
        from verl.utils.torch_dtypes import PrecisionType

        self.param_dtype = torch.bfloat16
        log_gpu_memory_usage("Before init rollout model", logger=logger)
        self.dtype = PrecisionType.to_dtype(self.param_dtype)

        self._build_rollout(trust_remote_code=self.config.model.get("trust_remote_code", False))
        # In the up-to-date verl, `_build_rollout` exposes the device mesh via
        # `self.rollout.device_mesh`. Older versions used `self.rollout_device_mesh`,
        # so guard the access for backwards compatibility.
        self.rollout_device_mesh = getattr(self.rollout, "device_mesh", None)
        log_gpu_memory_usage("After rollout init", logger=logger)
        get_torch_device().empty_cache()

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"), blocking=False)
    def async_generate_sequences(self, *args, **kwargs):
        return self.generate_sequences(*args, **kwargs)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_actor_weights_info(self, weights_info):
        assert self._is_rollout
        self._weights_info = weights_info
