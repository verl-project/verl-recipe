import logging
import os
import warnings
import torch
from omegaconf import DictConfig

try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.config import FSDPEngineConfig
from verl.workers.fsdp_workers import (
    ActorRolloutRefWorker,
    AsyncActorRolloutRefWorker,
    get_vl_model_vision_tower
)
from verl.utils.fsdp_utils import (
    get_init_weight_context_manager,
    load_fsdp_model_to_gpu,
    collect_lora_params,
    replace_lora_wrapper,
    offload_fsdp_model_to_cpu
)
from verl.utils.model import convert_weight_keys
from verl.utils.device import (
    get_device_id,
    get_device_name,
    get_nccl_backend,
    get_torch_device,
    set_expandable_segments,
)
from verl.utils.memory_utils import aggressive_empty_cache

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class MMActorRolloutRefWorker(ActorRolloutRefWorker):
    def __init__(self, config: DictConfig, role: str, **kwargs):
        super().__init__(config, role, **kwargs)

    def _dataloader(self, args=None):
        return None

    async def rollout_mode(self):
        """Context switch hybridengine to rollout mode."""
        aggressive_empty_cache(force_sync=True)

        log_gpu_memory_usage("Before load_fsdp_model_to_gpu", logger=logger)
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After load_fsdp_model_to_gpu", logger=logger)

        peft_config = None
        peft_model = getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        if hasattr(peft_model, "peft_config"):
            peft_config = peft_model.peft_config.get("default", None)
            params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.config.rollout.get("layered_summon", False),
                base_sync_done=self.base_sync_done,
            )
            if not self.base_sync_done:
                params = {replace_lora_wrapper(k, peft_config): v for k, v in params.items()}
        else:
            params = self.actor_module_fsdp.state_dict()

        params = convert_weight_keys(
            params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
        )

        for k in list(params.keys()):
            if "mlp.experts.gate_up_proj" in k or "mlp.experts.down_proj" in k:
                print(f"--> modify shape {k}")
                params[k] = params[k].transpose(1, 2).contiguous()

        # Special handling for LoRA with sleep_level=2:
        # When sleep_level=2, base model weights are destroyed during each sleep cycle.
        # separately collect and update LoRA weights and base model weights through their respective interfaces.
        # Here: params contains LoRA weights, base_model_params contains base model weights.
        # Only needed if the rollout engine actually sleeps/frees weights (free_cache_engine=True).
        if (
                peft_config is not None
                and getattr(self.rollout, "sleep_level", None) == 2
                and self.config.rollout.free_cache_engine
        ):
            base_model_params = collect_lora_params(
                module=self.actor_module_fsdp,
                layered_summon=self.layered_summon,
                base_sync_done=False,
            )
            base_model_params = {replace_lora_wrapper(k, peft_config): v for k, v in base_model_params.items()}
            base_model_params = convert_weight_keys(
                base_model_params, getattr(self.actor_module_fsdp, "_fsdp_wrapped_module", self.actor_module_fsdp)
            )

        log_gpu_memory_usage("Before offload_fsdp_model_to_cpu", logger=logger)
        # breakpoint()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        log_gpu_memory_usage("After offload_fsdp_model_to_cpu", logger=logger)

        set_expandable_segments(False)

        if peft_config is not None and self.base_sync_done:
            per_tensor_param = params.items() if isinstance(params, dict) else params  # Fixed: handle dict case
        else:
            device = get_device_id()  # used when fsdp2 set cpu_offload_policy
            per_tensor_param = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in params.items()
            )

        # QAT: quantize weights before sending to vLLM
        if self._qat_enabled:
            from verl.utils.qat.quantizer import QATQuantizer

            quantizer = QATQuantizer(
                mode=self.qat_config.mode,
                group_size=self.qat_config.group_size,
                ignore_patterns=self.qat_config.ignore_patterns,
                device=torch.device(get_device_id()),
                param_dtype=self._param_dtype,
            )
            per_tensor_param = quantizer.quantize_with_fusion(
                per_tensor_param,
                target_device=torch.device("cpu"),
            )
            aggressive_empty_cache(force_sync=True)

        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["weights"])
        log_gpu_memory_usage("After resume weights", logger=logger)

        if (
                peft_config is not None
                and getattr(self.rollout, "sleep_level", None) == 2
                and self.config.rollout.free_cache_engine
        ):
            per_tensor_base_params = (
                (name, param.to(device, non_blocking=True).full_tensor() if isinstance(param, DTensor) else param)
                for name, param in base_model_params.items()
            )
            await self.rollout.update_weights(per_tensor_base_params, base_sync_done=False)
            del base_model_params, per_tensor_base_params

        await self.rollout.update_weights(per_tensor_param, peft_config=peft_config,
                                          base_sync_done=self.base_sync_done)  # qxm
        log_gpu_memory_usage("After update_weights", logger=logger)
        del params, per_tensor_param
        aggressive_empty_cache(force_sync=True)
        if self.config.rollout.free_cache_engine:
            await self.rollout.resume(tags=["kv_cache"])
        log_gpu_memory_usage("After resume kv_cache", logger=logger)

        self.base_sync_done = True
        set_expandable_segments(True)

    def _build_model_optimizer(
            self,
            model_path,
            fsdp_config: FSDPEngineConfig,
            optim_config,
            override_model_config,
            use_remove_padding=False,
            use_fused_kernels=False,
            enable_gradient_checkpointing=False,
            trust_remote_code=False,
            use_liger=False,
            role="actor",
            enable_activation_offload=False,
            use_prefix_grouper=False,
            use_tiled_mlp=False,
            tiled_mlp_shards=4,
    ):
        from transformers import AutoConfig

        from verl.utils.model import get_generation_config, print_model_size, update_model_config
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        from typing import Any

        import yaml
        from mindspeed_mm.fsdp.params.argument import Arguments
        from mindspeed_mm.fsdp.params.utils import instantiate_dataclass

        mm_file_path = os.environ.get("MM_CONFIG_FILE")
        with open(os.path.abspath(mm_file_path), encoding="utf-8") as f:
            input_data: dict[str, dict[str, Any]] = yaml.safe_load(f)
        self.mm_args = instantiate_dataclass(Arguments, input_data)
        self.mm_args.training.compute_distributed_training(self.mm_args.parallel)

        # TiledMLP requires FSDP2 for correct gradient computation
        if use_tiled_mlp and self.config.actor.strategy == "fsdp":
            raise ValueError("TiledMLP requires FSDP2. Set `actor_rollout_ref.actor.strategy=fsdp2`.")

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        if self.config.model.get("custom_chat_template", None) is not None:
            if self.processor is not None:
                self.processor.chat_template = self.config.model.custom_chat_template
            else:
                self.tokenizer.chat_template = self.config.model.custom_chat_template

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        attn_implementation = override_model_config.get("attn_implementation", "flash_attention_2")
        actor_model_config = AutoConfig.from_pretrained(
            local_path, trust_remote_code=trust_remote_code, attn_implementation=attn_implementation
        )
        # which will be patched by _ulysses_flash_attention_forward, but errorly misses position_ids
        # Maybe support Ulysses in VisionAttention in the future and remove this patch
        if self.ulysses_sequence_parallel_size > 1 and hasattr(actor_model_config, "vision_config"):
            actor_model_config.vision_config._attn_implementation = "eager"

        # patch for qwen2.5-vl: when using flash_attention_3, set vision tower to use flash_attention_2
        # because the vision tower does not support flash_attention_3
        if (
                getattr(actor_model_config, "model_type", None) == "qwen2_5_vl"
                and attn_implementation == "flash_attention_3"
                and hasattr(actor_model_config, "vision_config")
        ):
            actor_model_config.vision_config._attn_implementation = "flash_attention_2"

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        if self.config.model.get("mtp", {}).get("enable", False):
            raise NotImplementedError("Right now,  MTP is not supported in FSDP")
        else:
            if hasattr(actor_model_config, "num_nextn_predict_layers"):
                actor_model_config.num_nextn_predict_layers = 0

        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from mindspeed_mm.fsdp.train.trainer import Trainer

            if role == "actor":
                self.mm_args.parallel.fsdp_plan.cpu_offload = False
            else:
                self.mm_args.parallel.fsdp_plan.cpu_offload = True
            trainer = Trainer(args=self.mm_args, dataloader_provider=self._dataloader)
            trainer.train_dataloader = None
            actor_module = trainer.model

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=actor_module)

            fused_kernel_options = self.config.model.get("fused_kernel_options", None)
            fused_kernels_backend = (
                fused_kernel_options.get("impl_backend", None) if fused_kernel_options is not None else None
            )

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
                fused_kernels_backend=fused_kernels_backend,
                use_prefix_grouper=use_prefix_grouper,
                use_tiled_mlp=use_tiled_mlp,
                tiled_mlp_shards=tiled_mlp_shards,
            )

            # some parameters may not in torch_dtype.
            actor_module.to(torch_dtype)

        self.use_orig_params = fsdp_config.get("use_orig_params", False)
        if self.config.actor.get("freeze_vision_tower", False):
            vision_tower = get_vl_model_vision_tower(actor_module)
            if vision_tower is not None:
                vision_tower.requires_grad_(False)
                self.use_orig_params = True
                if self.rank == 0:
                    print("[actor model] Vision tower is set to not trainable.")
            else:
                if self.rank == 0:
                    print("[actor model] No vision tower found.")

        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)
        actor_module_fsdp = actor_module

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        if role == "actor" and optim_config is not None:
            actor_optimizer = trainer.optimizer
            actor_lr_scheduler = trainer.lr_scheduler

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            trainer.optimizer = None
            trainer.lr_scheduler = None
            actor_optimizer = None
            actor_lr_scheduler = None

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config


class AsyncMMActorRolloutRefWorker(AsyncActorRolloutRefWorker, MMActorRolloutRefWorker):
    pass
