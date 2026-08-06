# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import importlib
import importlib.util
import logging
import re
import time
from contextlib import nullcontext

import hydra
import torch
import torch.distributed
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from tensordict import TensorDict
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.distributed.fsdp import CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import Dataset, DistributedSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel

import verl.utils.hdfs_io as hdfs_io
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, get_checkpoint_tracker_filename
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.dataset import SFTDataset
from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset
from verl.utils.device import get_device_id, get_device_name, is_cuda_available, is_npu_available
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_clip_grad_norm_,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.logger import log_with_rank
from verl.utils.profiler import log_gpu_memory_usage
from verl.utils.py_functional import convert_to_regular_types
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import get_cosine_schedule_with_warmup, get_wsd_schedule_with_warmup
from verl.utils.tracking import Tracking
from verl.utils.ulysses import (
    gather_outputs_and_unpad,
    get_ulysses_sequence_parallel_world_size,
    ulysses_pad_and_slice_inputs,
)
from verl.workers.config.optimizer import build_optimizer
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_SFT_LOGGING_LEVEL", "WARN"))


def extract_step(path):
    match = re.search(r"global_step_(\d+)", path)
    if match:
        return int(match.group(1))
    return None


def _optional_float(value, name: str):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid float for {name}: {value}") from exc


def _optional_str(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"", "none", "null"}:
        return None
    return str(value)


class FSDPSFTTrainer:
    def __init__(
        self,
        config,
        device_mesh: DeviceMesh,
        ulysses_device_mesh: DeviceMesh,
        tokenizer,
        train_dataset: Dataset,
        val_dataset: Dataset,
    ):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.tokenizer = tokenizer
        if self.config.data.chat_template is not None:
            raise ValueError("Apply Chat template from config is not supported yet.")
        self.loss_mode = str(self.config.trainer.get("loss_mode", "sft")).lower()
        self.asft_kl_coef = float(self.config.trainer.get("asft_kl_coef", 0.1))
        if self.loss_mode not in {"sft", "dft", "asft"}:
            raise ValueError(f"Unsupported loss_mode: {self.loss_mode}. Expected one of: sft, dft, asft")
        self.ref_model = None

        # Benchmark evaluation config (MCQ accuracy during training)
        self.benchmark_eval_dir = _optional_str(self.config.trainer.get("benchmark_eval_dir", None))
        self.benchmark_eval_max_samples = int(self.config.trainer.get("benchmark_eval_max_samples", 500))

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, "ulysses_sequence_parallel_size", 1)
        self.use_remove_padding = getattr(self.config, "use_remove_padding", False)
        if self.device_mesh.get_rank() == 0:
            print(f"Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}")
            print(f"Using remove padding: {self.use_remove_padding}")
            print(f"Using SFT loss_mode: {self.loss_mode}, asft_kl_coef: {self.asft_kl_coef}")

        self._build_dataloader(train_dataset, val_dataset)

        self.lora = self.config.model.get("lora_adapter_path") is not None or self.config.model.lora_rank > 0

        # Initialize resume-related variables
        self.resume_global_step = 0

        # build model
        self._build_model_optimizer()

        # Initialize checkpoint manager
        self._init_checkpoint_manager()

        self.load_checkpoint()

        if self.device_mesh.get_rank() == 0:
            print(self.config)
        self.device_name = self.config.trainer.device

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f"Normalize batch size by dp {dp_size}")

        assert self.config.data.train_batch_size % dp_size == 0, (
            f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"
        )

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self, train_dataset, val_dataset):
        # build dataset
        config = self.config
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank("dp")
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f"Using SP rank {rank} and size {world_size} for data distribution")
                print("Each SP rank gets different data, but the same data WITHIN the same rank")
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f"Using FSDP rank {rank} and size {world_size} for data distribution")

        # Set pin_memory_device when pin_memory is enabled.
        device_name = get_device_name()

        self.train_sampler = DistributedSampler(
            self.train_dataset, shuffle=True, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=config.data.train_batch_size,
            sampler=self.train_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

        self.val_sampler = DistributedSampler(
            self.val_dataset, shuffle=False, num_replicas=world_size, rank=rank, drop_last=True
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=config.data.micro_batch_size_per_gpu,
            sampler=self.val_sampler,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            pin_memory_device=device_name,
        )

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get("external_lib", None) is not None:
            # This is used to import external_lib into the huggingface systems
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage("Before model allocation", logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        torch_dtype = self.config.model.fsdp_config.get("model_dtype", "fp32")
        torch_dtype = PrecisionType.to_dtype(torch_dtype)
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        self.model_config = config
        if hasattr(self.model_config, "max_position_embeddings"):
            self.model_config.max_position_embeddings = max(
                self.model_config.max_position_embeddings, self.config.data.max_length
            )
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"

        # This may be very large
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not config.tie_word_embeddings, mesh=self.device_mesh
        )

        with init_context():
            attn_impl = "flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa"
            if self.device_mesh.get_rank() == 0 and attn_impl != "flash_attention_2":
                print("flash_attn not found, fallback to attn_implementation=sdpa")
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
                local_model_path,
                config=config,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
                trust_remote_code=trust_remote_code,
            )

            if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
                from verl.models.transformers.monkey_patch import apply_monkey_patch

                apply_monkey_patch(model=self.model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get("use_liger", False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

                _apply_liger_kernel_to_instance(model=self.model)

            if self.lora:
                self.model.enable_input_require_grads()

                lora_adapter_path = self.config.model.get("lora_adapter_path")
                if lora_adapter_path is not None:
                    from peft import PeftModel

                    print(f"Loading pre-trained LoRA adapter for sft from: {lora_adapter_path}")

                    local_adapter_path = copy_to_local(lora_adapter_path, use_shm=self.config.model.use_shm)

                    self.model = PeftModel.from_pretrained(self.model, local_adapter_path, is_trainable=True)
                    peft_config = self.model.peft_config["default"]
                    # Ensure task_type is TaskType enum, not string
                    if isinstance(peft_config.task_type, str):
                        peft_config.task_type = TaskType.CAUSAL_LM
                else:
                    # Convert config to regular Python types before creating PEFT model
                    lora_config = {
                        "task_type": TaskType.CAUSAL_LM,
                        "r": self.config.model.lora_rank,
                        "lora_alpha": self.config.model.lora_alpha,
                        "target_modules": convert_to_regular_types(self.config.model.target_modules),
                        "bias": "none",
                    }
                    self.model = get_peft_model(self.model, LoraConfig(**lora_config))
                self.model = self.model.to(torch_dtype)

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        log_gpu_memory_usage("After model allocation", logger=logger)

        mixed_precision = MixedPrecision(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            self.model,
            config=self.config.model.fsdp_config.wrap_policy,
            is_lora=self.lora,
        )

        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        fsdp_strategy = self.config.model.strategy
        if fsdp_strategy == "fsdp":
            self.fsdp_model = FSDP(
                self.model,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=False,
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_device_id(),
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert CPUOffloadPolicy is not None, "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True
            )

            fsdp_kwargs = {
                "mesh": self.device_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": True,
            }
            full_state = self.model.state_dict()
            apply_fsdp2(self.model, fsdp_kwargs, self.config.model.fsdp_config)
            fsdp2_load_full_state_dict(self.model, full_state, self.device_mesh, cpu_offload)
            self.fsdp_model = self.model
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        log_gpu_memory_usage("After FSDP wrapping", logger=logger)
        self._build_reference_model_if_needed(
            local_model_path=local_model_path,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            fsdp_strategy=fsdp_strategy,
            auto_wrap_policy=auto_wrap_policy,
            cpu_offload=cpu_offload,
            mixed_precision=mixed_precision,
        )

        self.optimizer = build_optimizer(self.fsdp_model.parameters(), self.config.optim)

        log_gpu_memory_usage("After initialize optimizer", logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = self.steps_per_epoch * self.config.trainer.total_epochs

        if self.device_mesh.get_rank() == 0:
            print(
                f"Number of steps/epoch {self.steps_per_epoch}, number of epochs "
                f"{self.config.trainer.total_epochs}, total number of steps {self.total_steps}"
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.lr_warmup_steps_ratio)

        if not hasattr(self.config.optim, "lr_scheduler") or self.config.optim.lr_scheduler == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        elif self.config.optim.lr_scheduler == "wsd":
            self.lr_scheduler = get_wsd_schedule_with_warmup(
                optimizer=self.optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=self.total_steps
            )
        else:
            raise ValueError(f"Unknown lr scheduler: {self.config.optim.lr_scheduler}")

    def _build_reference_model_if_needed(
        self,
        local_model_path: str,
        trust_remote_code: bool,
        torch_dtype: torch.dtype,
        fsdp_strategy: str,
        auto_wrap_policy,
        cpu_offload,
        mixed_precision,
    ):
        if self.loss_mode not in {"asft"}:
            return
        if fsdp_strategy != "fsdp":
            raise NotImplementedError("ASFT mode currently supports model.strategy=fsdp only.")

        if self.loss_mode == "asft":
            if self.device_mesh.get_rank() == 0:
                print("Building frozen reference model for ASFT KL term.")
            self.ref_model = self._build_frozen_fsdp_model(
                model_path=local_model_path,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch_dtype,
                auto_wrap_policy=auto_wrap_policy,
                cpu_offload=cpu_offload,
                mixed_precision=mixed_precision,
                role="ASFT reference",
            )
            log_gpu_memory_usage("After building ASFT reference model", logger=logger)
            return

    def _build_frozen_fsdp_model(
        self,
        model_path: str,
        trust_remote_code: bool,
        torch_dtype: torch.dtype,
        auto_wrap_policy,
        cpu_offload,
        mixed_precision,
        role: str,
    ):
        frozen_config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        if hasattr(frozen_config, "max_position_embeddings"):
            frozen_config.max_position_embeddings = max(
                frozen_config.max_position_embeddings, self.config.data.max_length
            )
        frozen_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=frozen_config,
            torch_dtype=torch_dtype,
            attn_implementation="flash_attention_2" if importlib.util.find_spec("flash_attn") is not None else "sdpa",
            trust_remote_code=trust_remote_code,
        )

        if self.use_remove_padding or self.config.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch

            apply_monkey_patch(model=frozen_model, ulysses_sp_size=self.config.ulysses_sequence_parallel_size)

        if self.config.model.get("use_liger", False):
            from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

            _apply_liger_kernel_to_instance(model=frozen_model)

        frozen_model.eval()
        for param in frozen_model.parameters():
            param.requires_grad_(False)

        fsdp_frozen_model = FSDP(
            frozen_model,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=get_device_id(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False,
        )
        fsdp_frozen_model.eval()
        for param in fsdp_frozen_model.parameters():
            param.requires_grad_(False)
        if self.device_mesh.get_rank() == 0:
            print(f"Built frozen model: {role}")
        return fsdp_frozen_model

    def _compute_loss_and_backward(self, batch, do_backward=True, n_micro_batches=1):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1

        # Move inputs to GPU and prepare loss mask
        input_ids = batch["input_ids"].to(self.device_name)
        attention_mask = batch["attention_mask"].to(self.device_name)
        position_ids = batch["position_ids"].to(self.device_name)
        loss_mask = batch.pop("loss_mask")[:, 1:].reshape(-1).to(self.device_name)
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context, torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            if not use_sp:
                # Standard forward pass without sequence parallel
                labels = input_ids[:, 1:].contiguous()
                output = self.fsdp_model(
                    input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
                )
                logits = output.logits

                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels.contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                token_loss = loss_fct(shift_logits, shift_labels)
                student_token_logp = -token_loss
                if self.loss_mode == "sft":
                    loss = token_loss
                elif self.loss_mode == "dft":
                    # DFT uses p(y_t|x_{<t}) as token weight, equivalent to exp(-CE).
                    token_weights = torch.exp(student_token_logp.detach())
                    loss = token_loss * token_weights
                elif self.loss_mode == "asft":
                    # DFT uses p(y_t|x_{<t}) as token weight, equivalent to exp(-CE).
                    token_weights = torch.exp(student_token_logp.detach())
                    dft_loss = token_loss * token_weights
                    if self.ref_model is None:
                        raise RuntimeError("ASFT mode requires a reference model, but ref_model is None.")
                    with torch.no_grad():
                        ref_output = self.ref_model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            use_cache=False,
                        )
                        ref_shift_logits = ref_output.logits[..., :-1, :].contiguous()
                        ref_shift_logits = ref_shift_logits.view(-1, ref_shift_logits.size(-1))
                    # Forward KL per the ASFT paper (Eq. 6): D_KL(p_ref || p_theta),
                    # where p_ref (a.k.a. P / pi_base) is the fixed reference and
                    # p_theta (a.k.a. Q / pi_theta) is the policy. The paper explicitly
                    # adopts the "P||Q" convention as forward KL and ablates against
                    # the mode-seeking reverse KL D_KL(p_theta || p_ref) (Sec 5, Fig 2).
                    # F.kl_div(input=log q, target=p) computes sum p * (log p - log q)
                    # = D_KL(target || input), so passing input=log p_theta, target=p_ref
                    # gives exactly D_KL(p_ref || p_theta) — i.e. paper's forward KL.
                    kl_div = F.kl_div(
                        F.log_softmax(shift_logits.float(), dim=-1),
                        F.log_softmax(ref_shift_logits.float(), dim=-1),
                        log_target=True,
                        reduction="none",
                    ).sum(dim=-1)
                    loss = dft_loss + self.asft_kl_coef * kl_div.to(dft_loss.dtype)
                else:
                    raise RuntimeError(f"Unexpected loss_mode: {self.loss_mode}")
                loss = loss * loss_mask.to(loss.device)
            else:
                if self.loss_mode != "sft":
                    raise NotImplementedError(
                        "DFT/ASFT loss with sequence parallel is not implemented. "
                        "Set `use_remove_padding=False` and `ulysses_sequence_parallel_size=1`."
                    )
                # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                # 1. All SP ranks will receive the *SAME* batch
                # 2. Different SP groups will receive *DIFFERENT* batches
                # This is implemented by the DistributedSampler

                batch_size, seqlen = input_ids.shape
                # Remove padding
                input_ids_rmpad, indices, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # Unpad position_ids to align rotary
                position_ids_rmpad = index_first_axis(
                    rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                ).transpose(0, 1)

                # Pad and slice inputs for sequence parallelism
                input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size()
                )
                # For computing loss
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                    input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size()
                )
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # Forward pass
                output = self.fsdp_model(
                    input_ids=input_ids_rmpad_sliced,
                    attention_mask=None,  # Not needed with flash attention varlen
                    position_ids=position_ids_rmpad_padded,
                    use_cache=False,
                )

                # Compute loss locally then aggregate
                logits_rmpad = output.logits.squeeze(0)
                input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                # Gather and unpad for sequence parallelism
                loss = gather_outputs_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # This is the loss collected from all ulysses ranks
                full_loss = pad_input(
                    hidden_states=loss.unsqueeze(-1), indices=indices, batch=batch_size, seqlen=seqlen
                )
                full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                full_loss = full_loss.reshape(-1)
                loss_mask = loss_mask.to(full_loss.device)
                loss = full_loss * loss_mask

            valid_token_this_rank = torch.sum(loss_mask)

            if self.config.data.balance_dp_token:
                torch.distributed.all_reduce(valid_token_this_rank)
                dp_size = self.ulysses_device_mesh.size("dp") if use_sp else torch.distributed.get_world_size()
            else:
                dp_size = 1

            loss = torch.sum(loss) / (valid_token_this_rank + 1e-8) * dp_size

            loss = loss / n_micro_batches  # normalize loss

            if do_backward:
                loss.backward()
            return loss

    def training_step(self, batch: TensorDict):
        start_time = time.time()

        self.fsdp_model.train()

        log_gpu_memory_usage("Before optimizer zero_grad", logger=logger)

        self.optimizer.zero_grad()

        log_gpu_memory_usage("After optimizer zero_grad", logger=logger)

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(batch=micro_batch, n_micro_batches=n_micro_batches)
            step_loss += loss.item()

        if self.config.model.strategy == "fsdp":
            grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        elif self.config.model.strategy == "fsdp2":
            grad_norm = fsdp2_clip_grad_norm_(self.fsdp_model.parameters(), max_norm=self.config.optim.clip_grad)
        else:
            raise NotImplementedError(f"not implement {self.config.model.strategy}")

        log_gpu_memory_usage("Before optimizer step", logger=logger)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        log_gpu_memory_usage("After optimizer step", logger=logger)

        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage("After offload weights", logger=logger)

        step_loss = torch.tensor(step_loss).to(self.device_name)

        # compute time spent per step
        end_time = time.time()
        spend_time_per_step = end_time - start_time

        if is_cuda_available:
            torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        elif is_npu_available:
            torch.distributed.all_reduce(step_loss)
            step_loss /= self.device_mesh.size(0)
        return {
            "train/loss": step_loss.detach().item(),
            "train/lr(1e-3)": lr * 1e3,
            "train/time(s)": spend_time_per_step,
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            if is_cuda_available:
                torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
            elif is_npu_available:
                torch.distributed.all_reduce(loss)
                loss /= self.device_mesh.size(0)
        return loss

    # ------------------------------------------------------------------
    # Generation-based validation metrics (Word-F1, Profile Coverage)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_word_f1(prediction: str, reference: str) -> float:
        """Bag-of-words F1 between prediction and reference."""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        if not pred_tokens or not ref_tokens:
            return 0.0
        common = pred_tokens & ref_tokens
        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0.0
        return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    @staticmethod
    def _compute_profile_coverage(generation: str, profile: str) -> float:
        """Fraction of meaningful profile keywords appearing in generation."""
        _STOP_WORDS = {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "shall",
            "can",
            "need",
            "dare",
            "ought",
            "used",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "through",
            "during",
            "before",
            "after",
            "above",
            "below",
            "between",
            "out",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "both",
            "either",
            "neither",
            "each",
            "every",
            "all",
            "any",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "only",
            "own",
            "same",
            "than",
            "too",
            "very",
            "just",
            "because",
            "if",
            "when",
            "where",
            "how",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "it",
            "its",
            "they",
            "them",
            "their",
            "about",
            "up",
            "down",
            "here",
            "there",
            "also",
        }
        profile_words = {
            w.lower().strip(".,!?;:\"'()[]{}")
            for w in profile.split()
            if len(w) > 2 and w.lower().strip(".,!?;:\"'()[]{}") not in _STOP_WORDS
        }
        if not profile_words:
            return 0.0
        gen_words = {w.lower().strip(".,!?;:\"'()[]{}") for w in generation.split()}
        return len(profile_words & gen_words) / len(profile_words)

    @torch.no_grad()
    def generation_validation(self, global_step: int, tracking=None, max_samples: int = 50):
        """Generate text from val samples and compute Word-F1 / Profile Coverage.

        Called only at *save steps* (every save_freq steps) to avoid slowing down training.
        All ranks participate in generation (required for FSDP2 all-gathers).
        Only rank 0 prepares prompts, computes metrics, and logs.

        For multi-turn SFT data the conversation format is:
            system (profile) -> assistant -> user -> assistant -> user -> ...
        where the role labels are swapped (assistant = real user, user = real assistant).

        Per-turn generation: for each conversation, we iterate over ALL assistant
        turns (not just the last one) to measure compounding error across turns.
        Each (prompt_up_to_turn_i, gt_turn_i, turn_index) is a separate sample.
        Total samples are capped at max_samples, uniformly sampled across turns.
        """
        rank = self.device_mesh.get_rank()
        self.fsdp_model.eval()

        # Check val_dataset has messages
        if not hasattr(self.val_dataset, "messages") or self.val_dataset.messages is None:
            if rank == 0:
                print("[gen_val] val_dataset has no 'messages' attribute, skipping generation validation.")
            self.fsdp_model.train()
            return

        if rank == 0:
            print(f"\n[gen_val] Running generation validation at step {global_step}...")

        messages_list = self.val_dataset.messages

        # Prepare tokenizer
        orig_padding_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        word_f1_scores = []
        p_cover_scores = []
        gen_lengths = []
        turn_indices = []  # track which assistant turn each sample belongs to
        successful_turn_indices = []  # only for successfully generated samples
        gt_texts = []
        profiles = []

        # Rank 0 prepares all prompts; other ranks will receive broadcast tensors
        prompt_input_ids_list = []
        prompt_attn_mask_list = []

        if rank == 0:
            # Collect all (prompt, gt, turn_index, profile) tuples across conversations
            all_candidates = []  # list of (prompt_messages, gt_text, turn_index, profile)
            for idx in range(len(messages_list)):
                messages = messages_list[idx]
                if not messages:
                    continue

                # Extract profile from system message
                profile = ""
                if messages[0].get("role") == "system":
                    profile = messages[0].get("content", "")

                # Find ALL assistant turn positions
                asst_turn_count = 0
                for msg_idx, msg in enumerate(messages):
                    if msg["role"] == "assistant":
                        gt_text = msg.get("content", "")
                        prompt_messages = messages[:msg_idx]
                        if prompt_messages and gt_text:
                            all_candidates.append((prompt_messages, gt_text, asst_turn_count, profile))
                        asst_turn_count += 1

            # Uniformly sample up to max_samples
            if len(all_candidates) > max_samples:
                import random

                rng = random.Random(global_step)
                all_candidates = rng.sample(all_candidates, max_samples)

            # Tokenize all candidates
            for prompt_messages, gt_text, turn_index, profile in all_candidates:
                try:
                    prompt_text = self.tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    inputs = self.tokenizer(
                        prompt_text,
                        return_tensors="pt",
                        truncation=True,
                        max_length=self.config.data.max_length - 256,
                    )
                    prompt_input_ids_list.append(inputs["input_ids"].squeeze(0))
                    prompt_attn_mask_list.append(inputs["attention_mask"].squeeze(0))
                    gt_texts.append(gt_text)
                    profiles.append(profile)
                    turn_indices.append(turn_index)
                except Exception as e:
                    print(f"[gen_val] tokenization failed (turn {turn_index}): {e}")
                    continue

            print(
                f"[gen_val] Prepared {len(prompt_input_ids_list)} samples across turns "
                f"(from {len(messages_list)} conversations)"
            )

        # Generate one sample at a time — all ranks must call generate together
        # Broadcast sample count from rank 0
        n_valid = torch.tensor([len(prompt_input_ids_list) if rank == 0 else 0], device=self.device_name)
        torch.distributed.broadcast(n_valid, src=0)
        n_valid = n_valid.item()

        for sample_idx in range(n_valid):
            try:
                # Rank 0 broadcasts input_ids and attention_mask
                if rank == 0:
                    ids = prompt_input_ids_list[sample_idx].to(self.device_name)
                    mask = prompt_attn_mask_list[sample_idx].to(self.device_name)
                    seq_len = torch.tensor([ids.shape[0]], device=self.device_name)
                else:
                    seq_len = torch.tensor([0], device=self.device_name)

                torch.distributed.broadcast(seq_len, src=0)
                sl = seq_len.item()

                if rank != 0:
                    ids = torch.zeros(sl, dtype=torch.long, device=self.device_name)
                    mask = torch.zeros(sl, dtype=torch.long, device=self.device_name)

                torch.distributed.broadcast(ids, src=0)
                torch.distributed.broadcast(mask, src=0)

                input_ids = ids.unsqueeze(0)  # (1, seq_len)
                attention_mask = mask.unsqueeze(0)

                # All ranks call generate. FSDP-wrapped modules don't expose
                # `generate` directly, so summon full params and call generate
                # on the underlying HF module.
                with FSDP.summon_full_params(self.fsdp_model, recurse=True, writeback=False):
                    inner_model = self.fsdp_model.module if hasattr(self.fsdp_model, "module") else self.fsdp_model
                    output_ids = inner_model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=256,
                        do_sample=False,
                        use_cache=False,
                    )

                # Only rank 0 decodes and computes metrics
                if rank == 0:
                    gen_ids = output_ids[0, input_ids.shape[1] :]
                    gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

                    gt_text = gt_texts[sample_idx]
                    profile = profiles[sample_idx]

                    word_f1_scores.append(self._compute_word_f1(gen_text, gt_text))
                    p_cover_scores.append(self._compute_profile_coverage(gen_text, profile) if profile else 0.0)
                    gen_lengths.append(len(gen_text.split()))
                    # Only record turn_index for successfully generated samples
                    successful_turn_indices.append(turn_indices[sample_idx])

            except Exception as e:
                if rank == 0:
                    print(f"[gen_val] sample {sample_idx} generation failed: {e}")
                continue

        # Aggregate and log (rank 0 only)
        if rank == 0 and word_f1_scores:
            avg_word_f1 = sum(word_f1_scores) / len(word_f1_scores)
            avg_p_cover = sum(p_cover_scores) / len(p_cover_scores)
            avg_gen_len = sum(gen_lengths) / len(gen_lengths)

            metrics = {
                "val/word_f1": avg_word_f1,
                "val/profile_coverage": avg_p_cover,
                "val/gen_length": avg_gen_len,
            }

            # Per-turn aggregation for compounding error curve
            from collections import defaultdict

            turn_f1 = defaultdict(list)
            turn_pcover = defaultdict(list)
            for i, t_idx in enumerate(successful_turn_indices):
                turn_f1[t_idx].append(word_f1_scores[i])
                turn_pcover[t_idx].append(p_cover_scores[i])

            # Log per-turn metrics
            for t_idx in sorted(turn_f1.keys()):
                avg_tf1 = sum(turn_f1[t_idx]) / len(turn_f1[t_idx])
                avg_tpc = sum(turn_pcover[t_idx]) / len(turn_pcover[t_idx])
                metrics[f"val/word_f1_turn_{t_idx}"] = avg_tf1
                metrics[f"val/p_cover_turn_{t_idx}"] = avg_tpc

            # Build per-turn summary string
            turn_strs = []
            for t_idx in sorted(turn_f1.keys()):
                n_t = len(turn_f1[t_idx])
                avg_tf1 = sum(turn_f1[t_idx]) / n_t
                turn_strs.append(f"t{t_idx}={avg_tf1:.2f}({n_t})")
            per_turn_summary = " ".join(turn_strs)

            print(
                f"[gen_val] step={global_step}  n={len(word_f1_scores)}  "
                f"word_f1={avg_word_f1:.4f}  p_cover={avg_p_cover:.4f}  "
                f"gen_len={avg_gen_len:.1f} | {per_turn_summary}"
            )
            if tracking is not None:
                tracking.log(data=metrics, step=global_step)
        elif rank == 0:
            print("[gen_val] No valid samples for generation validation.")

        # Restore state
        self.tokenizer.padding_side = orig_padding_side
        self.fsdp_model.train()

    def benchmark_validation(self, global_step: int, tracking=None):
        """Run MCQ benchmark evaluation during training using FSDP model forward pass.

        Loads test data from benchmark_eval_dir (medqa, mmlu, medmcqa jsonl files),
        tokenizes prompts, runs forward pass, picks the option letter (A/B/C/D) with
        highest logprob at the next-token position, and computes accuracy.

        Only rank 0 loads data and reports results; all ranks participate in forward pass
        (required for FSDP all-gathers).
        """
        if not self.benchmark_eval_dir:
            return

        import json as _json

        rank = self.device_mesh.get_rank()
        self.fsdp_model.eval()

        # Discover dataset files
        eval_dir = self.benchmark_eval_dir
        dataset_files = {}
        for name, filename in [
            ("medqa", "medqa_test.jsonl"),
            ("mmlu", "mmlu_medical_test.jsonl"),
            ("medmcqa", "medmcqa_test.jsonl"),
        ]:
            path = os.path.join(eval_dir, filename)
            if os.path.isfile(path):
                dataset_files[name] = path

        if not dataset_files and rank == 0:
            print(f"[bench_eval] No test files found in {eval_dir}, skipping.")
            self.fsdp_model.train()
            return

        from verl.utils.med_mcq import render_mcq_prompt

        # Helper to format questions
        def _format(item, dtype):
            if dtype == "medqa":
                return render_mcq_prompt(item["question"], item["options"])
            elif dtype == "mmlu":
                return render_mcq_prompt(item["question"], item["choices"])
            else:  # medmcqa
                return render_mcq_prompt(item["question"], [item["opa"], item["opb"], item["opc"], item["opd"]])

        def _answer(item, dtype):
            if dtype == "medqa":
                return item["answer_idx"]
            elif dtype == "mmlu":
                return chr(65 + item["answer"])
            else:  # medmcqa
                return chr(65 + item["cop"]) if item.get("cop", -1) != -1 else "A"

        # Get token IDs for A, B, C, D
        option_token_ids = {}
        for letter in ["A", "B", "C", "D"]:
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            option_token_ids[letter] = ids[-1]  # take last token in case of multi-token

        all_metrics = {}

        for dataset_name, dataset_path in dataset_files.items():
            # Rank 0 loads data
            if rank == 0:
                data = []
                with open(dataset_path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            data.append(_json.loads(line))
                # Subsample
                if len(data) > self.benchmark_eval_max_samples:
                    import random

                    rng = random.Random(42)
                    data = rng.sample(data, self.benchmark_eval_max_samples)

                prompts = [_format(item, dataset_name) for item in data]
                answers = [_answer(item, dataset_name) for item in data]
                n_samples = len(prompts)
            else:
                prompts = None
                answers = None
                n_samples = 0

            # Broadcast n_samples
            n_tensor = torch.tensor([n_samples], dtype=torch.long, device=self.device_name)
            torch.distributed.broadcast(n_tensor, src=0)
            n_samples = n_tensor.item()

            if n_samples == 0:
                continue

            # Rank 0 tokenizes, broadcasts
            if rank == 0:
                orig_padding_side = self.tokenizer.padding_side
                self.tokenizer.padding_side = "left"
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                encoded = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                    add_special_tokens=True,
                )
                input_ids = encoded["input_ids"].to(self.device_name)
                attention_mask = encoded["attention_mask"].to(self.device_name)
                self.tokenizer.padding_side = orig_padding_side
            else:
                # Receive shape first
                input_ids = None
                attention_mask = None

            # Broadcast tensor shapes then tensors
            if rank == 0:
                shape_tensor = torch.tensor(list(input_ids.shape), dtype=torch.long, device=self.device_name)
            else:
                shape_tensor = torch.zeros(2, dtype=torch.long, device=self.device_name)
            torch.distributed.broadcast(shape_tensor, src=0)
            bs, seq_len = shape_tensor[0].item(), shape_tensor[1].item()

            if rank != 0:
                input_ids = torch.zeros(bs, seq_len, dtype=torch.long, device=self.device_name)
                attention_mask = torch.zeros(bs, seq_len, dtype=torch.long, device=self.device_name)
            torch.distributed.broadcast(input_ids, src=0)
            torch.distributed.broadcast(attention_mask, src=0)

            # Forward pass in batches
            correct = 0
            total = n_samples
            micro_bs = max(1, min(16, self.config.data.micro_batch_size_per_gpu * 4))

            for start in range(0, bs, micro_bs):
                end = min(start + micro_bs, bs)
                batch_ids = input_ids[start:end]
                batch_mask = attention_mask[start:end]

                with torch.no_grad():
                    outputs = self.fsdp_model(
                        input_ids=batch_ids,
                        attention_mask=batch_mask,
                        use_cache=False,
                    )
                    # Get logits at last non-padding position for each sample
                    logits = outputs.logits  # (micro_bs, seq_len, vocab)

                if rank == 0:
                    for i in range(end - start):
                        # Find last non-padding token position
                        mask_row = batch_mask[i]
                        last_pos = mask_row.sum().item() - 1
                        next_token_logits = logits[i, last_pos, :]  # (vocab,)

                        # Get logprobs for A, B, C, D
                        best_letter = "A"
                        best_logit = float("-inf")
                        for letter, tid in option_token_ids.items():
                            if next_token_logits[tid].item() > best_logit:
                                best_logit = next_token_logits[tid].item()
                                best_letter = letter

                        if answers[start + i] == best_letter:
                            correct += 1

            torch.distributed.barrier()

            if rank == 0:
                acc = correct / total if total > 0 else 0.0
                all_metrics[f"bench/{dataset_name}_acc"] = acc
                print(f"[bench_eval] step={global_step} {dataset_name}: {acc:.4f} ({correct}/{total})")

        if rank == 0 and all_metrics:
            # Log average
            avg_acc = sum(all_metrics.values()) / len(all_metrics)
            all_metrics["bench/avg_acc"] = avg_acc
            print(f"[bench_eval] step={global_step} avg_acc={avg_acc:.4f}")
            if tracking is not None:
                tracking.log(data=all_metrics, step=global_step)

        self.fsdp_model.train()

    def save_checkpoint(self, step):
        """Save checkpoint using FSDPCheckpointManager with improved tracking"""
        from verl.utils.fs import local_mkdir_safe

        # Determine checkpoint path
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}")

        if self.device_mesh.get_rank() == 0:
            print(f"Saving checkpoint to: {local_global_step_folder}")

        # Get max checkpoints to keep
        max_ckpt_to_keep = getattr(self.config.trainer, "max_ckpt_to_keep", None)

        # Use checkpoint manager to save
        self.checkpoint_manager.save_checkpoint(
            local_path=local_global_step_folder, global_step=step, max_ckpt_to_keep=max_ckpt_to_keep
        )

        # Save dataloader state
        if self.device_mesh.get_rank() == 0:
            local_mkdir_safe(local_global_step_folder)
            dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")

            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = self.train_dataloader.state_dict()
            torch.save(dataloader_state_dict, dataloader_local_path)
            print(f"Saved dataloader state to: {dataloader_local_path}")

            # Update latest checkpoint tracker (atomic write)
            tracker_file = get_checkpoint_tracker_filename(self.config.trainer.default_local_dir)
            temp_tracker_file = tracker_file + ".tmp"
            with open(temp_tracker_file, "w") as f:
                f.write(str(step))
            os.rename(temp_tracker_file, tracker_file)
            print(f"Updated checkpoint tracker: {tracker_file}")

        # Copy to HDFS if configured
        if self.device_mesh.get_rank() == 0 and getattr(self.config.trainer, "default_hdfs_dir", None):
            hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
            hdfs_io.copy(src=local_global_step_folder, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)

        torch.distributed.barrier()

    def _init_checkpoint_manager(self):
        """Initialize checkpoint manager with proper configuration"""
        # Get checkpoint configuration from config, with defaults
        checkpoint_config = getattr(self.config.trainer, "checkpoint", {})

        # Set default values if not specified
        save_contents = checkpoint_config.get("save_contents", ["model", "optimizer", "extra"])
        load_contents = checkpoint_config.get("load_contents", save_contents)

        # Create checkpoint config dict
        checkpoint_config_dict = {
            "load_contents": load_contents,
            "save_contents": save_contents,
        }

        # Convert to DictConfig for compatibility
        checkpoint_config_dict = DictConfig(checkpoint_config_dict)

        # Initialize checkpoint manager
        self.checkpoint_manager = FSDPCheckpointManager(
            model=self.fsdp_model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            processing_class=self.tokenizer,
            checkpoint_config=checkpoint_config_dict,
        )

    def load_checkpoint(self):
        # Determine resume path based on configuration
        checkpoint_path = self._determine_resume_path()

        if checkpoint_path is None:
            return 0

        # extract resume step from checkpoint path
        resume_step = extract_step(checkpoint_path)
        if resume_step is None:
            log_with_rank(
                f"Warning: Could not extract step number from {checkpoint_path}, starting from step 0",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )
            return 0
        self.resume_global_step = resume_step

        # Use checkpoint manager to load model state
        self.checkpoint_manager.load_checkpoint(checkpoint_path)
        log_with_rank(
            f"Successfully loaded model checkpoint from {checkpoint_path} (step {resume_step})",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # Always load dataloader state for StatefulDataLoader
        self._load_dataloader_state(checkpoint_path)

        return resume_step

    def _load_dataloader_state(self, checkpoint_path: str):
        """Load dataloader state from checkpoint"""
        dataloader_path = os.path.join(checkpoint_path, "data.pt")

        if os.path.exists(dataloader_path):
            # Use StatefulDataLoader's built-in state dict functionality
            dataloader_state_dict = torch.load(dataloader_path, map_location="cpu", weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)

            log_with_rank(
                f"Successfully loaded dataloader state from {dataloader_path}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        else:
            log_with_rank(
                f"Warning: No dataloader state found at {dataloader_path}, will start from scratch",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                level=logging.WARNING,
                log_only_rank_0=True,
            )

    def _determine_resume_path(self):
        """Determine the path to resume from based on resume_mode configuration"""
        resume_mode = getattr(self.config.trainer, "resume_mode", "auto")
        resume_from_path = getattr(self.config.trainer, "resume_from_path", None)

        if resume_mode == "disable":
            return None
        elif resume_mode == "auto":
            if resume_from_path is not None:
                assert os.path.exists(resume_from_path), (
                    "resume_from_path must be null or an existing path when resume_mode is 'auto'"
                )
                assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
                return resume_from_path
            # Try to find the latest checkpoint in the default directory
            return self._find_latest_checkpoint()
        elif resume_mode == "resume_path":
            assert os.path.exists(resume_from_path), (
                "resume_from_path must be an existing path when resume_mode is 'resume_path'"
            )
            assert "global_step_" in resume_from_path, "resume_from_path must specify the global_steps"
            return resume_from_path
        else:
            raise ValueError(f"Invalid resume_mode: {resume_mode}. Must be 'auto', 'disable', or 'resume_path'")

    def _find_latest_checkpoint(self):
        """Find the latest checkpoint in the default local directory"""
        checkpoint_dir = self.config.trainer.default_local_dir

        if not os.path.exists(checkpoint_dir):
            return None

        latest_checkpoint = find_latest_ckpt_path(checkpoint_dir)

        if latest_checkpoint and self.device_mesh.get_rank() == 0:
            step_num = extract_step(latest_checkpoint)
            print(f"Found latest checkpoint: {latest_checkpoint} (step {step_num})")

        return latest_checkpoint

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        if rank == 0:
            tracking = Tracking(
                project_name=self.config.trainer.project_name,
                experiment_name=self.config.trainer.experiment_name,
                default_backend=self.config.trainer.logger,
                config=OmegaConf.to_container(self.config, resolve=True),
            )

        global_step = self.resume_global_step  # Start from resumed step
        last_valid_metric = None
        # compute the total training steps.
        # the total training steps in SFT is mainly for early exit
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        log_with_rank(
            f"Total training steps: {self.total_training_steps},",
            logger=logger,
            rank=self.device_mesh.get_rank(),
            log_only_rank_0=True,
        )

        # With StatefulDataLoader, we don't need to manually calculate epochs and steps
        # The dataloader will automatically resume from where it left off
        if global_step > 0:
            log_with_rank(
                f"StatefulDataLoader will automatically resume from global step: {global_step}",
                logger=logger,
                rank=self.device_mesh.get_rank(),
                log_only_rank_0=True,
            )

        # Calculate which epoch we're starting from for sampler.set_epoch()
        start_epoch = global_step // self.steps_per_epoch

        train_time = 0
        for epoch in range(start_epoch, self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)

            for step_in_epoch, data in enumerate(
                tqdm(
                    self.train_dataloader,
                    initial=global_step % self.steps_per_epoch if epoch == start_epoch else 0,
                    total=self.steps_per_epoch,
                    desc=f"Epoch {epoch + 1}/{self.config.trainer.total_epochs}",
                    disable=rank != 0,
                )
            ):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).to(self.device_name)
                metric = self.training_step(data)
                train_time += metric["train/time(s)"]
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                is_last_step = global_step >= self.total_training_steps
                is_valid_step = global_step % self.config.trainer.test_freq == 0
                is_save_step = global_step % self.config.trainer.save_freq == 0

                # early exit or validation step
                if is_last_step or (self.config.trainer.test_freq > 0 and is_valid_step):
                    # Perform validation
                    val_losses = []
                    for val_data in self.val_dataloader:
                        val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).to(
                            self.device_name
                        )
                        val_loss = self.validation_step(val_data)
                        val_losses.append(val_loss)
                    if rank == 0:
                        val_loss = torch.mean(torch.stack(val_losses))
                        metric = {"val/loss": val_loss.detach().item()}
                        tracking.log(data=metric, step=global_step)
                        last_valid_metric = metric
                    torch.distributed.barrier()

                if is_last_step or (self.config.trainer.save_freq > 0 and is_save_step):
                    # Run generation-based validation at save steps
                    self.generation_validation(
                        global_step=global_step,
                        tracking=tracking if rank == 0 else None,
                    )
                    # Run benchmark MCQ evaluation at save steps
                    self.benchmark_validation(
                        global_step=global_step,
                        tracking=tracking if rank == 0 else None,
                    )
                    self.save_checkpoint(step=global_step)

                if is_last_step:
                    if rank == 0:
                        print(f"Total time for train steps: {train_time:.2f}s")
                        print(f"Final validation metrics: {last_valid_metric}")
                    return


def run_sft(config):
    device_name = get_device_name()
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )
    # build tokenizer and datasets first
    from verl.utils import hf_tokenizer

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(
        config.data.train_files, config.data, tokenizer, max_samples=config.data.get("train_max_samples", -1)
    )
    val_dataset = create_sft_dataset(
        config.data.val_files, config.data, tokenizer, max_samples=config.data.get("val_max_samples", -1)
    )

    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )

    trainer.fit()

    destroy_global_process_group()


@hydra.main(config_path=".", config_name="sft_trainer", version_base=None)
def main(config):
    run_sft(config)


def create_sft_dataset(data_paths, data_config, tokenizer, max_samples=-1):
    """Create a dataset."""
    # build dataset
    # First check if a custom dataset class is specified
    if data_config.custom_cls.get("path", None):
        from verl.utils.import_utils import load_extern_type

        dataset_cls = load_extern_type(data_config.custom_cls.path, data_config.custom_cls.name)
    # Then check if multi-turn dataset should be used
    elif data_config.get("multiturn", {}).get("enable", False):
        dataset_cls = MultiTurnSFTDataset
    # Default to single-turn dataset
    else:
        dataset_cls = SFTDataset

    # Create datasets based on the selected class
    dataset = dataset_cls(parquet_files=data_paths, tokenizer=tokenizer, config=data_config, max_samples=max_samples)
    return dataset


if __name__ == "__main__":
    main()
