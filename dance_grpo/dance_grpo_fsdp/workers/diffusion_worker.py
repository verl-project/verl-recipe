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
DiffusionActorRolloutWorker – Ray remote worker for DanceGRPO.

This worker is the single distributed process that owns the DiT model weights
and coordinates rollout, reward, and actor update.  It follows the same
structural conventions as ``verl.workers.fsdp_workers.ActorRolloutRefWorker``:

  - ``__init__``               initialise distributed env, build models
  - ``generate_sequences``     delegate to :class:`DiffusionRollout`
  - ``update_actor``           delegate to :class:`DiffusionActor`
  - ``save_checkpoint``        save DiT weights (+ optionally optimizer)
  - ``load_checkpoint``        restore DiT weights
  - ``init_fixed_eval``        capture fixed evaluation prompts
  - ``run_fixed_eval``         run deterministic eval images every N steps

Heavy algorithm logic lives in :mod:`~recipe.dance_grpo.algorithms.grpo_diffusion`.
Reward computation lives in :class:`~recipe.dance_grpo.reward.HPSv3RewardManager`.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import re

import numpy as np
import torch
import torch.distributed
import torch.distributed as dist
from omegaconf import DictConfig
from PIL import Image
from recipe.dance_grpo.reward.hpsv3_reward_manager import HPSv3RewardManager
from recipe.dance_grpo.workers.diffusion_actor import DiffusionActor
from recipe.dance_grpo.workers.diffusion_rollout import DiffusionRollout
from torch.distributed._composable.fsdp import CPUOffloadPolicy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from transformers import AutoProcessor

from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import get_device_name, get_nccl_backend
from verl.utils.fsdp_utils import (
    fsdp2_load_full_state_dict,
    get_fsdp_full_state_dict,
    load_fsdp2_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp2_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.profiler import DistProfilerExtension
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

__all__ = ["DiffusionActorRolloutWorker"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

_device_name = get_device_name()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_device_mesh(world_size: int, fsdp_size: int):
    """Return an FSDP device mesh with shape ``[dp, fsdp]``."""
    if fsdp_size <= 0 or fsdp_size > world_size:
        fsdp_size = world_size
    return init_device_mesh(
        _device_name,
        mesh_shape=(world_size // fsdp_size, fsdp_size),
        mesh_dim_names=["dp", "fsdp"],
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


class DiffusionActorRolloutWorker(Worker, DistProfilerExtension):
    """Ray remote worker for DanceGRPO diffusion training.

    This worker owns:
      - A frozen MLLM (text/image encoder)
      - A trainable DiT (denoising transformer)
      - A frozen VAE decoder
      - A :class:`~recipe.dance_grpo.workers.diffusion_rollout.DiffusionRollout`
      - A :class:`~recipe.dance_grpo.workers.diffusion_actor.DiffusionActor`
      - A :class:`~recipe.dance_grpo.reward.HPSv3RewardManager`

    Args:
        config: Full training config (``actor_rollout_ref`` sub-node is used
            for actor/rollout settings; ``model`` for paths).
        role: Worker role string (``"hybrid"`` or ``"actor_rollout"``).
    """

    def __init__(self, config: DictConfig, role: str = "hybrid", **kwargs) -> None:
        Worker.__init__(self)

        self.config = config
        self.role = role
        self._is_actor = role in ("actor", "hybrid", "actor_rollout")
        self._is_rollout = role in ("rollout", "hybrid")

        # ---- Initialise distributed process group ---- #
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{_device_name}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # ---- Device meshes ---- #
        self.device_mesh = _create_device_mesh(
            world_size=world_size,
            fsdp_size=self.config.actor.fsdp_config.fsdp_size,
        )

        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size

        self.ulysses_device_mesh = None
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                _device_name,
                mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                mesh_dim_names=["dp", "sp"],
            )

        # ---- Dispatch registration ---- #
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor",
                dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(),
                is_collect=is_collect,
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=rank, is_collect=True)
        self._register_dispatch_collect_info("rollout", dp_rank=rank, is_collect=True)

        # ---- Ulysses sharding manager ---- #
        self.ulysses_sharding_manager = (
            FSDPUlyssesShardingManager(self.ulysses_device_mesh) if self.ulysses_sequence_parallel_size > 1 else None
        )

        # ---- Placeholders ---- #
        self.dit = None
        self.full_model = None
        self.rollout: DiffusionRollout | None = None
        self.actor: DiffusionActor | None = None
        self.reward_manager: HPSv3RewardManager | None = None
        self._fixed_eval_samples: list[dict] | None = None

        # ---- Build everything ---- #
        self._build_model_optimizer()

    # ------------------------------------------------------------------
    # Model / optimiser construction
    # ------------------------------------------------------------------

    def _apply_fsdp2(
        self,
        model,
        no_split_module_classes: list[str],
        is_train: bool = False,
        cpu_offload: bool = False,
        grad_checkpointing: bool = False,
    ):
        """Wrap *model* with FSDP2 (fully_shard), optional CPU offload and
        activation checkpointing."""
        offload_policy = CPUOffloadPolicy() if cpu_offload else None
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32) if is_train else None

        for _, module in model.named_modules():
            if module.__class__.__name__ in no_split_module_classes:
                if is_train and grad_checkpointing:
                    checkpoint_wrapper(module, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
                kwargs: dict = {}
                if cpu_offload:
                    kwargs["offload_policy"] = offload_policy
                if mp_policy is not None:
                    kwargs["mp_policy"] = mp_policy
                fully_shard(module, **kwargs)

        root_kwargs: dict = {}
        if cpu_offload:
            root_kwargs["offload_policy"] = offload_policy
        if mp_policy is not None:
            root_kwargs["mp_policy"] = mp_policy
        fully_shard(model, **root_kwargs)
        return model

    def _extract_and_wrap_models(self, model_id_or_path: str):
        """Load the Mammothmoda2 model and return FSDP2-wrapped sub-modules.

        Returns:
            ``(mllm_model, dit_model, vae_model, full_model, processor)``
        """
        from mammothmoda2.model import Mammothmoda2Model

        full_model = Mammothmoda2Model.from_pretrained(
            model_id_or_path, torch_dtype=torch.bfloat16, t2i_generate=True
        ).to(torch.bfloat16)
        processor = AutoProcessor.from_pretrained(model_id_or_path, t2i_generate=True, ar_height=32, ar_width=32)

        mllm_model = full_model.llm_model
        dit_model = full_model.gen_transformer
        vae_model = full_model.gen_vae

        # Freeze MLLM and VAE; train only DiT
        mllm_model.eval()
        for p in mllm_model.parameters():
            p.requires_grad = False

        vae_model.eval()
        for p in vae_model.parameters():
            p.requires_grad = False
        vae_model.to(_device_name)

        dit_model.train()
        freeze_non_attn_ffn = getattr(self.config.actor, "freeze_non_attn_ffn", False)
        if freeze_non_attn_ffn:
            import re as _re

            for p in dit_model.parameters():
                p.requires_grad = False
            _pat = _re.compile(
                r"\battn\b|\battention\b|self_attn|cross_attn|\.ff\.|\.ff1\.|" r"ff\.net|\bmlp\b|\.ffn\.|feed_forward"
            )
            for name, p in dit_model.named_parameters():
                if _pat.search(name):
                    p.requires_grad = True
        else:
            for p in dit_model.parameters():
                p.requires_grad = True

        mllm_no_split = getattr(mllm_model, "_no_split_modules", [])
        dit_no_split = getattr(dit_model, "_no_split_modules", [])

        cpu_offload = self.config.actor.fsdp_config.offload_policy
        grad_ckpt = self.config.model.get("enable_gradient_checkpointing", False)

        # Convert DiT to fp32 before FSDP2 so Adam operates on fp32 master weights.
        dit_model = dit_model.float()

        mllm_model = self._apply_fsdp2(mllm_model, mllm_no_split, is_train=False, cpu_offload=cpu_offload)
        dit_model = self._apply_fsdp2(
            dit_model, dit_no_split, is_train=True, cpu_offload=cpu_offload, grad_checkpointing=grad_ckpt
        )
        if grad_ckpt:
            dit_model.enable_gradient_checkpointing()

        return mllm_model, dit_model, vae_model, full_model, processor

    def _build_model_optimizer(self) -> None:
        """Initialise models, optimizer, lr scheduler, rollout, actor, reward."""
        mllm, dit, vae, full_model, processor = self._extract_and_wrap_models(self.config.model.path)

        self.dit = dit
        self.full_model = full_model

        # ---- Optimizer ---- #
        trainable_params = [p for p in dit.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.actor.optim.lr,
            betas=(0.9, 0.999),
            weight_decay=self.config.actor.optim.weight_decay,
            eps=1e-8,
        )

        # ---- LR scheduler ---- #
        from diffusers.optimization import get_scheduler

        lr_scheduler = get_scheduler(
            self.config.actor.optim.lr_scheduler.name,
            optimizer=optimizer,
            num_warmup_steps=self.config.actor.optim.lr_scheduler.num_warmup_steps,
            num_training_steps=self.config.actor.optim.lr_scheduler.num_training_steps,
            num_cycles=self.config.actor.optim.lr_scheduler.get("num_cycles", 1),
            power=self.config.actor.optim.lr_scheduler.get("power", 1.0),
        )

        # ---- DiffusionRollout ---- #
        rollout_config = self.config.rollout
        self.rollout = DiffusionRollout(
            mllm_model=mllm,
            dit_model=dit,
            vae_model=vae,
            full_model=full_model,
            processor=processor,
            config=rollout_config,
        )

        # ---- DiffusionActor ---- #
        actor_config = self.config.actor
        self.actor = DiffusionActor(
            config=actor_config,
            dit_module=dit,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        )
        # Attach the Ulysses sharding manager so update_policy can use it.
        if self.ulysses_sharding_manager is not None:
            self.actor._ulysses_sharding_manager = self.ulysses_sharding_manager

        # ---- Reward manager ---- #
        self.reward_manager = HPSv3RewardManager(
            checkpoint_path=self.config.model.reward_model_path,
            device=_device_name,
            reward_scale=0.1,
        )

    # ------------------------------------------------------------------
    # Rollout
    # ------------------------------------------------------------------

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, data: DataProto) -> DataProto:
        """Generate rollout sequences and compute rewards.

        Steps:
          1. Repeat prompts *n* times (one group per prompt).
          2. Run :meth:`DiffusionRollout.generate_sequences`.
          3. Score images with :meth:`HPSv3RewardManager.compute_reward`.
          4. Persist per-rank JSONL metadata.

        Args:
            data: DataProto with ``non_tensor_batch["prompts"]`` and
                ``meta_info["global_step"]`` / ``meta_info["experiment_name"]``.

        Returns:
            DataProto with trajectory tensors, conditioning, rewards, and images.
        """
        self.rollout._current_global_step = data.meta_info.get("global_step", 0)
        self.rollout._current_experiment_name = data.meta_info.get("experiment_name", "dance_grpo")

        # Repeat each prompt n times so the rollout generates n samples per prompt.
        data = data.repeat(repeat_times=self.config.rollout.n, interleave=True)

        with torch.no_grad():
            output = self.rollout.generate_sequences(data)
        data = data.union(output)

        # Score images
        data = self.reward_manager.compute_reward(data)

        # Persist rollout metadata
        self._save_rollout_metadata(data)

        return data

    # ------------------------------------------------------------------
    # Actor update
    # ------------------------------------------------------------------

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto) -> DataProto:
        """Run one policy update step.

        Delegates entirely to :meth:`DiffusionActor.update_policy`.

        Args:
            data: DataProto produced by :meth:`generate_sequences`.

        Returns:
            DataProto with ``meta_info["metrics"]``.
        """
        assert self._is_actor, "update_actor called on a non-actor worker"
        return self.actor.update_policy(data)

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self,
        local_path: str,
        hdfs_path: str | None = None,
        global_step: int = 0,
        max_ckpt_to_keep: int | None = None,
    ) -> None:
        """Save the DiT checkpoint (and optionally optimizer / lr_scheduler).

        On multi-GPU runs only model weights are saved (sharding the optimizer
        state dict across ranks is not yet implemented).

        Args:
            local_path: Directory where the checkpoint file will be written.
            hdfs_path: Unused (reserved for remote storage).
            global_step: Current training step (stored in the checkpoint dict).
            max_ckpt_to_keep: Unused (reserved for rotation logic).
        """
        assert self._is_actor

        use_offload = bool(self.config.actor.fsdp_config.get("offload_policy", False))
        if use_offload:
            load_fsdp2_model_to_gpu(self.dit)
            load_fsdp_optimizer(self.actor.optimizer)

        os.makedirs(local_path, exist_ok=True)
        save_contents = set(getattr(self.config.actor.checkpoint, "save_contents", ["model"]))
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        model_state = None
        if "model" in save_contents:
            model_state = get_fsdp_full_state_dict(self.dit, offload_to_cpu=True, rank0_only=True)

        if dist.get_rank() == 0:
            ckpt: dict = {"global_step": int(global_step)}
            if model_state is not None:
                ckpt["model"] = model_state
            if "optimizer" in save_contents and world_size == 1:
                ckpt["optimizer"] = self.actor.optimizer.state_dict()
            if "extra" in save_contents and world_size == 1:
                ckpt["lr_scheduler"] = self.actor.lr_scheduler.state_dict()
            if world_size > 1 and ("optimizer" in save_contents or "extra" in save_contents):
                logger.warning("Skipping optimizer/lr_scheduler save for world_size>1 (only model weights are saved).")
            ckpt_path = os.path.join(local_path, "actor_model.pt")
            torch.save(ckpt, ckpt_path)

        dist.barrier()

        if use_offload:
            offload_fsdp2_model_to_cpu(self.dit)
            offload_fsdp_optimizer(self.actor.optimizer)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(
        self,
        local_path: str | None,
        hdfs_path: str | None = None,
        del_local_after_load: bool = False,
    ) -> None:
        """Restore the DiT checkpoint from *local_path*.

        Args:
            local_path: Directory containing ``actor_model.pt``.  Pass
                ``None`` to skip loading (still handles offload correctly).
            hdfs_path: Unused.
            del_local_after_load: Unused.
        """
        use_offload = bool(self.config.actor.fsdp_config.get("offload_policy", False))

        if local_path is None:
            if use_offload:
                offload_fsdp2_model_to_cpu(self.dit)
                offload_fsdp_optimizer(self.actor.optimizer)
            return

        if use_offload:
            load_fsdp2_model_to_gpu(self.dit)
            load_fsdp_optimizer(self.actor.optimizer)

        file_path = os.path.join(local_path, "actor_model.pt")
        ckpt = torch.load(file_path, map_location="cpu") if dist.get_rank() == 0 else {}

        load_contents = set(getattr(self.config.actor.checkpoint, "load_contents", ["model"]))
        world_size = dist.get_world_size() if dist.is_initialized() else 1

        if "model" in load_contents:
            full_state = ckpt.get("model", {}) if dist.get_rank() == 0 else {}
            fsdp2_load_full_state_dict(
                self.dit,
                full_state,
                device_mesh=self.device_mesh["fsdp"],
                cpu_offload=self.config.actor.fsdp_config.offload_policy,
            )

        if "optimizer" in load_contents and world_size == 1 and dist.get_rank() == 0 and "optimizer" in ckpt:
            self.actor.optimizer.load_state_dict(ckpt["optimizer"])
        if "extra" in load_contents and world_size == 1 and dist.get_rank() == 0 and "lr_scheduler" in ckpt:
            self.actor.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        if world_size > 1 and dist.get_rank() == 0 and ("optimizer" in load_contents or "extra" in load_contents):
            logger.warning("Skipping optimizer/lr_scheduler load for world_size>1 (only model weights are restored).")

        if dist.is_initialized():
            dist.barrier()

        if use_offload:
            offload_fsdp2_model_to_cpu(self.dit)
            offload_fsdp_optimizer(self.actor.optimizer)

    # ------------------------------------------------------------------
    # Fixed evaluation
    # ------------------------------------------------------------------

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_fixed_eval(self, data: DataProto) -> DataProto:
        """Capture conditioning + initial noise for a set of fixed prompts.

        Should be called once before training starts.  The captured data is
        used by :meth:`run_fixed_eval` to render deterministic images after
        every weight update.

        Args:
            data: DataProto with ``non_tensor_batch["prompts"]``.

        Returns:
            Empty DataProto (side-effect only).
        """
        try:
            raw_prompts = data.non_tensor_batch.get("prompts", None)
            if raw_prompts is None or len(raw_prompts) == 0:
                logger.warning("init_fixed_eval: no prompts found, skipping.")
                return DataProto()

            num_eval = min(8, len(raw_prompts))
            self._fixed_eval_samples = []

            for p_idx in range(num_eval):
                prompt_str = str(raw_prompts[p_idx])
                single_data = DataProto.from_dict(
                    tensors={},
                    non_tensors={"prompts": np.array([prompt_str], dtype=object)},
                )
                single_data.meta_info.update(data.meta_info)

                with torch.no_grad():
                    output = self.rollout._generate_minibatch(single_data)

                # Fixed initial noise (broadcast from rank 0 for reproducibility)
                initial_noise = output.batch["all_latents"][0:1, 0].clone()
                if dist.is_initialized():
                    dist.broadcast(initial_noise, src=0)

                # Conditioning tensors
                conditioning: dict = {}
                for key in [
                    "text_hidden_states",
                    "text_attention_mask",
                    "negative_text_hidden_states",
                    "negative_text_attention_mask",
                    "image_hidden_states",
                    "image_attention_mask",
                ]:
                    if key in output.batch.keys():
                        conditioning[key] = output.batch[key][0:1].clone().cpu()
                for len_key in ["text_seq_len", "neg_seq_len", "image_seq_len"]:
                    if len_key in output.batch.keys():
                        conditioning[len_key] = int(output.batch[len_key][0].item())

                safe_prompt = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt_str).strip("_")[:20] or "prompt"
                self._fixed_eval_samples.append(
                    {
                        "noise": initial_noise.cpu(),
                        "conditioning": conditioning,
                        "prompt": prompt_str,
                        "safe_prompt": safe_prompt,
                    }
                )

        except Exception as exc:
            logger.warning("init_fixed_eval failed: %s", exc, exc_info=True)

        return DataProto()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def run_fixed_eval(self, data: DataProto) -> DataProto:
        """Render fixed evaluation images with the current DiT weights.

        Runs every 10 RL steps (``global_step % 10 == 0``).  Uses pure
        Euler ODE (no SDE noise) for reproducible visual comparison.

        Args:
            data: DataProto with ``meta_info["global_step"]`` and
                ``meta_info["experiment_name"]``.

        Returns:
            Empty DataProto (side-effect only: saves PNG files).
        """
        if not self._fixed_eval_samples:
            return DataProto()

        rank = dist.get_rank() if dist.is_initialized() else 0
        global_step = data.meta_info.get("global_step", 0)
        experiment_name = data.meta_info.get("experiment_name", "dance_grpo")

        if global_step % 10 != 0:
            return DataProto()

        output_dir = str(getattr(self.config.rollout, "output_dir", "logs"))
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(os.getcwd(), output_dir)

        cfg = self.config.rollout
        num_inference_steps = int(getattr(cfg, "num_inference_steps", 40))
        h = int(getattr(cfg, "height", 512))
        w = int(getattr(cfg, "width", 512))
        vae_scale_factor = int(getattr(cfg, "vae_scale_factor", 16))
        eval_cfg_scales = [1.0, 3.0]

        try:
            from mammothmoda2.model.mammothmoda2_dit import RotaryPosEmbedReal

            dev = _device_name
            freqs_cis = RotaryPosEmbedReal.get_freqs_real(
                self.dit.config.axes_dim_rope, self.dit.config.axes_lens, theta=10000
            )

            latent_h = 2 * h // vae_scale_factor
            latent_w = 2 * w // vae_scale_factor
            num_tokens = latent_h * latent_w
            timesteps = torch.linspace(0, 1, num_inference_steps + 1)
            m = (num_tokens**0.5) / 40.0
            timesteps = timesteps / (m + timesteps * (1.0 - m))
            timesteps = timesteps.to(dev)

            vae = self.full_model.gen_vae
            scaling_factor = getattr(vae.config, "scaling_factor", 0.18215)
            shift_factor = getattr(vae.config, "shift_factor", 0.0)

            eval_dir = os.path.join(output_dir, experiment_name, "fixed_eval")
            if rank == 0:
                os.makedirs(eval_dir, exist_ok=True)

            self.dit.eval()

            for sample in self._fixed_eval_samples:
                cond = sample["conditioning"]
                safe_prompt = sample["safe_prompt"]

                # Strip padding and move to device
                text_hs = cond["text_hidden_states"]
                text_am = cond["text_attention_mask"]
                if "text_seq_len" in cond:
                    tl = cond["text_seq_len"]
                    text_hs = text_hs[:, -tl:, :]
                    text_am = text_am[:, -tl:]
                text_hs = text_hs.to(dev, dtype=torch.bfloat16)
                text_am = text_am.to(dev)

                has_neg = "negative_text_hidden_states" in cond
                neg_hs = neg_am = None
                if has_neg:
                    neg_hs = cond["negative_text_hidden_states"]
                    neg_am = cond["negative_text_attention_mask"]
                    if "neg_seq_len" in cond:
                        nl = cond["neg_seq_len"]
                        neg_hs = neg_hs[:, -nl:, :]
                        neg_am = neg_am[:, -nl:]
                    neg_hs = neg_hs.to(dev, dtype=torch.bfloat16)
                    neg_am = neg_am.to(dev)

                has_img = "image_hidden_states" in cond
                img_hs = img_am = None
                if has_img:
                    img_hs = cond["image_hidden_states"]
                    img_am = cond["image_attention_mask"]
                    if "image_seq_len" in cond:
                        il = cond["image_seq_len"]
                        img_hs = img_hs[:, -il:, :]
                        img_am = img_am[:, -il:]
                    img_hs = img_hs.to(dev, dtype=torch.bfloat16)
                    img_am = img_am.to(dev)

                for cfg_scale in eval_cfg_scales:
                    use_cfg = cfg_scale > 1.0 and has_neg
                    latents = sample["noise"].clone().to(dev, dtype=torch.bfloat16)

                    with torch.no_grad(), torch.autocast(device_type=_device_name, dtype=torch.bfloat16):
                        for i in range(num_inference_steps):
                            t = timesteps[i]
                            t_next = timesteps[i + 1]
                            dt = (t_next - t).item()
                            ts = t.unsqueeze(0).expand(1)

                            if use_cfg:
                                cond_out = self.dit(
                                    hidden_states=latents,
                                    timestep=ts,
                                    text_hidden_states=text_hs,
                                    text_attention_mask=text_am,
                                    ar_image_hidden_states=img_hs,
                                    ar_image_attention_mask=img_am,
                                    freqs_cis=freqs_cis,
                                    return_dict=False,
                                )
                                if isinstance(cond_out, (list, tuple)):
                                    cond_out = cond_out[0]
                                uncond_out = self.dit(
                                    hidden_states=latents,
                                    timestep=ts,
                                    text_hidden_states=neg_hs,
                                    text_attention_mask=neg_am,
                                    freqs_cis=freqs_cis,
                                    return_dict=False,
                                )
                                if isinstance(uncond_out, (list, tuple)):
                                    uncond_out = uncond_out[0]
                                model_output = uncond_out + cfg_scale * (cond_out - uncond_out)
                            else:
                                model_output = self.dit(
                                    hidden_states=latents,
                                    timestep=ts,
                                    text_hidden_states=text_hs,
                                    text_attention_mask=text_am,
                                    ar_image_hidden_states=img_hs,
                                    ar_image_attention_mask=img_am,
                                    freqs_cis=freqs_cis,
                                    return_dict=False,
                                )
                                if isinstance(model_output, (list, tuple)):
                                    model_output = model_output[0]

                            # Pure ODE step (no noise)
                            latents = latents + dt * model_output.to(torch.bfloat16)

                    # VAE decode
                    latents_decode = latents.float() / scaling_factor + shift_factor
                    with torch.no_grad(), torch.autocast(device_type=_device_name, dtype=torch.bfloat16):
                        decoded = vae.decode(latents_decode.to(torch.bfloat16)).sample
                    decoded = decoded.float().cpu()
                    decoded = (decoded / 2 + 0.5).clamp(0, 1)
                    decoded = decoded.permute(0, 2, 3, 1).numpy()
                    decoded = (decoded * 255).round().astype(np.uint8)
                    pil_image = Image.fromarray(decoded[0])

                    if rank == 0:
                        img_path = os.path.join(
                            eval_dir,
                            f"step_{global_step:06d}_{safe_prompt}_cfg{cfg_scale:.1f}.png",
                        )
                        pil_image.save(img_path)

        except Exception as exc:
            logger.warning("run_fixed_eval step %d failed: %s", global_step, exc, exc_info=True)
        finally:
            self.dit.train()

        return DataProto()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_rollout_metadata(self, data: DataProto) -> None:
        """Write per-rank JSONL file with prompt / reward / image-path per sample."""
        try:
            rank = dist.get_rank() if dist.is_initialized() else 0
            global_step = data.meta_info.get("global_step", 0)
            experiment_name = data.meta_info.get("experiment_name", "dance_grpo")

            output_dir = str(getattr(self.config.rollout, "output_dir", "logs"))
            if not os.path.isabs(output_dir):
                output_dir = os.path.join(os.getcwd(), output_dir)
            step_dir = os.path.join(output_dir, experiment_name, f"step_{global_step:06d}")
            os.makedirs(step_dir, exist_ok=True)

            prompts = list(data.non_tensor_batch.get("prompts", []))
            image_paths = list(data.non_tensor_batch.get("image_paths", []))
            rewards = data.batch.get("rewards", None)

            lines = []
            for i in range(len(prompts)):
                entry = {
                    "index": i,
                    "rank": rank,
                    "global_step": global_step,
                    "prompt": str(prompts[i]) if i < len(prompts) else "",
                    "reward": float(rewards[i].item()) if rewards is not None and i < len(rewards) else None,
                    "image_path": str(image_paths[i]) if i < len(image_paths) else "",
                }
                lines.append(json.dumps(entry))

            meta_path = os.path.join(step_dir, f"rank_{rank}_metadata.jsonl")
            with open(meta_path, "w") as fh:
                fh.write("\n".join(lines))
        except Exception as exc:
            logger.warning("_save_rollout_metadata failed: %s", exc)
