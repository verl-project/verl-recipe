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


import dataclasses
import datetime
import logging
import random
import os
import pickle
import time

import cv2
import torch
import torch.distributed as dist
from PIL import Image
from megatron.core.enums import ModelType
from megatron.core.num_microbatches_calculator import init_num_microbatches_calculator
from megatron.core.optimizer import OptimizerConfig
from megatron.training.global_vars import get_args, set_args
from omegaconf import DictConfig
from torch.distributed.device_mesh import init_device_mesh
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.device import (
    get_device_name,
    get_nccl_backend,
    get_torch_device,
)
from verl.utils.profiler import DistProfilerExtension, log_gpu_memory_usage
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

device_name = get_device_name()


def set_random_seed(seed, only_rollout=False):
    import random
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if not only_rollout and get_torch_device().device_count() > 0:
        from megatron.core import tensor_parallel
        tensor_parallel.model_parallel_cuda_manual_seed(seed)


def video_first_frame_to_pil(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    pil_image = Image.fromarray(frame_rgb)

    cap.release()

    return pil_image


def create_device_mesh(world_size, fsdp_size):
    """Create device mesh for FSDP"""
    if fsdp_size <= 0 or fsdp_size > world_size:
        fsdp_size = world_size
    return init_device_mesh(device_name, mesh_shape=(world_size // fsdp_size, fsdp_size), mesh_dim_names=["dp", "fsdp"])


class DiffusionActorRolloutWorker(Worker, DistProfilerExtension):
    """
    Worker for diffusion action rollout and GRPO training
    This worker encapsulates:
    1. Rollout process with diffusion model sampling and log probability calculation
    2. Reward calculation logic using GRPO
    3. GRPO policy update with advantage clipping and KL divergence regularization
    4. Checkpointing and loading of model and optimizer states
    """

    def __init__(self, config: DictConfig, role='hybrid', **kwargs):
        log_gpu_memory_usage("Before Diffusion Worker init", logger=logger, level=logging.INFO)
        Worker.__init__(self)
        self.config = config
        self.role = role
        self._is_actor = role in ["actor", "hybrid", "actor_rollout"]
        self._is_rollout = role in ["rollout", "hybrid"]
        # Init mindspeed args
        import mindspeed.args_utils as args_utils
        with open(f'{self.config.actor.model_args_path}/mindspeed_args.pkl', 'rb') as f:
            args_utils._MINDSPEED_ARGS = pickle.load(f)
        # TODO Mindspeed patch init
        import mindspeed.megatron_adaptor
        with open(f'{self.config.actor.model_args_path}/mm_args.pkl', 'rb') as f:
            args = pickle.load(f)
        set_args(args)
        args = get_args()

        init_num_microbatches_calculator(
            args.rank,
            args.rampup_batch_size,
            args.global_batch_size,
            args.micro_batch_size,
            args.data_parallel_size,
            args.decrease_batch_size_if_needed,
        )
        # Initialize distributed training
        if not torch.distributed.is_initialized():
            rank = int(os.environ.get("RANK", 0))
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            torch.distributed.init_process_group(
                backend=f"cpu:gloo,{get_device_name()}:{get_nccl_backend()}",
                rank=rank,
                world_size=world_size,
                timeout=datetime.timedelta(seconds=self.config.get("nccl_timeout", 600)),
                init_method=os.environ.get("DIST_INIT_METHOD", None),
            )
            from megatron.core import parallel_state as mpu
            mpu.initialize_model_parallel()
        set_random_seed(seed=123, only_rollout=False)

        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()

        # Build device mesh for FSDP
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # Build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get("ulysses_sequence_parallel_size", 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh(
                device_name, mesh_shape=(dp, self.ulysses_sequence_parallel_size), mesh_dim_names=["dp", "sp"]
            )

        # Create training dispatch
        if self.ulysses_device_mesh is not None:
            is_collect = self.ulysses_device_mesh["sp"].get_local_rank() == 0
            self._register_dispatch_collect_info(
                "actor", dp_rank=self.ulysses_device_mesh["dp"].get_local_rank(), is_collect=is_collect
            )
        else:
            self._register_dispatch_collect_info("actor", dp_rank=rank, is_collect=True)

        self._register_dispatch_collect_info("rollout", dp_rank=rank, is_collect=True)
        self._register_dispatch_collect_info("reward", dp_rank=rank, is_collect=True)

        # 只有当序列并行大小大于1时才创建分片管理器
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        else:
            self.ulysses_sharding_manager = None

        # Initialize models
        self.actor_module = None
        self.actor_module_fsdp = None
        self.ref_module = None
        self.ref_module_fsdp = None
        self.reward_module = None

        # Initialize optimizers and schedulers
        self.actor_optimizer = None
        self.actor_lr_scheduler = None

        # Checkpoint management
        self.checkpoint_manager = None

        # Setup FSDP
        self._mm_build_model_optimizer()
        self._init_reward_module()
        log_gpu_memory_usage("After Diffusion Worker init", logger=logger, level=logging.INFO)

    def _mm_build_model_optimizer(self):
        """Setup FSDP for distributed training"""
        log_gpu_memory_usage("Before init_fsdp_module", logger=logger, level=logging.INFO)
        from mindspeed_mm.models.diffusion import DiffusionModel
        from mindspeed_mm.models.text_encoder import Tokenizer
        from mindspeed_mm.training import no_wd_decay_cond, scale_lr_cond
        from mindspeed.core.distributed.torch_fully_sharded_data_parallel.training import get_model
        from recipe.mm_dance_grpo.patches.sora_model import MMSoRAModel
        from recipe.mm_dance_grpo.rollout import HFRollout
        from recipe.mm_dance_grpo.actor import DataParallelPPOActor

        # mm的模型提供
        def mm_model_provider(pre_process=True, post_process=True):
            """Builds the model."""
            args = get_args()
            print("building SoRA model ...")
            model = MMSoRAModel(args.mm.model)
            return model

        # 初始化actor模型
        self.actor_module_fsdp = get_model(mm_model_provider, ModelType.encoder_or_decoder)[0]
        # dcp加载
        self.load_checkpoint()
        torch.distributed.fsdp.register_fsdp_forward_method(self.actor_module_fsdp.module, "forward")
        args = get_args()
        # 初始化actor模型的优化器
        from megatron.core.optimizer import get_megatron_optimizer
        from megatron.training.training import get_optimizer_param_scheduler
        kwargs = {}
        for f in dataclasses.fields(OptimizerConfig):
            if hasattr(args, f.name):
                kwargs[f.name] = getattr(args, f.name)
        config = OptimizerConfig(**kwargs)
        self.actor_optimizer = get_megatron_optimizer(config, [self.actor_module_fsdp], no_wd_decay_cond,
                                                      scale_lr_cond, args.lr_mult,
                                                      use_gloo_process_groups=args.enable_gloo_process_groups)
        self.actor_lr_scheduler = get_optimizer_param_scheduler(self.actor_optimizer)

        rollout_config = self.config.rollout
        actor_config = self.config.actor
        self.scheduler = DiffusionModel(args.mm.model.diffusion).get_model()
        self.tokenizer = Tokenizer(args.mm.model.tokenizer).get_tokenizer()
        self.rollout = HFRollout(self.actor_module_fsdp, rollout_config, self.scheduler, self.tokenizer)
        self.actor = DataParallelPPOActor(self.actor_module_fsdp, actor_config, self.scheduler, self.tokenizer)
        log_gpu_memory_usage("After init_fsdp_module", logger=logger, level=logging.INFO)

    def _init_reward_module(self):
        from hpsv3 import HPSv3RewardInferencer
        self.reward_module = HPSv3RewardInferencer(device='npu', checkpoint_path=self.config.model.reward_model_path)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="reward"))
    def compute_rm_score(self, data: DataProto):
        all_rewards = []
        latents = data.batch["all_latents"]
        images = data.non_tensor_batch["all_imgs"]
        prompts = data.non_tensor_batch['raw_prompt']
        reward_coeff = 0.1
        for i in range(latents.shape[0]):
            images_path = images[i]
            prompt = prompts[i][0]['content']
            global_steps = data.non_tensor_batch["global_steps"][i]
            with torch.no_grad():
                image = video_first_frame_to_pil(images_path)
                base, _ = os.path.splitext(images_path)
                png_path = base + '.png'
                image.save(png_path, format="PNG")
                hps_score = self.reward_module.reward([png_path], [prompt])
                if hps_score.ndim == 2:
                    hps_score = hps_score[:, 0]
                hps_score = reward_coeff * torch.tensor(hps_score, dtype=torch.float32).to('npu')
                # 如果只是推理，需要重命名后保存
                if self.config.rollout.only:
                    self.save_rollout_result(hps_score, prompt, images_path, global_steps)
                all_rewards.append(hps_score)
                print(f"-> The png {png_path}, reward is {hps_score}")
        data.batch["rewards"] = torch.cat(all_rewards, dim=0)
        return data

    def _compute_grpo_advantages(self, rewards):
        """Compute GRPO-specific advantages"""
        # Check if we should use group normalization or global normalization
        use_group = self.config.actor.get("use_group", False)
        reward_threshold = self.config.actor.get("reward_threshold", 0.0)

        # Compute advantages based on the chosen normalization method
        if use_group:
            advantages = torch.zeros_like(rewards)
            group_mean = rewards.mean()
            group_std = rewards.std() + 1e-8
            if group_mean < reward_threshold:
                advantages[:] = 0
            else:
                advantages[:] = (rewards - group_mean) / group_std
        else:
            advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        return advantages

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="rollout"))
    def generate_sequences(self, data: DataProto):
        log_gpu_memory_usage("before generate_sequences", logger=logger, level=logging.INFO)
        """Generate sequences using diffusion model with asynchronous processing"""
        # Ensure data has the required attributes for diffusion model
        # Call the rollout's generate_sequence method
        output = self.rollout.generate_sequences(data)
        # Union with original data to preserve all information
        data = data.repeat(repeat_times=self.config.rollout.n, interleave=True)
        data = data.union(output)
        log_gpu_memory_usage("After generate_sequences", logger=logger, level=logging.INFO)
        return data

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def update_actor(self, data: DataProto):
        """Update actor using GRPO"""
        log_gpu_memory_usage("Before update_actor", logger=logger)
        assert self._is_actor
        logger.info(
            f"param_offload: {self.config.actor.fsdp_config.param_offload}, optimizer_offload:{self.config.actor.fsdp_config.optimizer_offload} ")

        # 根据是否有序列并行决定是否使用分片管理器
        if self.ulysses_sharding_manager is not None:
            context_manager = self.ulysses_sharding_manager
        else:
            # 创建一个空的上下文管理器
            from contextlib import nullcontext
            context_manager = nullcontext()

        with context_manager:
            latents = data.batch["all_latents"]
            old_log_probs = data.batch["all_log_probs"]
            prompt_embeds = data.batch["prompt_embeds"]
            negative_prompt_embeds = data.batch["negative_prompt_embeds"]
            assert data.batch["rewards"] is not None
            rewards = data.batch["rewards"]

            # Get configuration
            actor_config = self.config.actor
            adv_clip_max = actor_config.ppo_adv_clip_max
            timestep_fraction = actor_config.timestep_fraction
            clip_range = actor_config.clip_range
            kl_coeff = actor_config.ppo_kl_coeff
            sampling_steps = actor_config.sampling_steps

            # Get sigma schedule
            timesteps = self.scheduler.timesteps
            train_timesteps = random.sample(range(len(timesteps)), int(len(timesteps) * timestep_fraction))

            # Initialize losses
            total_loss = 0.0
            total_policy_loss = 0.0
            total_kl_loss = 0.0

            # Perform GRPO update
            self.actor_module_fsdp.train()
            self.actor_optimizer.zero_grad()
            batch_size = latents.shape[0]
            grpo_size = self.config.rollout.n
            batch_index = torch.chunk(torch.arange(batch_size), batch_size // grpo_size)
            for i, batch_ind in enumerate(batch_index):
                sample_latents = latents[batch_ind]  # Keep batch dimension
                sample_log_probs = old_log_probs[batch_ind]
                sample_reward = rewards[batch_ind].to(torch.float32)
                sample_prompt_embeds = prompt_embeds[batch_ind]
                sample_negative_prompt_embeds = negative_prompt_embeds[batch_ind]
                sample_advantages = self._compute_grpo_advantages(sample_reward)
                mini_batchs_size = sample_latents.shape[0]
                log_gpu_memory_usage(f"update_actor batch {i}", logger=logger, level=logging.INFO)
                for i, timestep_idx in enumerate(train_timesteps):
                    start_time = time.time()
                    num_chunks = mini_batchs_size // actor_config.micro_batch_size
                    batch_indices = torch.chunk(torch.arange(mini_batchs_size), num_chunks)

                    for idx, batch_idx in enumerate(batch_indices):
                        log_probs_chunk = sample_log_probs[batch_idx]
                        latents_chunk = sample_latents[batch_idx]
                        prompt_embeds_chunk = sample_prompt_embeds[batch_ind]
                        negative_prompt_embeds_chunk = sample_negative_prompt_embeds[batch_ind]
                        current_latents = latents_chunk[:, timestep_idx]
                        next_latents = latents_chunk[:, timestep_idx + 1]

                        # Calculate new log probs
                        self.actor_module_fsdp.train()
                        new_log_probs = self.actor.forward_micro_batch(
                            current_latents,
                            next_latents,
                            timestep_idx,
                            prompt_embeds_chunk,
                            negative_prompt_embeds_chunk
                        )

                        # Clamp advantages
                        clamped_advantages = torch.clamp(sample_advantages[batch_idx], -adv_clip_max, adv_clip_max)
                        # Calculate policy loss
                        ratio = torch.exp(new_log_probs.npu() - log_probs_chunk.npu()[:, timestep_idx])
                        unclipped_loss = -clamped_advantages.npu() * ratio.npu()
                        clipped_loss = -clamped_advantages.npu() * torch.clamp(ratio, 1.0 - clip_range,
                                                                               1.0 + clip_range)
                        loss = torch.mean(torch.max(clipped_loss, unclipped_loss)) / (
                                latents.shape[0] * len(train_timesteps))
                        # Calculate KL loss if reference model is available

                        # Backward pass
                        loss.backward()
                        avg_loss = loss.detach().clone()
                        abs_loss = torch.abs(avg_loss)
                        dist.all_reduce(abs_loss, op=dist.ReduceOp.AVG)
                        total_loss += abs_loss.item()

                    rank = torch.distributed.get_rank()
                    if rank == 0:
                        end_time = time.time()
                        logger.info(
                            f"Step {i + 1}/{len(train_timesteps)}: ABS Loss {total_loss:.4f}, Time {end_time - start_time:.4f}")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module_fsdp.parameters(), actor_config.ppo_max_grad_norm).to(device='cpu')

            # wan2.2 optimizer
            self.actor_optimizer.step()
            # wan2.2 scheduler
            self.actor_lr_scheduler.step(batch_size)
            self.actor_optimizer.zero_grad()

            # Synchronize across processes
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

            reward_mean = torch.mean(rewards).detach().item()
            if torch.distributed.get_rank() == 0:
                logger.info(f"===>>> Loss {total_loss:.4f}, Reward_mean: {reward_mean:.4f}")

            # Create metrics dictionary
            metrics = {
                "total_loss": total_loss,
                "reward_mean": reward_mean,
                "grad_norm": grad_norm
            }
            output = DataProto(meta_info={"metrics": metrics})
            log_gpu_memory_usage(f"After update_actor", logger=logger, level=logging.INFO)
            return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # only support save and load ckpt for actor
        assert self._is_actor
        args = get_args()
        args.save = local_path
        from megatron.training.checkpointing import save_checkpoint
        from megatron.training.training import preprocess_common_state_dict
        num_floating_point_operations_so_far = None
        checkpointing_context = None
        optimizer = self.actor_optimizer
        opt_param_scheduler = self.actor_lr_scheduler
        save_checkpoint(global_step, [self.actor_module_fsdp], optimizer, opt_param_scheduler,
                        num_floating_point_operations_so_far, checkpointing_context,
                        non_persistent_ckpt=False, train_data_iterator=None,
                        preprocess_common_state_dict_fn=preprocess_common_state_dict)
        print('Success save checkpoint to local_path: ', local_path)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, local_path=None, hdfs_path=None, del_local_after_load=False):
        assert self._is_actor or (not self._is_actor and self._is_rollout), (
            f"Checkpoint loading is only supported for Actor or standalone Rollout Workers, but got "
            f"{self._is_actor} and {self._is_rollout}"
        )
        args = get_args()
        if local_path:
            args.load = local_path

        optimizer = self.actor_optimizer
        opt_param_scheduler = self.actor_lr_scheduler
        checkpointing_context = None
        from megatron.training.checkpointing import load_checkpoint
        load_checkpoint(
            [self.actor_module_fsdp], optimizer, opt_param_scheduler, checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=getattr(args, "use_torch_fsdp2", False) and args.ckpt_format == "torch_dist")
        print('Success load checkpoint from local_path: ', args.load)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def online_test(self, step: int):
        save_path = self.config.rollout.online.save.path
        if save_path is None:
            raise ValueError('the rollout online save path is None')
        os.makedirs(save_path, exist_ok=True)
        # 重复推理当前文本
        prompt = self.config.rollout.online.prompt
        images_path = self.rollout.generate_test(prompt, step, save_path)
        # 获取第一帧打分
        image = video_first_frame_to_pil(images_path)
        base, _ = os.path.splitext(images_path)
        png_path = base + '.png'
        image.save(png_path, format="PNG")
        reward_coeff = 0.1
        hps_score = reward_coeff * self.reward_module.reward([png_path], [prompt])
        if hps_score.ndim == 2:
            hps_score = hps_score[:, 0]
        # 记录打分结果
        context = {
            "reward": hps_score.item(),
            "images_path": images_path,
            "prompt": prompt
        }
        file_path = f"{save_path}/reward_result.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(context, ensure_ascii=False) + "\n")

    def save_rollout_result(self, hps_score, prompt, images_path, global_steps):
        save_path = self.config.rollout.result.save.path
        if save_path is None:
            raise ValueError('the rollout result save path is None')
        os.makedirs(save_path, exist_ok=True)
        import shutil
        # 移动到目标目录并重命名（增加后缀）
        dirname, filename = os.path.split(images_path)
        basename, ext = os.path.splitext(filename)
        # 构造新文件名，添加后缀
        new_filename = f"{basename}_step{global_steps}{ext}"
        dst_path = os.path.join(save_path, new_filename)
        shutil.move(images_path, dst_path)
        # 记录打分结果
        context = {
            "reward": hps_score.item(),
            "images_path": dst_path,
            "prompt": prompt
        }
        file_path = f"{save_path}/reward_result.jsonl"
        with open(file_path, "a", encoding="utf-8") as f:
            import json
            f.write(json.dumps(context, ensure_ascii=False) + "\n")


# Helper functions moved from train_grpo_edit.py
def omni_time_shift(shift, t):
    t = 1 - t
    t = (shift * t) / (1 + (shift - 1) * t)
    t = 1 - t
    return t
