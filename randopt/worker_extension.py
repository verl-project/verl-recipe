"""vLLM worker extension used by the RandOpt recipe.

The extension runs inside each vLLM worker process. It exposes small RPC
methods for applying deterministic Gaussian perturbations, updating one worker
from a list of seeds, and synchronizing updated weights across engines.
"""

import gc
import os
import random
import time

import numpy as np
import torch


def _stateless_init_process_group(master_address, master_port, rank, world_size, device):
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    process_group = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=rank, world_size=world_size
    )
    return PyNcclCommunicator(process_group, device=device)


class WorkerExtension:
    """RPC surface for RandOpt perturbation and inter-engine synchronization."""

    _VISUAL_PREFIXES = ("visual.", "model.visual.")

    def _set_seed(self, seed: int) -> None:
        seed = int(seed)
        self.local_seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _should_perturb(self, name: str) -> bool:
        if os.environ.get("RANDOPT_PERTURB_VISUAL", "0") == "1":
            return True
        return not name.startswith(self._VISUAL_PREFIXES)

    def perturb_self_weights(self, seed, noise_scale, negate=False):
        self._set_seed(seed)
        sign = -1.0 if negate else 1.0
        scale = float(noise_scale)

        for name, param in self.model_runner.model.named_parameters():
            if not self._should_perturb(name):
                continue
            generator = torch.Generator(device=param.device)
            generator.manual_seed(int(seed))
            noise = torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=generator)
            param.data.add_(sign * scale * noise)
            del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True

    def restore_self_weights(self, seed, noise_scale, negate=False):
        self._set_seed(seed)
        sign = -1.0 if negate else 1.0
        scale = float(noise_scale)

        for name, param in self.model_runner.model.named_parameters():
            if not self._should_perturb(name):
                continue
            generator = torch.Generator(device=param.device)
            generator.manual_seed(int(seed))
            noise = torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=generator)
            param.data.add_(-sign * scale * noise)
            del noise

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        return True

    def update_weights_from_seeds(self, seeds, coeffs, alpha, population_size):
        """Apply a normalized ES update on the current worker."""
        for name, param in self.model_runner.model.named_parameters():
            if not self._should_perturb(name):
                continue

            update_accumulator = torch.zeros_like(param.data, dtype=torch.float32)
            for seed, coeff in zip(seeds, coeffs, strict=False):
                generator = torch.Generator(device=param.device)
                generator.manual_seed(int(seed))
                noise = torch.randn(param.shape, dtype=param.dtype, device=param.device, generator=generator)
                update_accumulator.add_(noise.to(torch.float32), alpha=float(coeff))
                del noise

            update_accumulator.mul_(float(alpha) / int(population_size))
            param.data.add_(update_accumulator.to(param.dtype))
            del update_accumulator

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()
        return True

    def init_inter_engine_group(self, master_address: str, master_port: int, rank: int, world_size: int):
        self.inter_pg = _stateless_init_process_group(master_address, master_port, rank, world_size, self.device)
        return True

    def broadcast_all_weights(self, src_rank: int):
        for _, param in self.model_runner.model.named_parameters():
            self.inter_pg.broadcast(param, src=int(src_rank), stream=torch.cuda.current_stream())
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return True

    def save_self_weights_to_disk(self, filepath: str):
        state_dict = {name: param.detach().cpu() for name, param in self.model_runner.model.named_parameters()}
        torch.save(state_dict, filepath)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(0.1)
        return True
