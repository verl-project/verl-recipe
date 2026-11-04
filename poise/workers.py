# Copyright 2026 POISE authors
# SPDX-License-Identifier: Apache-2.0
"""Recipe-local engine registration; the upstream language-model engine stays intact."""

from dataclasses import replace

from verl.utils import tensordict_utils as tu
from verl.workers.engine import EngineRegistry
from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

from .features import capture_hidden


@EngineRegistry.register(model_type="poise", backend=["fsdp", "fsdp2"], device="cuda")
class PoiseFSDPEngine(FSDPEngineWithLMHead):
    def __init__(self, model_config, **kwargs):
        # The distinct registry key selects this engine, while HF still constructs a causal LM.
        model_config.model_type = "language_model"
        super().__init__(model_config=model_config, **kwargs)

    def forward_step(self, micro_batch, loss_function, forward_only):
        spec = tu.get_non_tensor_data(micro_batch, "poise_capture", default=None)
        if not forward_only or spec is None:
            return super().forward_step(micro_batch, loss_function, forward_only)
        if self.ulysses_sequence_parallel_size != 1:
            raise ValueError("The POISE recipe currently supports Ulysses sequence parallel size 1")
        with capture_hidden(
            self.module,
            input_ids=micro_batch["input_ids"],
            responses=micro_batch["responses"],
            response_mask=micro_batch["response_mask"],
            packed=tu.get_non_tensor_data(micro_batch, "use_remove_padding", default=True),
            **spec,
        ) as pooled:
            loss, output = super().forward_step(micro_batch, loss_function, forward_only)
        # veRL concatenates/reorders every model_output field across dynamic microbatches.
        output["model_output"].update(pooled)
        return loss, output


class PoiseTrainingWorker(TrainingWorker):
    def __init__(self, config):
        super().__init__(replace(config, model_type="poise"))


class PoiseActorRolloutWorker(ActorRolloutRefWorker):
    actor_worker_cls = PoiseTrainingWorker
