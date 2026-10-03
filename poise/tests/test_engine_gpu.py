# Copyright 2026 POISE authors
# SPDX-License-Identifier: Apache-2.0
"""FSDP2 feature extraction and the 16-step RLOO to POISE transition on tiny models."""

import os
from functools import partial

import numpy as np
import pytest
import torch
from recipe.poise.probe import ProbeBank, ProbeConfig
from recipe.poise.tests.test_features import nested
from tensordict import TensorDict

pytestmark = pytest.mark.skipif(
    os.getenv("POISE_TEST_GPU") != "1", reason="Set POISE_TEST_GPU=1 for the GPU smoke test"
)


@pytest.fixture(scope="module")
def process_group(tmp_path_factory):
    rank, world_size = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    device = torch.device("cuda", int(os.getenv("LOCAL_RANK", "0")))
    torch.cuda.set_device(device)
    torch.distributed.init_process_group(
        "nccl",
        init_method="env://" if world_size > 1 else f"file://{tmp_path_factory.mktemp('dist')}/rendezvous",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    yield world_size
    torch.distributed.destroy_process_group()


@pytest.mark.parametrize("model_type", ["qwen3", "olmo3"])
@pytest.mark.parametrize("packed", [False, True], ids=["sdpa", "flash_attention_2"])
def test_fsdp2_bootstrap_to_poise(tmp_path, model_type, packed, process_group):
    if packed:
        pytest.importorskip("flash_attn")

    from recipe.poise.workers import PoiseFSDPEngine
    from transformers import AutoModelForCausalLM, Olmo3Config, Qwen3Config

    from verl.trainer.config import CheckpointConfig
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import FSDPActorConfig, FSDPEngineConfig, FSDPOptimizerConfig, HFModelConfig
    from verl.workers.utils.losses import ppo_loss
    from verl.workers.utils.padding import response_from_nested

    world_size = process_group
    torch.manual_seed(42)
    config_class = Qwen3Config if model_type == "qwen3" else Olmo3Config
    hf_config = config_class(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model_path = tmp_path / "model"
    reference = AutoModelForCausalLM.from_config(hf_config)
    reference.save_pretrained(model_path)
    engine = PoiseFSDPEngine(
        model_config=HFModelConfig(
            path=str(model_path),
            load_tokenizer=False,
            enable_gradient_checkpointing=False,
            use_remove_padding=packed,
            override_config={"attn_implementation": "flash_attention_2" if packed else "sdpa"},
        ),
        engine_config=FSDPEngineConfig(strategy="fsdp2", use_torch_compile=False, model_dtype="fp32"),
        optimizer_config=FSDPOptimizerConfig(lr=1e-4, total_training_steps=17),
        checkpoint_config=CheckpointConfig(),
    )
    engine.initialize()
    initial_hooks = dict(engine.module.model.layers[0]._forward_hooks)
    probe = ProbeBank(
        ProbeConfig(layer=0, pool_tokens=2, prompt_pca_dim=2, response_pca_dim=2, buffer_rows={"math": 16})
    )
    loss_fn = partial(
        ppo_loss,
        config=FSDPActorConfig(
            strategy="fsdp2",
            ppo_mini_batch_size=2,
            rollout_n=2,
            use_dynamic_bsz=True,
            clip_ratio_low=0.2,
            clip_ratio_high=0.28,
            clip_ratio_c=10.0,
        ),
    )
    prompts = [[3, 4], [3, 4], [5, 6, 7], [5, 6, 7]]
    responses = [[8, 9, 2], [10, 2], [11, 2], [12, 13, 14, 2]]
    ids = [p + r for p, r in zip(prompts, responses, strict=True)]
    data = TensorDict(
        {
            "input_ids": nested(ids),
            "prompts": nested(prompts),
            "responses": nested(responses),
            "position_ids": nested([list(range(len(row))) for row in ids]),
            "response_mask": nested([[1] * len(r) for r in responses]),
            "loss_mask": nested([[1] * len(r) for r in responses]),
        },
        batch_size=4,
    )
    tu.assign_non_tensor(
        data,
        temperature=1.0,
        calculate_entropy=True,
        use_remove_padding=packed,
        use_dynamic_bsz=True,
        max_token_len_per_gpu=10,
        global_batch_size=4 * world_size,
        poise_capture={"layer": 0, "pool_tokens": 2, "think_end_ids": []},
    )
    reference = reference.cuda().eval()
    for step in range(17):
        with engine.eval_mode():
            output = engine.infer_batch(data)["model_output"]
        if step == 0:
            # Independent HF forwards check token alignment and dynamic microbatch restoration.
            for i, tokens in enumerate(ids):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = reference(torch.tensor([tokens], device="cuda"), output_hidden_states=True).hidden_states[
                        1
                    ][0]
                n = len(prompts[i])
                torch.testing.assert_close(
                    output["poise_prompt"][i], hidden[:n][-2:].mean(0).float(), atol=0.025, rtol=0.03
                )
                torch.testing.assert_close(
                    output["poise_response"][i], hidden[n:][-2:].mean(0).float(), atol=0.025, rtol=0.03
                )
        entropy = response_from_nested(output["entropy"], data["response_mask"].cuda())
        advantages, metrics = probe.advantages(
            uids=["a", "a", "b", "b"],
            domains=["math"] * 4,
            rewards=[1, 0, 1, 0],
            prompt=np.stack([x.cpu().numpy() for x in output["poise_prompt"].unbind()]),
            response=np.stack([x.cpu().numpy() for x in output["poise_response"].unbind()]),
            scalars=np.asarray([[float(e.mean()), 0, 0] for e in entropy.unbind()]),
        )
        assert metrics["poise/bootstrap"] == float(step < 16)
        data["old_log_probs"] = response_from_nested(output["log_probs"], data["response_mask"].cuda()).cpu()
        data["advantages"] = nested([[float(a)] * len(r) for a, r in zip(advantages, responses, strict=True)])
        with engine.train_mode():
            result = engine.train_batch(data, loss_function=loss_fn)
        assert "poise_prompt" not in result.get("model_output", {})
        assert torch.isfinite(torch.as_tensor(result["metrics"]["grad_norm"]))
        probe.commit()
    assert probe.completed_steps == 17
    assert dict(engine.module.model.layers[0]._forward_hooks) == initial_hooks
