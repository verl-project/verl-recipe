# Copyright 2026 POISE authors
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from recipe.poise.probe import ProbeBank, ProbeConfig
from recipe.poise.tests.test_features import CharTokenizer, nested
from recipe.poise.trainer import PoiseTrainer, validate_poise_config
from tensordict import NonTensorStack, TensorDict
from transfer_queue import KVBatchMeta

from verl.utils.config import validate_config


@pytest.mark.parametrize("name,layer,pool", [("qwen3_4b", 19, 10), ("olmo3_7b", 7, 32)])
def test_profiles_compose_and_validate(name, layer, pool):
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1] / "config"), version_base=None):
        config = compose(config_name=name, overrides=["data.train_files=train.parquet", "data.val_files=test.parquet"])
    probe = validate_poise_config(config)
    validate_config(config, use_reference_policy=False, use_critic=False)
    assert probe.bootstrap_steps == 16
    assert (probe.layer, probe.pool_tokens) == (layer, pool)
    config.actor_rollout_ref.rollout.n = 3
    with pytest.raises(ValueError, match="exactly two"):
        validate_poise_config(config)


def test_trainer_handles_nested_queue_data_and_padding(monkeypatch):
    trainer = object.__new__(PoiseTrainer)
    trainer.tokenizer = CharTokenizer()
    trainer.probes = ProbeBank(ProbeConfig(buffer_rows={"math": 8}))
    data = TensorDict(
        {
            "uid": NonTensorStack("a", "padding", "a"),
            "data_source": NonTensorStack("math", "padding", "math"),
            "responses": nested([[65, 66], [0], [67]]),
            "response_mask": nested([[1, 1], [0], [1]]),
            "rm_scores": nested([[0.0, 1.0], [0.0], [0.0]]),
            "entropy": nested([[0.5, 0.2], [0.0], [0.3]]),
            "poise_prompt": nested([[1.0, 2.0], [0.0, 0.0], [1.0, 2.0]]),
            "poise_response": nested([[3.0, 4.0], [0.0, 0.0], [5.0, 6.0]]),
        },
        batch_size=3,
    )
    meta = KVBatchMeta(keys=["a_0_0", "padding", "a_1_0"], partition_id="train", tags=[{}, {"is_padding": True}, {}])
    monkeypatch.setattr("recipe.poise.trainer.tq.kv_batch_get", lambda **kw: data.select(*kw["select_fields"]))
    saved = {}

    def put(**kwargs):
        saved.update(kwargs["fields"].to_dict())
        return meta

    monkeypatch.setattr("recipe.poise.trainer.tq.kv_batch_put", put)
    metrics = {}
    trainer._compute_advantage(meta, metrics)
    for actual, expected in zip(saved["advantages"].unbind(), [[1.0, 1.0], [0.0], [-1.0]], strict=True):
        np.testing.assert_array_equal(actual, expected)
    assert trainer.probes.completed_steps == 0
    trainer.probes.commit()
    assert len(trainer.probes.buffers["math"]) == 1
