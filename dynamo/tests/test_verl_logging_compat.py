"""Exercise the scoped logging fix with real logging and the upstream setup."""

import importlib.util
import io
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

TARGET = "verl.trainer.ppo.v1.trainer_separate_async"


@pytest.fixture
def compat(monkeypatch):
    logger = logging.getLogger(TARGET)
    monkeypatch.setattr(logger, "info", logging.Logger.info.__get__(logger))
    monkeypatch.setattr(logger, "disabled", False)
    level = logger.level
    logger.setLevel(logging.INFO)
    path = Path(__file__).resolve().parents[1] / "verl_logging_compat.py"
    spec = importlib.util.spec_from_file_location("_recipe_logging_compat_test", path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        yield module, logger
    finally:
        logger.setLevel(level)


def test_info_is_emitted_with_flush_and_preserves_metadata(compat, monkeypatch):
    _, logger = compat
    records = []
    monkeypatch.setattr(logger, "handle", records.append)
    logger.info("standalone replicas: %s", 2, flush=True, extra={"pool": "rollout"})
    assert len(records) == 1
    assert records[0].getMessage() == "standalone replicas: 2"
    assert records[0].levelno == logging.INFO
    assert records[0].pool == "rollout"
    assert records[0].pathname == __file__


def test_stream_handler_still_flushes(compat, monkeypatch):
    _, logger = compat
    stream = io.StringIO()
    flushes = []
    handler = logging.StreamHandler(stream)
    monkeypatch.setattr(handler, "flush", lambda: flushes.append(True))
    monkeypatch.setattr(logger, "handlers", [handler])
    monkeypatch.setattr(logger, "propagate", False)
    logger.info("setup ready", flush=True)
    assert stream.getvalue() == "setup ready\n"
    assert flushes == [True]


def test_other_loggers_and_errors_are_not_patched(compat):
    module, logger = compat
    original = logger.info
    module.install()
    assert logger.info is original
    unrelated = logging.Logger("unrelated", level=logging.INFO)
    with pytest.raises(TypeError, match="flush"):
        unrelated.info("unrelated", flush=True)
    with pytest.raises(TypeError, match="unknown_argument"):
        logger.info("bad argument", unknown_argument=True)


def test_pristine_upstream_setup_runs_with_info_enabled(compat, monkeypatch):
    pytest.importorskip("verl")
    # Keep the complete upstream _setup control flow and real INFO logging;
    # replace only resource allocation and model/checkpoint initialization.
    from omegaconf import OmegaConf

    from verl.trainer.ppo.v1 import trainer_separate_async as upstream

    records = []
    monkeypatch.setattr(upstream.logger, "handle", records.append)
    monkeypatch.setattr(upstream.PPOTrainer, "_setup", lambda self: None)
    manager = SimpleNamespace(get_replicas=lambda: ["rollout"], rollout_replicas=[])
    monkeypatch.setattr(upstream, "LLMServerManager", SimpleNamespace(create=lambda **kwargs: manager))
    monkeypatch.setattr(upstream, "omega_conf_to_dataclass", lambda config: config)
    monkeypatch.setattr(upstream, "CheckpointEngineManager", lambda **kwargs: kwargs)
    trainer = object.__new__(upstream.PPOTrainerSeparateAsync)
    trainer.config = OmegaConf.create(
        {
            "actor_rollout_ref": {
                "rollout": {
                    "prometheus": {"enable": False},
                    "checkpoint_engine": {"backend": "nccl"},
                }
            }
        }
    )
    trainer.llm_server_manager = SimpleNamespace(rollout_replicas=[])
    trainer.actor_rollout_wg = object()
    trainer._enable_hybrid_replicas = False
    with monkeypatch.context() as baseline:
        baseline.setattr(upstream.logger, "info", logging.Logger.info.__get__(upstream.logger))
        try:
            upstream.PPOTrainerSeparateAsync._setup(trainer)
        except TypeError as exc:
            # Older verl uses logger.info(..., flush=True); current images fixed it.
            assert "flush" in str(exc)
    upstream.PPOTrainerSeparateAsync._setup(trainer)
    assert trainer.current_mode is upstream.HybridEngineMode.TRAINER
    assert trainer.standalone_server_manager is manager
    assert trainer.standalone_checkpoint_manager["replicas"] == ["rollout"]
    assert any("hybrid replicas disabled" in record.getMessage() for record in records)
