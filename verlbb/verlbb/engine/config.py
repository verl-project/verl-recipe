"""Config objects for the VerlBB Bumblebee engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from verl.workers.config.engine import EngineConfig


@dataclass
class BumblebeeEngineConfig(EngineConfig):
    """Minimal VERL-facing config for the external Bumblebee engine."""

    strategy: str = "bumblebee"
    model_name: str = "auto"
    impl: str = "lite"

    tp: int = 1
    etp: int | None = None
    ep: int = 1
    pp: int = 1
    vpp: int = 1
    cp: int = 1

    attention_backend_override: str | None = "flash"
    router_aux_loss_coef: float | None = None
    impl_cfg: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.strategy != "bumblebee":
            raise ValueError(f"BumblebeeEngineConfig expects strategy='bumblebee', got {self.strategy!r}")
