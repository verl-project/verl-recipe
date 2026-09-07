"""Data models for the kube-rl SDK client."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Agent configuration to submit with a run request."""

    name: str | None = None
    import_path: str | None = None
    model_name: str | None = None
    override_timeout_sec: float | None = None
    override_setup_timeout_sec: float | None = None
    max_timeout_sec: float | None = None
    llm_proxy_url: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)


class VerifierConfig(BaseModel):
    """Verifier configuration to submit with a run request."""

    override_timeout_sec: float | None = None
    max_timeout_sec: float | None = None
    disable: bool = False


class RolloutDetail(BaseModel):
    """Rollout detail data for RL training."""

    prompt_token_ids: list[list[int]] | None = None
    completion_token_ids: list[list[int]] | None = None
    logprobs: list[list[float]] | None = None


class AgentRunResult(BaseModel):
    """Result of an agent run, containing RL-relevant fields."""

    run_id: str
    status: str = Field(description="completed | failed | timeout")
    rewards: dict[str, float | int] | None = None
    rollout_details: list[RolloutDetail] | None = None
    error: str | None = None
    result_uri: str | None = None
