"""Data models for the LLM proxy session recording."""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel, Field


class CompletionRecord(BaseModel):
    """Record of a single LLM completion call captured by the proxy."""

    request_messages: list[dict[str, Any]] = Field(
        description="Full messages array sent in this request"
    )
    completion_text: str = Field(description="Generated completion text")
    completion_token_ids: list[int] = Field(
        description="Token IDs of the generated completion"
    )
    completion_logprobs: list[float] = Field(
        description="Log probabilities for each generated token"
    )
    finish_reason: str | None = Field(
        default=None, description="Reason generation stopped: stop, tool_calls, length"
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None, description="Parsed tool calls from the completion, if any"
    )


class SessionRecord(BaseModel):
    """All recorded data for a single agent session (one rollout)."""

    session_id: str
    turns: list[CompletionRecord] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    completed: bool = False
