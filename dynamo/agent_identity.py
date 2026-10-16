"""Coding-agent program identity derived from request headers.

Dynamo's frontend normalizes these headers into ``agent_context`` (``lib/llm/src/protocols/agents.rs``);
on the training path they terminate at the Uni-Agent Gateway, so the recipe applies the same precedence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class _HeaderMapping:
    root_session: str
    child_session: str | None
    parent_session: str | None
    infer_parent_for_child: bool  # a child turn with no explicit parent has the root session as its parent


_DYNAMO_SESSION = "x-dynamo-session-id"
_DYNAMO_PARENT = "x-dynamo-parent-session-id"

_AGENT_HEADERS = (
    _HeaderMapping("x-claude-code-session-id", "x-claude-code-agent-id", "x-claude-code-parent-agent-id", True),
)


@dataclass(frozen=True)
class AgentIdentity:
    """A chain's own id (for a Claude Code subagent turn, the agent id) and the chain that dispatched it."""

    session_id: str
    parent_session_id: str | None = None


def _value(headers: Mapping[str, str], name: str) -> str | None:
    value = headers.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def agent_identity_from_headers(headers: Mapping[str, str]) -> AgentIdentity | None:
    """Derive the program identity from request headers, or None when none are set.

    Canonical ``x-dynamo-*`` headers win. Lookups are lower-case, so ``headers`` must be case-insensitive.
    """
    session_id = _value(headers, _DYNAMO_SESSION)
    if session_id is not None:
        parent = _value(headers, _DYNAMO_PARENT)
        return AgentIdentity(session_id, parent if parent != session_id else None)

    for mapping in _AGENT_HEADERS:
        root = _value(headers, mapping.root_session)
        if root is None:
            continue
        child = _value(headers, mapping.child_session) if mapping.child_session else None
        session_id = child or root
        is_child = session_id != root
        parent = _value(headers, mapping.parent_session) if mapping.parent_session else None
        if parent == session_id:
            parent = None
        if parent is not None and mapping.infer_parent_for_child and not is_child:
            parent = None  # a root turn has no parent; a stale header must not invent one
        if parent is None and mapping.infer_parent_for_child and is_child:
            parent = root
        return AgentIdentity(session_id, parent)
    return None


__all__ = ["AgentIdentity", "agent_identity_from_headers"]
