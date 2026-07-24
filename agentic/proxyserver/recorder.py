"""Thread-safe session data recorder for the LLM proxy."""

from __future__ import annotations

import logging
import threading
from typing import Any

from .models import CompletionRecord, SessionRecord

logger = logging.getLogger(__name__)


class SessionRecorder:
    """Manages session records for the LLM proxy.

    Thread-safe: multiple proxy request handlers may record completions
    concurrently for different sessions.
    """

    def __init__(self) -> None:
        self._sessions: dict[str, SessionRecord] = {}
        self._lock = threading.Lock()

    def create_session(self, session_id: str) -> None:
        """Create a new session for recording."""
        with self._lock:
            if session_id in self._sessions:
                logger.warning("Session %s already exists, resetting", session_id)
            self._sessions[session_id] = SessionRecord(session_id=session_id)

    def record_completion(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
        completion_text: str,
        token_ids: list[int],
        logprobs: list[float],
        finish_reason: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Record a single LLM completion for a session."""
        record = CompletionRecord(
            request_messages=messages,
            completion_text=completion_text,
            completion_token_ids=token_ids,
            completion_logprobs=logprobs,
            finish_reason=finish_reason,
            tool_calls=tool_calls,
        )
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                logger.warning(
                    "Session %s not found, auto-creating for recording", session_id
                )
                session = SessionRecord(session_id=session_id)
                self._sessions[session_id] = session
            session.turns.append(record)

    def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve session data. Returns None if not found."""
        with self._lock:
            return self._sessions.get(session_id)

    def mark_completed(self, session_id: str) -> None:
        """Mark a session as completed."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                session.completed = True

    def delete_session(self, session_id: str) -> None:
        """Remove a session and free its memory."""
        with self._lock:
            self._sessions.pop(session_id, None)

    def list_sessions(self) -> list[str]:
        """List all active session IDs."""
        with self._lock:
            return list(self._sessions.keys())
