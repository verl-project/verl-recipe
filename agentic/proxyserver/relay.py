"""WebSocket-based inference relay for standalone proxy deployment.

When the proxy is deployed as a standalone service (without direct access to
vLLM via Ray), it uses an ``InferenceRelay`` to forward completion requests
to connected framework-side workers over WebSocket.

Protocol
~~~~~~~~

All messages are JSON objects with a ``type`` field.

**Proxy → Worker** (pushed when an agent calls the proxy)::

    {
        "type": "completion_request",
        "request_id": "<uuid>",
        "session_id": "<trial-id>",
        "messages": [...],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 2048,
        "stop": null,
        "tools": null,
        "repetition_penalty": null,
        "stream": false
    }

**Worker → Proxy** (after inference completes)::

    {
        "type": "completion_response",
        "request_id": "<uuid>",
        "token_ids": [...],
        "log_probs": [...],
        "completion_text": "...",
        "content_text": "...",
        "tool_calls": [...] | null,
        "finish_reason": "stop" | "tool_calls" | "length",
        "error": null
    }

**Heartbeat** (bidirectional keep-alive)::

    {"type": "ping"}  →  {"type": "pong"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class _PendingRequest:
    """A pending inference request awaiting a worker response."""

    __slots__ = ("future", "created_at", "worker_id")

    def __init__(self, worker_id: str) -> None:
        self.future: asyncio.Future = asyncio.get_event_loop().create_future()
        self.created_at: float = time.monotonic()
        self.worker_id = worker_id


class InferenceRelay:
    """Manages WebSocket connections to framework-side inference workers.

    The relay maintains a pool of connected workers and dispatches incoming
    inference requests (from agent HTTP calls) to available workers in a
    round-robin fashion.  Responses are correlated back via ``request_id``.

    Thread-safety: all public methods are coroutine-safe and use asyncio
    primitives.  The relay is intended to be used from a single event loop
    (the FastAPI/Uvicorn loop).
    """

    def __init__(self, request_timeout: float = 300.0) -> None:
        """
        Args:
            request_timeout: Maximum seconds to wait for a worker response
                before timing out an inference request.
        """
        self.request_timeout = request_timeout

        # worker_id -> WebSocket
        self._workers: dict[str, WebSocket] = {}
        # Round-robin index
        self._rr_index: int = 0

        # request_id -> _PendingRequest
        self._pending: dict[str, _PendingRequest] = {}

        # Event that is set whenever at least one worker is connected.
        self._worker_available = asyncio.Event()

    # ------------------------------------------------------------------
    # Worker connection management
    # ------------------------------------------------------------------

    @property
    def num_workers(self) -> int:
        return len(self._workers)

    def register_worker(self, worker_id: str, ws: WebSocket) -> None:
        """Register a newly connected worker."""
        self._workers[worker_id] = ws
        self._worker_available.set()
        logger.info(
            "Worker %s connected (total: %d)", worker_id, len(self._workers)
        )

    def unregister_worker(self, worker_id: str) -> None:
        """Remove a disconnected worker and fail its pending requests."""
        self._workers.pop(worker_id, None)
        if not self._workers:
            self._worker_available.clear()

        # Fail any pending requests assigned to this worker
        failed_ids = [
            rid for rid, pr in self._pending.items()
            if pr.worker_id == worker_id
        ]
        for rid in failed_ids:
            pr = self._pending.pop(rid)
            if not pr.future.done():
                pr.future.set_exception(
                    RuntimeError(
                        f"Worker {worker_id} disconnected while processing "
                        f"request {rid}"
                    )
                )
        logger.info(
            "Worker %s disconnected (total: %d, failed requests: %d)",
            worker_id, len(self._workers), len(failed_ids),
        )

    # ------------------------------------------------------------------
    # Request dispatching
    # ------------------------------------------------------------------

    def _pick_worker(self) -> tuple[str, WebSocket]:
        """Pick the next worker using round-robin."""
        worker_ids = list(self._workers.keys())
        idx = self._rr_index % len(worker_ids)
        self._rr_index = idx + 1
        wid = worker_ids[idx]
        return wid, self._workers[wid]

    async def dispatch(
        self,
        *,
        session_id: str,
        messages: list[dict[str, Any]],
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        repetition_penalty: float | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Dispatch an inference request to a connected worker.

        Blocks (asynchronously) until the worker responds or the request
        times out.

        Returns:
            The worker's response dict containing ``token_ids``,
            ``log_probs``, ``completion_text``, ``content_text``,
            ``tool_calls``, ``finish_reason``.

        Raises:
            RuntimeError: No workers connected or request timed out.
        """
        # Wait for at least one worker (with timeout)
        if not self._workers:
            try:
                await asyncio.wait_for(
                    self._worker_available.wait(),
                    timeout=min(self.request_timeout, 30.0),
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "No inference workers connected to the proxy. "
                    "Ensure a framework-side worker is running and "
                    "connected via WebSocket."
                )

        worker_id, ws = self._pick_worker()
        request_id = uuid4().hex

        pending = _PendingRequest(worker_id=worker_id)
        self._pending[request_id] = pending

        # Build and send request message
        request_msg = {
            "type": "completion_request",
            "request_id": request_id,
            "session_id": session_id,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stop": stop,
            "tools": tools,
            "repetition_penalty": repetition_penalty,
            "stream": stream,
        }

        try:
            await ws.send_json(request_msg)
        except Exception as e:
            self._pending.pop(request_id, None)
            raise RuntimeError(
                f"Failed to send request to worker {worker_id}: {e}"
            )

        # Wait for the response
        try:
            result = await asyncio.wait_for(
                pending.future, timeout=self.request_timeout
            )
        except asyncio.TimeoutError:
            self._pending.pop(request_id, None)
            raise RuntimeError(
                f"Inference request {request_id} timed out after "
                f"{self.request_timeout}s (worker: {worker_id})"
            )

        return result

    # ------------------------------------------------------------------
    # Response handling (called from the WebSocket receive loop)
    # ------------------------------------------------------------------

    def deliver_response(self, request_id: str, response: dict[str, Any]) -> bool:
        """Deliver a worker's response to the waiting caller.

        Returns:
            True if the response was delivered, False if no pending request
            was found (e.g. it already timed out).
        """
        pending = self._pending.pop(request_id, None)
        if pending is None:
            logger.warning(
                "Received response for unknown/expired request %s", request_id
            )
            return False

        if not pending.future.done():
            error = response.get("error")
            if error:
                pending.future.set_exception(RuntimeError(f"Worker error: {error}"))
            else:
                pending.future.set_result(response)
        return True

    # ------------------------------------------------------------------
    # WebSocket handler (to be mounted as a FastAPI endpoint)
    # ------------------------------------------------------------------

    async def handle_worker_ws(self, ws: WebSocket) -> None:
        """Handle an incoming worker WebSocket connection.

        This coroutine runs for the lifetime of the connection, reading
        messages and dispatching them to the appropriate handler.
        """
        await ws.accept()

        # Read the initial handshake message to get worker_id
        try:
            init_msg = await asyncio.wait_for(ws.receive_json(), timeout=10.0)
        except Exception as e:
            logger.error("Worker handshake failed: %s", e)
            await ws.close(code=4000, reason="Handshake timeout or invalid")
            return

        if init_msg.get("type") != "worker_hello":
            await ws.close(code=4001, reason="Expected worker_hello")
            return

        worker_id = init_msg.get("worker_id", uuid4().hex)
        self.register_worker(worker_id, ws)

        # Send acknowledgment
        await ws.send_json({
            "type": "worker_hello_ack",
            "worker_id": worker_id,
            "status": "connected",
        })

        try:
            while True:
                data = await ws.receive_json()
                msg_type = data.get("type")

                if msg_type == "completion_response":
                    request_id = data.get("request_id")
                    if request_id:
                        self.deliver_response(request_id, data)
                    else:
                        logger.warning("Response without request_id: %s", data)

                elif msg_type == "ping":
                    await ws.send_json({"type": "pong"})

                elif msg_type == "pong":
                    pass  # heartbeat ack, ignore

                else:
                    logger.warning("Unknown message type from worker: %s", msg_type)

        except WebSocketDisconnect:
            logger.info("Worker %s WebSocket disconnected normally", worker_id)
        except Exception as e:
            logger.error("Worker %s WebSocket error: %s", worker_id, e)
        finally:
            self.unregister_worker(worker_id)
