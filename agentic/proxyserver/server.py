"""LLM Proxy server -- a Ray singleton that bridges OpenAI-compatible HTTP
requests to verl's vLLM rollout servers via their Ray actor handles.

Architecture
~~~~~~~~~~~~

::

    Third-party Agent
         |  (OpenAI /v1/chat/completions)
         v
    +----------------------------------+
    |  LLMProxyServer  (Ray singleton) |
    |  - FastAPI HTTP server           |
    |  - VLLMRayProvider (LiteLLM)     |
    |  - SessionRecorder per trial_id  |
    +---------------+------------------+
                    |  server.generate.remote(prompt_ids, ...)
                    v
             vLLM rollout servers

Each agent loop registers a *trial_id* with the proxy.  The agent is
given a URL of the form ``http://host:port/{trial_id}/v1`` so that
standard OpenAI SDK calls resolve to
``POST http://host:port/{trial_id}/v1/chat/completions``.

The actual vLLM interaction is handled by :class:`VLLMRayProvider`, a
LiteLLM ``CustomLLM`` provider that calls vLLM via Ray actor handles.
LiteLLM takes care of OpenAI response formatting, SSE streaming, and
logprobs construction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
from typing import Any
from uuid import uuid4

import litellm
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from transformers import AutoTokenizer

from verl.experimental.agent_loop.tool_parser import ToolParser

from .recorder import SessionRecorder
from .vllm_provider import VLLMRayProvider

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastAPI application builder
# ---------------------------------------------------------------------------


def _build_app(proxy: "LLMProxyServer") -> FastAPI:
    """Build the FastAPI application that serves as the OpenAI proxy."""

    app = FastAPI(title="LLM Proxy", version="0.3.0")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # ---- session management ----------------------------------------------

    @app.post("/sessions/{session_id}")
    async def create_session(session_id: str):
        proxy.recorder.create_session(session_id)
        return {"session_id": session_id, "status": "created"}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        session = proxy.recorder.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return session.model_dump()

    @app.post("/sessions/{session_id}/complete")
    async def complete_session(session_id: str):
        proxy.recorder.mark_completed(session_id)
        return {"session_id": session_id, "status": "completed"}

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        proxy.delete_session(session_id)
        return {"session_id": session_id, "status": "deleted"}

    # ---- OpenAI-compatible chat completions ------------------------------
    # Two URL patterns so the proxy works with both:
    #   1. Agents that call a fully custom endpoint
    #   2. Standard OpenAI SDK where base_url = http://host:port/{trial_id}/v1

    @app.post("/v1/chat/completions/{trial_id}")
    @app.post("/{trial_id}/v1/chat/completions")
    async def chat_completions(trial_id: str, request: Request):
        """Generate a chat completion via LiteLLM+vLLM and record data."""
        body = await request.json()
        messages: list[dict[str, Any]] = body.get("messages", [])
        is_streaming: bool = body.get("stream", False)

        # Build kwargs for litellm.acompletion via the custom provider
        completion_kwargs: dict[str, Any] = {
            "model": "verl-vllm/default",
            "messages": messages,
            "temperature": body.get("temperature", 1.0),
            "top_p": body.get("top_p", 1.0),
            "max_tokens": body.get("max_tokens", 2048),
            "stream": is_streaming,
            "extra_body": {"session_id": trial_id},
        }
        if body.get("tools"):
            completion_kwargs["tools"] = body["tools"]
        if "stop" in body:
            completion_kwargs["stop"] = body["stop"]
        if "repetition_penalty" in body:
            completion_kwargs["repetition_penalty"] = body["repetition_penalty"]

        try:
            response = await litellm.acompletion(**completion_kwargs)
        except Exception as e:
            logger.error("Generate failed for trial %s: %s", trial_id, e)
            return JSONResponse(
                status_code=500, content={"error": str(e)}
            )

        if is_streaming:
            return StreamingResponse(
                _stream_and_record(proxy, trial_id, messages, response),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        # Non-streaming: record and return
        _record_from_response(proxy, trial_id, messages, response)
        return JSONResponse(content=response.model_dump())

    return app


# ---------------------------------------------------------------------------
# Helpers for session recording
# ---------------------------------------------------------------------------


def _record_from_response(
    proxy: "LLMProxyServer",
    trial_id: str,
    messages: list[dict[str, Any]],
    response: Any,
) -> None:
    """Extract token_ids/logprobs from a LiteLLM ModelResponse and record."""
    choice = response.choices[0]

    # token_ids via provider_specific_fields
    psf = getattr(choice, "provider_specific_fields", None) or {}
    token_ids = psf.get("token_ids", [])

    # logprobs from standard ChoiceLogprobs
    logprobs_list: list[float] = []
    choice_logprobs = getattr(choice, "logprobs", None)
    if choice_logprobs and hasattr(choice_logprobs, "content") and choice_logprobs.content:
        logprobs_list = [t.logprob for t in choice_logprobs.content]

    # tool_calls
    tool_calls = None
    msg = choice.message
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        tool_calls = [
            tc.model_dump() if hasattr(tc, "model_dump") else tc
            for tc in msg.tool_calls
        ]

    content = ""
    if hasattr(msg, "content") and msg.content:
        content = msg.content

    proxy.recorder.record_completion(
        session_id=trial_id,
        messages=messages,
        completion_text=content,
        token_ids=token_ids,
        logprobs=logprobs_list,
        finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
        tool_calls=tool_calls,
    )


async def _stream_and_record(
    proxy: "LLMProxyServer",
    trial_id: str,
    messages: list[dict[str, Any]],
    stream,
) -> Any:
    """Iterate a LiteLLM streaming response, yield SSE chunks, and record.

    We accumulate token_ids and logprobs from the stream, then record the
    full completion once streaming finishes.

    Recording is placed in a ``finally`` block so that the completion is
    always persisted even when the client disconnects mid-stream (which
    causes the async generator to be closed via ``aclose()``).
    """
    collected_text_parts: list[str] = []
    collected_token_ids: list[int] = []
    collected_logprobs: list[float] = []
    finish_reason = "stop"
    tool_calls_collected: list[dict[str, Any]] = []

    try:
        async for chunk in stream:
            # Yield the raw SSE chunk to the client
            chunk_data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            yield f"data: {json.dumps(chunk_data)}\n\n"

            # Accumulate data for recording
            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0]

                # Text content
                if hasattr(delta, "delta") and hasattr(delta.delta, "content"):
                    if delta.delta.content:
                        collected_text_parts.append(delta.delta.content)

                # Tool calls
                if hasattr(delta, "delta") and hasattr(delta.delta, "tool_calls"):
                    if delta.delta.tool_calls:
                        for tc in delta.delta.tool_calls:
                            tc_dict = tc.model_dump() if hasattr(tc, "model_dump") else tc
                            tool_calls_collected.append(tc_dict)

                # Finish reason
                if hasattr(delta, "finish_reason") and delta.finish_reason:
                    finish_reason = delta.finish_reason

                # Provider-specific fields (token_ids on last chunk)
                psf = getattr(delta, "provider_specific_fields", None) or {}
                if "token_ids" in psf:
                    collected_token_ids = psf["token_ids"]

                # Logprobs
                chunk_logprobs = getattr(delta, "logprobs", None)
                if chunk_logprobs and hasattr(chunk_logprobs, "content") and chunk_logprobs.content:
                    for t in chunk_logprobs.content:
                        collected_logprobs.append(t.logprob)

        yield "data: [DONE]\n\n"
    finally:
        # Always record, even if the generator is closed early due to
        # client disconnect.  The data may be incomplete (e.g. missing
        # token_ids from the last chunk) but partial recording is still
        # preferable to losing the entire turn.
        proxy.recorder.record_completion(
            session_id=trial_id,
            messages=messages,
            completion_text="".join(collected_text_parts),
            token_ids=collected_token_ids,
            logprobs=collected_logprobs,
            finish_reason=finish_reason,
            tool_calls=tool_calls_collected if tool_calls_collected else None,
        )


# ---------------------------------------------------------------------------
# LLMProxyServer: the singleton server
# ---------------------------------------------------------------------------


class LLMProxyServer:
    """LLM Proxy server that holds vLLM server handles and serves
    OpenAI-compatible HTTP requests via LiteLLM.

    This server is designed to be a **singleton** in the Ray cluster.
    It is created once when the agent system starts and shared across
    all ``RemoteAgentLoop`` instances.

    Usage::

        proxy = LLMProxyServer(
            server_handles=[handle_1, handle_2],
            model_path="Qwen/Qwen2.5-7B-Instruct",
        )
        url = await proxy.start()          # http://0.0.0.0:PORT
        proxy.register_session("trial-1")
        # Agent uses base_url = f"{url}/trial-1/v1"
        # ...
        session = proxy.get_session_data("trial-1")
        await proxy.stop()
    """

    def __init__(
        self,
        server_handles: list,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 0,
        tool_format: str | None = None,
    ):
        """
        Args:
            server_handles: Ray actor handles of vLLM rollout servers.
            model_path: HuggingFace model name/path for the tokenizer.
            host: Bind address for the HTTP server.
            port: Port number. ``0`` means auto-select a free port.
            tool_format: Tool-call format (e.g. ``"hermes"``, ``"gpt-oss"``).
        """
        self.host = host
        self.port = port

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        self.recorder = SessionRecorder()

        # Initialize LiteLLM custom provider
        tool_parser: ToolParser | None = None
        if tool_format:
            tool_parser = ToolParser.get_tool_parser(tool_format, self.tokenizer)
            logger.info("Tool parser initialized: %s", tool_format)

        self._provider = VLLMRayProvider(
            server_handles=server_handles,
            tokenizer=self.tokenizer,
            tool_parser=tool_parser,
        )
        litellm.custom_provider_map = [
            {"provider": "verl-vllm", "custom_handler": self._provider}
        ]

        self.app = _build_app(self)

        self._server: uvicorn.Server | None = None
        self._serve_task: asyncio.Task | None = None
        self._actual_port: int | None = None

    # -- session management ------------------------------------------------

    def register_session(self, trial_id: str) -> None:
        """Register a new trial session for recording."""
        self.recorder.create_session(trial_id)

    def get_session_data(self, session_id: str):
        """Retrieve the recorded session data."""
        return self.recorder.get_session(session_id)

    def complete_session(self, session_id: str) -> None:
        self.recorder.mark_completed(session_id)

    def delete_session(self, session_id: str) -> None:
        self._provider.release_session(session_id)
        self.recorder.delete_session(session_id)

    # -- HTTP server lifecycle ---------------------------------------------

    @property
    def url(self) -> str | None:
        if self._actual_port is None:
            return None
        return f"http://{self.host}:{self._actual_port}"

    async def start(self, on_cancel_callback=None) -> str:
        """Start the HTTP server.  Returns the base URL.

        Args:
            on_cancel_callback: Optional callback function to call when the
                server task is cancelled. Useful for cleaning up singleton state.
        """
        if self.port == 0:
            self._actual_port = _find_free_port()
        else:
            self._actual_port = self.port

        config = uvicorn.Config(
            app=self.app,
            host=self.host,
            port=self._actual_port,
            log_level="info",
        )
        self._server = uvicorn.Server(config)
        self._server.install_signal_handlers = lambda: None

        async def _safe_serve():
            try:
                await self._server.serve()
            except asyncio.CancelledError:
                logger.warning(
                    "Proxy server task cancelled (cleaning up singleton state)"
                )
                if on_cancel_callback is not None:
                    try:
                        on_cancel_callback()
                    except Exception as e:
                        logger.error("Error in cancel callback: %s", e)
                raise

        self._serve_task = asyncio.create_task(_safe_serve())

        for _ in range(100):
            if self._server.started:
                break
            await asyncio.sleep(0.1)

        logger.info("LLM Proxy started at %s", self.url)
        return self.url

    async def stop(self) -> None:
        """Gracefully shut down the HTTP server."""
        if self._server is not None:
            self._server.should_exit = True
            if self._serve_task is not None:
                try:
                    await self._serve_task
                except asyncio.CancelledError:
                    logger.debug("Proxy serve task was cancelled during shutdown")
                self._serve_task = None
            self._server = None
            logger.info("LLM Proxy stopped")


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]
