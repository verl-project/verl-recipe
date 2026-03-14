# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Model Proxy Server for SWE Agent.

This module provides a lightweight HTTP proxy server that intercepts OpenAI-compatible
API calls from SWE-Agent and forwards them to VERL for processing.

The proxy implements an "anti-call" mechanism similar to ROCK's ModelService:
- SWE-Agent calls `/v1/chat/completions` → proxy suspends the request
- VERL calls `get_request()` to retrieve the request
- VERL generates a response and calls `send_response()`
- Proxy returns the OpenAI-format response to SWE-Agent
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Optional

from aiohttp import web

logger = logging.getLogger(__name__)


@dataclass
class ModelRequest:
    """Represents a model call request from SWE-Agent."""

    request_id: str
    messages: list[dict[str, Any]]  # OpenAI format messages
    model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    stream: bool = False
    extra_params: Optional[dict[str, Any]] = None


@dataclass
class ResponseState:
    """Tracks the lifecycle of a pending proxy response."""

    event: asyncio.Event
    state: Literal["pending", "completed", "failed"] = "pending"
    response_data: Optional[dict[str, Any]] = None
    error_message: Optional[str] = None


class ModelProxy:
    """Model call proxy server for intercepting SWE-Agent's OpenAI API calls.

    This proxy server:
    1. Listens on a configurable port for OpenAI-compatible requests
    2. Suspends incoming requests and queues them for VERL processing
    3. Provides control interfaces for VERL to retrieve requests and send responses
    4. Returns responses to SWE-Agent in OpenAI format

        Example:
            ```python
            proxy = ModelProxy()
            await proxy.start_server(port=0)

            # In VERL loop:
            request = await proxy.get_request()
            response = await generate_response(request.messages)
            await proxy.send_response(response, request=request)

            await proxy.stop_server()
            ```
    """

    def __init__(self, port: int = 0, host: str = "127.0.0.1"):
        """Initialize the model proxy.

        Args:
            port: Port to bind the HTTP server to. Defaults to 0 (let OS assign).
            host: Host address to bind to. Defaults to "127.0.0.1" (localhost only).
        """
        self.port = port
        self.host = host

        # Request queue: stores ModelRequest objects waiting for VERL processing
        self.request_queue: asyncio.Queue[ModelRequest] = asyncio.Queue()

        # Response storage: maps request_id -> response state
        self.response_storage: dict[str, ResponseState] = {}

        # Server components
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # Server state
        self._server_started = False
        self._stopping = False
        self._lock = asyncio.Lock()

    async def start_server(self, port: Optional[int] = None, max_retries: int = 1000) -> None:
        """Start the HTTP proxy server.

        If ``port == 0``, the OS picks an available ephemeral port atomically
        (recommended for high-concurrency startup). For fixed ports (``port > 0``),
        the server falls back to linear probing (port+1, port+2, ...).

        The default ``max_retries=1000`` covers large single-node deployments
        (e.g. hundreds of rollout workers).  Users can override this via
        ``proxy_config.max_port_retries`` in the YAML config.

        Args:
            port: Optional port override. If None, uses self.port.
            max_retries: Maximum number of consecutive ports to try when
                ``port > 0``. Defaults to 1000.

        Raises:
            RuntimeError: If server is already started or cannot find
                an available port within *max_retries* attempts.
        """
        async with self._lock:
            if self._server_started:
                raise RuntimeError("Server is already started")

            if port is not None:
                self.port = port

            self._stopping = False

            # Try to bind to port.
            initial_port = self.port
            for attempt in range(max_retries):
                try:
                    # Create aiohttp application
                    self.app = web.Application()
                    self.app.router.add_post("/v1/chat/completions", self._handle_chat_completion)

                    # Health check endpoint
                    self.app.router.add_get("/health", self._handle_health)

                    # Setup runner and site
                    self.runner = web.AppRunner(self.app)
                    await self.runner.setup()
                    self.site = web.TCPSite(self.runner, self.host, self.port)
                    await self.site.start()

                    # If binding to port 0, capture the actual assigned port.
                    self.port = self._resolve_bound_port()

                    self._server_started = True
                    logger.info(f"Model proxy server started on {self.host}:{self.port}")
                    return

                except OSError as e:
                    if e.errno == 98 and self.port > 0:  # Address already in use
                        logger.warning(f"Port {self.port} already in use, trying port {self.port + 1}")
                        self.port += 1

                        # Cleanup failed attempt
                        if self.runner:
                            await self.runner.cleanup()
                            self.runner = None
                        self.app = None
                        self.site = None
                    else:
                        raise

            # If we exhausted all retries
            raise RuntimeError(
                f"Failed to start server after {max_retries} attempts. Tried ports {initial_port} to {self.port - 1}."
            )

    def _resolve_bound_port(self) -> int:
        """Resolve actual bound port from aiohttp site after start()."""
        if self.site is None:
            raise RuntimeError("Server site is not initialized")

        server = getattr(self.site, "_server", None)
        sockets = getattr(server, "sockets", None)
        if not sockets:
            raise RuntimeError("Failed to resolve bound proxy port")

        return int(sockets[0].getsockname()[1])

    async def stop_server(self) -> None:
        """Stop the HTTP proxy server.

        This method gracefully shuts down the server and cleans up resources.
        """
        async with self._lock:
            if not self._server_started:
                logger.warning("Server is not started, skipping stop")
                return

            self._stopping = True
            self._fail_pending_requests("Model proxy server stopped")

            if self.site is not None:
                await self.site.stop()
                logger.info("Server site stopped")

            if self.runner is not None:
                await self.runner.cleanup()
                logger.info("Server runner cleaned up")

            # Clear pending state to avoid leaking across runs.
            self.request_queue = asyncio.Queue()
            self.response_storage.clear()

            self._server_started = False
            self._stopping = False
            logger.info("Model proxy server stopped")

    def _fail_pending_requests(self, error_message: str) -> None:
        """Fail and wake all pending requests."""
        for response_state in self.response_storage.values():
            if response_state.state != "pending":
                continue
            response_state.state = "failed"
            response_state.error_message = error_message
            response_state.event.set()

    async def _handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({"status": "ok", "service": "model_proxy"})

    async def _handle_chat_completion(self, request: web.Request) -> web.Response:
        """Handle OpenAI-compatible chat completion requests from SWE-Agent.

        This method:
        1. Parses the incoming request
        2. Creates a ModelRequest and queues it for VERL processing
        3. Waits for VERL to provide a response via send_response()
        4. Returns the response in OpenAI format

        Args:
            request: aiohttp request object containing the chat completion request.

        Returns:
            JSON response in OpenAI format.
        """
        request_id: Optional[str] = None
        try:
            if self._stopping:
                return web.json_response(
                    {"error": {"message": "Model proxy server is stopping", "type": "server_error"}}, status=503
                )

            # Parse request body
            data = await request.json()

            # Extract messages (required)
            messages = data.get("messages", [])
            if not messages:
                return web.json_response(
                    {"error": {"message": "messages field is required", "type": "invalid_request_error"}}, status=400
                )

            # Generate unique request ID
            request_id = str(uuid.uuid4())

            # Extract other parameters
            model = data.get("model")
            temperature = data.get("temperature")
            max_tokens = data.get("max_tokens")
            stream = data.get("stream", False)

            if stream:
                return web.json_response(
                    {"error": {"message": "Streaming is not supported by ModelProxy", "type": "invalid_request_error"}},
                    status=400,
                )

            # Store extra parameters
            extra_params = {
                k: v for k, v in data.items() if k not in ["messages", "model", "temperature", "max_tokens", "stream"]
            }

            # Create ModelRequest
            model_request = ModelRequest(
                request_id=request_id,
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                extra_params=extra_params,
            )

            logger.debug(f"Received request {request_id} with {len(messages)} messages")

            # Create response event for this request
            response_state = ResponseState(event=asyncio.Event())
            self.response_storage[request_id] = response_state

            # Queue the request for VERL processing
            await self.request_queue.put(model_request)

            # Wait for VERL to provide response
            await response_state.event.wait()

            # Retrieve response
            final_state = self.response_storage.pop(request_id, None)

            if final_state is None:
                logger.error(f"No response state for request {request_id}")
                return web.json_response(
                    {"error": {"message": "Internal server error: no response generated", "type": "server_error"}},
                    status=500,
                )

            if final_state.state == "failed":
                error_message = final_state.error_message or "Model proxy request failed"
                logger.warning(f"Request {request_id} failed: {error_message}")
                return web.json_response(
                    {"error": {"message": error_message, "type": "server_error"}},
                    status=503 if self._stopping else 500,
                )

            if final_state.state != "completed" or final_state.response_data is None:
                logger.error(f"Invalid response state for request {request_id}: {final_state.state}")
                return web.json_response(
                    {"error": {"message": "Internal server error: invalid response state", "type": "server_error"}},
                    status=500,
                )

            # Return OpenAI-format response
            return web.json_response(final_state.response_data)

        except asyncio.CancelledError:
            if request_id is not None:
                self.response_storage.pop(request_id, None)
            logger.warning("Request cancelled")
            raise
        except Exception as e:
            logger.exception(f"Error handling chat completion request: {e}")
            return web.json_response(
                {"error": {"message": f"Internal server error: {str(e)}", "type": "server_error"}}, status=500
            )

    async def get_request(self) -> ModelRequest:
        """Get the next model call request from the queue.

        This method is called by VERL to retrieve the next request from SWE-Agent.
        It blocks until a request is available.

        Returns:
            ModelRequest object containing the request details.

        Example:
            ```python
            request = await proxy.get_request()
            messages = request.messages
            # Process messages and generate response
            ```
        """
        request = await self.request_queue.get()
        logger.debug(f"Retrieved request {request.request_id} from queue")
        return request

    async def send_response(
        self,
        response: str,
        request: Optional[ModelRequest] = None,
        request_id: Optional[str] = None,
        finish_reason: str = "stop",
    ) -> None:
        """Send a response back to SWE-Agent for a specific request.

        This method is called by VERL after generating a response. It formats the
        response in OpenAI format and signals the waiting request handler.

        Args:
            response: The generated response text.
            request: Optional ModelRequest object. If provided, uses its request_id.
            request_id: Optional request ID. Required if request is not provided.
            finish_reason: Finish reason for the response. Defaults to "stop".

        Raises:
            KeyError: If request_id is not found in response storage.
            ValueError: If neither request nor request_id is provided.

        Example:
            ```python
            request = await proxy.get_request()
            response_text = await generate_response(request.messages)
            # Option 1: Pass the request object explicitly
            await proxy.send_response(response_text, request=request)
            # Option 2: Pass the request_id explicitly
            await proxy.send_response(response_text, request_id=request.request_id)
            ```
        """
        # Determine request_id
        if request is not None:
            request_id = request.request_id
        elif request_id is None:
            raise ValueError("Either request or request_id must be provided")

        if request_id not in self.response_storage:
            raise KeyError(f"Request ID {request_id} not found in response storage")

        response_state = self.response_storage[request_id]
        if response_state.state != "pending":
            raise RuntimeError(f"Request ID {request_id} is already in state {response_state.state}")

        # Format response in OpenAI format
        response_data = {
            "id": f"chatcmpl-{request_id}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "swe-agent-proxy",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": response}, "finish_reason": finish_reason}
            ],
            "usage": {
                "prompt_tokens": 0,  # Could be calculated if needed
                "completion_tokens": 0,  # Could be calculated if needed
                "total_tokens": 0,
            },
        }

        # Store response and signal event
        response_state.state = "completed"
        response_state.response_data = response_data
        response_state.error_message = None
        response_state.event.set()

        logger.debug(f"Sent response for request {request_id}")
