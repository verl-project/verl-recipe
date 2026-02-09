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
from typing import Any, Optional

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
    # Additional OpenAI API parameters
    extra_params: Optional[dict[str, Any]] = None

    def is_session_end(self) -> bool:
        """Check if this is a session end marker."""
        # Check if messages contain a session end indicator
        # This can be customized based on SWE-Agent's behavior
        return False


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
            # Pass only response (uses current request from get_request)
            await proxy.send_response(response)

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

        # Response storage: maps request_id -> (response_event, response_data)
        self.response_storage: dict[str, tuple[asyncio.Event, Optional[dict[str, Any]]]] = {}

        # Track the most recently retrieved request (for convenience when response is sent without request_id)
        self._current_request: Optional[ModelRequest] = None

        # Server components
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        # Server state
        self._server_started = False
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

            if self.site is not None:
                await self.site.stop()
                logger.info("Server site stopped")

            if self.runner is not None:
                await self.runner.cleanup()
                logger.info("Server runner cleaned up")

            # Clear pending state to avoid leaking across runs.
            self.request_queue = asyncio.Queue()
            self.response_storage.clear()
            self._current_request = None

            self._server_started = False
            logger.info("Model proxy server stopped")

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
        try:
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
            response_event = asyncio.Event()
            self.response_storage[request_id] = (response_event, None)

            # Queue the request for VERL processing
            await self.request_queue.put(model_request)

            # Wait for VERL to provide response
            await response_event.wait()

            # Retrieve response
            _, response_data = self.response_storage.pop(request_id, (None, None))

            if response_data is None:
                logger.error(f"No response data for request {request_id}")
                return web.json_response(
                    {"error": {"message": "Internal server error: no response generated", "type": "server_error"}},
                    status=500,
                )

            # Return OpenAI-format response
            return web.json_response(response_data)

        except asyncio.CancelledError:
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
        self._current_request = request  # Track current request for convenience
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
            # Option 1: Pass only response (uses current request from get_request)
            await proxy.send_response(response_text)
            # Option 2: Pass the request object explicitly
            await proxy.send_response(response_text, request=request)
            # Option 3: Pass the request_id explicitly
            await proxy.send_response(response_text, request_id=request.request_id)
            ```
        """
        # Determine request_id
        if request is not None:
            request_id = request.request_id
        elif request_id is None:
            # Use current request if available (for convenience)
            if self._current_request is not None:
                request_id = self._current_request.request_id
            else:
                raise ValueError(
                    "Either request, request_id must be provided, or a request must have been "
                    "retrieved via get_request() first"
                )

        if request_id not in self.response_storage:
            raise KeyError(f"Request ID {request_id} not found in response storage")

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
        response_event, _ = self.response_storage[request_id]
        self.response_storage[request_id] = (response_event, response_data)
        response_event.set()

        logger.debug(f"Sent response for request {request_id}")

    async def send_error_response(self, request_id: str, error_message: str, error_type: str = "server_error") -> None:
        """Send an error response back to SWE-Agent.

        Args:
            request_id: The unique ID of the request.
            error_message: Error message to return.
            error_type: Error type. Defaults to "server_error".
        """
        if request_id not in self.response_storage:
            raise KeyError(f"Request ID {request_id} not found in response storage")

        response_data = {"error": {"message": error_message, "type": error_type}}

        response_event, _ = self.response_storage[request_id]
        self.response_storage[request_id] = (response_event, response_data)
        response_event.set()

        logger.warning(f"Sent error response for request {request_id}: {error_message}")
