"""verl-specific integration: Ray actor wrapper and LiteLLM local-mode handler.

This module contains all verl / Ray / LiteLLM–specific code that was
previously mixed into ``server.py`` and ``proxy_server.py``.  The core
proxy server (``server.py``) is now framework-agnostic — this module
bridges it to verl's vLLM rollout infrastructure.

Usage from a recipe::

    from recipe.agentic.proxyserver.ray_actor import start_proxy_server

    proxy_url = start_proxy_server(
        load_balancer=trainer.llm_server_manager.global_load_balancer,
        model_path="Qwen/Qwen2.5-7B-Instruct",
    )

Usage from ``RemoteAgentLoop`` (any worker node)::

    from recipe.agentic.proxyserver.ray_actor import get_proxy_url

    url = get_proxy_url()  # lightweight Ray RPC
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

from .server import LLMProxyServer, build_openai_response

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

PROXY_ACTOR_NAME = "llm_proxy_server"

# Module-level handle to keep the non-detached actor alive (prevent GC).
_proxy_actor_handle = None


# ---------------------------------------------------------------------------
# LiteLLM-based local completion handler (verl-specific)
# ---------------------------------------------------------------------------


def _create_local_proxy(
    load_balancer: Any,
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 0,
    tool_format: str | None = "hermes",
    tool_parser_factory: Any = None,
) -> LLMProxyServer:
    """Create an LLMProxyServer in local mode with LiteLLM + Ray provider.

    Sets up the tokenizer, VLLMRayProvider, and LiteLLM custom-provider
    mapping, then injects a local completion handler into the generic
    proxy server.

    Args:
        load_balancer: Ray actor handle of verl's
            ``GlobalRequestLoadBalancer``. The provider acquires a vLLM
            server handle per request from this balancer.
        tool_parser_factory: Optional callable ``(format, tokenizer) -> tool_parser``.
            If not supplied, the function tries to import
            ``verl.experimental.agent_loop.tool_parser.ToolParser`` as a
            convenience default.  If that import fails, tool parsing is
            simply disabled.
    """
    import litellm
    from transformers import AutoTokenizer

    from .vllm_provider import VLLMRayProvider

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    tool_parser = None
    if tool_format:
        if tool_parser_factory is not None:
            tool_parser = tool_parser_factory(tool_format, tokenizer)
        else:
            try:
                from verl.experimental.agent_loop.tool_parser import ToolParser
                tool_parser = ToolParser.get_tool_parser(tool_format, tokenizer)
            except ImportError:
                logger.warning(
                    "verl.experimental.agent_loop.tool_parser not available; "
                    "tool parsing disabled"
                )
        if tool_parser is not None:
            logger.info("Tool parser initialized: %s", tool_format)

    provider = VLLMRayProvider(
        load_balancer=load_balancer,
        tokenizer=tokenizer,
        tool_parser=tool_parser,
    )
    litellm.custom_provider_map = [
        {"provider": "verl-vllm", "custom_handler": provider}
    ]

    # Build the handler closure that captures litellm + provider
    handler = _make_litellm_handler()

    proxy = LLMProxyServer(
        host=host,
        port=port,
        completion_handler=handler,
        on_session_deleted=lambda sid: provider.release_session(sid),
    )
    # Attach for code that needs direct access (e.g. agentic_main)
    proxy.tokenizer = tokenizer
    proxy._provider = provider

    return proxy


def _make_litellm_handler():
    """Build an async completion handler that routes via LiteLLM."""

    async def _handle_local_completion(
        proxy: LLMProxyServer,
        trial_id: str,
        messages: list[dict[str, Any]],
        body: dict[str, Any],
        is_streaming: bool,
    ):
        import litellm
        from fastapi.responses import JSONResponse, StreamingResponse

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
            return JSONResponse(status_code=500, content={"error": str(e)})

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

    return _handle_local_completion


# ---------------------------------------------------------------------------
# LiteLLM response recording helpers
# ---------------------------------------------------------------------------


def _record_from_response(
    proxy: LLMProxyServer,
    trial_id: str,
    messages: list[dict[str, Any]],
    response: Any,
) -> None:
    """Extract token_ids/logprobs from a LiteLLM ModelResponse and record."""
    choice = response.choices[0]

    psf = getattr(choice, "provider_specific_fields", None) or {}
    token_ids = psf.get("token_ids", [])

    logprobs_list: list[float] = []
    choice_logprobs = getattr(choice, "logprobs", None)
    if choice_logprobs and hasattr(choice_logprobs, "content") and choice_logprobs.content:
        logprobs_list = [t.logprob for t in choice_logprobs.content]

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
    proxy: LLMProxyServer,
    trial_id: str,
    messages: list[dict[str, Any]],
    stream,
) -> Any:
    """Iterate a LiteLLM streaming response, yield SSE chunks, and record."""
    collected_text_parts: list[str] = []
    collected_token_ids: list[int] = []
    collected_logprobs: list[float] = []
    finish_reason = "stop"
    tool_calls_collected: list[dict[str, Any]] = []

    try:
        async for chunk in stream:
            chunk_data = chunk.model_dump() if hasattr(chunk, "model_dump") else chunk
            yield f"data: {json.dumps(chunk_data)}\n\n"

            if hasattr(chunk, "choices") and chunk.choices:
                delta = chunk.choices[0]

                if hasattr(delta, "delta") and hasattr(delta.delta, "content"):
                    if delta.delta.content:
                        collected_text_parts.append(delta.delta.content)

                if hasattr(delta, "delta") and hasattr(delta.delta, "tool_calls"):
                    if delta.delta.tool_calls:
                        for tc in delta.delta.tool_calls:
                            tc_dict = tc.model_dump() if hasattr(tc, "model_dump") else tc
                            tool_calls_collected.append(tc_dict)

                if hasattr(delta, "finish_reason") and delta.finish_reason:
                    finish_reason = delta.finish_reason

                psf = getattr(delta, "provider_specific_fields", None) or {}
                if "token_ids" in psf:
                    collected_token_ids = psf["token_ids"]

                chunk_logprobs = getattr(delta, "logprobs", None)
                if chunk_logprobs and hasattr(chunk_logprobs, "content") and chunk_logprobs.content:
                    for t in chunk_logprobs.content:
                        collected_logprobs.append(t.logprob)

        yield "data: [DONE]\n\n"
    finally:
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
# Ray Actor
# ---------------------------------------------------------------------------


def _get_ray():
    """Lazy import of ray."""
    import ray
    return ray


def _get_actor_class():
    """Create the Ray remote actor class lazily."""
    ray = _get_ray()

    @ray.remote
    class ProxyServerActor:
        """Ray actor wrapping :class:`LLMProxyServer` (local mode),
        pinned to the head node."""

        def __init__(
            self,
            load_balancer: Any,
            model_path: str,
            host: str = "0.0.0.0",
            port: int = 0,
            tool_format: str | None = "hermes",
        ):
            self._lb = load_balancer
            self.proxy = _create_local_proxy(
                load_balancer=load_balancer,
                model_path=model_path,
                host=host,
                port=port,
                tool_format=tool_format,
            )
            self._url: str | None = None

        async def start(self) -> str:
            await self.proxy.start()
            actual_ip = ray.util.get_node_ip_address().strip("[]")
            actual_port = self.proxy._actual_port
            self._url = f"http://{actual_ip}:{actual_port}"
            logger.info("ProxyServerActor started at %s", self._url)
            return self._url

        def get_url(self) -> str | None:
            return self._url

        def get_load_balancer(self) -> Any:
            """Return the verl ``GlobalRequestLoadBalancer`` actor handle
            owned by this proxy.

            Used by ``RemoteAgentLoop._ensure_inference_worker`` (standalone
            proxy mode) to obtain a load-balancer handle, since the new
            ``LLMServerClient`` does not expose vLLM server handles directly.
            """
            return self._lb

        async def stop(self) -> None:
            if self.proxy is not None:
                await self.proxy.stop()
                logger.info("ProxyServerActor stopped")

    return ProxyServerActor


# ---------------------------------------------------------------------------
# Lightweight load-balancer registry (used in standalone-proxy mode)
# ---------------------------------------------------------------------------


def _get_registry_class():
    """Create the lightweight Ray named actor that just holds the LB handle.

    Used only when an external standalone proxy is configured
    (``PROXY_SERVER_URL`` set). In that case the heavyweight
    :class:`ProxyServerActor` is unnecessary — we only need a Ray-addressable
    place where ``RemoteAgentLoop._ensure_inference_worker`` can look up
    the verl ``GlobalRequestLoadBalancer`` handle to start an
    :class:`InferenceWorkerClient`.
    """
    ray = _get_ray()

    @ray.remote(num_cpus=0.01)
    class LoadBalancerRegistry:
        def __init__(self, load_balancer: Any):
            self._lb = load_balancer
            self._url: str | None = None  # no HTTP server in this mode

        def get_url(self) -> str | None:
            return self._url

        def get_load_balancer(self) -> Any:
            return self._lb

    return LoadBalancerRegistry


def start_lb_registry(load_balancer: Any) -> None:
    """Start a lightweight load-balancer registry as a Ray named actor.

    Registers under :data:`PROXY_ACTOR_NAME` so that
    :func:`RemoteAgentLoop._ensure_inference_worker` can call
    ``ray.get_actor(PROXY_ACTOR_NAME).get_load_balancer.remote()`` without
    knowing whether the full :class:`ProxyServerActor` or this lightweight
    registry was started.

    No-op if an actor with the same name already exists.
    """
    ray = _get_ray()

    try:
        ray.get_actor(PROXY_ACTOR_NAME)
        logger.info(
            "Reusing existing actor '%s' as load-balancer registry",
            PROXY_ACTOR_NAME,
        )
        return
    except ValueError:
        pass

    global _proxy_actor_handle
    Registry = _get_registry_class()
    head_node_id = ray.get_runtime_context().get_node_id()
    actor = Registry.options(
        name=PROXY_ACTOR_NAME,
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote(load_balancer)

    _proxy_actor_handle = actor
    logger.info(
        "Load-balancer registry started as named actor '%s' on head node %s",
        PROXY_ACTOR_NAME,
        head_node_id,
    )


# ---------------------------------------------------------------------------
# Convenience functions — Ray actor mode
# ---------------------------------------------------------------------------


def start_proxy_server(
    load_balancer: Any,
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 0,
    tool_format: str = "hermes",
) -> str:
    """Start the proxy server as a Ray named actor on the head node.

    If an actor with the same name already exists, its URL is returned
    immediately without creating a new one.

    Args:
        load_balancer: Ray actor handle of verl's
            ``GlobalRequestLoadBalancer``. The proxy uses it to route
            inference requests to vLLM rollout servers.
        model_path: HuggingFace model name/path for the tokenizer.
        host: Bind address for the HTTP server.
        port: Port number. ``0`` means auto-select a free port.
        tool_format: Tool-call format (e.g. ``"hermes"``).

    Returns:
        The cluster-routable proxy URL, e.g. ``http://10.0.1.5:9123``.
    """
    ray = _get_ray()

    try:
        existing = ray.get_actor(PROXY_ACTOR_NAME)
        url = ray.get(existing.get_url.remote())
        if url is not None:
            logger.info("Reusing existing proxy server actor at %s", url)
            return url
    except ValueError:
        pass

    global _proxy_actor_handle
    ProxyServerActor = _get_actor_class()
    head_node_id = ray.get_runtime_context().get_node_id()
    actor = ProxyServerActor.options(
        name=PROXY_ACTOR_NAME,
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote(load_balancer, model_path, host, port, tool_format)

    _proxy_actor_handle = actor

    url = ray.get(actor.start.remote())
    logger.info("Proxy server actor created at %s (head node %s)", url, head_node_id)
    return url


def get_proxy_url() -> str | None:
    """Get the URL of an already-running proxy server via Ray RPC.

    Returns ``None`` if the actor does not exist.
    """
    ray = _get_ray()
    try:
        actor = ray.get_actor(PROXY_ACTOR_NAME)
        return ray.get(actor.get_url.remote())
    except ValueError:
        return None


def stop_proxy_server() -> None:
    """Stop the proxy server actor and remove it from the cluster."""
    ray = _get_ray()
    global _proxy_actor_handle
    try:
        actor = ray.get_actor(PROXY_ACTOR_NAME)
        ray.get(actor.stop.remote())
        ray.kill(actor)
        logger.info("Proxy server actor stopped and killed")
    except ValueError:
        pass
    _proxy_actor_handle = None
