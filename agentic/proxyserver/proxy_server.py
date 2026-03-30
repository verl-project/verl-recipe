"""Standalone LLM Proxy Server management via Ray actor.

Extracts the proxy server lifecycle from ``RemoteAgentLoop`` into
a reusable module.  The proxy is wrapped in a **Ray named actor**
(``ProxyServerActor``) that is pinned to the head node via
``NodeAffinitySchedulingStrategy``.

Usage from a recipe::

    from verl.experimental.proxy_server import start_proxy_server

    proxy_url = start_proxy_server(
        server_handles=server_handles,
        model_path="Qwen/Qwen2.5-7B-Instruct",
    )

Usage from ``RemoteAgentLoop`` (or any worker node)::

    from verl.experimental.proxy_server import get_proxy_url

    url = get_proxy_url()  # lightweight Ray RPC to discover URL
    # Then interact via HTTP: POST /sessions/{id}, GET /sessions/{id}, ...
"""

from __future__ import annotations

import logging
import os

import ray

from .server import LLMProxyServer

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

PROXY_ACTOR_NAME = "llm_proxy_server"

# Module-level handle to keep the non-detached actor alive (prevent GC).
_proxy_actor_handle = None


# ---------------------------------------------------------------------------
# Ray Actor
# ---------------------------------------------------------------------------


@ray.remote
class ProxyServerActor:
    """Ray actor wrapping :class:`LLMProxyServer`, pinned to head node.

    The actor manages the full lifecycle of the HTTP proxy that bridges
    OpenAI-compatible requests to verl's vLLM rollout servers.
    """

    def __init__(
        self,
        server_handles: list,
        model_path: str,
        host: str = "0.0.0.0",
        port: int = 0,
        tool_format: str | None = "hermes",
    ):
        self.proxy = LLMProxyServer(
            server_handles=server_handles,
            model_path=model_path,
            host=host,
            port=port,
            tool_format=tool_format,
        )
        self._url: str | None = None

    async def start(self) -> str:
        """Start the proxy HTTP server and return a cluster-routable URL.

        Uses ``ray.util.get_node_ip_address()`` (the standard pattern in
        the verl codebase) to obtain an IP that is reachable from other
        nodes in the Ray cluster.
        """
        await self.proxy.start()

        actual_ip = ray.util.get_node_ip_address().strip("[]")
        actual_port = self.proxy._actual_port
        self._url = f"http://{actual_ip}:{actual_port}"
        logger.info("ProxyServerActor started at %s", self._url)
        return self._url

    def get_url(self) -> str | None:
        """Return the cluster-routable proxy URL, or ``None`` if not started."""
        return self._url

    async def stop(self) -> None:
        """Gracefully shut down the proxy HTTP server."""
        if self.proxy is not None:
            await self.proxy.stop()
            logger.info("ProxyServerActor stopped")


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def start_proxy_server(
    server_handles: list,
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 0,
    tool_format: str = "hermes",
) -> str:
    """Start the proxy server as a Ray named actor on the head node.

    If an actor with the same name already exists, its URL is returned
    immediately without creating a new one.

    This function is intended to be called from the recipe's
    ``TaskRunner.run()`` (which executes on the head node) **after**
    ``trainer.init_workers()`` has produced ``server_handles``.

    Args:
        server_handles: Ray actor handles of vLLM rollout servers.
        model_path: HuggingFace model name/path for the tokenizer.
        host: Bind address for the HTTP server.
        port: Port number. ``0`` means auto-select a free port.
        tool_format: Tool-call format (e.g. ``"hermes"``).

    Returns:
        The cluster-routable proxy URL, e.g. ``http://10.0.1.5:9123``.
    """
    # Fast path: reuse existing actor
    try:
        existing = ray.get_actor(PROXY_ACTOR_NAME)
        url = ray.get(existing.get_url.remote())
        if url is not None:
            logger.info("Reusing existing proxy server actor at %s", url)
            return url
    except ValueError:
        pass

    # Create new actor, pinned to the current (head) node
    global _proxy_actor_handle
    head_node_id = ray.get_runtime_context().get_node_id()
    actor = ProxyServerActor.options(
        name=PROXY_ACTOR_NAME,
        scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
            node_id=head_node_id,
            soft=False,
        ),
    ).remote(server_handles, model_path, host, port, tool_format)

    # Keep the handle at module level to prevent GC of the non-detached actor.
    _proxy_actor_handle = actor

    url = ray.get(actor.start.remote())
    logger.info("Proxy server actor created at %s (head node %s)", url, head_node_id)
    return url


def get_proxy_url() -> str | None:
    """Get the URL of an already-running proxy server.

    Performs a lightweight Ray RPC to the named actor.  Returns ``None``
    if the actor does not exist.

    This is the primary entry point for ``RemoteAgentLoop`` on any node
    to discover the proxy address.
    """
    try:
        actor = ray.get_actor(PROXY_ACTOR_NAME)
        return ray.get(actor.get_url.remote())
    except ValueError:
        return None


def stop_proxy_server() -> None:
    """Stop the proxy server actor and remove it from the cluster."""
    global _proxy_actor_handle
    try:
        actor = ray.get_actor(PROXY_ACTOR_NAME)
        ray.get(actor.stop.remote())
        ray.kill(actor)
        logger.info("Proxy server actor stopped and killed")
    except ValueError:
        pass
    _proxy_actor_handle = None
