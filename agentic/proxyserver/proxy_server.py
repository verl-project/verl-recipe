"""Standalone LLM Proxy Server — framework-agnostic relay service.

Starts the proxy as an independent HTTP + WebSocket service in relay
mode.  No framework-specific dependencies (no Ray, no PyTorch, no
transformers).  Any framework's inference workers can connect to
``/ws/worker`` and service inference requests pushed by the proxy.

Usage::

    # CLI
    python -m recipe.agentic.proxyserver.proxy_server \\
        --host 0.0.0.0 --port 8080

    # Programmatic
    from recipe.agentic.proxyserver.proxy_server import run_standalone_proxy
    run_standalone_proxy(host="0.0.0.0", port=8080)
"""

from __future__ import annotations

import asyncio
import logging

from .server import LLMProxyServer

logger = logging.getLogger(__name__)


def run_standalone_proxy(
    host: str = "0.0.0.0",
    port: int = 8080,
    relay_request_timeout: float = 300.0,
    log_level: str = "INFO",
) -> None:
    """Run the proxy as a standalone service in relay mode.

    This starts the HTTP + WebSocket server and blocks until interrupted.
    No framework-specific dependencies are required.  Framework-side
    inference workers must connect to ``ws://{host}:{port}/ws/worker``
    to service requests.

    Args:
        host: Bind address for the HTTP server.
        port: Port number.
        relay_request_timeout: Timeout (seconds) for relay-mode requests.
        log_level: Logging level (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

    proxy = LLMProxyServer(
        host=host,
        port=port,
        relay_request_timeout=relay_request_timeout,
    )

    async def _run():
        url = await proxy.start()
        logger.info(
            "Standalone proxy started at %s (relay mode)\n"
            "  Workers connect to: ws://%s:%d/ws/worker\n"
            "  Health check:       %s/health",
            url, host, port, url,
        )
        try:
            while True:
                await asyncio.sleep(3600)
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            await proxy.stop()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass


def main() -> None:
    """CLI entry point for running the proxy as a standalone service."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the LLM Proxy as a standalone relay service.",
    )
    parser.add_argument(
        "--host", default="0.0.0.0",
        help="Bind address (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port number (default: 8080)",
    )
    parser.add_argument(
        "--relay-timeout", type=float, default=300.0,
        help="Timeout (seconds) for relay requests (default: 300)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    run_standalone_proxy(
        host=args.host,
        port=args.port,
        relay_request_timeout=args.relay_timeout,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
