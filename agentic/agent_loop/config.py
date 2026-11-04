"""Configuration for RemoteAgentLoop — all values read from environment variables.

Environment variables
~~~~~~~~~~~~~~~~~~~~~
HARBOR_SERVER_URL                   Harbor server URL (default: http://localhost:8080)
HARBOR_TIMEOUT                      Timeout in seconds for harbor requests (default: 1800)
REMOTE_AGENT_NAME                   Agent name to run on harbor
REMOTE_AGENT_IMPORT_PATH            Python import path for the agent class
REMOTE_MODEL_NAME                   Model name passed to the remote agent
REMOTE_AGENT_KWARGS                 JSON-encoded dict of extra agent kwargs
REMOTE_AGENT_MAX_RETRIES            Max retry attempts on failure (default: 3)
REMOTE_AGENT_RETRY_BASE_DELAY       Base delay in seconds for exponential backoff (default: 2.0)
REMOTE_AGENT_TASK_PATH_TEMPLATE     Task path template with {instance_id} placeholder
REMOTE_AGENT_ENVIRONMENT_OVERRIDES  JSON-encoded dict of env vars forwarded to remote agent
REMOTE_AGENT_ENVIRONMENT_KWARGS     JSON-encoded dict of kwargs forwarded to the environment
                                    constructor (e.g. namespace, registry, use_sandbox_claim …)
REMOTE_AGENT_ENVIRONMENT_IMPORT_PATH
                                    Python import path of the harbor environment class used
                                    when ``use_local_trial`` is true. Format: ``module:Class``.
                                    Default: ``harbor.environments.ack:ACKEnvironment``.
                                    Use ``harbor.environments.docker.docker:DockerEnvironment``
                                    for local Docker-based runs without a Kubernetes cluster.
REMOTE_AGENT_USE_LOCAL_TRIAL        If "1" or "true", run Trial locally instead of via Harbor
                                    HTTP server (default: false)
PROXY_SERVER_URL                    URL of a standalone proxy server (e.g. http://proxy:8080).
                                    When set, the agent loop skips Ray-based proxy discovery
                                    and connects to this URL directly.  An inference worker
                                    is also started automatically to bridge vLLM to the proxy.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class RemoteAgentConfig:
    """Typed configuration for the remote agent loop, populated from env vars."""

    harbor_server_url: str = "http://localhost:8080"
    harbor_timeout: float = 1800.0

    agent_name: Optional[str] = None
    agent_import_path: Optional[str] = None
    model_name: Optional[str] = None
    agent_kwargs: dict[str, Any] = field(default_factory=dict)

    max_retries: int = 3
    retry_base_delay: float = 2.0

    task_path_template: str = "/home/verl/dataset-tasks/{instance_id}"

    environment_overrides: dict[str, str] = field(default_factory=dict)
    environment_kwargs: dict[str, Any] = field(default_factory=dict)
    environment_import_path: str = "harbor.environments.ack:ACKEnvironment"
    use_local_trial: bool = False

    # Standalone proxy mode
    proxy_server_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "RemoteAgentConfig":
        """Build config by reading environment variables."""
        kwargs: dict[str, Any] = {}

        if v := os.getenv("HARBOR_SERVER_URL"):
            kwargs["harbor_server_url"] = v
        if v := os.getenv("HARBOR_TIMEOUT"):
            kwargs["harbor_timeout"] = float(v)

        if v := os.getenv("REMOTE_AGENT_NAME"):
            kwargs["agent_name"] = v
        if v := os.getenv("REMOTE_AGENT_IMPORT_PATH"):
            kwargs["agent_import_path"] = v
        if v := os.getenv("REMOTE_MODEL_NAME"):
            kwargs["model_name"] = v
        if v := os.getenv("REMOTE_AGENT_KWARGS"):
            kwargs["agent_kwargs"] = json.loads(v)

        if v := os.getenv("REMOTE_AGENT_MAX_RETRIES"):
            kwargs["max_retries"] = int(v)
        if v := os.getenv("REMOTE_AGENT_RETRY_BASE_DELAY"):
            kwargs["retry_base_delay"] = float(v)

        if v := os.getenv("REMOTE_AGENT_TASK_PATH_TEMPLATE"):
            kwargs["task_path_template"] = v
        if v := os.getenv("REMOTE_AGENT_ENVIRONMENT_OVERRIDES"):
            kwargs["environment_overrides"] = json.loads(v)
        if v := os.getenv("REMOTE_AGENT_ENVIRONMENT_KWARGS"):
            kwargs["environment_kwargs"] = json.loads(v)
        if v := os.getenv("REMOTE_AGENT_ENVIRONMENT_IMPORT_PATH"):
            kwargs["environment_import_path"] = v
        if v := os.getenv("REMOTE_AGENT_USE_LOCAL_TRIAL"):
            kwargs["use_local_trial"] = v.strip().lower() in ("1", "true", "yes")
        if v := os.getenv("PROXY_SERVER_URL"):
            kwargs["proxy_server_url"] = v

        return cls(**kwargs)
