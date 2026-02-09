# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Config Validator for SWE Agent.

This module provides validation logic for SWE Agent configuration,
improving testability and separation of concerns.
"""

from typing import Any, Dict, Optional


class ConfigValidationError(ValueError):
    """Exception raised when config validation fails."""
    pass


class AgentConfigValidator:
    """Validator for SWE Agent configuration.

    Validates proxy_config and sandbox_config to ensure all required
    fields are present and have valid values.

    Example:
        >>> config = {"proxy_config": {"port": 8080}, "sandbox_config": {}}
        >>> validator = AgentConfigValidator(config)
        >>> validator.validate()  # Raises ConfigValidationError if invalid
    """

    def __init__(self, config: Optional[Dict[str, Any]]):
        """Initialize validator with configuration.

        Args:
            config: Agent configuration dictionary.
        """
        self.config = config

    def validate(self) -> None:
        """Validate the configuration.

        Raises:
            ConfigValidationError: If configuration is invalid.
        """
        # Check if config exists
        if self.config is None:
            raise ConfigValidationError(
                "SWE Agent Loop requires agent configuration. "
                "Please provide config in actor_rollout_ref.rollout.agent."
            )

        # Validate proxy_config
        self._validate_proxy_config()

        # Validate sandbox_config
        self._validate_sandbox_config()

    def _validate_proxy_config(self) -> None:
        """Validate proxy configuration.

        Raises:
            ConfigValidationError: If proxy_config is invalid.
        """
        proxy_config = self.config.get("proxy_config", {})

        if not isinstance(proxy_config, dict):
            raise ConfigValidationError("proxy_config must be a dict")

        # Validate port
        proxy_port = proxy_config.get("port", 8080)
        if not isinstance(proxy_port, int) or proxy_port < 1 or proxy_port > 65535:
            raise ConfigValidationError(
                f"proxy_config.port must be a valid port number (1-65535), got: {proxy_port}"
            )

    def _validate_sandbox_config(self) -> None:
        """Validate sandbox configuration.

        Raises:
            ConfigValidationError: If sandbox_config is invalid.
        """
        sandbox_config = self.config.get("sandbox_config", {})

        if not isinstance(sandbox_config, dict):
            raise ConfigValidationError("sandbox_config must be a dict")

        # Validate timeout
        swe_agent_timeout = sandbox_config.get("swe_agent_timeout")
        if swe_agent_timeout is not None:
            if not isinstance(swe_agent_timeout, (int, float)) or swe_agent_timeout <= 0:
                raise ConfigValidationError(
                    f"sandbox_config.swe_agent_timeout must be a positive number, got: {swe_agent_timeout}"
                )

        # Validate max_steps
        max_steps = sandbox_config.get("max_steps")
        if max_steps is not None:
            if not isinstance(max_steps, int) or max_steps <= 0:
                raise ConfigValidationError(
                    f"sandbox_config.max_steps must be a positive integer, got: {max_steps}"
                )
