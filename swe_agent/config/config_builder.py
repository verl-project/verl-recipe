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
SWE Agent Configuration Builder.

This module provides a reusable builder class for generating SWE-Agent YAML configurations.
It eliminates duplication between subprocess and Docker execution modes.
"""

import json
from typing import Any, Dict, Optional
import yaml


def _to_native(obj):
    """Recursively convert OmegaConf/DictConfig/ListConfig objects to native Python types.
    
    This prevents YAML serialization from producing Python-specific tags like
    !!python/object:omegaconf.listconfig.ListConfig that SWE-Agent's parser cannot handle.
    """
    # Try OmegaConf conversion first (if available)
    try:
        from omegaconf import OmegaConf, DictConfig, ListConfig
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    
    # Fallback: recursively convert dicts and lists
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(item) for item in obj]
    return obj


class SWEAgentConfigBuilder:
    """Builder for SWE-Agent configuration files.

    This builder encapsulates the logic for generating SWE-Agent YAML configurations,
    providing a clean, testable interface that eliminates duplication across execution modes.

    Example:
        >>> builder = SWEAgentConfigBuilder(
        ...     instance_id="test-123",
        ...     repo_path="/workspace/repo",
        ...     output_dir="/tmp/output",
        ...     model_proxy_port=8080,
        ...     max_steps=50,
        ...     execution_timeout=300,
        ... )
        >>> config = builder.build()
        >>> yaml_str = builder.to_yaml()
    """

    # Default templates - optimized for small models with explicit output format and submission flow
    # Key design:
    #   1. system_template specifies thought_action format (single code block only)
    #   2. instance_template emphasizes final submission via submit command
    DEFAULT_SYSTEM_TEMPLATE = """You are a helpful assistant that can interact with a computer to solve tasks.

IMPORTANT: Every response MUST follow this exact format:

DISCUSSION
Your reasoning about what to do next.

```
exactly_one_command_here
```

Rules:
- Include EXACTLY ONE code block (``` ```) per response
- The code block must be the LAST thing in your response
- The code block contains the bash command or tool command to execute
- Do NOT put example outputs or other code blocks in your response
- When you are done, run the `submit` command to submit your changes"""

    DEFAULT_INSTANCE_TEMPLATE = """<uploaded_files>
{{working_dir}}
</uploaded_files>
I've uploaded a python code repository in the directory {{working_dir}}.

<pr_description>
{{problem_statement}}
</pr_description>

Implement the necessary changes to satisfy the <pr_description>.
Do NOT modify any test files.

Steps:
1. Explore the repo with `ls` and `cat` to understand the code
2. Make the required changes using `str_replace_editor` or bash commands
3. Verify your changes work
4. Run `submit` to submit your patch

You MUST run `submit` when you are done to generate the final patch."""

    DEFAULT_NEXT_STEP_TEMPLATE = "OBSERVATION:\n{{observation}}"

    DEFAULT_NEXT_STEP_NO_OUTPUT_TEMPLATE = "Your command ran successfully and did not produce any output."

    DEFAULT_TOOL_BUNDLES = [
        {"path": "tools/registry"},
        {"path": "tools/edit_anthropic"},
        {"path": "tools/review_on_submit_m"},
        {"path": "tools/diff_state"},
    ]

    DEFAULT_ENV_VARIABLES = {
        "PAGER": "cat",
        "MANPAGER": "cat",
        "LESS": "-R",
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
        "GIT_PAGER": "cat",
    }

    DEFAULT_REGISTRY_VARIABLES = {
        "USE_FILEMAP": "true",
    }

    def __init__(
        self,
        instance_id: str,
        repo_path: str,
        output_dir: str,
        model_proxy_port: int,
        # All parameters below are required — values come from swe_agent_config.yaml.
        max_steps: int,
        execution_timeout: int,
        custom_templates: Optional[Dict[str, str]] = None,
        custom_env_variables: Optional[Dict[str, str]] = None,
        custom_registry_variables: Optional[Dict[str, str]] = None,
        custom_tool_bundles: Optional[list] = None,
        parse_function_type: str = "thought_action",
        max_requeries: int = 5,
        max_observation_length: int = 85000,
        enable_bash_tool: bool = True,
        max_input_tokens: int = 0,  # 0 = disabled; set to vLLM max_model_len to prevent context overflow
        # Deployment config
        deployment_type: str = "docker",
        # Docker config (only used when deployment_type="docker")
        docker_image: str = "swerex-python:3.11",
        docker_memory_limit: str = "8g",
        docker_startup_timeout: float = 180.0,
        docker_remove_container: bool = True,
    ):
        """Initialize the config builder.

        Args:
            instance_id: Unique instance identifier.
            repo_path: Path to the repository.
            output_dir: Directory for SWE-Agent output.
            model_proxy_port: Port where ModelProxy is listening.
            max_steps: Maximum number of model calls (per_instance_call_limit).
            execution_timeout: Timeout for tool execution in seconds.
            custom_templates: Optional custom templates to override defaults.
            custom_env_variables: Optional custom environment variables.
            custom_registry_variables: Optional custom registry variables.
            custom_tool_bundles: Optional custom tool bundles.
            parse_function_type: Parse function type (e.g. thought_action).
            max_observation_length: Max observation length for templates.
            enable_bash_tool: Whether to enable SWE-Agent bash tool.
            docker_image: Docker image to use for SWE-Agent execution.
            docker_memory_limit: Memory limit for Docker container (e.g., "8g").
            docker_startup_timeout: Timeout for Docker container startup in seconds.
            docker_remove_container: Whether to remove Docker container after execution.
                Set to False to keep containers for debugging.
        """
        self.instance_id = instance_id
        self.repo_path = repo_path
        self.output_dir = output_dir
        self.model_proxy_port = model_proxy_port
        self.max_steps = max_steps
        self.execution_timeout = execution_timeout
        # Convert OmegaConf objects to native Python types upfront
        self.custom_templates = _to_native(custom_templates or {})
        self.custom_env_variables = _to_native(custom_env_variables or {})
        self.custom_registry_variables = _to_native(custom_registry_variables or {})
        self.custom_tool_bundles = _to_native(custom_tool_bundles) if custom_tool_bundles is not None else None
        self.parse_function_type = parse_function_type
        self.max_requeries = max_requeries
        self.max_observation_length = max_observation_length
        self.enable_bash_tool = enable_bash_tool
        self.max_input_tokens = max_input_tokens
        # Deployment config
        self.deployment_type = deployment_type
        # Docker config
        self.docker_image = docker_image
        self.docker_memory_limit = docker_memory_limit
        self.docker_startup_timeout = docker_startup_timeout
        self.docker_remove_container = docker_remove_container

    def build(self) -> Dict[str, Any]:
        """Build the SWE-Agent configuration dictionary.

        Returns:
            Configuration dictionary ready for YAML serialization.
        """
        # Build templates
        templates = {
            "system_template": self.custom_templates.get(
                "system_template", self.DEFAULT_SYSTEM_TEMPLATE
            ),
            "instance_template": self.custom_templates.get(
                "instance_template", self.DEFAULT_INSTANCE_TEMPLATE
            ),
            "next_step_template": self.custom_templates.get(
                "next_step_template", self.DEFAULT_NEXT_STEP_TEMPLATE
            ),
            "next_step_no_output_template": self.custom_templates.get(
                "next_step_no_output_template", self.DEFAULT_NEXT_STEP_NO_OUTPUT_TEMPLATE
            ),
            "max_observation_length": self.max_observation_length,
        }

        # Build environment variables
        env_variables = {**self.DEFAULT_ENV_VARIABLES, **self.custom_env_variables}

        # Build registry variables
        registry_variables = {
            **self.DEFAULT_REGISTRY_VARIABLES,
            **self.custom_registry_variables,
        }

        # Build tool bundles
        tool_bundles = (
            self.custom_tool_bundles
            if self.custom_tool_bundles is not None
            else self.DEFAULT_TOOL_BUNDLES
        )

        # Build deployment configuration
        if self.deployment_type == "local":
            # Local deployment: runs tools directly on host, no Docker overhead
            # Much faster startup but no isolation between parallel instances
            deployment_config = {
                "type": "local",
            }
        else:
            # Docker deployment: provides isolation between parallel SWE-Agent instances
            # This prevents FileExistsError when multiple instances try to install tools
            # Note: sweagent must be executed with cwd set to a directory without 'docker' subdir
            # to avoid YAML parsing issues (see swe_agent_loop.py)
            deployment_config = {
                "type": "docker",
                "image": self.docker_image,
                "docker_args": [
                    f"--memory={self.docker_memory_limit}",
                    # Use host.docker.internal to access host services (like ModelProxy)
                    # This avoids --network=host which causes port conflicts between containers
                    "--add-host", "host.docker.internal:host-gateway",
                ],
                "startup_timeout": self.docker_startup_timeout,
                "remove_container": self.docker_remove_container,
            }

        config = {
            "output_dir": self.output_dir,
            "env": {
                "repo": {
                    "path": self.repo_path,
                    "type": "local",
                },
                "deployment": deployment_config,
                "name": f"verl-swe-{self.instance_id}",  # Unique name per instance
            },
            # Note: problem_statement text will be passed via CLI --problem_statement.text
            # We don't include it in the config file
            "agent": {
                "templates": templates,
                "tools": {
                    "execution_timeout": self.execution_timeout,
                    "env_variables": env_variables,
                    "bundles": tool_bundles,
                    "registry_variables": registry_variables,
                    "enable_bash_tool": self.enable_bash_tool,
                    # Use thought_action for text-based models (Qwen)
                    # function_calling requires OpenAI API tool_calls field which Qwen doesn't support
                    "parse_function": {"type": self.parse_function_type},
                },
                "max_requeries": self.max_requeries,
                "history_processors": [{"type": "cache_control", "last_n_messages": 2}],
                "model": {
                    "name": "openai/verl-model",  # Name doesn't matter, we intercept all calls
                    "per_instance_cost_limit": 0,
                    "per_instance_call_limit": self.max_steps,
                    "total_cost_limit": 0,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    # SWE-Agent checks token count BEFORE sending to LLM.
                    # If exceeded → ContextWindowExceededError → exit_context
                    # (agent auto-submits current patch and exits cleanly).
                    # Set to vLLM max_model_len to prevent vLLM crashes.
                    "max_input_tokens": self.max_input_tokens,
                    # SWE-Agent runs on HOST (not in container), so use 127.0.0.1
                    # The container is only used for tool execution via swerex
                    "api_base": f"http://127.0.0.1:{self.model_proxy_port}/v1",
                    "api_key": "verl-swe-agent-key",  # Dummy key, ModelProxy doesn't validate
                },
            },
        }

        # Convert all OmegaConf objects to native Python types to prevent
        # YAML serialization issues (!!python/object:omegaconf.listconfig.ListConfig)
        return _to_native(config)

    def to_yaml(self) -> str:
        """Convert configuration to YAML string.

        Returns:
            YAML string representation of the configuration.
        """
        config = self.build()
        return yaml.dump(config, default_flow_style=False, allow_unicode=True)

    def to_file(self, file_path: str) -> None:
        """Write configuration to a YAML file.

        Args:
            file_path: Path where the YAML file should be written.
        """
        config = self.build()
        with open(file_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
