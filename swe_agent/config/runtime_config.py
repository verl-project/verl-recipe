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
SWE-Agent Runtime Configuration & CLI YAML Builder.

Provides:
  - ``SWEAgentRuntimeConfig`` — dataclass holding all parameters for one
    SWE-Agent instance inside the VERL agent loop.
  - ``build_runtime_config`` — factory that builds the config from YAML kwargs.
  - ``apply_data_overrides`` — per-instance data-affine overrides.
  - ``SWEAgentYAMLBuilder`` — generates the YAML consumed by ``sweagent run``.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_HISTORY_PROCESSORS: list[dict[str, Any]] = [{"type": "cache_control", "last_n_messages": 2}]

_DEFAULT_TOOL_BUNDLES: list[dict[str, str]] = [
    {"path": "tools/registry"},
    {"path": "tools/edit_anthropic"},
    {"path": "tools/review_on_submit_m"},
    {"path": "tools/diff_state"},
]

_DEFAULT_ENV_VARIABLES: dict[str, str] = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
    "GIT_PAGER": "cat",
}

_DEFAULT_REGISTRY_VARIABLES: dict[str, str] = {
    "USE_FILEMAP": "true",
}

_DEFAULT_TEMPLATES: dict[str, Any] = {
    "system_template": (
        "You are a helpful assistant that can interact with a computer to solve tasks.\n\n"
        "IMPORTANT: Every response MUST follow this exact format:\n\n"
        "DISCUSSION\nYour reasoning about what to do next.\n\n"
        "```\nexactly_one_command_here\n```\n\n"
        "Rules:\n"
        "- Include EXACTLY ONE code block (``` ```) per response\n"
        "- The code block must be the LAST thing in your response\n"
        "- The code block contains the bash command or tool command to execute\n"
        "- Do NOT put example outputs or other code blocks in your response\n"
        "- When you are done, run the `submit` command to submit your changes"
    ),
    "instance_template": (
        "<uploaded_files>\n{{working_dir}}\n</uploaded_files>\n"
        "I've uploaded a python code repository in the directory {{working_dir}}.\n\n"
        "<pr_description>\n{{problem_statement}}\n</pr_description>\n\n"
        "Implement the necessary changes to satisfy the <pr_description>.\n"
        "Do NOT modify any test files.\n\n"
        "Steps:\n"
        "1. Explore the repo with `ls` and `cat` to understand the code\n"
        "2. Make the required changes using `str_replace_editor` or bash commands\n"
        "   - `str_replace_editor` requires positional args: <command> <path> (no --path flag)\n"
        '   - Example: str_replace_editor str_replace /testbed/file.py --old_str "<exact old>" --new_str "<new>"\n'
        "   - Quote arguments carefully when strings contain spaces or newlines\n"
        "3. Verify your changes work\n"
        "4. Run `submit` to submit your patch\n\n"
        "You MUST run `submit` when you are done to generate the final patch."
    ),
    "next_step_template": "OBSERVATION:\n{{observation}}",
    "next_step_no_output_template": "Your command ran successfully and did not produce any output.",
    "max_observation_length": 85000,
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ConfigValidationError(ValueError):
    """Exception raised when runtime config validation fails."""


def _validate_proxy_config(proxy_config: dict[str, Any]) -> None:
    if not isinstance(proxy_config, dict):
        raise ConfigValidationError("proxy_config must be a dict")
    proxy_port = proxy_config.get("port", 0)
    if not isinstance(proxy_port, int) or proxy_port < 0 or proxy_port > 65535:
        raise ConfigValidationError(f"proxy_config.port must be a valid port number (0-65535), got: {proxy_port}")


def _validate_sandbox_config(sandbox_config: dict[str, Any]) -> None:
    if not isinstance(sandbox_config, dict):
        raise ConfigValidationError("sandbox_config must be a dict")
    swe_agent_timeout = sandbox_config.get("swe_agent_timeout")
    if swe_agent_timeout is not None and (not isinstance(swe_agent_timeout, (int, float)) or swe_agent_timeout <= 0):
        raise ConfigValidationError(
            f"sandbox_config.swe_agent_timeout must be a positive number, got: {swe_agent_timeout}"
        )
    max_steps = sandbox_config.get("max_steps")
    if max_steps is not None and (not isinstance(max_steps, int) or max_steps <= 0):
        raise ConfigValidationError(f"sandbox_config.max_steps must be a positive integer, got: {max_steps}")
    max_parallel_tasks = sandbox_config.get("max_parallel_tasks_per_worker")
    if max_parallel_tasks is not None and (not isinstance(max_parallel_tasks, int) or max_parallel_tasks < 0):
        raise ConfigValidationError(
            f"sandbox_config.max_parallel_tasks_per_worker must be a non-negative integer, got: {max_parallel_tasks}"
        )


# ---------------------------------------------------------------------------
# Type coercion helpers
# ---------------------------------------------------------------------------

_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off", ""}


def _to_bool(value: Any, default: bool) -> bool:
    """Convert common bool-like values safely."""
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in _TRUE_VALUES:
            return True
        if normalized in _FALSE_VALUES:
            return False
    logger.warning(f"Invalid bool value {value!r}, fallback to default={default}")
    return default


def _coerce_like(value: Any, original_value: Any) -> Any:
    """Coerce *value* to the type of *original_value* when possible."""
    if isinstance(original_value, bool):
        return _to_bool(value, original_value)
    target_type = type(original_value)
    try:
        return target_type(value)
    except (TypeError, ValueError):
        logger.warning(
            f"Failed to cast override value {value!r} to {target_type.__name__}; "
            f"keeping original value {original_value!r}"
        )
        return original_value


def _to_native_history_processors(value: Any) -> Optional[list[dict[str, Any]]]:
    """Convert history_processors config to a native ``list[dict]`` if valid."""
    if value is None:
        return None
    items = None
    if isinstance(value, list):
        items = value
    elif hasattr(value, "__iter__") and not isinstance(value, (str, bytes, dict)):
        items = list(value)
    if items is not None:
        normalized: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, dict):
                normalized.append(dict(item))
            elif hasattr(item, "items"):
                normalized.append(dict(item))
            else:
                logger.warning(f"Ignoring invalid history processor item: {item!r}")
        return normalized
    logger.warning(f"Invalid history_processors value: {value!r}; using default")
    return None


def _to_native(obj: Any) -> Any:
    """Recursively convert OmegaConf objects to native Python types for YAML serialization."""
    try:
        from omegaconf import DictConfig, ListConfig, OmegaConf

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
    except ImportError:
        pass
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(item) for item in obj]
    return obj


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class SWEAgentRuntimeConfig:
    """Structured runtime configuration for one SWE-Agent execution.

    Infrastructure (fixed per deployment):
      proxy_port, max_port_retries, proxy_timeout, swe_agent_timeout,
      execution_timeout, max_parallel_tasks_per_worker, output_dir,
      docker_memory_limit, docker_startup_timeout, docker_remove_container

    Data-affine (may vary per task/dataset instance):
      max_steps, max_turns, docker_image, templates, tool_bundles,
      parse_function_type
    """

    # --- Proxy ---
    proxy_port: int = 0
    max_port_retries: int = 1000
    proxy_timeout: int = 600

    # --- Sandbox (infrastructure) ---
    swe_agent_timeout: int = 1800
    execution_timeout: int = 300
    max_parallel_tasks_per_worker: int = 0
    output_dir: str = ""
    docker_memory_limit: str = "8g"
    docker_startup_timeout: float = 180.0
    docker_remove_container: bool = True

    # --- Sandbox (data-affine) ---
    max_steps: int = 30
    max_turns: int = 15
    docker_image: str = "swerex-python:3.11-preinstalled"

    # --- Agent (data-affine) ---
    templates: dict[str, Any] = field(default_factory=dict)
    tool_bundles: Optional[list[dict[str, str]]] = None
    tool_env_variables: dict[str, str] = field(default_factory=dict)
    tool_registry_variables: dict[str, str] = field(default_factory=dict)
    parse_function_type: str = "thought_action"
    enable_bash_tool: bool = True
    max_requeries: int = 5
    history_processors: list[dict[str, Any]] = field(default_factory=lambda: deepcopy(_DEFAULT_HISTORY_PROCESSORS))


# ---------------------------------------------------------------------------
# Factory: build from YAML kwargs
# ---------------------------------------------------------------------------


def build_runtime_config(yaml_kwargs: dict[str, Any]) -> SWEAgentRuntimeConfig:
    """Build a ``SWEAgentRuntimeConfig`` baseline from YAML kwargs.

    Args:
        yaml_kwargs: Keyword arguments from the YAML config file, passed via
                     ``hydra.utils.instantiate``.  Contains keys like
                     ``proxy_config``, ``sandbox_config``, ``agent``.

    Returns:
        Built ``SWEAgentRuntimeConfig`` baseline.
    """
    yaml_proxy = dict(yaml_kwargs.get("proxy_config", {}) or {})
    yaml_sandbox = dict(yaml_kwargs.get("sandbox_config", {}) or {})
    yaml_agent = dict(yaml_kwargs.get("agent", {}) or {})
    yaml_templates = dict(yaml_agent.get("templates", {}) or {})
    yaml_tools = dict(yaml_agent.get("tools", {}) or {})
    yaml_history_processors = yaml_agent.get("history_processors", None)

    templates = {**deepcopy(_DEFAULT_TEMPLATES), **yaml_templates}
    history_processors = _to_native_history_processors(yaml_history_processors)

    # Validate before type coercion.
    _validate_proxy_config(yaml_proxy)
    _validate_sandbox_config(yaml_sandbox)

    parse_fn = yaml_tools.get("parse_function", {}) or {}

    # Resolve output_dir
    raw_output_dir = yaml_sandbox.get("output_dir", "") or os.path.join(os.getcwd(), "swe_agent_outputs")
    output_dir = os.path.abspath(os.path.expanduser(str(raw_output_dir)))

    cfg = SWEAgentRuntimeConfig(
        # Proxy
        proxy_port=int(yaml_proxy.get("port", 0)),
        max_port_retries=int(yaml_proxy.get("max_port_retries", 1000)),
        proxy_timeout=int(yaml_proxy.get("timeout", 600)),
        # Sandbox — infrastructure
        swe_agent_timeout=int(yaml_sandbox.get("swe_agent_timeout", 1800)),
        execution_timeout=int(yaml_sandbox.get("execution_timeout", 300)),
        max_parallel_tasks_per_worker=int(yaml_sandbox.get("max_parallel_tasks_per_worker", 0)),
        output_dir=output_dir,
        docker_memory_limit=str(yaml_sandbox.get("docker_memory_limit", "8g")),
        docker_startup_timeout=float(yaml_sandbox.get("docker_startup_timeout", 180.0)),
        docker_remove_container=_to_bool(yaml_sandbox.get("docker_remove_container", True), True),
        # Sandbox — data-affine
        max_steps=int(yaml_sandbox.get("max_steps", 30)),
        max_turns=int(yaml_sandbox.get("max_turns", 15)),
        docker_image=str(yaml_sandbox.get("docker_image", "swerex-python:3.11-preinstalled")),
        # Agent — data-affine
        templates=dict(templates),
        tool_bundles=yaml_tools.get("bundles", deepcopy(_DEFAULT_TOOL_BUNDLES)),
        tool_env_variables={**_DEFAULT_ENV_VARIABLES, **dict(yaml_tools.get("env_variables", {}) or {})},
        tool_registry_variables={**_DEFAULT_REGISTRY_VARIABLES, **dict(yaml_tools.get("registry_variables", {}) or {})},
        parse_function_type=str(parse_fn.get("type", "thought_action")),
        enable_bash_tool=_to_bool(yaml_tools.get("enable_bash_tool", True), True),
        max_requeries=int(yaml_agent.get("max_requeries", 5) or 5),
        history_processors=history_processors
        if history_processors is not None
        else deepcopy(_DEFAULT_HISTORY_PROCESSORS),
    )

    os.makedirs(cfg.output_dir, exist_ok=True)
    logger.info(
        f"SWEAgentRuntimeConfig built: "
        f"max_turns={cfg.max_turns}, max_steps={cfg.max_steps}, "
        f"output_dir={cfg.output_dir}"
    )
    return cfg


# ---------------------------------------------------------------------------
# Per-instance data-affine overrides
# ---------------------------------------------------------------------------


def apply_data_overrides(
    base: SWEAgentRuntimeConfig,
    extra_info: dict[str, Any],
) -> SWEAgentRuntimeConfig:
    """Create a per-instance copy of *base* with data-affine overrides applied.

    Reads ``extra_info["sandbox_overrides"]`` and ``extra_info["agent_overrides"]``
    and applies them on top of *base*.  Returns a new object — *base* is not mutated.
    """
    sandbox_overrides = extra_info.get("sandbox_overrides", {})
    agent_overrides = extra_info.get("agent_overrides", {})

    # Defensive: parquet deserialization may produce JSON strings
    if isinstance(sandbox_overrides, str):
        try:
            sandbox_overrides = json.loads(sandbox_overrides)
        except (json.JSONDecodeError, TypeError):
            sandbox_overrides = {}
    if isinstance(agent_overrides, str):
        try:
            agent_overrides = json.loads(agent_overrides)
        except (json.JSONDecodeError, TypeError):
            agent_overrides = {}

    if not sandbox_overrides and not agent_overrides:
        return base

    cfg = deepcopy(base)

    _SANDBOX_FIELDS = {
        "max_steps",
        "max_turns",
        "docker_image",
        "swe_agent_timeout",
        "execution_timeout",
        "max_parallel_tasks_per_worker",
        "docker_memory_limit",
        "docker_startup_timeout",
        "docker_remove_container",
    }
    for k, v in sandbox_overrides.items():
        if k in _SANDBOX_FIELDS and hasattr(cfg, k):
            old = getattr(cfg, k)
            setattr(cfg, k, _coerce_like(v, old) if old is not None else v)
            logger.debug(f"Data override: {k}={v} (was {old})")

    if agent_overrides.get("templates"):
        cfg.templates = {**cfg.templates, **agent_overrides["templates"]}
    if agent_overrides.get("tool_bundles"):
        cfg.tool_bundles = agent_overrides["tool_bundles"]
    if agent_overrides.get("parse_function_type"):
        cfg.parse_function_type = agent_overrides["parse_function_type"]
    if "enable_bash_tool" in agent_overrides:
        cfg.enable_bash_tool = _to_bool(agent_overrides["enable_bash_tool"], cfg.enable_bash_tool)
    if "history_processors" in agent_overrides:
        parsed = _to_native_history_processors(agent_overrides["history_processors"])
        if parsed is not None:
            cfg.history_processors = parsed

    return cfg


# ---------------------------------------------------------------------------
# SWE-Agent CLI YAML Builder
# ---------------------------------------------------------------------------


class SWEAgentYAMLBuilder:
    """Generates the YAML config consumed by ``sweagent run --config``.

    Use ``from_config()`` to construct from a ``SWEAgentRuntimeConfig``::

        builder = SWEAgentYAMLBuilder.from_config(
            cfg, instance_id="test-123", repo_path="/workspace/repo",
            output_dir="/tmp/out", model_proxy_port=8080,
        )
        yaml_str = builder.to_yaml()
    """

    def __init__(
        self,
        instance_id: str,
        repo_path: str,
        output_dir: str,
        model_proxy_port: int,
        cfg: SWEAgentRuntimeConfig,
        max_input_tokens: int = 0,
        repo_type: str = "local",
        repo_base_commit: str = "HEAD",
        preexisting_repo_reset: bool = True,
    ):
        self._id = instance_id
        self._repo = repo_path
        self._out = output_dir
        self._port = model_proxy_port
        self._cfg = cfg
        self._max_input_tokens = max_input_tokens
        self._repo_type = repo_type
        self._repo_base_commit = repo_base_commit
        self._preexisting_repo_reset = preexisting_repo_reset

    @classmethod
    def from_config(
        cls,
        cfg: SWEAgentRuntimeConfig,
        *,
        instance_id: str,
        repo_path: str,
        output_dir: str,
        model_proxy_port: int,
        max_input_tokens: int = 0,
        repo_type: str = "local",
        repo_base_commit: str = "HEAD",
        preexisting_repo_reset: bool = True,
    ) -> SWEAgentYAMLBuilder:
        """Construct a builder from a runtime config."""
        return cls(
            instance_id=instance_id,
            repo_path=repo_path,
            output_dir=output_dir,
            model_proxy_port=model_proxy_port,
            cfg=cfg,
            max_input_tokens=max_input_tokens,
            repo_type=repo_type,
            repo_base_commit=repo_base_commit,
            preexisting_repo_reset=preexisting_repo_reset,
        )

    def build(self) -> dict[str, Any]:
        """Build the SWE-Agent configuration dictionary."""
        cfg = self._cfg
        tpl = cfg.templates

        templates = {
            "system_template": tpl.get("system_template", _DEFAULT_TEMPLATES["system_template"]),
            "instance_template": tpl.get("instance_template", _DEFAULT_TEMPLATES["instance_template"]),
            "next_step_template": tpl.get("next_step_template", _DEFAULT_TEMPLATES["next_step_template"]),
            "next_step_no_output_template": tpl.get(
                "next_step_no_output_template", _DEFAULT_TEMPLATES["next_step_no_output_template"]
            ),
            "max_observation_length": tpl.get("max_observation_length", _DEFAULT_TEMPLATES["max_observation_length"]),
        }

        env_variables = {**_DEFAULT_ENV_VARIABLES, **cfg.tool_env_variables}
        registry_variables = {**_DEFAULT_REGISTRY_VARIABLES, **cfg.tool_registry_variables}
        tool_bundles = cfg.tool_bundles if cfg.tool_bundles is not None else _DEFAULT_TOOL_BUNDLES

        if self._repo_type == "preexisting":
            repo_config: dict[str, Any] = {
                "type": "preexisting",
                "repo_name": self._repo,
                "base_commit": self._repo_base_commit,
                "reset": self._preexisting_repo_reset,
            }
        elif self._repo_type == "github":
            github_url = self._repo
            if github_url and not github_url.startswith("http://") and not github_url.startswith("https://"):
                github_url = f"https://github.com/{github_url}"
            repo_config = {
                "type": "github",
                "github_url": github_url,
                "base_commit": self._repo_base_commit,
            }
        else:
            repo_config = {
                "path": self._repo,
                "type": "local",
                "base_commit": self._repo_base_commit,
            }

        config = {
            "output_dir": self._out,
            "env": {
                "repo": repo_config,
                "deployment": {
                    "type": "docker",
                    "image": cfg.docker_image,
                    "docker_args": [
                        f"--memory={cfg.docker_memory_limit}",
                        "--add-host",
                        "host.docker.internal:host-gateway",
                        "--label",
                        f"verl.instance_id={self._id}",
                    ],
                    "startup_timeout": cfg.docker_startup_timeout,
                    "remove_container": cfg.docker_remove_container,
                },
                "name": f"verl-swe-{self._id}",
            },
            "agent": {
                "templates": templates,
                "tools": {
                    "execution_timeout": cfg.execution_timeout,
                    "env_variables": env_variables,
                    "bundles": tool_bundles,
                    "registry_variables": registry_variables,
                    "enable_bash_tool": cfg.enable_bash_tool,
                    "parse_function": {"type": cfg.parse_function_type},
                },
                "max_requeries": cfg.max_requeries,
                "history_processors": cfg.history_processors or _DEFAULT_HISTORY_PROCESSORS,
                "model": {
                    "name": "openai/verl-model",
                    "per_instance_cost_limit": 0,
                    "per_instance_call_limit": cfg.max_steps,
                    "total_cost_limit": 0,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_input_tokens": self._max_input_tokens,
                    "api_base": f"http://127.0.0.1:{self._port}/v1",
                    "api_key": "verl-swe-agent-key",
                },
            },
        }

        return _to_native(config)

    def to_yaml(self) -> str:
        """Serialize to YAML string."""
        return yaml.dump(self.build(), default_flow_style=False, allow_unicode=True)
