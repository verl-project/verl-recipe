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

Nested OmegaConf structured configs mirror the YAML layout so that
``build_runtime_config`` is just ``OmegaConf.merge(schema, yaml)``.
All defaults live in the dataclasses — no module-level constant dicts.
"""

from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Nested structured dataclasses — mirror the YAML layout
# ---------------------------------------------------------------------------


@dataclass
class ProxyConfig:
    port: int = 0
    max_port_retries: int = 1000
    timeout: int = 600


@dataclass
class SandboxConfig:
    docker_mode: str = "host"
    swe_agent_timeout: int = 1800
    execution_timeout: int = 300
    max_parallel_tasks_per_worker: int = 0
    output_dir: str = ""
    docker_memory_limit: str = "8g"
    docker_startup_timeout: float = 180.0
    docker_remove_container: bool = True
    max_model_calls_per_instance: int = 15
    docker_image: str = "swerex-python:3.11-preinstalled"


@dataclass
class ToolsConfig:
    execution_timeout: int = 300
    env_variables: dict[str, str] = field(
        default_factory=lambda: {
            "PAGER": "cat",
            "MANPAGER": "cat",
            "LESS": "-R",
            "PIP_PROGRESS_BAR": "off",
            "TQDM_DISABLE": "1",
            "GIT_PAGER": "cat",
        }
    )
    bundles: list[dict[str, str]] = field(
        default_factory=lambda: [
            {"path": "tools/registry"},
            {"path": "tools/edit_anthropic"},
            {"path": "tools/review_on_submit_m"},
            {"path": "tools/diff_state"},
        ]
    )
    registry_variables: dict[str, str] = field(default_factory=lambda: {"USE_FILEMAP": "true"})
    enable_bash_tool: bool = True
    parse_function: dict[str, str] = field(default_factory=lambda: {"type": "thought_action"})


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


@dataclass
class AgentConfig:
    max_requeries: int = 5
    templates: dict[str, Any] = field(default_factory=lambda: deepcopy(_DEFAULT_TEMPLATES))
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    history_processors: list[dict[str, Any]] = field(
        default_factory=lambda: [{"type": "cache_control", "last_n_messages": 2}]
    )


@dataclass
class SWEAgentRuntimeConfig:
    """Top-level config. Nesting matches the YAML structure."""

    proxy_config: ProxyConfig = field(default_factory=ProxyConfig)
    sandbox_config: SandboxConfig = field(default_factory=SandboxConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dict(val: Any) -> dict:
    """Coerce *val* to a plain ``dict``, handling JSON strings and OmegaConf."""
    if isinstance(val, str):
        try:
            return json.loads(val)
        except (json.JSONDecodeError, TypeError):
            return {}
    if isinstance(val, DictConfig):
        result = OmegaConf.to_container(val, resolve=True)
        return result if isinstance(result, dict) else {}
    if isinstance(val, dict):
        return val
    return {}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_runtime_config(yaml_kwargs: dict[str, Any]) -> SWEAgentRuntimeConfig:
    """Build config by merging YAML kwargs onto the structured schema."""
    raw = OmegaConf.to_container(OmegaConf.create(yaml_kwargs), resolve=True)
    if not isinstance(raw, dict):
        raw = {}
    raw.pop("name", None)
    raw.pop("_target_", None)

    # Backward compat: max_steps -> max_model_calls_per_instance
    sandbox_cfg = raw.get("sandbox_config", {})
    if (
        isinstance(sandbox_cfg, dict)
        and "max_steps" in sandbox_cfg
        and "max_model_calls_per_instance" not in sandbox_cfg
    ):
        sandbox_cfg["max_model_calls_per_instance"] = sandbox_cfg["max_steps"]

    schema = OmegaConf.structured(SWEAgentRuntimeConfig)
    merged = OmegaConf.merge(schema, OmegaConf.create(raw))
    cfg: SWEAgentRuntimeConfig = OmegaConf.to_object(merged)  # type: ignore[assignment]

    out = cfg.sandbox_config.output_dir or os.path.join(os.getcwd(), "swe_agent_outputs")
    cfg.sandbox_config.output_dir = os.path.abspath(os.path.expanduser(out))
    os.makedirs(cfg.sandbox_config.output_dir, exist_ok=True)
    return cfg


# ---------------------------------------------------------------------------
# Per-instance overrides
# ---------------------------------------------------------------------------

_SANDBOX_FIELDS = frozenset(SandboxConfig.__dataclass_fields__)
_AGENT_OVERRIDE_FIELDS = frozenset(("templates", "tools", "history_processors"))


def apply_data_overrides(
    base: SWEAgentRuntimeConfig,
    extra_info: dict[str, Any],
) -> SWEAgentRuntimeConfig:
    """Per-instance copy of *base* with data-affine overrides. *base* is not mutated."""
    sandbox_ov = _ensure_dict(extra_info.get("sandbox_overrides", {}))
    if "max_steps" in sandbox_ov and "max_model_calls_per_instance" not in sandbox_ov:
        sandbox_ov["max_model_calls_per_instance"] = sandbox_ov["max_steps"]
    agent_ov = _ensure_dict(extra_info.get("agent_overrides", {}))
    if not sandbox_ov and not agent_ov:
        return base

    patch: dict[str, Any] = {}
    if sandbox_ov:
        sandbox_patch = {k: v for k, v in sandbox_ov.items() if k in _SANDBOX_FIELDS}
        if sandbox_patch:
            patch["sandbox_config"] = sandbox_patch
    if agent_ov:
        agent_patch = {k: v for k, v in agent_ov.items() if k in _AGENT_OVERRIDE_FIELDS}
        if "history_processors" in agent_patch and not isinstance(agent_patch["history_processors"], list):
            agent_patch.pop("history_processors", None)
        if agent_patch:
            patch["agent"] = agent_patch

    if not patch:
        return base

    base_cfg = OmegaConf.structured(base)
    return OmegaConf.to_object(OmegaConf.merge(base_cfg, OmegaConf.create(patch)))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# SWE-Agent CLI YAML Builder
# ---------------------------------------------------------------------------


def build_sweagent_yaml(
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
) -> str:
    """Generate the YAML string consumed by ``sweagent run --config``."""
    sb = cfg.sandbox_config
    ag = cfg.agent
    tools = ag.tools

    docker_args = [
        f"--memory={sb.docker_memory_limit}",
        "--add-host",
        "host.docker.internal:host-gateway",
        "--label",
        f"verl.instance_id={instance_id}",
    ]
    if sb.docker_mode == "host":
        docker_args.extend(["--network", "host"])

    if repo_type == "preexisting":
        repo_config = {
            "type": "preexisting",
            "repo_name": repo_path,
            "base_commit": repo_base_commit,
            "reset": preexisting_repo_reset,
        }
    else:
        repo_config = {"path": repo_path, "type": "local", "base_commit": repo_base_commit}

    pf = tools.parse_function

    config = {
        "output_dir": output_dir,
        "env": {
            "repo": repo_config,
            "deployment": {
                "type": "docker",
                "image": sb.docker_image,
                "docker_args": docker_args,
                "startup_timeout": sb.docker_startup_timeout,
                "remove_container": sb.docker_remove_container,
            },
            "name": f"verl-swe-{instance_id}",
        },
        "agent": {
            "templates": ag.templates,
            "tools": {
                "execution_timeout": tools.execution_timeout,
                "env_variables": tools.env_variables,
                "bundles": tools.bundles,
                "registry_variables": tools.registry_variables,
                "enable_bash_tool": tools.enable_bash_tool,
                "parse_function": pf if isinstance(pf, dict) else {"type": "thought_action"},
            },
            "max_requeries": ag.max_requeries,
            "history_processors": ag.history_processors,
            "model": {
                "name": "openai/verl-model",
                "per_instance_cost_limit": 0,
                "per_instance_call_limit": sb.max_model_calls_per_instance,
                "total_cost_limit": 0,
                "temperature": 0.0,
                "top_p": 1.0,
                "max_input_tokens": max_input_tokens,
                "api_base": f"http://127.0.0.1:{model_proxy_port}/v1",
                "api_key": "verl-swe-agent-key",
            },
        },
    }

    def _native(obj: Any) -> Any:
        if isinstance(obj, DictConfig):
            return OmegaConf.to_container(obj, resolve=True)
        return obj

    config["agent"]["templates"] = _native(config["agent"]["templates"])
    config["agent"]["tools"]["env_variables"] = _native(config["agent"]["tools"]["env_variables"])
    config["agent"]["tools"]["bundles"] = _native(config["agent"]["tools"]["bundles"])
    config["agent"]["tools"]["registry_variables"] = _native(config["agent"]["tools"]["registry_variables"])
    config["agent"]["history_processors"] = _native(config["agent"]["history_processors"])

    return yaml.dump(config, default_flow_style=False, allow_unicode=True)
