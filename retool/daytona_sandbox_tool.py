"""Daytona-backed ReTool code interpreter."""

import asyncio
import logging
import os
import re
from typing import Any
from uuid import uuid4

from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
)
from verl.utils.rollout_trace import rollout_trace_op

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

code_pattern = re.compile(r"```python(.*?)```", re.DOTALL)


def _build_code_interpreter_schema() -> OpenAIFunctionToolSchema:
    """Return the OpenAI tool schema exposed to the model."""
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="code_interpreter",
            description="A tool for executing Python code in a Daytona sandbox.",
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "code": OpenAIFunctionPropertySchema(
                        type="string",
                        description="The Python code to execute.",
                    )
                },
                required=["code"],
            ),
        ),
    )


def _load_async_daytona_sdk():
    """Import the async Daytona SDK lazily so the backend stays optional."""
    try:
        from daytona import AsyncDaytona, CreateSandboxFromSnapshotParams, DaytonaConfig
    except ImportError as exc:
        raise ImportError(
            "CustomDaytonaSandboxTool requires the optional 'daytona' dependency. "
            "Install it with `pip install daytona`."
        ) from exc

    return AsyncDaytona, DaytonaConfig, CreateSandboxFromSnapshotParams


def _resolve_daytona_auth(config: dict[str, Any]) -> dict[str, str]:
    """Resolve Daytona auth from config or environment."""
    api_key = config.get("api_key") or os.getenv("DAYTONA_API_KEY")
    jwt_token = config.get("jwt_token") or os.getenv("DAYTONA_JWT_TOKEN")

    if not api_key and not jwt_token:
        raise ValueError("CustomDaytonaSandboxTool requires DAYTONA_API_KEY or DAYTONA_JWT_TOKEN")

    auth = {}
    if api_key:
        auth["api_key"] = api_key
    if jwt_token:
        auth["jwt_token"] = jwt_token
    return auth


def _normalize_code(code: str) -> str:
    """Mirror the existing ReTool Sandbox Fusion preprocessing."""
    matches = code_pattern.findall(code)
    if matches:
        code = matches[0].strip()

    # Some scripts omit an explicit print, so append one to the final line.
    lines = code.split("\n")
    for i, line in reversed(list(enumerate(lines))):
        if line == "":
            continue
        if not lines[i].startswith("print"):
            lines[i] = f"print({line})"
        break
    return "\n".join(lines)


class CustomDaytonaSandboxTool(BaseTool):
    """Execute Python code inside one Daytona sandbox per rollout trajectory."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None):
        tool_schema = tool_schema or _build_code_interpreter_schema()
        super().__init__(config, tool_schema)

        AsyncDaytona, DaytonaConfig, CreateSandboxFromSnapshotParams = _load_async_daytona_sdk()
        auth_config = _resolve_daytona_auth(config)
        self._create_sandbox_params_cls = CreateSandboxFromSnapshotParams

        resolved_config = {**config, **auth_config}
        client_config = {}
        for key in ("api_key", "api_url", "target", "jwt_token", "organization_id"):
            value = resolved_config.get(key)
            if value is not None:
                client_config[key] = value

        self._daytona = AsyncDaytona(DaytonaConfig(**client_config)) if client_config else AsyncDaytona()
        self._sandboxes: dict[str, Any] = {}
        self._instance_dict: dict[str, dict] = {}

        self.rate_limit = config.get("rate_limit", 32)
        self.default_timeout = config.get("default_timeout", 30)
        self._create_timeout = config.get("create_timeout", 60)
        self._delete_timeout = config.get("delete_timeout", 60)
        self._auto_stop_interval = config.get("auto_stop_interval", 15)
        self._auto_delete_interval = config.get("auto_delete_interval", 30)
        self._name_prefix = config.get("name_prefix", "verl-daytona")
        self._base_labels = dict(config.get("labels") or {})
        self._snapshot = config.get("snapshot")
        self._language = config.get("language", "python")
        self._env_vars = dict(config.get("env_vars") or {})
        self._semaphore = asyncio.Semaphore(self.rate_limit) if config.get("enable_global_rate_limit", True) else None

        if self._language != "python":
            raise ValueError("CustomDaytonaSandboxTool currently supports only language='python'")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        """Return the OpenAI tool schema."""
        return self.tool_schema

    async def _rate_limited(self, coro):
        """Await a coroutine under the optional rate limiter."""
        if self._semaphore is None:
            return await coro
        async with self._semaphore:
            return await coro

    async def create(self, instance_id: str | None = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a Daytona sandbox for one rollout trajectory."""
        if instance_id is None:
            instance_id = str(uuid4())

        labels = {
            **self._base_labels,
            "framework": "verl",
            "backend": "daytona",
            "tool": self.name,
            "instance_id": instance_id,
        }
        params = self._create_sandbox_params_cls(
            name=f"{self._name_prefix}-{instance_id[:8]}",
            language=self._language,
            snapshot=self._snapshot,
            env_vars=self._env_vars or None,
            labels=labels,
            auto_stop_interval=self._auto_stop_interval,
            auto_delete_interval=self._auto_delete_interval,
        )

        sandbox = await self._rate_limited(self._daytona.create(params, timeout=self._create_timeout))
        self._sandboxes[instance_id] = sandbox
        self._instance_dict[instance_id] = {"reward": []}
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        """Execute normalized Python code in the existing Daytona sandbox."""
        code = parameters.get("code", "")
        timeout = parameters.get("timeout", self.default_timeout)
        envs = parameters.get("envs")
        language = parameters.get("language")

        if language is not None and language != "python":
            raise ValueError("CustomDaytonaSandboxTool only supports Python code execution")

        if not isinstance(code, str):
            code = str(code)
        code = _normalize_code(code)

        if envs is not None and not isinstance(envs, dict):
            raise ValueError("envs must be a dictionary of string environment variables")

        sandbox = self._sandboxes[instance_id]
        result = await self._rate_limited(sandbox.code_interpreter.run_code(code, timeout=timeout, envs=envs))

        output = result.stdout + result.stderr
        if result.error is not None:
            error_text = f"{result.error.name}: {result.error.value}"
            if result.error.traceback:
                error_text = f"{error_text}\n{result.error.traceback}"
            output = f"{output.rstrip()}\n{error_text}".strip()

        self._instance_dict[instance_id]["reward"].append(output)
        metrics = {
            "sandbox_id": sandbox.id,
            "stdout_chars": len(result.stdout),
            "stderr_chars": len(result.stderr),
            "had_error": result.error is not None,
            "error_name": None if result.error is None else result.error.name,
        }
        return ToolResponse(text=output), 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs) -> list[str]:
        """Return the collected tool outputs for the trajectory."""
        return self._instance_dict[instance_id]["reward"]

    async def release(self, instance_id: str, **kwargs) -> None:
        """Delete the Daytona sandbox and clean up local state."""
        sandbox = self._sandboxes.pop(instance_id)
        await self._rate_limited(sandbox.delete(timeout=self._delete_timeout))
        del self._instance_dict[instance_id]

    async def close(self) -> None:
        """Close the underlying Daytona client."""
        await self._daytona.close()
