from __future__ import annotations

import asyncio
import importlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    ToolListWrap,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__name__)

SUBMIT_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
MAX_TOKENS_PER_GENERATION = 16384
COMMAND_TIMEOUT = 120.0
STARTUP_TIMEOUT = 600.0
AGENT_TIMEOUT = 3600.0
VERIFIER_TIMEOUT = 600.0
SANDBOX_TIMEOUT = STARTUP_TIMEOUT + AGENT_TIMEOUT + VERIFIER_TIMEOUT + 60.0
OBS_MAX_CHARS = 10000
OBS_HEAD_CHARS = 5000
OBS_TAIL_CHARS = 5000
OBS_TOO_LONG_HINT = (
    "The output of your last command was too long.\n"
    "Please try a different command that produces less output.\n"
    "If you're looking at a file you can try use head, tail or sed to view a\n"
    "smaller number of lines selectively. If you're using grep or find and it\n"
    "produced too much output, you can use a more selective search pattern.\n"
    "If you really need to see something from the full command's output, you\n"
    "can redirect output to a file and then search in that file.\n"
)


@dataclass
class TmaxInteraction:
    prompt_ids: list[int]
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    extra_fields: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    num_turns: int = 0

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_ids + self.response_ids

    def append_assistant(self, token_ids: list[int], logprobs: list[float]) -> None:
        self.response_ids.extend(token_ids)
        self.response_mask.extend([1] * len(token_ids))
        self.response_logprobs.extend(logprobs)

    def append_observation(self, token_ids: list[int]) -> None:
        self.response_ids.extend(token_ids)
        self.response_mask.extend([0] * len(token_ids))
        self.response_logprobs.extend([0.0] * len(token_ids))


def truncate_output(output: str) -> str:
    if len(output) <= OBS_MAX_CHARS:
        return output
    elided = len(output) - OBS_HEAD_CHARS - OBS_TAIL_CHARS
    return (
        f"{OBS_TOO_LONG_HINT}\n\n"
        f"---- HEAD ({OBS_HEAD_CHARS} chars) ----\n"
        f"{output[:OBS_HEAD_CHARS]}\n"
        f"---- {elided} chars elided ----\n"
        f"---- TAIL ({OBS_TAIL_CHARS} chars) ----\n"
        f"{output[-OBS_TAIL_CHARS:]}"
    )


def format_error(error: str) -> str:
    return (
        f"Format error: {error}\n\n"
        "Please always provide EXACTLY ONE call to the `bash` tool. If you want to\n"
        "end the task, please issue the command `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`\n"
        "via the `bash` tool, with no other content in the command.\n"
    )


class TmaxAgentLoop(AgentLoopBase):
    def __init__(self, *args, tools: ToolListWrap | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        tool_list = tools.tools if tools else []
        self.tool_schemas = [tool.tool_schema for tool in tool_list]
        self.tool_schema_dicts = [
            schema.model_dump(exclude_unset=True, exclude_none=True) for schema in self.tool_schemas
        ]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.response_length = self.rollout_config.response_length
        self.max_turns = self.rollout_config.multi_turn.max_assistant_turns

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        prompt_ids = self._render_messages(messages, add_generation_prompt=True)
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeding prompt_length={self.rollout_config.prompt_length}"
            )
        interaction = TmaxInteraction(prompt_ids=prompt_ids)
        if "global_steps" in kwargs:
            global_steps = int(kwargs["global_steps"])
            interaction.extra_fields["min_global_steps"] = global_steps
            interaction.extra_fields["max_global_steps"] = global_steps
        environment = None

        try:
            try:
                environment_class = importlib.import_module("recipe.tmax.modal_sandbox").ModalSandboxEnvironment
                environment = environment_class(
                    kwargs["extra_info"]["task_binary"],
                    sandbox_timeout=SANDBOX_TIMEOUT,
                )
                await asyncio.wait_for(environment.setup(), timeout=STARTUP_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("TMax environment setup timed out after %s seconds", STARTUP_TIMEOUT)
                return self._build_output(interaction)
            except Exception as exc:
                logger.warning("TMax environment setup failed: %s", exc, exc_info=True)
                return self._build_output(interaction)

            try:
                submitted = await asyncio.wait_for(
                    self._run_turns(interaction, uuid4().hex, environment, sampling_params),
                    timeout=AGENT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("TMax agent interaction timed out after %s seconds", AGENT_TIMEOUT)
                return self._build_output(interaction)
            except Exception as exc:
                logger.warning("TMax agent interaction failed: %s", exc, exc_info=True)
                return self._build_output(interaction)

            if submitted:
                try:
                    verifier_reward, _ = await asyncio.wait_for(
                        environment.run_verifier(timeout=VERIFIER_TIMEOUT),
                        timeout=VERIFIER_TIMEOUT,
                    )
                    interaction.reward = max(0.0, min(1.0, float(verifier_reward)))
                except asyncio.TimeoutError:
                    logger.warning("TMax verifier timed out after %s seconds", VERIFIER_TIMEOUT)
                    return self._build_output(interaction)
                except Exception as exc:
                    logger.warning("TMax verifier failed: %s", exc, exc_info=True)
                    return self._build_output(interaction)
        finally:
            if environment is not None:
                try:
                    await environment.cleanup()
                except Exception as exc:
                    logger.warning("TMax environment cleanup failed: %s", exc)

        return self._build_output(interaction)

    def _build_output(self, interaction: TmaxInteraction) -> AgentLoopOutput:
        self._ensure_non_empty_response(interaction)
        interaction.extra_fields["reward_extra_info"] = {"acc": interaction.reward}
        return AgentLoopOutput(
            prompt_ids=interaction.prompt_ids,
            response_ids=interaction.response_ids[: self.response_length],
            response_mask=interaction.response_mask[: self.response_length],
            response_logprobs=interaction.response_logprobs[: self.response_length],
            reward_score=interaction.reward,
            num_turns=interaction.num_turns,
            metrics=AgentLoopMetrics(),
            extra_fields=interaction.extra_fields,
        )

    async def _run_turns(
        self,
        interaction: TmaxInteraction,
        request_id: str,
        environment: Any,
        sampling_params: dict[str, Any],
    ) -> bool:
        for turn in range(self.max_turns):
            interaction.num_turns = turn + 1
            response_remaining = self.response_length - len(interaction.response_ids)
            max_tokens = min(MAX_TOKENS_PER_GENERATION, response_remaining)
            if max_tokens <= 0:
                return False

            output = await self._generate(
                interaction.token_ids,
                request_id,
                sampling_params,
                max_tokens,
            )
            min_global_steps = output.extra_fields["min_global_steps"]
            max_global_steps = output.extra_fields["max_global_steps"]
            if turn == 0:
                interaction.extra_fields["min_global_steps"] = min_global_steps
            interaction.extra_fields["max_global_steps"] = max_global_steps

            token_ids = output.token_ids[:max_tokens]
            logprobs = list(output.log_probs or [])[: len(token_ids)]
            logprobs.extend([0.0] * (len(token_ids) - len(logprobs)))
            interaction.append_assistant(token_ids, logprobs)

            _, tool_calls = await self.tool_parser.extract_tool_calls(token_ids, self.tool_schemas)
            if not tool_calls or tool_calls[0].name != "bash":
                return False
            terminal_output, submitted = await self._execute_commands(environment, tool_calls[:1])
            if submitted:
                return True
            if not await self._append_observation(interaction, terminal_output):
                return False

        return False

    async def _generate(
        self,
        prompt_ids: list[int],
        request_id: str,
        sampling_params: dict[str, Any],
        max_tokens: int,
    ) -> TokenOutput:
        params = dict(sampling_params)
        params["max_tokens"] = max_tokens
        return await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=params,
        )

    async def _execute_commands(
        self,
        environment: Any,
        commands: list[FunctionCall],
    ) -> tuple[str, bool]:
        outputs = []
        for tool_call in commands:
            try:
                arguments = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            command = arguments.get("command", "")
            if not command:
                outputs.append(format_error("'command' parameter is required."))
                continue
            try:
                result = await environment.exec(command, timeout=COMMAND_TIMEOUT)
                raw_output = result.stdout or ""
                if result.stderr:
                    raw_output += f"\n{result.stderr}" if raw_output else result.stderr
                exit_code = result.return_code
            except TimeoutError as exc:
                logger.warning("Command timed out: %r (%s)", command, exc)
                raw_output = f"Command timed out after {COMMAND_TIMEOUT:.0f}s.\n"
                exit_code = 124
            except Exception as exc:
                error = f"Step '{tool_call.name}' failed: {exc}. Args: {arguments}"
                logger.warning(error)
                outputs.append(error)
                continue
            if SUBMIT_MARKER in raw_output:
                return "", True
            output = truncate_output(raw_output) if raw_output else "(no output)"
            outputs.append(f"{output}\n\n(exit_code={exit_code})")
        return "\n\n".join(outputs), False

    async def _append_observation(self, interaction: TmaxInteraction, text: str) -> bool:
        merge_result, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(
            [],
            [{"role": "tool", "content": text, "name": "bash"}],
            interaction.token_ids,
            interaction.response_mask,
            interaction.response_logprobs,
            tools=self.tool_schema_dicts,
        )
        response_ids = merge_result.token_ids[len(interaction.prompt_ids) :]
        if len(response_ids) >= self.response_length:
            return False
        interaction.response_ids = response_ids
        interaction.response_mask = response_mask
        interaction.response_logprobs = response_logprobs or []
        return True

    def _render_messages(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool,
    ) -> list[int]:
        return self.tokenizer.apply_chat_template(
            messages,
            tools=self.tool_schema_dicts,
            add_generation_prompt=add_generation_prompt,
            tokenize=True,
            return_dict=False,
            **self.apply_chat_template_kwargs,
        )

    def _ensure_non_empty_response(self, interaction: TmaxInteraction) -> None:
        if interaction.response_ids:
            return
        interaction.response_ids = [self.tokenizer.eos_token_id or 0]
        interaction.response_mask = [0]
        interaction.response_logprobs = [0.0]
