from __future__ import annotations

import asyncio
import copy
import importlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import torch

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

MAX_TOKENS_PER_GENERATION = 2048
MAX_TERMINAL_OUTPUT_CHARS = 50000
STARTUP_TIMEOUT = 600.0
AGENT_TIMEOUT = 600.0
VERIFIER_TIMEOUT = 300.0
SANDBOX_TIMEOUT = STARTUP_TIMEOUT + AGENT_TIMEOUT + VERIFIER_TIMEOUT + 60.0


class EchoAgentLoopOutput(AgentLoopOutput):
    aux_token_loss_mask: list[int]

    def as_dict(self) -> dict[str, Any]:
        output = super().as_dict()
        output["aux_token_loss_mask"] = torch.as_tensor(output["aux_token_loss_mask"], dtype=torch.int64)
        return output


@dataclass
class EchoInteraction:
    prompt_messages: list[dict[str, Any]]
    prompt_ids: list[int]
    completion_messages: list[dict[str, Any]] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    aux_token_loss_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] = field(default_factory=list)
    extra_fields: dict[str, Any] = field(default_factory=dict)
    reward: float = 0.0
    num_turns: int = 0

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self.prompt_messages + self.completion_messages

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_ids + self.response_ids

    def append_assistant(self, token_ids: list[int], logprobs: list[float], tokenizer) -> None:
        self.completion_messages.append(
            {"role": "assistant", "content": tokenizer.decode(token_ids, skip_special_tokens=True)}
        )
        self.response_ids.extend(token_ids)
        self.response_mask.extend([1] * len(token_ids))
        self.aux_token_loss_mask.extend([0] * len(token_ids))
        self.response_logprobs.extend(logprobs)

    def append_observation(
        self,
        token_ids: list[int],
        aux_token_loss_mask: list[int],
        role: str,
        text: str,
    ) -> None:
        self.completion_messages.append({"role": role, "content": text})
        self.response_ids.extend(token_ids)
        self.response_mask.extend([0] * len(token_ids))
        self.aux_token_loss_mask.extend(aux_token_loss_mask)
        self.response_logprobs.extend([0.0] * len(token_ids))


def truncate_output(output: str) -> str:
    if len(output) <= MAX_TERMINAL_OUTPUT_CHARS:
        return output
    message = f"\n[Output truncated: showing first {MAX_TERMINAL_OUTPUT_CHARS} of {len(output)} characters]"
    return output[:MAX_TERMINAL_OUTPUT_CHARS] + message


class EchoAgentLoop(AgentLoopBase):
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
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        newline_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        self.turn_end_tokens = [self.im_end_id, newline_ids[-1]]

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = copy.deepcopy(list(kwargs["raw_prompt"]))
        prompt_ids = self._render_messages(messages, add_generation_prompt=True)
        if len(prompt_ids) > self.rollout_config.prompt_length:
            raise ValueError(
                f"Prompt has {len(prompt_ids)} tokens, exceeding prompt_length={self.rollout_config.prompt_length}"
            )
        interaction = EchoInteraction(prompt_messages=messages, prompt_ids=prompt_ids)
        if "global_steps" in kwargs:
            global_steps = int(kwargs["global_steps"])
            interaction.extra_fields["min_global_steps"] = global_steps
            interaction.extra_fields["max_global_steps"] = global_steps
        environment = None

        try:
            try:
                environment_class = importlib.import_module("recipe.echo.modal_sandbox").ModalSandboxEnvironment
                environment = environment_class(
                    kwargs["extra_info"]["task_binary"],
                    sandbox_timeout=SANDBOX_TIMEOUT,
                )
                await asyncio.wait_for(environment.setup(), timeout=STARTUP_TIMEOUT)
            except asyncio.TimeoutError:
                logger.warning("ECHO environment setup timed out after %s seconds", STARTUP_TIMEOUT)
                return self._build_output(interaction)
            except Exception as exc:
                logger.warning("ECHO environment setup failed: %s", exc, exc_info=True)
                return self._build_output(interaction)

            try:
                await asyncio.wait_for(
                    self._run_turns(interaction, uuid4().hex, environment, sampling_params),
                    timeout=AGENT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                logger.warning("ECHO agent interaction timed out after %s seconds", AGENT_TIMEOUT)
                return self._build_output(interaction)
            except Exception as exc:
                logger.warning("ECHO agent interaction failed: %s", exc, exc_info=True)
                return self._build_output(interaction)

            try:
                verifier_reward, _ = await asyncio.wait_for(
                    environment.run_verifier(timeout=VERIFIER_TIMEOUT),
                    timeout=VERIFIER_TIMEOUT,
                )
                interaction.reward = float(verifier_reward)
            except asyncio.TimeoutError:
                logger.warning("ECHO verifier timed out after %s seconds", VERIFIER_TIMEOUT)
                return self._build_output(interaction)
            except Exception as exc:
                logger.warning("ECHO verifier failed: %s", exc, exc_info=True)
                return self._build_output(interaction)
        finally:
            if environment is not None:
                try:
                    await environment.cleanup()
                except Exception as exc:
                    logger.warning("ECHO environment cleanup failed: %s", exc)

        return self._build_output(interaction)

    def _build_output(self, interaction: EchoInteraction) -> EchoAgentLoopOutput:
        self._ensure_non_empty_response(interaction)
        return EchoAgentLoopOutput(
            prompt_ids=interaction.prompt_ids,
            response_ids=interaction.response_ids[: self.response_length],
            response_mask=interaction.response_mask[: self.response_length],
            aux_token_loss_mask=interaction.aux_token_loss_mask[: self.response_length],
            response_logprobs=interaction.response_logprobs[: self.response_length],
            reward_score=interaction.reward,
            num_turns=interaction.num_turns,
            metrics=AgentLoopMetrics(),
            extra_fields=interaction.extra_fields,
        )

    async def _run_turns(
        self,
        interaction: EchoInteraction,
        request_id: str,
        environment: Any,
        sampling_params: dict[str, Any],
    ) -> None:
        for turn in range(self.max_turns):
            interaction.num_turns = turn + 1
            response_remaining = self.response_length - len(interaction.response_ids)
            max_tokens = min(MAX_TOKENS_PER_GENERATION, response_remaining)
            if max_tokens <= 0:
                break

            output = await self._generate(
                interaction.token_ids,
                request_id,
                sampling_params,
                max_tokens,
            )
            min_global_steps = output.extra_fields.get("min_global_steps")
            if min_global_steps is not None:
                interaction.extra_fields["min_global_steps"] = min(
                    interaction.extra_fields.get("min_global_steps", min_global_steps),
                    min_global_steps,
                )
            max_global_steps = output.extra_fields.get("max_global_steps")
            if max_global_steps is not None:
                interaction.extra_fields["max_global_steps"] = max(
                    interaction.extra_fields.get("max_global_steps", max_global_steps),
                    max_global_steps,
                )

            token_ids = output.token_ids[:max_tokens]
            logprobs = list(output.log_probs or [])[: len(token_ids)]
            logprobs.extend([0.0] * (len(token_ids) - len(logprobs)))
            interaction.append_assistant(token_ids, logprobs, self.tokenizer)

            response_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            warnings = self._format_warnings(response_text)
            _, tool_calls = await self.tool_parser.extract_tool_calls(token_ids, self.tool_schemas)
            if not tool_calls:
                if not self._append_observation(
                    interaction,
                    "",
                    "No <tool_call> found in response.",
                ):
                    break
                continue

            is_done = any(tool_call.name == "done" for tool_call in tool_calls)
            commands = [tool_call for tool_call in tool_calls if tool_call.name != "done"][:1]
            terminal_output = await self._execute_commands(environment, commands)

            if is_done:
                break

            warning_text = ""
            if warnings:
                warning_text = "WARNINGS:\n" + "\n".join(f"- {warning}" for warning in warnings) + "\n\n"
            if not commands:
                terminal_output = "No commands provided. Please provide commands to execute or set done."
            if not self._append_observation(interaction, warning_text, terminal_output):
                break

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
    ) -> str:
        outputs = []
        for tool_call in commands:
            try:
                arguments = json.loads(tool_call.arguments)
            except (json.JSONDecodeError, TypeError):
                outputs.append(f"Command '{tool_call.arguments}' skipped due to parse error: invalid arguments")
                continue
            if tool_call.name != "bash":
                outputs.append(
                    f"Command '{tool_call.arguments}' skipped due to Unknown tool "
                    f"'{tool_call.name}'. Only 'bash' is supported."
                )
                continue
            command = arguments.get("command", "")
            timeout = arguments.get("timeout", 30)
            if not command:
                outputs.append(
                    f"Command '{tool_call.arguments}' skipped due to no command provided. "
                    "Please provide a command in correct format to execute."
                )
                continue
            try:
                result = await environment.exec(command, timeout=float(timeout))
            except (RuntimeError, TimeoutError) as exc:
                logger.warning("Command failed or timed out: %r (%s)", command, exc)
                outputs.append(f"Command '{command}' timed out after {timeout} seconds.\n\n(exit_code=-1)")
                continue
            except Exception as exc:
                outputs.append(f"Command '{command}' execution error: {exc}\n\n(exit_code=-1)")
                continue

            raw_output = result.stdout or ""
            if result.stderr:
                raw_output += f"\n{result.stderr}"
            status = "executed successfully" if result.return_code == 0 else "failed"
            outputs.append(
                f"Command '{command}' {status}. Output: {truncate_output(raw_output)}"
                f"\n\n(exit_code={result.return_code})"
            )
        return "\n\n".join(outputs)

    def _append_observation(
        self,
        interaction: EchoInteraction,
        warning_text: str,
        environment_text: str,
    ) -> bool:
        self._append_turn_end(interaction)
        available = self.response_length - len(interaction.response_ids)
        if available <= 0:
            return False

        text = warning_text + environment_text
        warning_end, content_end = self._observation_spans(
            interaction,
            warning_text,
            environment_text,
        )
        token_ids = self._observation_tokens(interaction, text)
        aux_token_loss_mask = [0] * len(token_ids)
        for index in range(warning_end, content_end):
            aux_token_loss_mask[index] = 1
        truncated = len(token_ids) > available
        interaction.append_observation(
            token_ids[:available],
            aux_token_loss_mask[:available],
            "tool",
            text,
        )
        return not truncated

    def _observation_spans(
        self,
        interaction: EchoInteraction,
        warning_text: str,
        environment_text: str,
    ) -> tuple[int, int]:
        empty = self._message_delta(interaction.messages, "")
        warning = self._message_delta(interaction.messages, warning_text)
        full = self._message_delta(interaction.messages, warning_text + environment_text)
        content_start = self._common_prefix_length(empty, full)
        content_end = len(full) - self._common_suffix_length(empty, full, content_start)
        if not warning_text:
            return content_start, content_end
        warning_start = self._common_prefix_length(empty, warning)
        warning_end = len(warning) - self._common_suffix_length(empty, warning, warning_start)
        return warning_end, content_end

    def _observation_tokens(
        self,
        interaction: EchoInteraction,
        text: str,
    ) -> list[int]:
        before = self._render_messages(interaction.messages, add_generation_prompt=False)
        messages = interaction.messages + [{"role": "tool", "content": text}]
        after_full = self._render_messages(messages, add_generation_prompt=True)
        return after_full[len(before) :]

    def _message_delta(self, messages: list[dict[str, Any]], text: str) -> list[int]:
        before = self._render_messages(messages, add_generation_prompt=False)
        after = self._render_messages(
            messages + [{"role": "tool", "content": text}],
            add_generation_prompt=False,
        )
        return after[len(before) :]

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

    def _append_turn_end(self, interaction: EchoInteraction) -> None:
        tail = interaction.response_ids[-len(self.turn_end_tokens) :]
        if tail == self.turn_end_tokens:
            return
        tokens = self.turn_end_tokens[1:] if tail and tail[-1] == self.im_end_id else self.turn_end_tokens
        interaction.response_ids.extend(tokens)
        interaction.response_mask.extend([0] * len(tokens))
        interaction.aux_token_loss_mask.extend([0] * len(tokens))
        interaction.response_logprobs.extend([0.0] * len(tokens))

    @staticmethod
    def _format_warnings(response: str) -> list[str]:
        response = "<think>\n" + response
        warnings = []
        for tag in ("think", "tool_call"):
            if response.count(f"<{tag}>") > response.count(f"</{tag}>"):
                warnings.append(f"Unclosed <{tag}> tag")
            if re.search(rf"<{tag}>\s*</{tag}>", response):
                warnings.append(f"Empty <{tag}> block")
        return warnings

    def _ensure_non_empty_response(self, interaction: EchoInteraction) -> None:
        if interaction.response_ids:
            return
        interaction.response_ids = [0]
        interaction.response_mask = [0]
        interaction.aux_token_loss_mask = [0]
        interaction.response_logprobs = [0.0]

    @staticmethod
    def _common_prefix_length(first: list[int], second: list[int]) -> int:
        length = 0
        for first_token, second_token in zip(first, second, strict=False):
            if first_token != second_token:
                break
            length += 1
        return length

    @staticmethod
    def _common_suffix_length(first: list[int], second: list[int], prefix_length: int) -> int:
        max_length = min(len(first), len(second)) - prefix_length
        length = 0
        while length < max_length and first[-1 - length] == second[-1 - length]:
            length += 1
        return length
