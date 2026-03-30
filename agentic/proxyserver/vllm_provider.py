"""LiteLLM CustomLLM provider that routes requests to vLLM rollout servers
via Ray RPC.

This replaces the hand-written OpenAI response formatting, streaming SSE
generation, and logprobs content building that previously lived in
``server.py``.  LiteLLM handles all of that automatically once we return
a properly populated ``ModelResponse``.

Usage::

    from recipe.agentic.proxyserver.vllm_provider import VLLMRayProvider

    provider = VLLMRayProvider(server_handles, tokenizer, tool_parser)
    litellm.custom_provider_map = [
        {"provider": "verl-vllm", "custom_handler": provider}
    ]

    response = await litellm.acompletion(
        model="verl-vllm/default",
        messages=[...],
        logprobs=True,
    )
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable, Optional, Union
from uuid import uuid4

import httpx
from litellm.llms.custom_httpx.http_handler import (AsyncHTTPHandler,
                                                    HTTPHandler)
from litellm.llms.custom_llm import CustomLLM, CustomLLMError
from litellm.types.utils import (ChatCompletionTokenLogprob, ChoiceLogprobs,
                                 Choices, GenericStreamingChunk, Message,
                                 Usage)
from litellm.utils import ModelResponse

logger = logging.getLogger(__name__)


class VLLMRayProvider(CustomLLM):
    """LiteLLM custom provider that calls vLLM via Ray actor handles.

    This encapsulates the core generate logic (tokenize -> Ray RPC -> decode)
    and returns standard ``litellm.ModelResponse`` objects.  LiteLLM then
    takes care of OpenAI-compatible JSON formatting, SSE streaming, etc.
    """

    def __init__(
        self,
        server_handles: list,
        tokenizer: Any,
        tool_parser: Any | None = None,
    ) -> None:
        super().__init__()
        self.server_handles = list(server_handles)
        self.tokenizer = tokenizer
        self.tool_parser = tool_parser

        # Sticky-session load balancing: session_id -> server index
        self._sticky: dict[str, int] = {}
        self._request_counts: list[int] = [0] * len(self.server_handles)

    # ------------------------------------------------------------------
    # Load balancing
    # ------------------------------------------------------------------

    def _choose_server(self, session_id: str | None):
        """Pick a server handle using sticky sessions."""
        key = session_id or "default"
        if key in self._sticky:
            return self.server_handles[self._sticky[key]]

        idx = min(
            range(len(self._request_counts)),
            key=lambda i: self._request_counts[i],
        )
        self._sticky[key] = idx
        self._request_counts[idx] += 1
        return self.server_handles[idx]

    def release_session(self, session_id: str) -> None:
        """Remove sticky-session binding for the given session."""
        self._sticky.pop(session_id, None)

    # ------------------------------------------------------------------
    # Core generation via Ray RPC
    # ------------------------------------------------------------------

    async def _generate(
        self,
        messages: list[dict[str, Any]],
        session_id: str | None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        tools: list[dict] | None = None,
        repetition_penalty: float | None = None,
    ) -> tuple[list[int], list[float], str | None, str]:
        """Tokenize, call vLLM via Ray, return raw results.

        Returns:
            ``(token_ids, log_probs, stop_reason, completion_text)``
        """
        normalized = _normalize_messages(messages)
        prompt_ids: list[int] = self.tokenizer.apply_chat_template(
            normalized,
            tools=tools,
            add_generation_prompt=True,
            tokenize=True,
        )

        sampling_params: dict[str, Any] = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "logprobs": True,
        }
        if stop:
            sampling_params["stop"] = stop
        if repetition_penalty is not None:
            sampling_params["repetition_penalty"] = repetition_penalty

        server = self._choose_server(session_id)
        request_id = uuid4().hex

        ref = server.generate.remote(
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        import ray

        output = await asyncio.to_thread(ray.get, ref)

        token_ids = list(output.token_ids)
        log_probs = list(output.log_probs) if output.log_probs else []
        stop_reason = getattr(output, "stop_reason", None)

        completion_text = self.tokenizer.decode(
            token_ids, skip_special_tokens=False
        )

        return token_ids, log_probs, stop_reason, completion_text

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_logprobs(
        self, token_ids: list[int], log_probs: list[float]
    ) -> ChoiceLogprobs | None:
        """Build LiteLLM ``ChoiceLogprobs`` from raw token data."""
        if not log_probs:
            return None
        content = []
        for tid, lp in zip(token_ids, log_probs):
            tok_str = self.tokenizer.decode([tid])
            content.append(
                ChatCompletionTokenLogprob(
                    token=tok_str,
                    logprob=lp,
                    bytes=list(tok_str.encode("utf-8", errors="replace")),
                    top_logprobs=[],
                )
            )
        return ChoiceLogprobs(content=content)

    async def _parse_tool_calls(
        self, token_ids: list[int], completion_text: str
    ) -> tuple[str, list[dict[str, Any]] | None, str]:
        """Parse tool calls if tool_parser is configured.

        Returns:
            ``(content_text, openai_tool_calls_or_none, finish_reason)``
        """
        if self.tool_parser is None:
            return completion_text, None, "stop"

        content_text, parsed_calls = await self.tool_parser.extract_tool_calls(
            token_ids
        )
        if not parsed_calls:
            return content_text, None, "stop"

        openai_tool_calls = []
        for tc in parsed_calls:
            call_id = f"call_{uuid4().hex[:24]}"
            openai_tool_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments,
                    },
                }
            )
        return content_text, openai_tool_calls, "tool_calls"

    # ------------------------------------------------------------------
    # LiteLLM CustomLLM interface
    # ------------------------------------------------------------------

    async def acompletion(  # noqa: PLR0913
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ModelResponse:
        """Async chat completion via vLLM Ray RPC."""
        temperature = optional_params.get("temperature", 1.0)
        top_p = optional_params.get("top_p", 1.0)
        max_tokens = optional_params.get("max_tokens", 2048)
        stop = optional_params.get("stop")
        tools = optional_params.get("tools")
        repetition_penalty = optional_params.get("repetition_penalty")

        extra_body = optional_params.get("extra_body", {})
        session_id = (
            extra_body.get("session_id") if isinstance(extra_body, dict) else None
        )

        try:
            token_ids, log_probs, stop_reason, completion_text = (
                await self._generate(
                    messages=messages,
                    session_id=session_id,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    tools=tools,
                    repetition_penalty=repetition_penalty,
                )
            )
        except Exception as e:
            raise CustomLLMError(status_code=500, message=str(e))

        # Parse tool calls
        content_text, tool_calls_list, finish_reason = await self._parse_tool_calls(
            token_ids, completion_text
        )
        # Override finish_reason if vLLM provided a meaningful one
        if stop_reason in ("stop", "length"):
            finish_reason = stop_reason

        # Build logprobs
        logprobs_obj = self._build_logprobs(token_ids, log_probs)

        # Build message
        message_dict: dict[str, Any] = {"role": "assistant"}
        if tool_calls_list:
            message_dict["content"] = (
                content_text.strip()
                if content_text and content_text.strip()
                else None
            )
            message_dict["tool_calls"] = tool_calls_list
        else:
            message_dict["content"] = content_text

        model_response.choices = [
            Choices(
                finish_reason=finish_reason,
                index=0,
                message=Message(**message_dict),
                logprobs=logprobs_obj,
                provider_specific_fields={"token_ids": token_ids},
            )
        ]
        model_response.model = model
        model_response.usage = Usage(
            prompt_tokens=0,
            completion_tokens=len(token_ids),
            total_tokens=len(token_ids),
        )

        return model_response

    async def astreaming(  # noqa: PLR0913
        self,
        model: str,
        messages: list,
        api_base: str,
        custom_prompt_dict: dict,
        model_response: ModelResponse,
        print_verbose: Callable,
        encoding,
        api_key,
        logging_obj,
        optional_params: dict,
        acompletion=None,
        litellm_params=None,
        logger_fn=None,
        headers={},
        timeout: Optional[Union[float, httpx.Timeout]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> AsyncIterator[GenericStreamingChunk]:
        """Async streaming -- generates fully then yields per-token.

        vLLM Ray RPC returns the full response at once, so we simulate
        streaming by yielding one ``GenericStreamingChunk`` per token.
        """
        temperature = optional_params.get("temperature", 1.0)
        top_p = optional_params.get("top_p", 1.0)
        max_tokens = optional_params.get("max_tokens", 2048)
        stop = optional_params.get("stop")
        tools = optional_params.get("tools")
        repetition_penalty = optional_params.get("repetition_penalty")

        extra_body = optional_params.get("extra_body", {})
        session_id = (
            extra_body.get("session_id") if isinstance(extra_body, dict) else None
        )

        try:
            token_ids, log_probs, stop_reason, completion_text = (
                await self._generate(
                    messages=messages,
                    session_id=session_id,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    stop=stop,
                    tools=tools,
                    repetition_penalty=repetition_penalty,
                )
            )
        except Exception as e:
            raise CustomLLMError(status_code=500, message=str(e))

        finish_reason = (
            stop_reason
            if stop_reason in ("stop", "length", "tool_calls")
            else "stop"
        )

        for i, tid in enumerate(token_ids):
            tok_text = self.tokenizer.decode([tid])
            is_last = i == len(token_ids) - 1

            chunk: GenericStreamingChunk = {
                "text": tok_text,
                "is_finished": is_last,
                "finish_reason": finish_reason if is_last else "",
                "usage": (
                    {
                        "completion_tokens": len(token_ids),
                        "prompt_tokens": 0,
                        "total_tokens": len(token_ids),
                    }
                    if is_last
                    else None
                ),
                "index": 0,
                "tool_use": None,
                "provider_specific_fields": (
                    {"token_ids": token_ids} if is_last else None
                ),
            }
            yield chunk


# ---------------------------------------------------------------------------
# Message normalization
# ---------------------------------------------------------------------------


def _normalize_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Normalize message content fields for ``apply_chat_template``.

    Handles:
    * Multi-part ``content`` lists (OpenAI vision format) -> plain strings.
    * ``content: null`` in assistant messages with ``tool_calls``.
    * ``tool`` role messages are passed through with ``tool_call_id``.
    """
    normalized = []
    for msg in messages:
        msg = dict(msg)
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif "text" in part:
                        text_parts.append(part["text"])
                elif isinstance(part, str):
                    text_parts.append(part)
            msg["content"] = "\n".join(text_parts) if text_parts else ""

        if role == "assistant" and "tool_calls" in msg:
            if msg.get("content") is None:
                msg["content"] = ""

        normalized.append(msg)
    return normalized
