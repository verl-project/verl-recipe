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
"""Step-wise agent loop for the NeMo Gym `interactive_browser` environment.

The sibling `nemo_gym` recipe delegates a whole rollout to NeMo Gym through
`RolloutCollectionHelper`. A browser rollout is long-horizon and every step can
fail for infrastructure reasons (session lost, page navigation timeout, judge
unreachable), so this recipe keeps the loop on the verl side instead: one
`BaseTool` per browser action, verl's native `ToolAgentLoop` driving the state
machine, and per-step deadlines the trainer can enforce.

Environment sessions are cookie-scoped. NeMo Gym assigns a session id in
`SessionMiddleware` (`nemo_gym/server_utils.py`), so one `aiohttp.ClientSession`
per rollout — with its own cookie jar — is exactly one environment session.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from typing import Any

import aiohttp
from recipe.nemo_gym.browser.judge import JudgeResult, judge_rollout
from transformers.utils import get_json_schema

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__file__)

# Tool endpoints exposed by resources_servers/interactive_browser in NeMo Gym.
_ACTIONS = {
    "navigate": ("/browser_navigate", ("url",)),
    "click": ("/browser_click", ("element_id",)),
    "type": ("/browser_type", ("element_id", "text")),
    "observe": ("/browser_observe", ()),
    "finish": ("/browser_finish", ("answer",)),
}

_INVALID_RESET = "environment_seed_session_failed"
_INVALID_VERIFY = "environment_verify_failed"
_INVALID_JUDGE = "environment_judge_failed"
_INVALID_ABORTED = "generation_aborted_timeout"


def _clean_answer(content: str, generation_truncated: bool) -> tuple[str, str]:
    """Extract the user-visible answer, never promoting an unfinished reasoning block."""
    text = (content or "").strip()
    if generation_truncated:
        return "", "generation_truncated"
    if re.search(r"<think\b", text, flags=re.IGNORECASE):
        text = re.sub(r"<think\b[^>]*>.*?</think\s*>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
        if re.search(r"</?think\b", text, flags=re.IGNORECASE):
            return "", "no_final_answer"
    text = re.sub(r"(?:<\|im_start\|>\s*assistant\s*|<\|im_end\|>|<\|endoftext\|>)", "", text).strip()
    return (text, "complete") if text else ("", "no_final_answer")


class BrowserTool(BaseTool):
    """One live browser per rollout, driven through the NeMo Gym HTTP contract.

    The tool owns the environment session for the whole episode: `/seed_session`
    on first use, `browser_*` per action, `/verify` at the end. Failures are
    classified rather than raised, so an infrastructure fault is distinguishable
    from a policy that simply did not solve the task.
    """

    _sessions: dict[str, dict[str, Any]] = {}

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None):
        super().__init__(config, tool_schema)
        self.base_url = str(config.get("resources_server_url") or os.environ.get("NEMO_GYM_BROWSER_URL", "")).rstrip(
            "/"
        )
        if not self.base_url:
            raise ValueError(
                "browser tool needs `resources_server_url` in its config or NEMO_GYM_BROWSER_URL in the environment; "
                "point it at the interactive_browser resources server started by `gym env start`."
            )
        self.seed_timeout_s = float(config.get("seed_timeout_s", 120.0))
        self.step_timeout_s = float(config.get("step_timeout_s", 35.0))
        self.verify_timeout_s = float(config.get("verify_timeout_s", 90.0))
        self.max_observation_chars = int(config.get("max_observation_chars", 8000))
        _BROWSER_TOOLS.append(self)

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        def browser(action: str, url: str = "", element_id: int = -1, text: str = "", answer: str = "") -> str:
            """Operate a live browser.

            Args:
                action: One of observe, navigate, click, type, finish.
                url: Target URL for navigate.
                element_id: Element id from the most recent observation, for click and type.
                text: Text to type, for type.
                answer: Final answer to report, for finish.
            """
            return ""

        return OpenAIFunctionToolSchema(**get_json_schema(browser))

    async def _request(self, record: dict[str, Any], path: str, payload: dict[str, Any], timeout: float) -> dict:
        client: aiohttp.ClientSession = record["client"]
        async with client.post(
            f"{self.base_url}{path}", json=payload, timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            body = await response.json()
            if response.status >= 400:
                raise RuntimeError(f"{path} -> HTTP {response.status}: {body}")
            return body

    async def _session(self, agent_data: AgentData) -> dict[str, Any]:
        """Return this rollout's environment session, seeding it on first use."""
        request_id = agent_data.request_id
        record = self._sessions.get(request_id)
        if record is not None:
            return record

        create = agent_data.tools_kwargs["browser"]["create_kwargs"]
        # One cookie jar per rollout == one environment session on the server.
        record = {
            "client": aiohttp.ClientSession(),
            "task": dict(create),
            "events": [],
            "tool_calls": 0,
            "seed_error": "",
            "tool": self,
        }
        started = time.perf_counter()
        try:
            await self._request(
                record,
                "/seed_session",
                {
                    "initial_url": create.get("initial_url", "about:blank"),
                    "verifier_metadata": create.get("verifier_metadata") or {},
                },
                timeout=self.seed_timeout_s,
            )
        except Exception as exc:
            # A failed seed is a classified rollout outcome, not a crashed loop.
            record["seed_error"] = f"{type(exc).__name__}: {exc}"
        finally:
            agent_data.metrics["browser_seed_s"] = time.perf_counter() - started
        self._sessions[request_id] = record
        return record

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs: Any) -> tuple[ToolResponse, ...]:
        del instance_id
        agent_data: AgentData = kwargs["agent_data"]
        record = await self._session(agent_data)
        record["tool_calls"] += 1
        action = str(parameters.get("action", "")).strip().lower()

        if record["seed_error"]:
            text = f"ERROR_ENVIRONMENT_SESSION: {record['seed_error']}"
        elif action not in _ACTIONS:
            text = f"ERROR_POLICY_TOOL: action must be one of {sorted(_ACTIONS)}"
        else:
            path, fields = _ACTIONS[action]
            payload = {field: parameters.get(field) for field in fields if parameters.get(field) not in (None, "", -1)}
            missing = [field for field in fields if field not in payload]
            if missing:
                text = f"ERROR_POLICY_TOOL: {action} requires {missing}"
            else:
                started = time.perf_counter()
                try:
                    body = await self._request(record, path, payload, timeout=self.step_timeout_s)
                    text = str(body.get("observation") or "")
                    if body.get("error"):
                        # The environment returns tool errors to the model on purpose.
                        text = f"{text}\nERROR: {body['error']}".strip()
                    if body.get("done"):
                        record["finished"] = True
                        record["answer"] = str(parameters.get("answer") or "")
                except Exception as exc:
                    record["step_error"] = f"{type(exc).__name__}: {exc}"
                    text = f"ERROR_ENVIRONMENT_STEP: {record['step_error']}"
                finally:
                    metric = "browser_observe_s" if action == "observe" else "browser_act_s"
                    agent_data.metrics[metric] = agent_data.metrics.get(metric, 0.0) + (time.perf_counter() - started)

        if len(text) > self.max_observation_chars:
            text = text[: self.max_observation_chars] + "\n...(truncated)"
        record["events"].append({"action": json.dumps(parameters, ensure_ascii=False), "observation": text})
        return ToolResponse(text=text), 0.0, {}

    @classmethod
    async def finalize(cls, agent_data: AgentData, final_answer: str, answer_status: str) -> tuple[float, dict]:
        """Score the rollout and release the environment session exactly once."""
        record = cls._sessions.pop(agent_data.request_id, None)
        if record is None:
            return 0.0, {"reason": "no_environment_session", "invalid_sample": True, "invalid_reason": _INVALID_RESET}

        tool: BrowserTool = record["tool"]
        info: dict[str, Any] = {
            "tool_calls": record["tool_calls"],
            "answer_status": answer_status,
            "finished_by_model": bool(record.get("finished")),
        }
        if record["seed_error"]:
            info.update(reason="seed_failed", invalid_sample=True, invalid_reason=_INVALID_RESET)
            await cls._close(record)
            return 0.0, info

        reward = 0.0
        try:
            task = record["task"]
            if task.get("verifier_metadata"):
                body = await tool._request(
                    record,
                    "/verify",
                    {"verifier_metadata": task["verifier_metadata"]},
                    timeout=tool.verify_timeout_s,
                )
                reward = float(body.get("reward", 0.0))
                info["reason"] = "environment_verifier"
            else:
                # The environment's built-in verifier only supports deterministic
                # specs; open-ended tasks are scored by the recipe's judge.
                verdict: JudgeResult = await judge_rollout(
                    question=task.get("question", ""),
                    events=record["events"],
                    final_answer=final_answer,
                    answer_status=answer_status,
                )
                reward = verdict.reward
                info["reason"] = verdict.reason
                if not verdict.usable:
                    info.update(invalid_sample=True, invalid_reason=_INVALID_JUDGE)
        except Exception as exc:
            info.update(
                reason=f"verify_failed:{type(exc).__name__}", invalid_sample=True, invalid_reason=_INVALID_VERIFY
            )
        finally:
            await cls._close(record)
        return reward, info

    @staticmethod
    async def _close(record: dict[str, Any]) -> None:
        """Drop the cookie jar so the environment session is no longer referenced.

        NeMo Gym releases the browser inside `verify()`; a rollout that never got
        there leaves it to the environment's own reclaim path. See
        NVIDIA-NeMo/Gym#2609 for the teardown hook this would use once it exists.
        """
        client: aiohttp.ClientSession = record["client"]
        try:
            await client.close()
        except Exception as exc:  # pragma: no cover - close is best effort
            logger.warning("failed to close environment session client: %r", exc)


_BROWSER_TOOLS: list[BaseTool] = []


@register("nemo_gym_browser_agent")
class BrowserToolAgentLoop(ToolAgentLoop):
    """`ToolAgentLoop` with an episode deadline and environment-failure classification.

    Two behaviours the base loop does not provide:

    * a whole-episode deadline, because a browser rollout can stall in a way no
      single tool timeout catches;
    * an explicit `env_invalid` flag on every sample, so the trainer can exclude
      infrastructure failures from GRPO group statistics instead of treating them
      as a policy that scored zero. A rollout that is invalid is resampled first;
      only one that stays invalid reaches the batch, loss-masked and flagged.
    """

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        attempts = 1 + max(0, int(os.environ.get("NEMO_GYM_BROWSER_ENV_RETRIES", "1")))
        output: AgentLoopOutput | None = None
        for attempt in range(1, attempts + 1):
            output = await self._run_episode(sampling_params, **kwargs)
            if not output.extra_fields.get("env_invalid"):
                return output
            if attempt < attempts:
                logger.warning(
                    "environment-invalid rollout (%s); resampling %d/%d",
                    output.extra_fields.get("env_invalid_reason", "unknown"),
                    attempt + 1,
                    attempts,
                )
                output.extra_fields["env_retry_attempts"] = attempt + 1
        assert output is not None
        return output

    async def _run_episode(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentLoopOutput:
        import uuid

        episode_timeout_s = float(os.environ.get("NEMO_GYM_BROWSER_EPISODE_TIMEOUT_S", "180"))
        action_max_tokens = int(os.environ.get("NEMO_GYM_BROWSER_ACTION_MAX_TOKENS", "1024"))

        messages = list(kwargs["raw_prompt"])
        multimodal = await self.process_multi_modal_info(messages)
        agent_data = AgentData(
            messages,
            multimodal.get("images"),
            multimodal.get("videos"),
            multimodal.get("audios"),
            self._get_mm_processor_kwargs(multimodal.get("audios")),
            {},
            uuid.uuid4().hex,
            kwargs.get("tools_kwargs", {}),
        )
        agent_data._active_tools, agent_data._active_tool_schemas = self.tools, self.tool_schemas

        state = AgentState.PENDING
        timed_out = False
        truncated = False
        try:
            async with asyncio.timeout(episode_timeout_s):
                while state != AgentState.TERMINATED:
                    if state == AgentState.PENDING:
                        state = await self._handle_pending_state(agent_data, sampling_params)
                    elif state == AgentState.GENERATING:
                        turn_params = dict(sampling_params)
                        configured = turn_params.pop("max_new_tokens", turn_params.get("max_tokens"))
                        turn_params["max_tokens"] = min(
                            int(configured) if configured is not None else action_max_tokens, action_max_tokens
                        )
                        before = sum(agent_data.response_mask)
                        state = await self._handle_generating_state(agent_data, turn_params)
                        truncated = (sum(agent_data.response_mask) - before) >= int(turn_params["max_tokens"])
                    elif state == AgentState.TOOL_CALLING:
                        state = await self._handle_tool_calling_state(agent_data)
                    else:
                        state = await self._handle_processing_tools_state(agent_data)
        except TimeoutError:
            timed_out = True

        final_answer, answer_status = "", "episode_timed_out" if timed_out else "no_final_answer"
        if not timed_out and agent_data.response_ids:
            schemas = [tool.tool_schema for tool in getattr(agent_data, "_active_tools", self.tools).values()]
            try:
                content, tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, schemas)
                answer_status = "tool_call_only" if tool_calls else answer_status
                if not tool_calls:
                    final_answer, answer_status = _clean_answer(content or "", truncated)
            except Exception:
                answer_status = "parse_error"

        reward, info = await BrowserTool.finalize(agent_data, final_answer, answer_status)

        response_length = len(agent_data.response_mask)
        prompt_ids = agent_data.prompt_ids[:-response_length] if response_length else list(agent_data.prompt_ids)
        response_ids = (agent_data.prompt_ids[-response_length:] if response_length else [])[: self.response_length]
        response_mask = list(agent_data.response_mask[: self.response_length])
        response_logprobs = list(agent_data.response_logprobs[: self.response_length])

        if not response_ids:
            # Keep the trajectory in its GRPO group without fabricating a gradient
            # for a generation that never happened.
            eos_token_id = self.tokenizer.eos_token_id
            if eos_token_id is None:
                raise RuntimeError("tokenizer has no eos_token_id for an empty agent response")
            response_ids, response_mask, response_logprobs = [int(eos_token_id)], [0], [0.0]
            if timed_out:
                # An empty response after the deadline is infrastructure; an empty
                # response from a first-token stop is a real policy outcome and
                # must stay in the baseline.
                info.setdefault("invalid_sample", True)
                info.setdefault("invalid_reason", _INVALID_ABORTED)

        env_invalid = bool(info.get("invalid_sample"))
        if env_invalid:
            response_mask = [0] * len(response_mask)

        extra_fields = dict(agent_data.extra_fields)
        # Materialize the flag on every sample: a non-tensor key only reaches the
        # batch when at least one sample carries it, and the trainer must be able
        # to tell "no invalid samples" from "flags unavailable".
        extra_fields["env_invalid"] = env_invalid
        extra_fields["env_invalid_reason"] = str(info.get("invalid_reason") or "")
        extra_fields["reward_extra_info"] = info

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs or None,
            reward_score=reward,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            extra_fields=extra_fields,
        )
