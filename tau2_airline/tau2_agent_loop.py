"""Custom veRL agent loop for tau2-bench airline (agent <-> user-sim <-> env).

Registered as ``tau2_agent``. It reuses veRL's tested token/mask machinery from
``ToolAgentLoop`` (generation, incremental tokenization, response masking) and
makes exactly three domain-specific changes:

  1. tools come from tau2 (``Tau2Bridge``), executed against a per-trajectory
     private env -- NOT veRL's FunctionTool/BaseTool dispatch;
  2. when the policy emits *plain text* (no tool call), that text is a message
     to the USER: we call the tau2 user simulator and feed its reply back as a
     masked turn, instead of TERMINATING (which is what ToolAgentLoop does);
  3. at the end we score the finished trajectory with tau2's evaluator and put
     the result in ``output.reward_score`` (so veRL's reward worker is skipped).

Bridge configuration is read from environment variables so we do not have to
touch veRL's OmegaConf schema:

    TAU2_DOMAIN            (default "airline")
    TAU2_USER_LLM          (default "openai/usersim")
    TAU2_USER_API_BASE     (default "http://127.0.0.1:18001/v1")
    TAU2_USER_API_KEY      (default "dummy")
    TAU2_USER_TEMPERATURE  (default "0.0")
    TAU2_EVAL_TYPE         (default "all")
    TAU2_MAX_ERRORS        (default "10")
"""

import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from rollout_context import rollout_scope
from tau2.data_model.simulation import TerminationReason

# tau2 bridge + isolation live next to this file on PYTHONPATH.
from tau2_bridge import Tau2Bridge

from verl.experimental.agent_loop.agent_loop import AgentLoopOutput, register
from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class Tau2State(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    PROCESSING_USER = "processing_user"
    TERMINATED = "terminated"


def _build_bridge() -> Tau2Bridge:
    # Two usersim backends:
    #  - local vLLM: TAU2_USER_LLM=openai/usersim + TAU2_USER_API_BASE=http://127.0.0.1:18001/v1
    #  - hosted API: TAU2_USER_LLM=openrouter/<model> + empty TAU2_USER_API_BASE
    #    (litellm routes "openrouter/*" via OPENROUTER_API_KEY from the env).
    llm_args = {"temperature": float(os.getenv("TAU2_USER_TEMPERATURE", "0.0"))}
    api_base = os.getenv("TAU2_USER_API_BASE", "").strip()
    if api_base:
        llm_args["api_base"] = api_base
        llm_args["api_key"] = os.getenv("TAU2_USER_API_KEY", "dummy")
    return Tau2Bridge(
        domain=os.getenv("TAU2_DOMAIN", "airline"),
        user_llm=os.getenv("TAU2_USER_LLM", "openai/usersim"),
        user_llm_args=llm_args,
        evaluation_type=os.getenv("TAU2_EVAL_TYPE", "all"),
        max_errors=int(os.getenv("TAU2_MAX_ERRORS", "10")),
    )


@register("tau2_agent")
class Tau2AgentLoop(ToolAgentLoop):
    # One bridge per worker process, shared across concurrent trajectories
    # (it is stateless except for read-only task/DB templates).
    _bridge: Optional[Tau2Bridge] = None

    def __init__(self, *args, **kwargs):
        # Init veRL machinery (tool_parser, response_length, turn_separator, ...).
        # We do NOT use veRL tools, so drop any passed tool list.
        kwargs["tools"] = None
        super().__init__(*args, **kwargs)

        if Tau2AgentLoop._bridge is None:
            Tau2AgentLoop._bridge = _build_bridge()
        self.bridge = Tau2AgentLoop._bridge

        # Offer the policy the tau2 tool schemas (override the empty veRL set).
        self.tool_schemas = list(self.bridge.tool_schemas)
        # Pre-build typed schema objects once for the tool parser (static set).
        self.tool_schema_objs = [OpenAIFunctionToolSchema(**s) for s in self.tool_schemas]
        # Bound how many total assistant/user turns a trajectory may take. Fall
        # back to sane defaults if the config left them unset.
        self.max_assistant_turns = self.max_assistant_turns or 30
        self.max_user_turns = self.max_user_turns or 30

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        extra_info = kwargs.get("extra_info", {}) or {}
        task_id = extra_info.get("task_id")
        if task_id is None:
            raise ValueError("tau2_agent requires extra_info.task_id in the dataset row")

        request_id = uuid4().hex
        metrics: dict[str, Any] = {}

        # Build the isolated trajectory (one blocking user-sim call) off-loop.
        with simple_timer("tau2_start", metrics):
            traj = await self.loop.run_in_executor(None, self.bridge.start_trajectory, str(task_id))

        messages = self.bridge.initial_messages(traj)
        agent_data = AgentData(
            messages=messages,
            image_data=None,
            video_data=None,
            audio_data=None,
            mm_processor_kwargs={},
            metrics=metrics,
            request_id=request_id,
            tools_kwargs={},
        )
        agent_data._active_tools = {}
        agent_data._active_tool_schemas = self.tool_schemas
        agent_data.extra_fields["tau2_traj"] = traj
        # Default terminal state: an unfinished conversation scores 0 (honest).
        agent_data.extra_fields["termination_reason"] = TerminationReason.MAX_STEPS
        # Text the policy last produced, stashed by the generating handler.
        agent_data.extra_fields["assistant_content"] = ""

        # ContextVar isolation for the async portion of this trajectory.
        # A single trajectory must never crash the whole batch: an occasional
        # user-sim hiccup / tau2 edge case is caught here, the trajectory is
        # terminated with an invalid terminal state (→ reward 0, honest), and
        # the real traceback is logged so the root cause stays visible. Without
        # this, the raised exception propagates through Ray and surfaces as an
        # opaque UnserializableException that kills the run.
        with rollout_scope(request_id, payload=traj):
            state = Tau2State.PENDING
            while state != Tau2State.TERMINATED:
                try:
                    if state == Tau2State.PENDING:
                        state = await self._handle_pending_state(agent_data, sampling_params)
                    elif state == Tau2State.GENERATING:
                        state = await self._handle_generating_state(agent_data, sampling_params)
                    elif state == Tau2State.PROCESSING_TOOLS:
                        state = await self._handle_processing_tools_state(agent_data)
                    elif state == Tau2State.PROCESSING_USER:
                        state = await self._handle_processing_user_state(agent_data)
                    else:
                        logger.error(f"Invalid state: {state}")
                        state = Tau2State.TERMINATED
                except Exception:
                    import traceback

                    logger.warning(
                        f"tau2 trajectory {request_id} failed in {state}; terminating "
                        f"gracefully (reward 0):\n{traceback.format_exc()}"
                    )
                    agent_data.extra_fields["termination_reason"] = TerminationReason.MAX_STEPS
                    state = Tau2State.TERMINATED

        # Guard the empty-response edge case (failure before any generation):
        # veRL requires a non-empty response. Emit a single eos token so the
        # sample is well-formed and simply scores 0.
        if not agent_data.response_mask:
            eos = self.tokenizer.eos_token_id or 0
            agent_data.prompt_ids.append(eos)
            agent_data.response_mask.append(1)
            if agent_data.response_logprobs is not None and agent_data.response_logprobs:
                agent_data.response_logprobs.append(0.0)

        # Score the trajectory (pure function of messages+task) off-loop.
        termination_reason = agent_data.extra_fields["termination_reason"]
        with simple_timer("tau2_reward", metrics):
            reward = await self.loop.run_in_executor(None, self.bridge.compute_reward, traj, termination_reason)

        # Finalize token-level output (identical slicing to ToolAgentLoop).
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data={},
            response_logprobs=(
                agent_data.response_logprobs[: self.response_length] if agent_data.response_logprobs else None
            ),
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            reward_score=reward["reward"],
        )
        output.extra_fields.update(
            {
                # Raw terminal task success as a top-level scalar, kept separate
                # from any shaped/process reward that may be added later.  This is
                # the unshaped tau2 evaluator verdict, so group-level filtering and
                # outcome metrics stay correct even under reward shaping.
                "outcome_binary": float(reward["reward"] >= 1.0 - 1e-6),
                "tau2_reward_breakdown": reward["reward_breakdown"],
                "tau2_termination": reward["termination"],
                "num_tau2_errors": traj.num_errors,
            }
        )
        return output

    # -- states ----------------------------------------------------------
    async def _handle_pending_state(self, agent_data, sampling_params) -> Tau2State:
        prompt_ids = await self.apply_chat_template(agent_data.messages, tools=self.tool_schemas)
        agent_data.prompt_ids = prompt_ids
        return Tau2State.GENERATING

    async def _handle_generating_state(self, agent_data, sampling_params) -> Tau2State:
        # Halt generation right after a tool call so we can act on it.
        if self.tool_parser.stop_token_ids:
            stop_token_ids = list(set((sampling_params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
            sampling_params = {**sampling_params, "stop_token_ids": stop_token_ids}

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=None,
                video_data=None,
                audio_data=None,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)  # policy tokens: trained
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Length / turn caps -> stop (leaves termination_reason = MAX_STEPS).
        if len(agent_data.response_mask) >= self.response_length:
            return Tau2State.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return Tau2State.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return Tau2State.TERMINATED

        # Parse the generated turn.
        assistant_content, tool_calls = await self.tool_parser.extract_tool_calls(
            agent_data.response_ids, self.tool_schema_objs
        )
        agent_data.tool_calls = tool_calls
        agent_data.extra_fields["assistant_content"] = assistant_content or ""

        if tool_calls:
            return Tau2State.PROCESSING_TOOLS
        # Plain text -> a message to the user.
        return Tau2State.PROCESSING_USER

    async def _handle_processing_tools_state(self, agent_data) -> Tau2State:
        traj = agent_data.extra_fields["tau2_traj"]

        # Parse veRL FunctionCall objects -> plain dicts for the bridge.
        calls: list[dict] = []
        for fc in agent_data.tool_calls[: self.max_parallel_calls]:
            try:
                args = json.loads(fc.arguments) if fc.arguments else {}
                if not isinstance(args, dict):
                    args = {}
            except (json.JSONDecodeError, TypeError):
                args = {}
            calls.append({"name": fc.name, "arguments": args, "id": fc.tool_call_id})

        responses = await self.loop.run_in_executor(None, self.bridge.execute_tool_calls, traj, calls)

        # Build tool-role messages and tokenize them as a masked turn.
        add_messages: list[dict] = []
        for fc, resp in zip(agent_data.tool_calls[: self.max_parallel_calls], responses, strict=True):
            text = resp["content"]
            if text and len(text) > self.max_tool_response_length:
                text = text[: self.max_tool_response_length] + "...(truncated)"
            msg = {"role": "tool", "content": text}
            if fc.tool_call_id is not None:
                msg["tool_call_id"] = fc.tool_call_id
            add_messages.append(msg)

        response_ids = await self.apply_chat_template(add_messages, images=None, videos=None, remove_system_prompt=True)
        response_ids = self.turn_separator + response_ids

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return Tau2State.TERMINATED
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)  # tool tokens: not trained
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.messages.extend(add_messages)

        # Too many malformed/errored tool calls -> give up (scores 0).
        if traj.num_errors >= self.bridge.max_errors:
            agent_data.extra_fields["termination_reason"] = TerminationReason.TOO_MANY_ERRORS
            return Tau2State.TERMINATED
        return Tau2State.GENERATING

    async def _handle_processing_user_state(self, agent_data) -> Tau2State:
        traj = agent_data.extra_fields["tau2_traj"]
        assistant_text = agent_data.extra_fields["assistant_content"]

        reply = await self.loop.run_in_executor(None, self.bridge.user_respond, traj, assistant_text)

        if reply["stop"]:
            # User is satisfied / transferring -> valid terminal state.
            agent_data.extra_fields["termination_reason"] = TerminationReason.USER_STOP
            return Tau2State.TERMINATED

        user_msg = {"role": "user", "content": reply["content"]}
        response_ids = await self.apply_chat_template([user_msg], images=None, videos=None, remove_system_prompt=True)
        response_ids = self.turn_separator + response_ids

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return Tau2State.TERMINATED
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)  # user tokens: not trained
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.messages.append(user_msg)
        agent_data.user_turns += 1

        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return Tau2State.TERMINATED
        return Tau2State.GENERATING
