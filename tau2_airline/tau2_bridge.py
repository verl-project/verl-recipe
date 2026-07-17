"""Bridge between tau2-bench (airline domain) and a veRL token-level agent loop.

This module contains ALL tau2 domain logic, deliberately decoupled from veRL so
it can be unit-tested on CPU with a mock LLM (see ``test_tau2_loop_offline.py``).
The veRL agent loop (``tau2_agent_loop.py``) owns tokenization / masking and
calls into this bridge for four things:

    A. build the (static) OpenAI tool schemas the policy is offered           -> build_tool_schemas()
    B. start a fresh, isolated trajectory (env + task + user simulator)        -> start_trajectory()
    C. execute a batch of tool calls against this trajectory's private env     -> execute_tool_calls()
    D. get the next user-simulator turn (or detect a STOP)                     -> user_respond()
    E. score the finished trajectory with tau2's own evaluator (pure fn)       -> compute_reward()

Design notes
------------
* Per-trajectory isolation: each :class:`Tau2Trajectory` owns a deep-copied
  ``FlightDB`` so concurrent trajectories never share mutable state. The DB is
  copied from a single template loaded once per worker (avoids re-parsing the
  7 MB ``db.json`` per sample).
* Blocking calls: the user simulator talks to an OpenAI-compatible endpoint via
  litellm, which is *synchronous*. Every method here is therefore plain sync;
  the async loop is responsible for offloading them with ``run_in_executor`` so
  it never blocks the event loop.
* Reward is a pure function of (messages, task): tau2's evaluator rebuilds a
  fresh env and replays the message trajectory to compare DB hashes, then string
  -matches ``communicate_info``. No LLM is involved for the airline domain (all
  50 base tasks use reward_basis = {DB, COMMUNICATE}), so scoring is fast and
  deterministic.
"""

from __future__ import annotations

import copy
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

# tau2 imports. ``import tau2`` triggers registry population (domains/users).
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    ToolCall,
    ToolMessage,
)
from tau2.data_model.simulation import SimulationRun, TerminationReason
from tau2.data_model.tasks import Task
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.environment import get_environment as airline_get_environment
from tau2.domains.airline.utils import AIRLINE_DB_PATH
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator import EvaluationType, evaluate_simulation
from tau2.orchestrator.orchestrator import DEFAULT_FIRST_AGENT_MESSAGE
from tau2.registry import registry
from tau2.user.user_simulator import UserSimulator

# tau2 caps a half-duplex sim's tool errors; mirror it so a policy that spams
# malformed tool calls terminates instead of looping forever.
DEFAULT_MAX_ERRORS = 10


@dataclass
class Tau2Trajectory:
    """Isolated tau2 state for one rollout sample."""

    task: Task
    env: Environment
    user_sim: UserSimulator
    user_state: Any
    # Structured tau2 message log, fed verbatim to the evaluator at the end.
    # Starts with the canned agent greeting so it matches the orchestrator's
    # trajectory exactly (the greeting is fixed text, never policy-generated).
    tau2_messages: list[Message] = field(default_factory=list)
    num_errors: int = 0

    def system_text(self) -> str:
        """The airline policy text that becomes the system prompt."""
        return self.env.get_policy()


class Tau2Bridge:
    """Worker-scoped factory + operations for tau2 airline trajectories."""

    def __init__(
        self,
        *,
        domain: str = "airline",
        user_llm: str = "openai/usersim",
        user_llm_args: Optional[dict] = None,
        evaluation_type: str = "all",
        max_errors: int = DEFAULT_MAX_ERRORS,
    ):
        assert domain == "airline", "Only the airline domain is wired up here."
        self.domain = domain
        self.user_llm = user_llm
        # e.g. {"api_base": "http://127.0.0.1:18001/v1", "api_key": "dummy",
        #       "temperature": 0.0}
        self.user_llm_args = dict(user_llm_args or {})
        self.evaluation_type = EvaluationType(evaluation_type)
        self.max_errors = max_errors

        # Load the DB template once; deep-copy per trajectory for isolation.
        self._db_template: FlightDB = FlightDB.load(AIRLINE_DB_PATH)

        # Static schemas: airline tool signatures do not depend on DB contents,
        # so build them once from a throwaway env.
        _template_env = airline_get_environment(db=copy.deepcopy(self._db_template))
        self._tool_schemas, self._tool_names = self._extract_schemas(_template_env)
        self._greeting_text = DEFAULT_FIRST_AGENT_MESSAGE.content or "Hi! How can I help you today?"

        # Cache task_id -> Task for O(1) lookup at trajectory start.
        self._tasks: dict[str, Task] = {}
        for split in (None,):  # None => all airline tasks
            for t in registry.get_tasks_loader(self.domain)(split):
                self._tasks[str(t.id)] = t
        logger.info(
            f"Tau2Bridge ready: {len(self._tasks)} {self.domain} tasks, "
            f"{len(self._tool_names)} tools, eval={self.evaluation_type.value}"
        )

    # -- A. schemas -------------------------------------------------------
    @staticmethod
    def _extract_schemas(env: Environment) -> tuple[list[dict], set[str]]:
        schemas, names = [], set()
        for tool in env.get_tools():
            schemas.append(tool.openai_schema)  # {"type":"function","function":{...}}
            names.add(tool.name)
        return schemas, names

    @property
    def tool_schemas(self) -> list[dict]:
        return self._tool_schemas

    @property
    def tool_names(self) -> set[str]:
        return self._tool_names

    @property
    def greeting_text(self) -> str:
        return self._greeting_text

    # -- B. trajectory start ---------------------------------------------
    def start_trajectory(self, task_id: str) -> Tau2Trajectory:
        """Create a fresh, isolated trajectory for ``task_id``.

        Builds a private env (deep-copied DB + task initial state), a user
        simulator seeded from the task scenario, and primes the conversation
        with the canned greeting + the user's first (LLM-generated) request.

        NOTE: this performs ONE blocking user-simulator LLM call. Call it from
        an executor thread.
        """
        task = self._tasks.get(str(task_id))
        if task is None:
            raise KeyError(f"Unknown airline task_id: {task_id!r}")

        env = airline_get_environment(db=copy.deepcopy(self._db_template))

        # Apply the task's initial state exactly like Orchestrator.initialize().
        init = task.initial_state
        if init is not None:
            env.set_state(
                initialization_data=init.initialization_data,
                initialization_actions=init.initialization_actions,
                message_history=list(init.message_history or []),
            )

        # User simulator: airline tasks have no user tools, so tools=None.
        try:
            user_tools = env.get_user_tools(include=task.user_tools) or None
        except Exception:
            user_tools = None
        user_sim = UserSimulator(
            llm=self.user_llm,
            instructions=str(task.user_scenario),
            tools=user_tools,
            llm_args=dict(self.user_llm_args),
        )
        user_state = user_sim.get_init_state(message_history=[])

        # Seed: agent greets (fixed text), user responds with the real request.
        greeting = AssistantMessage(role="assistant", content=self._greeting_text)
        user_msg, user_state = user_sim.generate_next_message(greeting, user_state)

        traj = Tau2Trajectory(
            task=task,
            env=env,
            user_sim=user_sim,
            user_state=user_state,
            tau2_messages=[greeting, user_msg],
        )
        return traj

    def initial_messages(self, traj: Tau2Trajectory) -> list[dict]:
        """OpenAI-format seed messages the policy conditions on for turn 1.

        [system(airline policy), assistant(greeting), user(first request)].
        The greeting is included for fidelity with tau2's orchestrator; it is
        masked (role != assistant-generated) on the token side.
        """
        first_user = traj.tau2_messages[-1]
        return [
            {"role": "system", "content": traj.system_text()},
            {"role": "assistant", "content": self._greeting_text},
            {"role": "user", "content": first_user.content or ""},
        ]

    # -- C. tool execution ------------------------------------------------
    def execute_tool_calls(self, traj: Tau2Trajectory, calls: list[dict]) -> list[dict]:
        """Run parsed tool calls against this trajectory's private env.

        ``calls`` is a list of ``{"name": str, "arguments": dict, "id": str}``.
        Records one AssistantMessage(tool_calls=...) followed by one ToolMessage
        per call into ``traj.tau2_messages`` (ids matched, as tau2's evaluator
        requires), and returns ``[{"content": str, "error": bool}, ...]`` for the
        loop to tokenize as (masked) tool responses.
        """
        tool_calls: list[ToolCall] = []
        for c in calls:
            tool_calls.append(
                ToolCall(
                    id=c.get("id") or uuid.uuid4().hex,
                    name=c["name"],
                    arguments=c.get("arguments") or {},
                    requestor="assistant",
                )
            )
        # One assistant message carrying all tool calls (content=None: tau2
        # forbids a message with both text and tool_calls).
        traj.tau2_messages.append(AssistantMessage(role="assistant", content=None, tool_calls=tool_calls))

        responses: list[dict] = []
        for tc in tool_calls:
            tool_msg: ToolMessage = traj.env.get_response(tc)  # handles errors + JSON
            traj.tau2_messages.append(tool_msg)
            if tool_msg.error:
                traj.num_errors += 1
            responses.append({"content": tool_msg.content or "", "error": bool(tool_msg.error)})
        return responses

    # -- D. user turn -----------------------------------------------------
    def user_respond(self, traj: Tau2Trajectory, assistant_text: str) -> dict:
        """Record the policy's text turn, then get the user simulator's reply.

        Returns ``{"content": str, "stop": bool}``. ``stop`` is True when the
        user emitted a STOP/TRANSFER/OUT_OF_SCOPE signal (conversation over).

        NOTE: one blocking user-simulator LLM call. Run in an executor.
        """
        agent_msg = AssistantMessage(role="assistant", content=assistant_text or "")
        traj.tau2_messages.append(agent_msg)

        user_msg, traj.user_state = traj.user_sim.generate_next_message(agent_msg, traj.user_state)
        traj.tau2_messages.append(user_msg)
        stop = UserSimulator.is_stop(user_msg)
        return {"content": user_msg.content or "", "stop": stop}

    # -- E. reward --------------------------------------------------------
    def compute_reward(self, traj: Tau2Trajectory, termination_reason: TerminationReason) -> dict:
        """Score the finished trajectory with tau2's evaluator.

        Returns ``{"reward": float, "reward_breakdown": dict, "termination": str}``.
        If the trajectory did not end in a valid terminal state
        (AGENT_STOP / USER_STOP), tau2 returns reward 0.0 by design -- we mirror
        that here rather than inventing a score.
        """
        sim = SimulationRun(
            id=uuid.uuid4().hex,
            task_id=str(traj.task.id),
            start_time="1970-01-01T00:00:00",
            end_time="1970-01-01T00:00:00",
            duration=0.0,
            termination_reason=termination_reason,
            messages=list(traj.tau2_messages),
            reward_info=None,
        )
        try:
            reward_info = evaluate_simulation(
                simulation=sim,
                task=traj.task,
                evaluation_type=self.evaluation_type,
                solo_mode=False,
                domain=self.domain,
            )
        except Exception as e:  # never let a scoring bug crash the rollout
            logger.warning(f"tau2 reward evaluation failed for task {traj.task.id}: {e}")
            return {"reward": 0.0, "reward_breakdown": {}, "termination": termination_reason.value}

        breakdown = {}
        if reward_info.reward_breakdown:
            breakdown = {str(k): float(v) for k, v in reward_info.reward_breakdown.items()}
        return {
            "reward": float(reward_info.reward),
            "reward_breakdown": breakdown,
            "termination": termination_reason.value,
        }
