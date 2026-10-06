from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
    ToolListWrap,
)
from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.utils.rollout_trace import rollout_trace_op

from .trajectory import Snapshot, Trajectory

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except ValueError:
        logger.warning("Invalid %s; using default %s", name, default)
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except ValueError:
        logger.warning("Invalid %s; using default %s", name, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Invalid %s; using default %s", name, default)
    return default


MAX_TURNS = _env_int("MAX_TURNS", 32)
MAX_TOKENS_PER_GENERATION = _env_int("MAX_TOKENS_PER_GENERATION", 2048)
MAX_TERMINAL_OUTPUT_CHARS = _env_int("MAX_TERMINAL_OUTPUT_CHARS", 10000)
STARTUP_TIMEOUT = _env_float("STARTUP_TIMEOUT", 120.0)
AGENT_TIMEOUT = _env_float("AGENT_TIMEOUT", 360.0)
VERIFIER_TIMEOUT = _env_float("VERIFIER_TIMEOUT", 120.0)
MODAL_SANDBOX_TIMEOUT = _env_float(
    "MODAL_SANDBOX_TIMEOUT",
    AGENT_TIMEOUT + VERIFIER_TIMEOUT + 60.0,
    minimum=1.0,
)
DAYTONA_SANDBOX_TIMEOUT = _env_float(
    "DAYTONA_SANDBOX_TIMEOUT",
    AGENT_TIMEOUT + VERIFIER_TIMEOUT + 60.0,
    minimum=1.0,
)
SANDBOX_PROVIDER = os.environ.get("SANDBOX_PROVIDER", "modal").strip().lower()
SPECULATION_DEPTH = _env_int("SPECULATION_DEPTH", 1)
SPECULATION_BREADTH = _env_int("SPECULATION_BREADTH", 1)
MAX_SPEC_OBS_TOKENS = _env_int("MAX_SPEC_OBS_TOKENS", 2048)
ENABLE_SPECULATION = _env_bool("ENABLE_SPECULATION", True)
ENABLE_THINKING = _env_bool("ENABLE_THINKING", False)
WORLD_MODEL_WARMUP_STEPS = _env_int("WORLD_MODEL_WARMUP_STEPS", 0, minimum=0)
SPECULATE_DURING_VALIDATION = _env_bool("SPECULATE_DURING_VALIDATION", False)
NO_TOOL_CALL_ERROR = """No valid tool call.

To run a command:
<tool_call>
<function=bash>
<parameter=command>
...
</parameter>
</function>
</tool_call>

To finish:
<tool_call>
<function=done>
</function>
</tool_call>"""
_NAN = float("nan")


def _sandbox_environment_factory(provider: str) -> Callable[[bytes], Any]:
    if provider == "modal":
        try:
            from .modal_sandbox import ModalSandboxEnvironment
        except ModuleNotFoundError as exc:
            if exc.name == "modal":
                raise RuntimeError("SANDBOX_PROVIDER=modal requires the 'modal' package") from exc
            raise
        return lambda task_binary: ModalSandboxEnvironment(
            task_binary,
            sandbox_timeout=MODAL_SANDBOX_TIMEOUT,
        )
    if provider == "daytona":
        try:
            from .daytona_sandbox import DaytonaSandboxEnvironment
        except ModuleNotFoundError as exc:
            if exc.name == "daytona":
                raise RuntimeError("SANDBOX_PROVIDER=daytona requires the 'daytona' package") from exc
            raise
        return lambda task_binary: DaytonaSandboxEnvironment(
            task_binary,
            sandbox_timeout=DAYTONA_SANDBOX_TIMEOUT,
        )
    raise ValueError(f"Unknown SANDBOX_PROVIDER={provider!r}; expected 'modal' or 'daytona'")


def _mmm(prefix: str, values: list[float]) -> dict[str, float]:
    if not values:
        return {f"{prefix}/min": _NAN, f"{prefix}/mean": _NAN, f"{prefix}/max": _NAN}
    return {
        f"{prefix}/min": float(min(values)),
        f"{prefix}/mean": float(sum(values) / len(values)),
        f"{prefix}/max": float(max(values)),
    }


def _request_metrics(rows: list[dict[str, Any]], prefix: str) -> dict[str, float]:
    rows = [row for row in rows if row]

    def column(name: str) -> list[float]:
        return [float(row[name]) for row in rows if row.get(name) is not None]

    result: dict[str, float] = {}
    result.update(_mmm(f"{prefix}queue_sec_per_call", column("queue_sec")))
    result.update(_mmm(f"{prefix}prefill_sec_per_call", column("prefill_sec")))
    result.update(_mmm(f"{prefix}decode_sec_per_call", column("decode_sec")))

    prefill_tps = []
    decode_tps = []
    cache_frac = []
    for row in rows:
        prompt = row.get("prompt_tokens")
        cached = row.get("cached_tokens")
        generated = row.get("gen_tokens")
        prefill = row.get("prefill_sec")
        decode = row.get("decode_sec")
        if prompt is not None and cached is not None and prefill:
            prefill_tps.append((prompt - cached) / prefill)
        if generated is not None and decode:
            decode_tps.append(max(generated - 1, 0) / decode)
        if prompt:
            cache_frac.append((cached or 0) / prompt)
    result.update(_mmm(f"{prefix}prefill_toks_per_sec", prefill_tps))
    result.update(_mmm(f"{prefix}decode_toks_per_sec", decode_tps))
    result.update(_mmm(f"{prefix}cache_hit_frac", cache_frac))
    result[f"{prefix}request_count"] = float(len(rows))
    result[f"{prefix}completed_request_count"] = float(len(rows))

    prompt_tokens = sum(column("prompt_tokens"))
    cached_tokens = sum(column("cached_tokens"))
    generated_tokens = sum(column("gen_tokens"))
    prefill_sec = sum(column("prefill_sec"))
    decode_sec = sum(column("decode_sec"))
    result[f"{prefix}prefill_toks_per_sec/aggregate"] = (
        (prompt_tokens - cached_tokens) / prefill_sec if prefill_sec else _NAN
    )
    result[f"{prefix}decode_toks_per_sec/aggregate"] = (
        max(generated_tokens - len(rows), 0.0) / decode_sec if decode_sec else _NAN
    )
    result[f"{prefix}cache_hit_frac/aggregate"] = cached_tokens / prompt_tokens if prompt_tokens else _NAN
    result[f"{prefix}completed_generated_tokens_total"] = float(generated_tokens)
    return result


@dataclass
class PendingTurn:
    kind: str
    snap_pre_action: Snapshot
    snap_post_turn: Snapshot | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    snap_pre_observation: Snapshot | None = None
    command: dict[str, Any] | None = None
    draft_text: str = ""
    exec_task: asyncio.Task | None = None
    spec_task: asyncio.Task | None = None
    spec_start: float | None = None
    exec_start: float | None = None
    exec_end: float | None = None
    action_result: ActionResult | None = None


@dataclass(frozen=True)
class ActionResult:
    token_ids: list[int]
    logprobs: list[float]
    trace: dict[str, Any]
    output_extra: dict[str, Any]
    completed_at: float


@dataclass(frozen=True)
class WorldModelResult:
    token_ids: list[int]
    request_metrics: dict[str, Any]
    call_sec: float
    max_tokens: int
    completed_at: float


@dataclass(frozen=True)
class ExecutionResult:
    observation: str
    outcome: str
    completed_at: float


@dataclass(frozen=True)
class ActionProducer:
    task: asyncio.Task
    snap_pre_action: Snapshot
    depth: int


@dataclass
class BreadthCandidate:
    observation: str
    ranks: list[int]
    wm_token_count: int
    policy_prompt_ids: list[int]
    policy_task: asyncio.Task | None


@dataclass(frozen=True)
class TaskInfo:
    phase: str
    started_at: float


@dataclass
class RolloutStats:
    turns: list[dict[str, Any]] = field(default_factory=list)
    all_turns: list[dict[str, Any]] = field(default_factory=list)
    policy_requests: list[dict[str, Any]] = field(default_factory=list)
    wm_requests: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "error"
    verifier_error: str | None = None
    setup_sec: float | None = None
    agent_sec: float | None = None
    verifier_sec: float | None = None
    pipeline_sec: float | None = None
    block_sec: float = 0.0
    overlap_sec: float = 0.0
    frontier_depth_area: float = 0.0
    max_frontier_depth: int = 0
    cancellation_counts: dict[str, int] = field(default_factory=dict)
    cancellation_secs: dict[str, float] = field(default_factory=dict)
    phase_secs: dict[str, float] = field(default_factory=dict)
    exact_with_descendant_work: int = 0
    exact_with_action_ready: int = 0
    exact_with_policy_inflight: int = 0
    exact_with_active_descendant_spec: int = 0
    tasks_remaining_at_exit: int = 0


class TaskRegistry:
    def __init__(self, stats: RolloutStats) -> None:
        self._stats = stats
        self._tasks: dict[asyncio.Task, TaskInfo] = {}

    def create(self, awaitable: Any, phase: str) -> asyncio.Task:
        task = asyncio.create_task(awaitable)
        self._tasks[task] = TaskInfo(phase=phase, started_at=time.monotonic())
        return task

    def tasks(self) -> set[asyncio.Task]:
        return set(self._tasks)

    def count(self) -> int:
        return len(self._tasks)

    def phase(self, task: asyncio.Task) -> str | None:
        info = self._tasks.get(task)
        return info.phase if info is not None else None

    def active_phases(self) -> set[str]:
        return {info.phase for task, info in self._tasks.items() if not task.done()}

    def result(self, task: asyncio.Task) -> Any:
        self._tasks.pop(task, None)
        return task.result()

    async def cancel(self, task: asyncio.Task | None, reason: str) -> None:
        if task is None:
            return
        info = self._tasks.get(task)
        if info is None:
            await asyncio.gather(task, return_exceptions=True)
            return
        if not task.done():
            key = f"{reason}/{info.phase}"
            self._stats.cancellation_counts[key] = self._stats.cancellation_counts.get(key, 0) + 1
            self._stats.cancellation_secs[key] = self._stats.cancellation_secs.get(key, 0.0) + (
                time.monotonic() - info.started_at
            )
            task.cancel()
        try:
            await asyncio.gather(task, return_exceptions=True)
        finally:
            self._tasks.pop(task, None)

    async def cancel_all(self, reason: str) -> None:
        await asyncio.gather(
            *(self.cancel(task, reason) for task in list(self._tasks)),
            return_exceptions=True,
        )


class TerminalObsSpecAgentLoop(AgentLoopBase):
    def __init__(
        self,
        *args,
        tools: ToolListWrap | None = None,
        wm_server_manager=None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.eval_report_level = str(self.config.trainer.get("eval_report_level", "off"))
        self.wm_server_manager = wm_server_manager
        self.sandbox_provider = SANDBOX_PROVIDER
        self._create_sandbox_environment = _sandbox_environment_factory(self.sandbox_provider)
        if ENABLE_SPECULATION and self.wm_server_manager is None:
            raise ValueError("ENABLE_SPECULATION requires a separate world_model_actor")
        if SPECULATION_BREADTH > 1 and SPECULATION_DEPTH != 1:
            raise ValueError("SPECULATION_BREADTH > 1 currently requires SPECULATION_DEPTH=1")
        tool_list = tools.tools if tools else []
        self.tool_schemas = [tool.tool_schema for tool in tool_list]
        self.tool_schema_dicts = [
            schema.model_dump(exclude_unset=True, exclude_none=True) for schema in self.tool_schemas
        ]
        self.tool_parser = ToolParser.get_tool_parser(
            self.rollout_config.multi_turn.format,
            self.tokenizer,
        )
        self.response_length = self.rollout_config.response_length

    def _detailed_reporting(self) -> bool:
        return self.eval_report_level == "detailed" and bool(getattr(self, "_validate", False))

    def _make_trajectory(self, raw_prompt: list[dict[str, Any]]) -> Trajectory:
        return Trajectory(
            self.tokenizer,
            raw_prompt,
            tools=self.tool_schema_dicts,
            enable_thinking=ENABLE_THINKING,
            spec=True,
            apply_chat_template_kwargs=self.apply_chat_template_kwargs,
        )

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        trajectory = self._make_trajectory(list(kwargs["raw_prompt"]))
        environment = self._create_sandbox_environment(kwargs["extra_info"]["task_binary"])
        stats = RolloutStats()
        reward = 0.0
        num_turns = 0
        extra_fields: dict[str, Any] = {}
        if "global_steps" in kwargs:
            global_steps = int(kwargs["global_steps"])
            extra_fields["min_global_steps"] = global_steps
            extra_fields["max_global_steps"] = global_steps

        try:
            started_at = time.monotonic()
            try:
                await asyncio.wait_for(environment.setup(), timeout=STARTUP_TIMEOUT)
            except asyncio.TimeoutError:
                stats.stop_reason = "setup_timeout"
                return self._build_output(trajectory, reward, num_turns, extra_fields, stats)
            except Exception as exc:
                logger.warning("Terminal environment setup failed: %s", exc, exc_info=True)
                stats.stop_reason = "setup_error"
                return self._build_output(trajectory, reward, num_turns, extra_fields, stats)
            stats.setup_sec = time.monotonic() - started_at

            started_at = time.monotonic()
            try:
                num_turns, stats.stop_reason = await asyncio.wait_for(
                    self._run_turns(
                        trajectory,
                        uuid4().hex,
                        environment,
                        sampling_params,
                        extra_fields,
                        stats,
                    ),
                    timeout=AGENT_TIMEOUT,
                )
            except asyncio.TimeoutError:
                stats.stop_reason = "agent_timeout"
            except Exception as exc:
                logger.warning("Terminal agent interaction failed: %s", exc, exc_info=True)
                stats.stop_reason = "error"
            finally:
                stats.agent_sec = time.monotonic() - started_at

            if stats.stop_reason == "done":
                started_at = time.monotonic()
                try:
                    verifier_reward, stats.verifier_error = await asyncio.wait_for(
                        environment.run_verifier(timeout=VERIFIER_TIMEOUT),
                        timeout=VERIFIER_TIMEOUT,
                    )
                    reward = float(verifier_reward)
                except asyncio.TimeoutError:
                    stats.verifier_error = "verifier_timeout"
                except Exception as exc:
                    logger.warning("Terminal verifier failed: %s", exc, exc_info=True)
                    stats.verifier_error = "verifier_error"
                finally:
                    stats.verifier_sec = time.monotonic() - started_at
        finally:
            try:
                await environment.cleanup()
            except Exception as exc:
                logger.warning("Terminal environment cleanup failed: %s", exc)

        return self._build_output(trajectory, reward, num_turns, extra_fields, stats)

    def _build_output(
        self,
        trajectory: Trajectory,
        reward: float,
        num_turns: int,
        extra_fields: dict[str, Any],
        stats: RolloutStats,
    ) -> AgentLoopOutput:
        response_ids = trajectory.ids[: self.response_length]
        response_mask = trajectory.mask[: self.response_length]
        world_mask = trajectory.world_mask[: self.response_length]
        response_logprobs = trajectory.logprobs[: self.response_length]
        if not response_ids:
            response_ids = [self.tokenizer.eos_token_id or 0]
            response_mask = [0]
            world_mask = [0]
            response_logprobs = [0.0]

        preemptions = [turn["num_preempted"] for turn in stats.turns if turn.get("num_preempted") is not None]
        if self._detailed_reporting():
            extra_fields["agent_trace"] = self._agent_trace(stats)
        return AgentLoopOutput(
            prompt_ids=trajectory.prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=reward,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(
                generate_sequences=sum(turn.get("gen_sec", 0.0) for turn in stats.turns),
                tool_calls=sum(turn.get("exec_sec") or 0.0 for turn in stats.turns),
                compute_score=stats.verifier_sec or 0.0,
                num_preempted=sum(preemptions) if preemptions else -1,
            ),
            extra_fields={
                **extra_fields,
                "world_loss_masks": world_mask,
                "stop_reason": stats.stop_reason,
                "correct": reward >= 0.5,
                "agent_metrics": self._agent_metrics(
                    stats,
                    reward >= 0.5,
                    response_mask,
                    world_mask,
                ),
            },
        )

    async def _run_turns(
        self,
        trajectory: Trajectory,
        request_id: str,
        environment: Any,
        sampling_params: dict[str, Any],
        extra_fields: dict[str, Any],
        stats: RolloutStats,
    ) -> tuple[int, str]:
        if self._speculation_active():
            if SPECULATION_BREADTH > 1:
                return await self._run_breadth_speculative(
                    trajectory,
                    request_id,
                    environment,
                    sampling_params,
                    extra_fields,
                    stats,
                )
            return await self._run_speculative(
                trajectory,
                request_id,
                environment,
                sampling_params,
                extra_fields,
                stats,
            )
        return await self._run_sequential(
            trajectory,
            request_id,
            environment,
            sampling_params,
            extra_fields,
            stats,
        )

    def _speculation_active(self) -> bool:
        if not ENABLE_SPECULATION:
            return False
        if getattr(self, "_validate", False):
            return SPECULATE_DURING_VALIDATION
        return int(getattr(self, "_global_steps", 0)) > WORLD_MODEL_WARMUP_STEPS

    async def _run_sequential(
        self,
        trajectory: Trajectory,
        request_id: str,
        environment: Any,
        sampling_params: dict[str, Any],
        extra_fields: dict[str, Any],
        stats: RolloutStats,
    ) -> tuple[int, str]:
        previous_exec_end = None
        for turn_index in range(MAX_TURNS):
            produced = await self._generate_action(
                trajectory,
                request_id,
                sampling_params,
                extra_fields,
                stats,
                depth=0,
            )
            if produced is None:
                return turn_index, "max_response_length"
            token_ids, trace = produced
            _, tool_calls = await self.tool_parser.extract_tool_calls(
                token_ids,
                self.tool_schemas,
            )
            if not tool_calls:
                trace["parse_error"] = True
                stats.turns.append(trace)
                trajectory.add_observation(NO_TOOL_CALL_ERROR, is_env=False)
                continue

            tool_call = tool_calls[0]
            if tool_call.name == "done":
                stats.turns.append(trace)
                return turn_index + 1, "done"
            arguments = self._bash_arguments(tool_call)
            if arguments is None:
                trace["parse_error"] = True
                stats.turns.append(trace)
                trajectory.add_observation(NO_TOOL_CALL_ERROR, is_env=False)
                continue

            if self._detailed_reporting():
                trace["command"] = dict(arguments)
            exec_start = time.monotonic()
            if previous_exec_end is not None:
                trace["inter_exec_gap_sec"] = exec_start - previous_exec_end
            observation, outcome = await self._execute(environment, arguments)
            previous_exec_end = time.monotonic()
            trace["exec_sec"] = previous_exec_end - exec_start
            trace["exec_outcome"] = outcome
            if self._detailed_reporting():
                trace["observation"] = observation
            stats.turns.append(trace)
            trajectory.add_observation(observation, is_env=True)
        return MAX_TURNS, "max_turns"

    async def _request_action(
        self,
        prompt_ids: list[int],
        request_id: str,
        sampling_params: dict[str, Any],
        stats: RolloutStats,
        *,
        depth: int,
        max_tokens: int,
    ) -> ActionResult:
        started_at = time.monotonic()
        output = await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params={
                **sampling_params,
                "max_tokens": max_tokens,
                "logprobs": 1,
            },
        )
        completed_at = time.monotonic()
        gen_sec = completed_at - started_at
        output_extra = dict(output.extra_fields or {})
        request_metrics = dict(output_extra.pop("request_metrics", {}))
        token_ids = list(output.token_ids)[:max_tokens]
        request_metrics.setdefault("gen_tokens", len(token_ids))
        request_metrics["call_sec"] = gen_sec
        policy_request_index = len(stats.policy_requests)
        if self._detailed_reporting():
            request_metrics["request_index"] = policy_request_index
        stats.policy_requests.append(request_metrics)

        logprobs = list(output.log_probs or [])[: len(token_ids)]
        logprobs.extend([0.0] * (len(token_ids) - len(logprobs)))
        trace = {
            "gen_sec": gen_sec,
            "gen_tokens": len(token_ids),
            "num_preempted": getattr(output, "num_preempted", None),
            "budget_hit": len(token_ids) >= MAX_TOKENS_PER_GENERATION,
            "parse_error": False,
            "exec_sec": None,
            "inter_exec_gap_sec": None,
            "exec_outcome": None,
            "spec_attempted": False,
            "spec_exec_won": False,
            "spec_failed": False,
            "spec_match": False,
            "spec_sec": None,
            "spec_obs_tokens": 0,
            "spec_decode_truncated": False,
            "spec_obs_overran": False,
            "spec_depth": depth,
        }
        if self._detailed_reporting():
            trace["policy_request_index"] = policy_request_index
            trace["action"] = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        stats.all_turns.append(trace)
        return ActionResult(
            token_ids=token_ids,
            logprobs=logprobs,
            trace=trace,
            output_extra=output_extra,
            completed_at=completed_at,
        )

    @staticmethod
    def _merge_output_steps(extra_fields: dict[str, Any], output_extra: dict[str, Any]) -> None:
        for key in ("min_global_steps", "max_global_steps"):
            value = output_extra.get(key)
            if value is None:
                continue
            existing = extra_fields.get(key)
            if existing is None:
                extra_fields[key] = value
            elif key == "min_global_steps":
                extra_fields[key] = min(existing, value)
            else:
                extra_fields[key] = max(existing, value)

    @staticmethod
    def _apply_action_result(trajectory: Trajectory, result: ActionResult) -> None:
        trajectory.add_generation(result.token_ids, result.logprobs)

    async def _generate_action(
        self,
        trajectory: Trajectory,
        request_id: str,
        sampling_params: dict[str, Any],
        extra_fields: dict[str, Any],
        stats: RolloutStats,
        *,
        depth: int,
    ) -> tuple[list[int], dict[str, Any]] | None:
        max_tokens = min(
            MAX_TOKENS_PER_GENERATION,
            self.response_length - len(trajectory.ids),
        )
        if max_tokens <= 0:
            return None
        result = await self._request_action(
            list(trajectory.tokens),
            request_id,
            sampling_params,
            stats,
            depth=depth,
            max_tokens=max_tokens,
        )
        self._apply_action_result(trajectory, result)
        self._merge_output_steps(extra_fields, result.output_extra)
        return result.token_ids, result.trace

    async def _request_world_model(
        self,
        prompt_ids: list[int],
        max_tokens: int,
        stats: RolloutStats,
        *,
        candidate_rank: int = 0,
        temperature: float = 0.0,
    ) -> WorldModelResult:
        params: dict[str, Any] = {
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": -1,
            "n": 1,
            "logprobs": False,
            "max_tokens": max_tokens,
            "stop": ["</tool_response>"],
        }
        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(im_end_id, int) and im_end_id >= 0:
            params["stop_token_ids"] = [im_end_id]
        started_at = time.monotonic()
        output = await self.wm_server_manager.generate(
            request_id=f"wm-{uuid4().hex}",
            prompt_ids=prompt_ids,
            sampling_params=params,
        )
        completed_at = time.monotonic()
        call_sec = completed_at - started_at
        request_metrics = dict((output.extra_fields or {}).get("request_metrics", {}))
        token_ids = list(output.token_ids)
        request_metrics.setdefault("gen_tokens", len(token_ids))
        request_metrics["call_sec"] = call_sec
        request_metrics["candidate_rank"] = candidate_rank
        request_metrics["sampling_temperature"] = temperature
        wm_request_index = len(stats.wm_requests)
        if self._detailed_reporting():
            request_metrics["request_index"] = wm_request_index
        stats.wm_requests.append(request_metrics)
        return WorldModelResult(
            token_ids=token_ids,
            request_metrics=request_metrics,
            call_sec=call_sec,
            max_tokens=max_tokens,
            completed_at=completed_at,
        )

    def _decode_draft(self, token_ids: list[int]) -> str:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
        return text.split("</tool_response>")[0]

    def _draft_overran(self, token_ids: list[int]) -> bool:
        text = self.tokenizer.decode(token_ids, skip_special_tokens=False)
        return not text.rstrip().endswith("</tool_response>")

    def _bash_arguments(self, tool_call: Any) -> dict[str, Any] | None:
        if tool_call.name != "bash":
            return None
        try:
            arguments = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError):
            return None
        return arguments if arguments.get("command") else None

    async def _timed_execute(
        self,
        environment: Any,
        arguments: dict[str, Any],
    ) -> ExecutionResult:
        observation, outcome = await self._execute(environment, arguments)
        return ExecutionResult(observation=observation, outcome=outcome, completed_at=time.monotonic())

    def _start_execution(
        self,
        turn: PendingTurn,
        environment: Any,
        registry: TaskRegistry,
        previous_exec_end: float | None,
    ) -> None:
        turn.exec_start = time.monotonic()
        if previous_exec_end is not None:
            turn.trace["inter_exec_gap_sec"] = turn.exec_start - previous_exec_end
        turn.exec_task = registry.create(
            self._timed_execute(environment, turn.command or {}),
            "exec",
        )

    def _start_frontier_speculation(
        self,
        trajectory: Trajectory,
        turn: PendingTurn,
        stats: RolloutStats,
        registry: TaskRegistry,
    ) -> None:
        if turn.kind != "command_frontier":
            raise RuntimeError(f"Cannot start speculation for turn kind {turn.kind}")
        wm_prompt = trajectory.get_spec_prompt()
        max_tokens = max(
            1,
            min(MAX_SPEC_OBS_TOKENS, self.response_length - len(trajectory.ids)),
        )
        turn.kind = "command_speculating"
        turn.spec_start = time.monotonic()
        turn.trace["spec_attempted"] = True
        turn.spec_task = registry.create(
            self._request_world_model(wm_prompt, max_tokens, stats),
            "wm",
        )

    async def _materialize_action(
        self,
        trajectory: Trajectory,
        producer: ActionProducer,
        result: ActionResult,
        stats: RolloutStats,
        registry: TaskRegistry,
    ) -> PendingTurn:
        if trajectory.snapshot() != producer.snap_pre_action:
            raise RuntimeError("Trajectory changed while policy generation was in flight")
        self._apply_action_result(trajectory, result)
        trace = result.trace
        _, tool_calls = await self.tool_parser.extract_tool_calls(result.token_ids, self.tool_schemas)
        if not tool_calls:
            trace["parse_error"] = True
            trajectory.add_observation(NO_TOOL_CALL_ERROR, is_env=False)
            return PendingTurn(
                "synthetic",
                producer.snap_pre_action,
                trajectory.snapshot(),
                trace=trace,
                action_result=result,
            )
        tool_call = tool_calls[0]
        if tool_call.name == "done":
            return PendingTurn(
                "done",
                producer.snap_pre_action,
                trajectory.snapshot(),
                trace=trace,
                action_result=result,
            )
        arguments = self._bash_arguments(tool_call)
        if arguments is None:
            trace["parse_error"] = True
            trajectory.add_observation(NO_TOOL_CALL_ERROR, is_env=False)
            return PendingTurn(
                "synthetic",
                producer.snap_pre_action,
                trajectory.snapshot(),
                trace=trace,
                action_result=result,
            )
        if self._detailed_reporting():
            trace["command"] = dict(arguments)

        snap_pre_observation = trajectory.snapshot()
        if producer.depth >= SPECULATION_DEPTH:
            return PendingTurn(
                "command_frontier",
                producer.snap_pre_action,
                snap_pre_observation,
                trace=trace,
                snap_pre_observation=snap_pre_observation,
                command=arguments,
                action_result=result,
            )

        wm_prompt = trajectory.get_spec_prompt()
        max_tokens = max(
            1,
            min(MAX_SPEC_OBS_TOKENS, self.response_length - len(trajectory.ids)),
        )
        spec_start = time.monotonic()
        trace["spec_attempted"] = True
        return PendingTurn(
            "command_speculating",
            producer.snap_pre_action,
            None,
            trace=trace,
            snap_pre_observation=snap_pre_observation,
            command=arguments,
            spec_task=registry.create(
                self._request_world_model(wm_prompt, max_tokens, stats),
                "wm",
            ),
            spec_start=spec_start,
            action_result=result,
        )

    def _finish_speculation(
        self,
        trajectory: Trajectory,
        turn: PendingTurn,
        result: WorldModelResult,
    ) -> None:
        draft_text = self._decode_draft(result.token_ids)
        turn.trace.update(
            {
                "spec_sec": result.completed_at - (turn.spec_start or result.completed_at),
                "spec_call_sec": result.call_sec,
                "spec_obs_tokens": len(result.token_ids),
                "spec_decode_truncated": len(result.token_ids) >= result.max_tokens,
                "spec_obs_overran": self._draft_overran(result.token_ids),
            }
        )
        if self._detailed_reporting():
            turn.trace["wm_request_index"] = result.request_metrics["request_index"]
            turn.trace["draft_observation"] = draft_text
        trajectory.add_observation(draft_text, is_env=True)
        turn.kind = "command"
        turn.draft_text = draft_text
        turn.snap_post_turn = trajectory.snapshot()
        turn.spec_task = None

    @staticmethod
    def _mark_speculation_failed(turn: PendingTurn) -> None:
        turn.trace["spec_failed"] = True
        turn.kind = "command_deferred"
        turn.spec_task = None

    def _commit_turn(
        self,
        turn: PendingTurn,
        stats: RolloutStats,
        extra_fields: dict[str, Any],
    ) -> None:
        stats.turns.append(turn.trace)
        if turn.action_result is not None:
            self._merge_output_steps(extra_fields, turn.action_result.output_extra)

    @staticmethod
    def _record_wait_phase(stats: RolloutStats, phases: set[str], elapsed: float) -> None:
        ordered = [phase for phase in ("exec", "policy", "wm") if phase in phases]
        key = "_and_".join(ordered) if ordered else "idle"
        stats.phase_secs[key] = stats.phase_secs.get(key, 0.0) + elapsed
        if "exec" in phases:
            if "policy" in phases or "wm" in phases:
                stats.overlap_sec += elapsed
            else:
                stats.block_sec += elapsed

    async def _cancel_descendants(
        self,
        pending: deque[PendingTurn],
        producer: ActionProducer | None,
        registry: TaskRegistry,
        reason: str,
    ) -> None:
        if producer is not None:
            await registry.cancel(producer.task, reason)
        for turn in list(pending)[1:]:
            await registry.cancel(turn.spec_task, reason)
            await registry.cancel(turn.exec_task, reason)

    async def _run_breadth_speculative(
        self,
        trajectory: Trajectory,
        request_id: str,
        environment: Any,
        sampling_params: dict[str, Any],
        extra_fields: dict[str, Any],
        stats: RolloutStats,
    ) -> tuple[int, str]:
        if SPECULATION_DEPTH != 1:
            raise ValueError("Breadth speculation currently requires SPECULATION_DEPTH=1")

        registry = TaskRegistry(stats)
        committed_snapshot = trajectory.snapshot()
        prefetched_action: ActionResult | None = None
        committed_turns = 0
        cleanup_reason = "teardown"
        pipeline_start = time.monotonic()
        last_exec_end = None

        try:
            while committed_turns < MAX_TURNS:
                if prefetched_action is None:
                    max_tokens = min(
                        MAX_TOKENS_PER_GENERATION,
                        self.response_length - len(trajectory.ids),
                    )
                    if max_tokens <= 0:
                        return committed_turns, "max_response_length"
                    action_result = await self._request_action(
                        list(trajectory.tokens),
                        request_id,
                        sampling_params,
                        stats,
                        depth=0,
                        max_tokens=max_tokens,
                    )
                else:
                    action_result = prefetched_action
                    prefetched_action = None

                self._apply_action_result(trajectory, action_result)
                trace = action_result.trace
                _, tool_calls = await self.tool_parser.extract_tool_calls(action_result.token_ids, self.tool_schemas)
                if not tool_calls:
                    trace["parse_error"] = True
                    trajectory.add_observation(NO_TOOL_CALL_ERROR, is_env=False)
                    stats.turns.append(trace)
                    self._merge_output_steps(extra_fields, action_result.output_extra)
                    committed_snapshot = trajectory.snapshot()
                    committed_turns += 1
                    continue

                tool_call = tool_calls[0]
                if tool_call.name == "done":
                    stats.turns.append(trace)
                    self._merge_output_steps(extra_fields, action_result.output_extra)
                    committed_snapshot = trajectory.snapshot()
                    return committed_turns + 1, "done"

                arguments = self._bash_arguments(tool_call)
                if arguments is None:
                    trace["parse_error"] = True
                    trajectory.add_observation(NO_TOOL_CALL_ERROR, is_env=False)
                    stats.turns.append(trace)
                    self._merge_output_steps(extra_fields, action_result.output_extra)
                    committed_snapshot = trajectory.snapshot()
                    committed_turns += 1
                    continue

                if self._detailed_reporting():
                    trace["command"] = dict(arguments)

                exec_start = time.monotonic()
                if last_exec_end is not None:
                    trace["inter_exec_gap_sec"] = exec_start - last_exec_end
                exec_task = registry.create(
                    self._timed_execute(environment, arguments),
                    "exec",
                )

                wm_prompt = trajectory.get_spec_prompt()
                max_spec_tokens = max(
                    1,
                    min(MAX_SPEC_OBS_TOKENS, self.response_length - len(trajectory.ids)),
                )
                trace["spec_attempted"] = True
                trace["spec_breadth_configured"] = SPECULATION_BREADTH
                wm_tasks: dict[asyncio.Task, int] = {}
                for rank in range(SPECULATION_BREADTH):
                    temperature = 0.0 if rank == 0 else 1.0
                    task = registry.create(
                        self._request_world_model(
                            wm_prompt,
                            max_spec_tokens,
                            stats,
                            candidate_rank=rank,
                            temperature=temperature,
                        ),
                        "wm",
                    )
                    wm_tasks[task] = rank

                candidates: dict[tuple[int, ...], BreadthCandidate] = {}
                completed_candidates = 0
                failed_candidates = 0
                duplicate_candidates = 0
                completed_wm_tokens = 0
                completed_call_secs: list[float] = []
                any_decode_truncated = False
                any_observation_overran = False

                while not exec_task.done():
                    waiters = {exec_task, *wm_tasks}
                    done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                    if exec_task in done:
                        break
                    for wm_task in sorted(
                        (task for task in done if task in wm_tasks),
                        key=lambda task: wm_tasks[task],
                    ):
                        rank = wm_tasks.pop(wm_task)
                        try:
                            wm_result = registry.result(wm_task)
                        except Exception:
                            failed_candidates += 1
                            continue

                        completed_candidates += 1
                        completed_wm_tokens += len(wm_result.token_ids)
                        completed_call_secs.append(wm_result.call_sec)
                        any_decode_truncated |= len(wm_result.token_ids) >= wm_result.max_tokens
                        any_observation_overran |= self._draft_overran(wm_result.token_ids)
                        draft_text = self._decode_draft(wm_result.token_ids)
                        body_ids = tuple(trajectory.observation_body_ids(draft_text))
                        existing = candidates.get(body_ids)
                        if existing is not None:
                            existing.ranks.append(rank)
                            existing.ranks.sort()
                            duplicate_candidates += 1
                            continue

                        policy_prompt_ids = trajectory.get_policy_prompt_after_observation(draft_text)
                        branch_response_length = len(policy_prompt_ids) - len(trajectory.prompt_ids)
                        branch_max_tokens = min(
                            MAX_TOKENS_PER_GENERATION,
                            self.response_length - branch_response_length,
                        )
                        policy_task = None
                        if branch_max_tokens > 0:
                            policy_task = registry.create(
                                self._request_action(
                                    policy_prompt_ids,
                                    f"{request_id}-breadth-{committed_turns}-{rank}",
                                    sampling_params,
                                    stats,
                                    depth=1,
                                    max_tokens=branch_max_tokens,
                                ),
                                "policy",
                            )
                        candidates[body_ids] = BreadthCandidate(
                            observation=draft_text,
                            ranks=[rank],
                            wm_token_count=len(wm_result.token_ids),
                            policy_prompt_ids=policy_prompt_ids,
                            policy_task=policy_task,
                        )

                exec_result = registry.result(exec_task)
                last_exec_end = exec_result.completed_at
                trace["exec_sec"] = exec_result.completed_at - exec_start
                trace["exec_outcome"] = exec_result.outcome
                if self._detailed_reporting():
                    trace["observation"] = exec_result.observation
                    trace["draft_observations"] = [
                        {
                            "ranks": list(candidate.ranks),
                            "observation": candidate.observation,
                        }
                        for candidate in candidates.values()
                    ]

                await asyncio.gather(
                    *(registry.cancel(task, "exec_won") for task in list(wm_tasks)),
                    return_exceptions=True,
                )
                wm_tasks.clear()

                true_body_ids = tuple(trajectory.observation_body_ids(exec_result.observation))
                selected = candidates.get(true_body_ids)
                trace.update(
                    {
                        "spec_match": selected is not None,
                        "spec_exec_won": completed_candidates == 0 and failed_candidates == 0,
                        "spec_failed": completed_candidates == 0 and failed_candidates > 0,
                        "spec_sec": max(completed_call_secs) if completed_call_secs else None,
                        "spec_obs_tokens": completed_wm_tokens,
                        "spec_decode_truncated": any_decode_truncated,
                        "spec_obs_overran": any_observation_overran,
                        "spec_breadth_completed": completed_candidates,
                        "spec_breadth_unique": len(candidates),
                        "spec_breadth_duplicates": duplicate_candidates,
                        "spec_breadth_policy_launched": sum(
                            candidate.policy_task is not None for candidate in candidates.values()
                        ),
                        "spec_breadth_match_rank": min(selected.ranks) if selected is not None else None,
                        "spec_matched_obs_tokens": selected.wm_token_count if selected is not None else 0,
                        "spec_breadth_greedy_match": selected is not None and 0 in selected.ranks,
                        "spec_breadth_sampled_match": selected is not None and any(rank > 0 for rank in selected.ranks),
                    }
                )

                await asyncio.gather(
                    *(
                        registry.cancel(candidate.policy_task, "breadth_prune" if selected is not None else "rollback")
                        for candidate in candidates.values()
                        if candidate is not selected
                    ),
                    return_exceptions=True,
                )

                trajectory.add_observation(exec_result.observation, is_env=True)
                stats.turns.append(trace)
                self._merge_output_steps(extra_fields, action_result.output_extra)
                committed_snapshot = trajectory.snapshot()
                committed_turns += 1

                if selected is None or selected.policy_task is None:
                    continue
                if list(trajectory.tokens) != selected.policy_prompt_ids:
                    raise RuntimeError("Matched breadth candidate produced a different policy prompt")

                trace["spec_breadth_selected_policy_ready"] = selected.policy_task.done()
                try:
                    await selected.policy_task
                    prefetched_action = registry.result(selected.policy_task)
                except Exception:
                    trace["spec_breadth_selected_policy_failed"] = True
                    prefetched_action = None

            return MAX_TURNS, "max_turns"
        except asyncio.CancelledError:
            cleanup_reason = "timeout"
            raise
        finally:
            await registry.cancel_all(cleanup_reason)
            stats.tasks_remaining_at_exit = registry.count()
            trajectory.restore(committed_snapshot)
            stats.pipeline_sec = time.monotonic() - pipeline_start

    async def _run_speculative(
        self,
        trajectory: Trajectory,
        request_id: str,
        environment: Any,
        sampling_params: dict[str, Any],
        extra_fields: dict[str, Any],
        stats: RolloutStats,
    ) -> tuple[int, str]:
        pending: deque[PendingTurn] = deque()
        producer: ActionProducer | None = None
        registry = TaskRegistry(stats)
        committed_snapshot = trajectory.snapshot()
        committed_turns = 0
        stop_reason = "error"
        pipeline_start = time.monotonic()
        last_exec_end = None
        frontier_depth = 0
        frontier_changed_at = pipeline_start
        cleanup_reason = "teardown"

        def update_frontier(new_depth: int) -> None:
            nonlocal frontier_depth, frontier_changed_at
            now = time.monotonic()
            stats.frontier_depth_area += frontier_depth * (now - frontier_changed_at)
            frontier_depth = new_depth
            frontier_changed_at = now

        def can_launch_producer() -> bool:
            if producer is not None or committed_turns + len(pending) >= MAX_TURNS:
                return False
            if not pending:
                return True
            if len(pending) - 1 >= SPECULATION_DEPTH:
                return False
            return pending[-1].kind in {"command", "synthetic"}

        try:
            while committed_turns < MAX_TURNS:
                update_frontier(max(len(pending) - 1, 0))

                if pending and pending[0].kind == "budget":
                    stop_reason = "max_response_length"
                    break

                if pending and pending[0].kind in {"synthetic", "done"}:
                    head = pending.popleft()
                    self._commit_turn(head, stats, extra_fields)
                    if head.snap_post_turn is None:
                        raise RuntimeError("Committed non-command turn has no snapshot")
                    committed_snapshot = head.snap_post_turn
                    committed_turns += 1
                    if head.kind == "done":
                        stop_reason = "done"
                        break
                    continue

                eligible_frontiers = [
                    turn
                    for depth, turn in enumerate(pending)
                    if turn.kind == "command_frontier" and depth < SPECULATION_DEPTH
                ]
                if eligible_frontiers:
                    if len(eligible_frontiers) != 1 or eligible_frontiers[0] is not pending[-1]:
                        raise RuntimeError("Expected one terminal frontier turn")
                    self._start_frontier_speculation(
                        trajectory,
                        eligible_frontiers[0],
                        stats,
                        registry,
                    )

                if pending and pending[0].kind in {"command", "command_speculating", "command_deferred"}:
                    head = pending[0]
                    if head.exec_task is None:
                        self._start_execution(head, environment, registry, last_exec_end)

                if can_launch_producer():
                    snap_pre_action = trajectory.snapshot()
                    max_tokens = min(
                        MAX_TOKENS_PER_GENERATION,
                        self.response_length - len(trajectory.ids),
                    )
                    if max_tokens <= 0:
                        pending.append(PendingTurn("budget", snap_pre_action, snap_pre_action))
                        continue
                    depth = len(pending)
                    producer = ActionProducer(
                        task=registry.create(
                            self._request_action(
                                list(trajectory.tokens),
                                request_id,
                                sampling_params,
                                stats,
                                depth=depth,
                                max_tokens=max_tokens,
                            ),
                            "policy",
                        ),
                        snap_pre_action=snap_pre_action,
                        depth=depth,
                    )

                waiters = registry.tasks()
                if not waiters:
                    raise RuntimeError("Speculative scheduler has no runnable task")
                phases = registry.active_phases()
                wait_started_at = time.monotonic()
                done, _ = await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
                self._record_wait_phase(stats, phases, time.monotonic() - wait_started_at)

                head = pending[0] if pending else None
                if head is not None and head.exec_task is not None and head.exec_task in done:
                    exec_result = registry.result(head.exec_task)
                    head.exec_task = None
                    head.exec_end = exec_result.completed_at
                    head.trace["exec_sec"] = exec_result.completed_at - (head.exec_start or exec_result.completed_at)
                    head.trace["exec_outcome"] = exec_result.outcome
                    if self._detailed_reporting():
                        head.trace["observation"] = exec_result.observation
                    last_exec_end = exec_result.completed_at

                    if head.kind == "command_speculating":
                        spec_result = None
                        spec_failed = False
                        if head.spec_task is not None and head.spec_task.done():
                            try:
                                spec_result = registry.result(head.spec_task)
                            except Exception:
                                spec_failed = True
                            head.spec_task = None
                        if spec_result is not None and spec_result.completed_at <= exec_result.completed_at:
                            self._finish_speculation(trajectory, head, spec_result)
                        else:
                            if head.spec_task is not None:
                                await registry.cancel(head.spec_task, "exec_won")
                                head.spec_task = None
                            head.trace["spec_exec_won"] = not spec_failed
                            head.trace["spec_failed"] = spec_failed
                            trajectory.add_observation(exec_result.observation, is_env=True)
                            head.snap_post_turn = trajectory.snapshot()
                            pending.popleft()
                            self._commit_turn(head, stats, extra_fields)
                            committed_snapshot = head.snap_post_turn
                            committed_turns += 1
                            continue

                    if head.kind == "command_deferred":
                        trajectory.add_observation(exec_result.observation, is_env=True)
                        head.snap_post_turn = trajectory.snapshot()
                        pending.popleft()
                        self._commit_turn(head, stats, extra_fields)
                        committed_snapshot = head.snap_post_turn
                        committed_turns += 1
                        continue

                    if head.kind != "command":
                        raise RuntimeError(f"Unexpected executed turn kind: {head.kind}")
                    head.trace["spec_match"] = trajectory.observation_body_ids(
                        head.draft_text
                    ) == trajectory.observation_body_ids(exec_result.observation)
                    if head.trace["spec_match"]:
                        if len(pending) > 1 or producer is not None:
                            stats.exact_with_descendant_work += 1
                        if len(pending) > 1:
                            stats.exact_with_action_ready += 1
                            if pending[1].spec_task is not None and not pending[1].spec_task.done():
                                stats.exact_with_active_descendant_spec += 1
                        if producer is not None and not producer.task.done():
                            stats.exact_with_policy_inflight += 1
                        pending.popleft()
                        self._commit_turn(head, stats, extra_fields)
                        if head.snap_post_turn is None:
                            raise RuntimeError("Exact turn has no speculative snapshot")
                        committed_snapshot = head.snap_post_turn
                        committed_turns += 1
                        continue

                    await self._cancel_descendants(pending, producer, registry, "rollback")
                    producer = None
                    trajectory.restore(head.snap_pre_observation or head.snap_pre_action)
                    trajectory.add_observation(exec_result.observation, is_env=True)
                    head.snap_post_turn = trajectory.snapshot()
                    self._commit_turn(head, stats, extra_fields)
                    committed_snapshot = head.snap_post_turn
                    committed_turns += 1
                    pending.clear()
                    continue

                completed_spec_turn = next(
                    (turn for turn in pending if turn.spec_task is not None and turn.spec_task in done),
                    None,
                )
                if completed_spec_turn is not None:
                    spec_task = completed_spec_turn.spec_task
                    completed_spec_turn.spec_task = None
                    try:
                        spec_result = registry.result(spec_task)
                    except Exception:
                        self._mark_speculation_failed(completed_spec_turn)
                    else:
                        self._finish_speculation(trajectory, completed_spec_turn, spec_result)
                    continue

                if producer is not None and producer.task in done:
                    completed_producer = producer
                    producer = None
                    action_result = registry.result(completed_producer.task)
                    turn = await self._materialize_action(
                        trajectory,
                        completed_producer,
                        action_result,
                        stats,
                        registry,
                    )
                    pending.append(turn)
                    stats.max_frontier_depth = max(
                        stats.max_frontier_depth,
                        max(len(pending) - 1, 0),
                    )
                    continue

                raise RuntimeError("Speculative scheduler did not process a completed task")
            else:
                stop_reason = "max_turns"
        except asyncio.CancelledError:
            cleanup_reason = "timeout"
            raise
        finally:
            await registry.cancel_all(cleanup_reason)
            stats.tasks_remaining_at_exit = registry.count()
            update_frontier(max(len(pending) - 1, 0))
            trajectory.restore(committed_snapshot)
            stats.pipeline_sec = time.monotonic() - pipeline_start
        return committed_turns, stop_reason

    def _agent_trace(self, stats: RolloutStats) -> dict[str, Any]:
        committed_ids = {id(turn) for turn in stats.turns}
        return {
            "turns": stats.turns,
            "discarded_turns": [turn for turn in stats.all_turns if id(turn) not in committed_ids],
            "policy_requests": stats.policy_requests,
            "wm_requests": stats.wm_requests,
        }

    def _agent_metrics(
        self,
        stats: RolloutStats,
        correct: bool,
        response_mask: list[int],
        world_mask: list[int],
    ) -> dict[str, float]:
        turns = stats.turns
        executed = [turn for turn in turns if turn.get("exec_outcome") is not None]
        count = len(turns)

        def turn_rate(key: str, value: Any = True) -> float:
            return sum(1 for turn in turns if turn.get(key) == value) / count if count else 0.0

        def exec_rate(outcome: str) -> float:
            return (
                sum(1 for turn in executed if turn.get("exec_outcome") == outcome) / len(executed) if executed else 0.0
            )

        result: dict[str, float] = {"solve_rate": float(correct)}
        for reason in (
            "done",
            "max_turns",
            "max_response_length",
            "setup_timeout",
            "setup_error",
            "agent_timeout",
            "error",
        ):
            result[f"stop_reason/{reason}"] = float(stats.stop_reason == reason)
        for error in ("verifier_timeout", "verifier_error"):
            result[f"verifier_error/{error}"] = float(stats.verifier_error == error)
        result["parse_error_rate"] = turn_rate("parse_error")
        result["exec_timeout_rate"] = exec_rate("timeout")
        result["exec_error_rate"] = exec_rate("error")
        result["nonzero_exit_rate"] = exec_rate("nonzero")
        result["gen_budget_hit_rate"] = turn_rate("budget_hit")
        result.update(_mmm("num_tool_calls", [float(len(executed))]))
        result.update(_mmm("gen_tokens_per_call", [float(t["gen_tokens"]) for t in turns]))
        result.update(_mmm("gen_sec_per_call", [float(t["gen_sec"]) for t in turns]))
        result.update(
            _mmm(
                "exec_sec_per_call",
                [float(t["exec_sec"]) for t in executed if t.get("exec_sec") is not None],
            )
        )
        result.update(
            _mmm(
                "inter_exec_gap_sec",
                [float(t["inter_exec_gap_sec"]) for t in executed if t.get("inter_exec_gap_sec") is not None],
            )
        )
        result.update(_mmm("completion_tokens_unmasked", [float(sum(response_mask))]))
        result.update(
            _mmm(
                "obs_tokens_masked",
                [float(len(response_mask) - sum(response_mask))],
            )
        )
        result.update(_mmm("world_target_tokens", [float(sum(world_mask))]))
        result["world_target_frac"] = float(sum(world_mask)) / len(response_mask) if response_mask else 0.0
        for name in ("setup_sec", "agent_sec", "verifier_sec"):
            value = getattr(stats, name)
            result.update(_mmm(name, [float(value)] if value is not None else []))
        result.update(_request_metrics(stats.policy_requests, ""))
        result.update(self._spec_metrics(stats))
        return result

    def _spec_metrics(self, stats: RolloutStats) -> dict[str, float]:
        attempted = [turn for turn in stats.turns if turn.get("spec_attempted")]
        exec_won = [turn for turn in attempted if turn.get("spec_exec_won")]
        failed = [turn for turn in attempted if turn.get("spec_failed")]
        used = [turn for turn in attempted if not turn.get("spec_exec_won") and not turn.get("spec_failed")]
        exact = [turn for turn in used if turn.get("spec_match")]
        rolled_back = [turn for turn in used if not turn.get("spec_match")]
        result: dict[str, float] = {
            "spec/attempt_count": float(len(attempted)),
            "spec/exec_won_count": float(len(exec_won)),
            "spec/exec_won_frac": len(exec_won) / len(attempted) if attempted else _NAN,
            "spec/failed_count": float(len(failed)),
            "spec/failed_frac": len(failed) / len(attempted) if attempted else _NAN,
            "spec/exact_count": float(len(exact)),
            "spec/exact_frac": len(exact) / len(used) if used else _NAN,
            "spec/rollback_count": float(len(rolled_back)),
            "spec/rollback_frac": len(rolled_back) / len(used) if used else _NAN,
            "spec/decode_truncated_frac": (
                sum(bool(t.get("spec_decode_truncated")) for t in used) / len(used) if used else _NAN
            ),
            "spec/obs_overran_frac": (sum(bool(t.get("spec_obs_overran")) for t in used) / len(used) if used else _NAN),
            "spec/block_sec_total": stats.block_sec,
            "spec/overlap_sec_total": stats.overlap_sec,
            "spec/pipeline_sec_total": (stats.pipeline_sec if stats.pipeline_sec is not None else _NAN),
            "spec/max_frontier_depth": float(stats.max_frontier_depth),
            "spec/frontier_depth_time_mean": (
                stats.frontier_depth_area / stats.pipeline_sec if stats.pipeline_sec else _NAN
            ),
            "spec/exact_with_descendant_work_count": float(stats.exact_with_descendant_work),
            "spec/exact_with_action_ready_count": float(stats.exact_with_action_ready),
            "spec/exact_with_policy_inflight_count": float(stats.exact_with_policy_inflight),
            "spec/exact_with_active_descendant_spec_count": float(stats.exact_with_active_descendant_spec),
            "spec/tasks_remaining_at_exit": float(stats.tasks_remaining_at_exit),
        }
        breadth_attempted = [turn for turn in attempted if turn.get("spec_breadth_configured") is not None]
        if breadth_attempted:
            matched_ranks = [
                turn["spec_breadth_match_rank"]
                for turn in breadth_attempted
                if turn["spec_breadth_match_rank"] is not None
            ]
            selected_policy_turns = [
                turn for turn in breadth_attempted if turn.get("spec_breadth_selected_policy_ready") is not None
            ]
            result.update(
                {
                    "spec/breadth_configured_mean": sum(turn["spec_breadth_configured"] for turn in breadth_attempted)
                    / len(breadth_attempted),
                    "spec/breadth_completed_mean": sum(turn["spec_breadth_completed"] for turn in breadth_attempted)
                    / len(breadth_attempted),
                    "spec/breadth_unique_mean": sum(turn["spec_breadth_unique"] for turn in breadth_attempted)
                    / len(breadth_attempted),
                    "spec/breadth_duplicates_total": float(
                        sum(turn["spec_breadth_duplicates"] for turn in breadth_attempted)
                    ),
                    "spec/breadth_policy_launched_mean": sum(
                        turn["spec_breadth_policy_launched"] for turn in breadth_attempted
                    )
                    / len(breadth_attempted),
                    "spec/breadth_greedy_match_frac": sum(
                        bool(turn["spec_breadth_greedy_match"]) for turn in breadth_attempted
                    )
                    / len(breadth_attempted),
                    "spec/breadth_sampled_only_match_frac": sum(
                        bool(turn["spec_breadth_sampled_match"]) and not bool(turn["spec_breadth_greedy_match"])
                        for turn in breadth_attempted
                    )
                    / len(breadth_attempted),
                    "spec/breadth_match_rank_mean": (
                        sum(matched_ranks) / len(matched_ranks) if matched_ranks else _NAN
                    ),
                    "spec/breadth_selected_policy_ready_frac": (
                        sum(bool(turn["spec_breadth_selected_policy_ready"]) for turn in selected_policy_turns)
                        / len(selected_policy_turns)
                        if selected_policy_turns
                        else _NAN
                    ),
                }
            )
        for reason in ("exec_won", "rollback", "breadth_prune", "timeout", "teardown"):
            for phase in ("policy", "wm", "exec"):
                key = f"{reason}/{phase}"
                prefix = f"spec/cancel/{reason}/{phase}"
                result[f"{prefix}_count"] = float(stats.cancellation_counts.get(key, 0))
                result[f"{prefix}_sec_total"] = float(stats.cancellation_secs.get(key, 0.0))
        for phase in (
            "idle",
            "exec",
            "policy",
            "wm",
            "exec_and_policy",
            "exec_and_wm",
            "policy_and_wm",
            "exec_and_policy_and_wm",
        ):
            result[f"spec/phase/{phase}_sec_total"] = float(stats.phase_secs.get(phase, 0.0))
        result.update(
            _mmm(
                "spec/sec_per_call",
                [float(t["spec_sec"]) for t in used if t.get("spec_sec") is not None],
            )
        )
        for depth in range(SPECULATION_DEPTH + 1):
            at_depth = [turn for turn in attempted if int(turn.get("spec_depth", 0)) == depth]
            used_at_depth = [turn for turn in at_depth if not turn.get("spec_exec_won") and not turn.get("spec_failed")]
            exact_at_depth = [turn for turn in used_at_depth if turn.get("spec_match")]
            prefix = f"spec/depth_{depth}"
            result[f"{prefix}/attempt_count"] = float(len(at_depth))
            result[f"{prefix}/exact_count"] = float(len(exact_at_depth))
            result[f"{prefix}/rollback_count"] = float(len(used_at_depth) - len(exact_at_depth))
            result[f"{prefix}/exact_frac"] = len(exact_at_depth) / len(used_at_depth) if used_at_depth else _NAN

        all_policy_tokens = sum(
            turn.get("gen_tokens", 0) for turn in stats.all_turns if int(turn.get("spec_depth", 0)) > 0
        )
        committed_policy_tokens = sum(
            turn.get("gen_tokens", 0) for turn in stats.turns if int(turn.get("spec_depth", 0)) > 0
        )
        all_wm_tokens = sum(row.get("gen_tokens", 0) for row in stats.wm_requests)
        verified_wm_tokens = sum(turn.get("spec_obs_tokens", 0) for turn in used)
        committed_wm_tokens = sum(turn.get("spec_matched_obs_tokens", turn.get("spec_obs_tokens", 0)) for turn in exact)
        result.update(
            {
                "spec/policy_tokens_completed_ahead": float(all_policy_tokens),
                "spec/policy_tokens_completed_discarded": float(max(all_policy_tokens - committed_policy_tokens, 0)),
                "spec/policy_tokens_generated_ahead": float(all_policy_tokens),
                "spec/policy_tokens_committed_ahead": float(committed_policy_tokens),
                "spec/policy_tokens_discarded": float(max(all_policy_tokens - committed_policy_tokens, 0)),
                "spec/wm_tokens_completed": float(all_wm_tokens),
                "spec/wm_tokens_completed_discarded": float(max(all_wm_tokens - committed_wm_tokens, 0)),
                "spec/wm_tokens_generated": float(all_wm_tokens),
                "spec/wm_tokens_verified": float(verified_wm_tokens),
                "spec/wm_tokens_committed": float(committed_wm_tokens),
                "spec/wm_tokens_discarded": float(max(all_wm_tokens - committed_wm_tokens, 0)),
            }
        )
        result.update(_request_metrics(stats.wm_requests, "spec/"))
        return result

    async def _execute(
        self,
        environment: Any,
        arguments: dict[str, Any],
    ) -> tuple[str, str]:
        command = arguments["command"]
        timeout = float(arguments.get("timeout", 30))
        try:
            result = await environment.exec(command, timeout=timeout)
        except TimeoutError:
            return f"command timed out after {timeout:g}s\n(exit_code=-1)", "timeout"
        except Exception:
            return "command failed to execute\n(exit_code=-1)", "error"

        streams = []
        if result.stdout:
            streams.append(result.stdout.rstrip("\n"))
        if result.stderr:
            streams.append(result.stderr.rstrip("\n"))
        output = "\n".join(streams)
        if len(output) > MAX_TERMINAL_OUTPUT_CHARS:
            output = output[:MAX_TERMINAL_OUTPUT_CHARS].rstrip("\n") + "\n[output truncated]"
        if not output:
            output = "(no output)"
        outcome = "ok" if result.return_code == 0 else "nonzero"
        return f"{output}\n(exit_code={result.return_code})", outcome
