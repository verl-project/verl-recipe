"""Per-trajectory rollout context isolation via :mod:`contextvars`.

In an async multi-turn rollout, a single ``AgentLoop`` instance services many
concurrent trajectories (one ``asyncio.Task`` per sample). Any per-trajectory
state -- the tau2 ``Environment`` holding a mutable ``FlightDB``, the user
simulator and its running conversation state -- MUST NOT leak across those
concurrent tasks, or trajectory A's tool call would mutate trajectory B's DB
and corrupt both the rollout and the reward.

``contextvars.ContextVar`` gives us exactly this isolation: a value ``set()``
inside one ``asyncio.Task`` is invisible to sibling tasks, because each task
runs with its own copy of the context. This module wraps that primitive in a
small, explicit API so the agent loop can bind the current trajectory once and
any code running *in the async portion* of that trajectory can recover it with
``current_rollout()`` -- without threading the object through every call.

Note on threads: values set here are visible to awaited sub-coroutines within
the same task, but NOT automatically to ``loop.run_in_executor`` worker threads
(a fresh thread does not inherit the caller's context). For executor-bound work
(the blocking litellm user-simulator call, synchronous tau2 tool execution) we
therefore pass the trajectory object *explicitly*. The ContextVar is the clean
in-async accessor; explicit passing is the cross-thread contract. Keeping both
honest is the whole point of the isolation design.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

# The single module-level ContextVar. Default ``None`` means "no active
# trajectory" -- calling current_rollout() outside a scope is a programming
# error and raises, rather than silently returning stale state.
_ROLLOUT_CTX: contextvars.ContextVar[Optional[RolloutContext]] = contextvars.ContextVar(
    "verl_rollout_ctx", default=None
)


@dataclass
class RolloutContext:
    """Isolated state for exactly one rollout trajectory.

    Attributes:
        request_id: The vLLM request id for this trajectory (unique per sample).
        payload: Arbitrary per-trajectory object. In the tau2 integration this
            holds the :class:`Tau2Trajectory` (env + task + user simulator +
            structured message log). Typed as ``Any`` to keep this module free
            of any tau2 import so it can be unit-tested on its own.
    """

    request_id: str
    payload: Any = None
    # Free-form scratch space for metrics / debugging, isolated per trajectory.
    scratch: dict[str, Any] = field(default_factory=dict)


@contextmanager
def rollout_scope(request_id: str, payload: Any = None) -> Iterator[RolloutContext]:
    """Bind a fresh :class:`RolloutContext` for the duration of the ``with`` block.

    Usage::

        with rollout_scope(request_id, payload=traj) as ctx:
            ...  # anything in here, and any coroutine it awaits, sees ctx

    The token-based reset guarantees the previous value is restored even if the
    body raises, so nested / sequential scopes never bleed into one another.
    """
    ctx = RolloutContext(request_id=request_id, payload=payload)
    token = _ROLLOUT_CTX.set(ctx)
    try:
        yield ctx
    finally:
        _ROLLOUT_CTX.reset(token)


def current_rollout() -> RolloutContext:
    """Return the active :class:`RolloutContext`, or raise if none is bound."""
    ctx = _ROLLOUT_CTX.get()
    if ctx is None:
        raise RuntimeError(
            "current_rollout() called with no active rollout_scope(). "
            "This usually means code ran in an executor thread that did not "
            "inherit the async context -- pass the trajectory explicitly there."
        )
    return ctx


def try_current_rollout() -> Optional[RolloutContext]:
    """Like :func:`current_rollout` but returns ``None`` instead of raising."""
    return _ROLLOUT_CTX.get()
