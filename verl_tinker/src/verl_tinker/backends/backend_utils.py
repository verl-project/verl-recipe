from __future__ import annotations

import time
from typing import Any

import ray
from ray.util.state import get_actor, get_placement_group


_DEFAULT_SHUTDOWN_TIMEOUT_S = 60.0
_SHUTDOWN_POLL_INTERVAL_S = 0.5


def kill_ray_actors_and_wait(
    actors: list[Any],
    *,
    logger,
    description: str,
    ray_module=ray,
    timeout_s: float = _DEFAULT_SHUTDOWN_TIMEOUT_S,
) -> None:
    """Force-kill Ray actors and wait until Ray no longer reports them alive."""

    unique: list[tuple[Any, str | None]] = []
    seen: set[str | int] = set()
    for actor in actors:
        if actor is None:
            continue
        actor_id = _actor_id_hex(actor)
        dedup_key: str | int = actor_id if actor_id is not None else id(actor)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        unique.append((actor, actor_id))

    for actor, actor_id in unique:
        try:
            ray_module.kill(actor, no_restart=True)
        except Exception as exc:
            logger.warning("Failed to kill %s actor %s: %s", description, actor_id or actor, exc)

    actor_ids = [actor_id for _, actor_id in unique if actor_id is not None]
    _wait_for_actor_exit(actor_ids, logger=logger, description=description, timeout_s=timeout_s)


def remove_placement_groups_and_wait(
    placement_groups: list[Any],
    *,
    logger,
    description: str,
    ray_module=ray,
    timeout_s: float = _DEFAULT_SHUTDOWN_TIMEOUT_S,
) -> None:
    """Remove Ray placement groups and wait until Ray marks them removed."""

    unique: list[tuple[Any, str | None]] = []
    seen: set[str | int] = set()
    for pg in placement_groups:
        if pg is None:
            continue
        pg_id = _placement_group_id_hex(pg)
        dedup_key: str | int = pg_id if pg_id is not None else id(pg)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        unique.append((pg, pg_id))

    for pg, pg_id in unique:
        try:
            ray_module.util.remove_placement_group(pg)
        except Exception as exc:
            logger.warning("Failed to remove %s placement group %s: %s", description, pg_id or pg, exc)

    pg_ids = [pg_id for _, pg_id in unique if pg_id is not None]
    _wait_for_placement_group_removal(pg_ids, logger=logger, description=description, timeout_s=timeout_s)


def _wait_for_actor_exit(actor_ids: list[str], *, logger, description: str, timeout_s: float) -> None:
    if not actor_ids:
        return
    pending = set(actor_ids)
    deadline = time.monotonic() + timeout_s
    while pending and time.monotonic() < deadline:
        pending = {actor_id for actor_id in pending if _actor_state(actor_id) not in {None, "DEAD"}}
        if pending:
            time.sleep(_SHUTDOWN_POLL_INTERVAL_S)
    if pending:
        raise TimeoutError(f"Timed out waiting for {description} actors to exit: {sorted(pending)}")
    logger.info("Confirmed %s Ray actors exited", description)


def _wait_for_placement_group_removal(pg_ids: list[str], *, logger, description: str, timeout_s: float) -> None:
    if not pg_ids:
        return
    pending = set(pg_ids)
    deadline = time.monotonic() + timeout_s
    terminal_states = {None, "REMOVED", "DEAD"}
    while pending and time.monotonic() < deadline:
        pending = {pg_id for pg_id in pending if _placement_group_state(pg_id) not in terminal_states}
        if pending:
            time.sleep(_SHUTDOWN_POLL_INTERVAL_S)
    if pending:
        raise TimeoutError(f"Timed out waiting for {description} placement groups to be removed: {sorted(pending)}")
    logger.info("Confirmed %s placement groups removed", description)


def _actor_id_hex(actor: Any) -> str | None:
    try:
        actor_id = actor._actor_id.hex()
    except Exception:
        return None
    return actor_id if isinstance(actor_id, str) else None


def _placement_group_id_hex(placement_group: Any) -> str | None:
    try:
        pg_id = placement_group.id.hex()
    except Exception:
        return None
    return pg_id if isinstance(pg_id, str) else None


def _state_value(state: Any) -> str | None:
    if state is None:
        return None
    if isinstance(state, dict):
        return state.get("state")
    value = getattr(state, "state", None)
    return value if isinstance(value, str) else None


def _actor_state(actor_id: str) -> str | None:
    try:
        return _state_value(get_actor(actor_id, timeout=1))
    except Exception:
        return "UNKNOWN"


def _placement_group_state(pg_id: str) -> str | None:
    try:
        return _state_value(get_placement_group(pg_id, timeout=1))
    except Exception:
        return "UNKNOWN"
