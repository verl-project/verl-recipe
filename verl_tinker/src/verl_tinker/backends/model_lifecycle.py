from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import Enum


class ModelRole(str, Enum):
    ACTOR = "actor"
    REF = "ref"
    ROLLOUT = "rollout"


@dataclass
class ModelLifecycle:
    """Track which colocated model roles are currently resident on GPU."""

    enable_offload: bool
    available_roles: set[ModelRole]
    awake_roles: set[ModelRole] = field(default_factory=set)

    @classmethod
    def create(
        cls,
        *,
        enable_offload: bool,
        has_rollout: bool,
        has_ref: bool,
        actor_awake: bool = True,
    ) -> "ModelLifecycle":
        available_roles = {ModelRole.ACTOR}
        if has_rollout:
            available_roles.add(ModelRole.ROLLOUT)
        if has_ref:
            available_roles.add(ModelRole.REF)

        if not enable_offload:
            return cls(
                enable_offload=False,
                available_roles=available_roles,
                awake_roles=set(available_roles),
            )

        awake_roles = {ModelRole.ACTOR} if actor_awake else set()
        if actor_awake and has_ref:
            awake_roles.add(ModelRole.REF)
        return cls(
            enable_offload=True,
            available_roles=available_roles,
            awake_roles=awake_roles & available_roles,
        )

    def mark_awake(self, role: ModelRole) -> None:
        if role in self.available_roles:
            self.awake_roles.add(role)

    def mark_asleep(self, role: ModelRole) -> None:
        self.awake_roles.discard(role)

    def prepare(
        self,
        required_roles: Iterable[ModelRole],
        *,
        sleep_role: Callable[[ModelRole], None],
        wake_role: Callable[[ModelRole], None],
    ) -> None:
        """Ensure only the required roles are awake when server offload is enabled."""

        if not self.enable_offload:
            return

        required = set(required_roles) & self.available_roles

        for role in list(self.awake_roles - required):
            sleep_role(role)
            self.mark_asleep(role)

        for role in required - self.awake_roles:
            wake_role(role)
            self.mark_awake(role)
