# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

import logging
from typing import Any

from verl.single_controller.base.decorator import Dispatch, register
from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker

logger = logging.getLogger("ray")


def profiler_state(profiler) -> dict[str, Any]:
    config = getattr(profiler, "config", None)
    state = {
        "exists": profiler is not None,
        "enabled": getattr(config, "enable", None),
        "tool": getattr(config, "tool", None),
        "save_path": getattr(config, "save_path", None),
        "rank_allowed": None,
    }
    check_this_rank = getattr(profiler, "check_this_rank", None)
    if check_this_rank is not None:
        try:
            state["rank_allowed"] = check_this_rank()
        except Exception as exc:
            state["rank_allowed"] = f"error: {exc}"
    return state


class TinkerProfilingActorRolloutRefWorker(TinkerActorRolloutRefWorker):
    """Tinker worker that starts profiling on the inner actor worker too."""

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def start_profile(self, **kwargs) -> None:
        actor = getattr(self, "actor", None)
        logger.info(
            "[profiler] worker start_profile rank=%s kwargs=%s outer=%s has_actor=%s inner=%s",
            getattr(self, "rank", None),
            kwargs,
            profiler_state(getattr(self, "profiler", None)),
            actor is not None,
            profiler_state(getattr(actor, "profiler", None)) if actor is not None else None,
        )
        super().start_profile(**kwargs)
        actor = getattr(self, "actor", None)
        if actor is not None:
            actor.start_profile(**kwargs)
        logger.info(
            "[profiler] worker start_profile done rank=%s outer=%s inner=%s",
            getattr(self, "rank", None),
            profiler_state(getattr(self, "profiler", None)),
            profiler_state(getattr(actor, "profiler", None)) if actor is not None else None,
        )

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def stop_profile(self) -> None:
        actor = getattr(self, "actor", None)
        logger.info(
            "[profiler] worker stop_profile rank=%s outer=%s has_actor=%s inner=%s",
            getattr(self, "rank", None),
            profiler_state(getattr(self, "profiler", None)),
            actor is not None,
            profiler_state(getattr(actor, "profiler", None)) if actor is not None else None,
        )
        if actor is not None:
            actor.stop_profile()
        super().stop_profile()
        actor = getattr(self, "actor", None)
        logger.info(
            "[profiler] worker stop_profile done rank=%s outer=%s inner=%s",
            getattr(self, "rank", None),
            profiler_state(getattr(self, "profiler", None)),
            profiler_state(getattr(actor, "profiler", None)) if actor is not None else None,
        )
