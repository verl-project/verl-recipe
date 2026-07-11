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

from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf
from verl_tinker.backends.profiler_util import TinkerProfilingActorRolloutRefWorker

from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker


class TestWorkerProfiling:
    def test_profile_lifecycle_targets_inner_actor_worker(self):
        worker = object.__new__(TinkerProfilingActorRolloutRefWorker)
        worker.actor = MagicMock(name="inner_actor")

        with (
            patch.object(TinkerActorRolloutRefWorker, "start_profile") as mock_outer_start,
            patch.object(TinkerActorRolloutRefWorker, "stop_profile") as mock_outer_stop,
        ):
            worker.start_profile(role="actor")
            worker.stop_profile()

        mock_outer_start.assert_not_called()
        worker.actor.start_profile.assert_called_once_with(role="actor")
        worker.actor.stop_profile.assert_called_once_with()
        mock_outer_stop.assert_not_called()

    def test_profile_lifecycle_handles_uninitialized_actor(self):
        worker = object.__new__(TinkerProfilingActorRolloutRefWorker)
        worker.actor = None

        with (
            patch.object(TinkerActorRolloutRefWorker, "start_profile") as mock_outer_start,
            patch.object(TinkerActorRolloutRefWorker, "stop_profile") as mock_outer_stop,
        ):
            worker.start_profile(role="actor")
            worker.stop_profile()

        mock_outer_start.assert_called_once_with(role="actor")
        mock_outer_stop.assert_called_once_with()

    def test_install_inner_actor_profiler_uses_actor_profiler_config(self):
        worker = object.__new__(TinkerProfilingActorRolloutRefWorker)
        worker.config = OmegaConf.create(
            {
                "actor": {
                    "profiler": {
                        "enable": True,
                        "tool": "torch",
                        "save_path": "outputs/profile",
                        "tool_config": {
                            "torch": {
                                "_target_": "verl.utils.profiler.config.TorchProfilerToolConfig",
                                "contents": ["cuda", "cpu"],
                            }
                        },
                    }
                }
            }
        )
        worker.actor = MagicMock(name="inner_actor")
        worker.actor.rank = 0

        worker._install_inner_actor_profiler()

        profiler = worker.actor.profiler
        assert profiler.check_enable()
        assert profiler.check_this_rank()
        assert profiler.config.tool == "torch"
        assert profiler.config.save_path == "outputs/profile"
        assert list(profiler.tool_config.contents) == ["cuda", "cpu"]
