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

from verl_tinker.backends.profiler_util import TinkerProfilingActorRolloutRefWorker

from verl.workers.engine_workers_tinker import TinkerActorRolloutRefWorker


class TestWorkerProfiling:
    def test_profile_lifecycle_forwards_to_inner_actor_worker(self):
        worker = object.__new__(TinkerProfilingActorRolloutRefWorker)
        worker.actor = MagicMock(name="inner_actor")

        with (
            patch.object(TinkerActorRolloutRefWorker, "start_profile") as mock_outer_start,
            patch.object(TinkerActorRolloutRefWorker, "stop_profile") as mock_outer_stop,
        ):
            worker.start_profile(role="actor")
            worker.stop_profile()

        mock_outer_start.assert_called_once_with(role="actor")
        worker.actor.start_profile.assert_called_once_with(role="actor")
        worker.actor.stop_profile.assert_called_once_with()
        mock_outer_stop.assert_called_once_with()

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
