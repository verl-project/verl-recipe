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

"""Unit tests for TinkerServer using fake Tinker backend.

These tests mock the Tinker API (ServiceClient, SamplingClient) and
get_shared_checkpoint_path() to test TinkerServer without Ray or real Tinker.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fake_tinker import (
    FakeModelInput,
    FakeSamplingParams,
    FakeServiceClient,
)
from omegaconf import OmegaConf

from verl_recipes.verl_tinker_connector.tinker_rollout import TinkerServer


# ==================== Fixtures ====================


@pytest.fixture
def tinker_server():
    """Create a TinkerServer with mock config."""
    config = OmegaConf.create({"tinker": {"model_name": "test-model"}})
    model_config = OmegaConf.create({"path": "/fake/model/path"})
    return TinkerServer(config=config, model_config=model_config)


@pytest.fixture
def fake_service_client():
    """Create a FakeServiceClient for mocking."""
    return FakeServiceClient()


# ==================== TinkerServer Tests ====================


class TestTinkerServerInitialize:
    """Tests for TinkerServer.initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_creates_sampling_client(self, tinker_server, fake_service_client):
        """Test that initialize() creates a sampling client from checkpoint path."""
        tinker_config = {"model_name": "Qwen/Qwen3-4B-Instruct-2507", "api_key": "tml-test-key"}

        with (
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.tinker.ServiceClient",
                return_value=fake_service_client,
            ),
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.get_shared_checkpoint_path",
                new_callable=AsyncMock,
                return_value="tinker://fake-checkpoint-path",
            ),
        ):
            await tinker_server.initialize(tinker_config)

        assert tinker_server._initialized is True
        assert tinker_server._service_client is fake_service_client
        assert tinker_server._sampling_client is not None
        assert len(fake_service_client.sampling_clients) == 1

    @pytest.mark.asyncio
    async def test_initialize_is_idempotent(self, tinker_server, fake_service_client):
        """Test that calling initialize() twice only initializes once."""
        tinker_config = {"model_name": "test-model", "api_key": "tml-test-key"}

        with (
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.tinker.ServiceClient",
                return_value=fake_service_client,
            ),
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.get_shared_checkpoint_path",
                new_callable=AsyncMock,
                return_value="tinker://fake-checkpoint-path",
            ),
        ):
            await tinker_server.initialize(tinker_config)
            await tinker_server.initialize(tinker_config)

        # Should only create one sampling client
        assert len(fake_service_client.sampling_clients) == 1


class TestTinkerServerGenerate:
    """Tests for TinkerServer.generate()."""

    @pytest.mark.asyncio
    async def test_generate_without_initialize_raises(self, tinker_server):
        """Test that generate() raises if not initialized."""
        with pytest.raises(RuntimeError, match="not initialized"):
            await tinker_server.generate(
                prompt_ids=[1, 2, 3],
                sampling_params={},
                request_id="test-request",
            )

    @pytest.mark.asyncio
    async def test_generate_returns_token_output(self, tinker_server, fake_service_client):
        """Test that generate() returns TokenOutput with correct format."""
        tinker_config = {"model_name": "test-model", "api_key": "tml-test-key"}

        with (
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.tinker.ServiceClient",
                return_value=fake_service_client,
            ),
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.get_shared_checkpoint_path",
                new_callable=AsyncMock,
                return_value="tinker://fake-checkpoint-path",
            ),
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.tinker.SamplingParams",
                FakeSamplingParams,
            ),
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.tinker.ModelInput",
                FakeModelInput,
            ),
        ):
            await tinker_server.initialize(tinker_config)

            result = await tinker_server.generate(
                prompt_ids=[1, 2, 3, 4, 5],
                sampling_params={"temperature": 0.7, "max_tokens": 20},
                request_id="test-request-1",
            )

        # Check token_ids: FakeSamplingClient returns [1000, 1001, ..., 1019] for max_tokens=20
        assert result.token_ids == list(range(1000, 1020))

        # Check log_probs: FakeSamplingClient returns [-0.1, -0.2, ..., -2.0]
        expected_logprobs = [-0.1 * (i + 1) for i in range(20)]
        assert result.log_probs == expected_logprobs

        # TinkerServer transforms "length" to "completed"
        assert result.stop_reason == "completed"


class TestTinkerServerWakeUp:
    """Tests for TinkerServer.wake_up()."""

    @pytest.mark.asyncio
    async def test_wake_up_refreshes_sampling_client(self, tinker_server, fake_service_client):
        """Test that wake_up() creates a new sampling client with updated checkpoint."""
        tinker_config = {"model_name": "test-model", "api_key": "tml-test-key"}

        with (
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.tinker.ServiceClient",
                return_value=fake_service_client,
            ),
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.get_shared_checkpoint_path",
                new_callable=AsyncMock,
                return_value="tinker://initial-checkpoint",
            ),
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.tinker.SamplingParams",
                FakeSamplingParams,
            ),
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.tinker.ModelInput",
                FakeModelInput,
            ),
        ):
            await tinker_server.initialize(tinker_config)
            initial_client = tinker_server._sampling_client

        # Now wake_up with new checkpoint
        with patch(
            "verl_recipes.verl_tinker_connector.tinker_rollout.get_shared_checkpoint_path",
            new_callable=AsyncMock,
            return_value="tinker://updated-checkpoint",
        ):
            await tinker_server.wake_up()

        # Should have created a new sampling client
        assert len(fake_service_client.sampling_clients) == 2
        assert tinker_server._sampling_client is not initial_client

    @pytest.mark.asyncio
    async def test_wake_up_handles_no_checkpoint(self, tinker_server, fake_service_client):
        """Test that wake_up() handles missing checkpoint gracefully."""
        tinker_config = {"model_name": "test-model", "api_key": "tml-test-key"}

        with (
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.tinker.ServiceClient",
                return_value=fake_service_client,
            ),
            patch(
                "verl_recipes.verl_tinker_connector.tinker_rollout.get_shared_checkpoint_path",
                new_callable=AsyncMock,
                return_value="tinker://initial-checkpoint",
            ),
        ):
            await tinker_server.initialize(tinker_config)
            initial_client = tinker_server._sampling_client

        # wake_up with no checkpoint available
        with patch(
            "verl_recipes.verl_tinker_connector.tinker_rollout.get_shared_checkpoint_path",
            new_callable=AsyncMock,
            return_value=None,
        ):
            await tinker_server.wake_up()

        # Should keep the old client
        assert tinker_server._sampling_client is initial_client


class TestTinkerServerNoOps:
    """Tests for TinkerServer no-op methods."""

    @pytest.mark.asyncio
    async def test_sleep_is_noop(self, tinker_server):
        """Test that sleep() is a no-op."""
        await tinker_server.sleep()

    @pytest.mark.asyncio
    async def test_clear_kv_cache_is_noop(self, tinker_server):
        """Test that clear_kv_cache() is a no-op."""
        await tinker_server.clear_kv_cache()
