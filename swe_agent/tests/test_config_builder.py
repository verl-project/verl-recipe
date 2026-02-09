# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import pytest
from recipe.swe_agent.config.config_builder import SWEAgentConfigBuilder


def test_config_builder_creates_valid_yaml():
    """Test that config builder produces valid SWE-Agent YAML."""
    builder = SWEAgentConfigBuilder(
        instance_id="test-123",
        repo_path="/workspace/repo",
        output_dir="/tmp/output",
        model_proxy_port=8080,
        max_steps=50,
        execution_timeout=300,
    )

    config = builder.build()

    assert config["output_dir"] == "/tmp/output"
    assert config["env"]["repo"]["path"] == "/workspace/repo"
    assert config["agent"]["model"]["api_base"] == "http://127.0.0.1:8080/v1"
    assert config["agent"]["model"]["per_instance_call_limit"] == 50
    assert config["agent"]["tools"]["execution_timeout"] == 300


def test_config_builder_to_yaml_string():
    """Test YAML serialization."""
    builder = SWEAgentConfigBuilder(
        instance_id="test-123",
        repo_path="/workspace",
        output_dir="/tmp",
        model_proxy_port=8080,
        max_steps=50,
        execution_timeout=300,
    )

    yaml_str = builder.to_yaml()

    assert "output_dir: /tmp" in yaml_str
    assert isinstance(yaml_str, str)


def test_config_builder_custom_templates():
    """Test custom template support."""
    builder = SWEAgentConfigBuilder(
        instance_id="test",
        repo_path="/workspace",
        output_dir="/tmp",
        model_proxy_port=8080,
        max_steps=50,
        execution_timeout=300,
        custom_templates={
            "system_template": "Custom system prompt",
        }
    )

    config = builder.build()
    assert config["agent"]["templates"]["system_template"] == "Custom system prompt"
