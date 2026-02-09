import pytest
from recipe.swe_agent.config.config_validator import AgentConfigValidator, ConfigValidationError


def test_validator_accepts_valid_config():
    """Test that validator accepts valid configuration."""
    config = {
        "proxy_config": {"port": 8080},
        "sandbox_config": {
            "swe_agent_timeout": 1800,
            "max_steps": 100,
        },
    }

    validator = AgentConfigValidator(config)
    validator.validate()  # Should not raise


def test_validator_rejects_invalid_port():
    """Test that validator rejects invalid port."""
    config = {
        "proxy_config": {"port": 99999},  # Invalid port
        "sandbox_config": {},
    }

    validator = AgentConfigValidator(config)

    with pytest.raises(ConfigValidationError, match="port must be a valid port number"):
        validator.validate()


def test_validator_rejects_negative_timeout():
    """Test that validator rejects negative timeout."""
    config = {
        "proxy_config": {"port": 8080},
        "sandbox_config": {"swe_agent_timeout": -100},
    }

    validator = AgentConfigValidator(config)

    with pytest.raises(ConfigValidationError, match="timeout must be a positive number"):
        validator.validate()


def test_validator_rejects_missing_config():
    """Test that validator rejects missing config."""
    validator = AgentConfigValidator(None)

    with pytest.raises(ConfigValidationError, match="requires agent configuration"):
        validator.validate()
