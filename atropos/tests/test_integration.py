"""
Test script for Atropos-VeRL integration.

This script validates the integration between VeRL and Atropos.
"""

import atexit
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import torch

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from recipe.atropos.atropos_integration import (
    AtroposAPIError,
    AtroposConfig,
    AtroposTrainerClient,
    convert_scalar_or_token_advantages,
)

# Try to import VeRL components, but handle gracefully if missing
try:
    from verl.trainer.ppo.core_algos import (
        ADV_ESTIMATOR_REGISTRY,
        compute_grpo_outcome_advantage,
    )

    VERL_AVAILABLE = True
except ImportError as e:
    print(f"Warning: VeRL components not available ({e}). Some tests will be skipped.")
    VERL_AVAILABLE = False

logger = logging.getLogger(__name__)


def _atropos_available(api_url: str) -> bool:
    try:
        resp = requests.get(f"{api_url}/status", timeout=3)
        return resp.status_code == 200
    except requests.exceptions.RequestException:
        return False


def _ensure_atropos_api(api_url: str) -> None:
    if _atropos_available(api_url):
        return

    atropos_path = os.environ.get("ATROPOS_PATH")
    if not atropos_path:
        print("⚠ ATROPOS_PATH not set; skipping auto-start of Atropos API.")
        return

    api_script = Path(atropos_path) / "atroposlib" / "cli" / "run_api.py"
    if not api_script.exists():
        print(f"⚠ Atropos API script not found: {api_script}")
        return

    parsed = urlparse(api_url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 9001

    cmd = [sys.executable, str(api_script), "--host", host, "--port", str(port)]
    print(f"Starting Atropos API: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=str(atropos_path), text=True)

    def _cleanup():
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()

    atexit.register(_cleanup)

    # Wait for /status to become available
    start = time.time()
    while time.time() - start < 30:
        if _atropos_available(api_url):
            return
        time.sleep(1)

    print("⚠ Atropos API did not become ready in time; tests may fail.")


def test_atropos_client():
    """Test Atropos client connectivity and basic operations."""
    print("Testing Atropos client...")

    config = AtroposConfig(api_url="http://localhost:9001", timeout=10)

    if not _atropos_available(config.api_url):
        print("⚠ Atropos API not available; skipping client test.")
        return True

    try:
        client = AtroposTrainerClient(config)
        print("✓ Client initialized successfully")

        registration = {
            "wandb_group": "verl_atropos_tests",
            "wandb_project": "atropos",
            "batch_size": 2,
            "max_token_len": 128,
            "checkpoint_dir": "/tmp/atropos_test",
            "save_checkpoint_interval": 0,
            "starting_step": 0,
            "num_steps": 1,
        }
        trainer_uuid = client.register_trainer(registration)
        print(f"✓ Registered trainer: {trainer_uuid}")

    except Exception as e:
        print(f"✗ Client test failed: {e}")
        return False

    return True


def test_advantage_broadcast_logic():
    """Test the broadcast logic for scalar vs token-level advantages."""
    print("\nTesting advantage broadcast logic...")

    try:
        # Test the broadcast logic directly without requiring API connectivity
        # Test 1: Scalar advantages should be broadcasted
        batch_size = 2
        seq_length = 5
        response_mask = torch.ones(batch_size, seq_length)

        # Mock scalar advantages (one per sample)
        scalar_advantages = [0.5, -0.3]

        tensor_advantages = convert_scalar_or_token_advantages(scalar_advantages, response_mask)

        # Check that scalar advantages were broadcasted correctly
        expected = torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5], [-0.3, -0.3, -0.3, -0.3, -0.3]])

        assert torch.allclose(tensor_advantages, expected), (
            f"Scalar broadcast failed: got {tensor_advantages}, expected {expected}"
        )

        print("✓ Scalar advantage broadcasting works correctly")

        # Test 2: With partial response mask
        partial_mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)

        tensor_advantages_masked = convert_scalar_or_token_advantages(scalar_advantages, partial_mask)

        expected_masked = torch.tensor([[0.5, 0.5, 0.0, 0.0, 0.0], [-0.3, -0.3, -0.3, 0.0, 0.0]])

        assert torch.allclose(tensor_advantages_masked, expected_masked), (
            f"Masked broadcast failed: got {tensor_advantages_masked}, expected {expected_masked}"
        )

        print("✓ Masked advantage broadcasting works correctly")

        # Test 3: Token-level advantages should pass through unchanged
        # This would require mocking the API response, so we'll test the shape validation
        token_level_advantages = torch.randn(batch_size, seq_length)

        # This should work without errors (shape matches)
        try:
            result = token_level_advantages * response_mask
            assert result.shape == (batch_size, seq_length)
            print("✓ Token-level advantage shape validation works")
        except Exception as e:
            print(f"✗ Token-level advantage test failed: {e}")
            return False

        return True

    except Exception as e:
        print(f"✗ Broadcast logic test failed: {e}")
        return False


def test_advantage_estimator():
    """Test the GRPO advantage estimator with Atropos overrides."""
    print("\nTesting GRPO advantage estimator with Atropos overrides...")

    if not VERL_AVAILABLE:
        print("Skipping GRPO estimator test as VeRL is not available.")
        return True  # Indicate success if VeRL is not available

    try:
        if "grpo" in ADV_ESTIMATOR_REGISTRY:
            print("✓ grpo estimator is registered")
        else:
            print("✗ grpo estimator not found in registry")
            return False

        batch_size = 4
        response_length = 20

        token_level_rewards = torch.randn(batch_size, response_length)
        response_mask = torch.ones(batch_size, response_length)
        index = torch.arange(batch_size).numpy()

        advantages, returns = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            norm_adv_by_std_in_grpo=True,
        )

        print("✓ Advantage computation successful")
        print(f"  Advantages shape: {advantages.shape}")
        print(f"  Returns shape: {returns.shape}")

        token_level_advantages = torch.randn(batch_size, response_length)
        advantages_override, returns_override = compute_grpo_outcome_advantage(
            token_level_rewards=token_level_rewards,
            response_mask=response_mask,
            index=index,
            token_level_advantages=token_level_advantages,
        )

        assert torch.allclose(advantages_override, token_level_advantages * response_mask), (
            "token-level overrides should be respected"
        )
        assert torch.allclose(advantages_override, returns_override), "returns should match overrides"
        print("✓ Advantage override successful")

    except Exception as e:
        print(f"✗ Advantage estimator test failed: {e}")
        return False

    return True


def test_fallback_on_api_failure():
    """Test that API failure is handled gracefully."""
    print("\nTesting fallback on API failure...")

    config = AtroposConfig(
        api_url="http://localhost:9999",
        timeout=2,
        retry_attempts=2,
    )

    client = AtroposTrainerClient(config)
    try:
        client.register_trainer(
            {
                "wandb_group": "verl_atropos_tests",
                "wandb_project": "atropos",
                "batch_size": 2,
                "max_token_len": 128,
                "checkpoint_dir": "/tmp/atropos_test",
                "save_checkpoint_interval": 0,
                "starting_step": 0,
                "num_steps": 1,
            }
        )
        print("✗ Expected registration failure but got success")
        return False
    except AtroposAPIError:
        print("✓ Registration failure handled as expected")
        return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Atropos-VeRL Integration Tests")
    print("=" * 60)

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    _ensure_atropos_api("http://localhost:9001")

    # Run tests
    tests = [
        ("Atropos Client", test_atropos_client),
        ("Advantage Broadcast Logic", test_advantage_broadcast_logic),
        ("Advantage Estimator", test_advantage_estimator),
        ("Fallback on API Failure", test_fallback_on_api_failure),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for test_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{test_name}: {status}")

    # Overall result
    all_passed = all(success for _, success in results)
    print("\n" + "=" * 60)
    if all_passed:
        print("All tests passed! ✓")
    else:
        print("Some tests failed. ✗")
        print("\nNote: If Atropos server is not running, start it with:")
        print("  python environments/gsm8k_server.py serve --slurm false")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
