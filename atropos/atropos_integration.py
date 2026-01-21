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

"""
Atropos-VeRL Integration Module

This module provides the core integration between VeRL and Atropos environments,
using the documented Atropos trainer API (register + batch polling).
"""

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import requests
import torch

logger = logging.getLogger(__name__)


class AtroposAPIError(Exception):
    """Raised when Atropos API operations fail"""
    pass


@dataclass
class AtroposConfig:
    """Configuration for Atropos integration"""
    api_url: str = "http://localhost:9001"
    timeout: int = 30
    retry_attempts: int = 10
    retry_delay: float = 0.5
    max_wait_time: float = 30.0
    use_advantages: bool = True
    fallback_to_standard: bool = True


class AtroposTrainerClient:
    """
    Client for interacting with the Atropos trainer API.

    This client follows the documented trainer flow:
    - POST /register
    - GET /batch
    - GET /status
    """

    def __init__(self, config: AtroposConfig):
        self.config = config
        self.session = requests.Session()
        self.trainer_uuid: Optional[str] = None

    def is_available(self) -> bool:
        try:
            response = self.session.get(f"{self.config.api_url}/status", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def register_trainer(self, registration: dict[str, Any]) -> str:
        try:
            response = self.session.post(
                f"{self.config.api_url}/register",
                json=registration,
                timeout=self.config.timeout,
            )
        except requests.exceptions.RequestException as e:
            raise AtroposAPIError(f"Failed to register with Atropos: {e}") from e

        if response.status_code != 200:
            raise AtroposAPIError(
                f"Atropos /register failed: {response.status_code} - {response.text}"
            )

        result = response.json()
        trainer_uuid = result.get("uuid") or result.get("trainer_uuid") or result.get("trainer_id")
        if not trainer_uuid:
            raise AtroposAPIError(f"Atropos /register response missing uuid: {result}")

        self.trainer_uuid = trainer_uuid
        logger.info("Registered trainer with Atropos: %s", trainer_uuid)
        return trainer_uuid

    def get_batch(self) -> list[dict[str, Any]]:
        last_error = None
        cumulative_wait_time = 0.0

        for attempt in range(self.config.retry_attempts):
            try:
                response = self.session.get(
                    f"{self.config.api_url}/batch",
                    timeout=self.config.timeout,
                )

                if response.status_code == 200:
                    result = response.json()
                    return result.get("batch", [])

                last_error = f"Atropos /batch status {response.status_code}: {response.text}"
                if 400 <= response.status_code < 500:
                    break
            except requests.exceptions.RequestException as e:
                last_error = f"Failed to poll Atropos /batch: {e}"

            if attempt < self.config.retry_attempts - 1:
                wait_time = self.config.retry_delay
                if cumulative_wait_time + wait_time > self.config.max_wait_time:
                    break
                time.sleep(wait_time)
                cumulative_wait_time += wait_time

        raise AtroposAPIError(f"Failed to get batch from Atropos: {last_error}")


def convert_scalar_or_token_advantages(
    advantages: list[float] | list[list[float]] | torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Convert response-level advantages to token-level using response mask."""
    if isinstance(advantages, torch.Tensor):
        adv_list = advantages.tolist()
    else:
        adv_list = advantages

    batch_size, seq_len = response_mask.shape
    token_advantages = torch.zeros(
        batch_size,
        seq_len,
        dtype=torch.float32,
        device=response_mask.device,
    )

    for i, adv in enumerate(adv_list):
        mask = response_mask[i].float()
        if isinstance(adv, (list, tuple)):
            adv_tensor = torch.as_tensor(adv, dtype=torch.float32, device=response_mask.device)
            if adv_tensor.ndim != 1:
                raise ValueError(f"Unexpected advantage shape for sample {i}: {adv_tensor.shape}")
            length = min(seq_len, adv_tensor.shape[0])
            token_advantages[i, :length] = adv_tensor[:length]
            token_advantages[i] *= mask
        else:
            token_advantages[i] = float(adv) * mask

    return token_advantages
