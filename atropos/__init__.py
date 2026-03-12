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
Atropos Recipe for VERL.
GRPO integration with optional token-level advantage overrides from Atropos.
"""

from .atropos_integration import AtroposConfig, AtroposTrainerClient
from .grpo_atropos_trainer import RayGRPOAtroposTrainer

__all__ = [
    "AtroposConfig",
    "AtroposTrainerClient",
    "RayGRPOAtroposTrainer",
]
