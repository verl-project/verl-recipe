# Copyright 2026 Huawei Technologies Co., Ltd. and/or its affiliates
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

from omegaconf import DictConfig

from verl.experimental.fully_async_policy.message_queue import MessageQueue, MessageQueueClient

logger = logging.getLogger(__name__)


def create_reward_queue(config: DictConfig, max_queue_size: int = 1000):
    # return MessageQueue.remote(config, max_queue_size, name="RewardQueue")
    return MessageQueue.options(name="RewardQueue").remote(config, max_queue_size)


class RewardQueueClient(MessageQueueClient):
    pass
