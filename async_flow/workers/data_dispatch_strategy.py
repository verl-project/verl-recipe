#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.#
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Optional

import torch
import torch.distributed as dist
from recipe.async_flow.utils.transfer_queue.tq_client import TransferQueueClient

from verl.utils.device import get_device_id, get_device_name

logger = logging.getLogger(__name__)


class EngineBackend(Enum):
    """并行后端类型枚举。

    定义不同的并行策略，用于决定 TQ 数据的 dispatch/collect 行为
    """

    NONE = auto()
    FSDP = auto()
    VLLM = auto()
    MEGATRON = auto()


class DispatchStrategy(ABC):
    """并行策略基类，定义 dispatch/collect 接口。"""

    @abstractmethod
    def is_master_for_dispatch(self) -> bool:
        """判断当前 rank 是否为 dispatch master（负责从 TQ 获取数据）。"""
        pass

    @abstractmethod
    def is_master_for_collect(self) -> bool:
        """判断当前 rank 是否为 collect master（负责向 TQ 写入数据）。"""
        pass

    @abstractmethod
    async def dispatch_data(
        self,
        *,
        tq_client: TransferQueueClient,
        fetch_params: dict[str, Any],
    ) -> tuple[Optional[dict], Optional[list[int]]]:
        """从 TQ 获取数据并分发到所有需要的 rank。

        Args:
            tq_client: TransferQueueClient 实例
            fetch_params: 传递给 get_experience_async 的参数

        Returns:
            (payload, indexes): 所有参与计算的 rank 都会收到数据
        """
        pass

    @abstractmethod
    async def collect_data(
        self,
        *,
        tq_client: TransferQueueClient,
        put_params: dict[str, Any],
    ) -> None:
        """收集结果并写入 TQ。

        Args:
            tq_client: TransferQueueClient 实例
            put_params: 传递给 put_experience_async 的参数，包含 data_dict, indexes, version 等
        """
        pass


class NoneStrategy(DispatchStrategy):
    """无并行策略，单进程直接访问 TQ。"""

    def is_master_for_dispatch(self) -> bool:
        return True

    def is_master_for_collect(self) -> bool:
        return True

    async def dispatch_data(
        self,
        *,
        tq_client: TransferQueueClient,
        fetch_params: dict[str, Any],
    ) -> tuple[Optional[dict], Optional[list[int]]]:
        payload, indexes = await tq_client.get_experience_async(**fetch_params)
        return payload, indexes

    async def collect_data(
        self,
        *,
        tq_client: TransferQueueClient,
        put_params: dict[str, Any],
    ) -> None:
        # 各 rank 独立写回自己的结果
        await tq_client.put_experience_async(**put_params)


class FSDPStrategy(DispatchStrategy):
    """FSDP 数据并行策略。

    每个 rank 独立从 TQ 获取不同的数据，独立写回结果。
    TQ 需要保证不同 rank 获取到不同的数据（通过 consumer 机制）。

    可选 pre-dispatch 就绪检查：当提供 ``query_usable_count_fn`` 时，
    ``dispatch_data`` 会先轮询 TQ，跨 rank all_reduce(MIN) 同步，
    确认可用样本 >= ``needed_count`` 后才真正 fetch 数据。
    """

    def __init__(
        self,
        rank: int = 0,
        world_size: int = 1,
        query_usable_count_fn: Optional[Callable[[], Any]] = None,
        needed_count_fn: Optional[Callable[[], int]] = None,
    ):
        self.rank = rank
        self.world_size = world_size
        # pre-dispatch readiness check 回调
        self._query_usable_count_fn = query_usable_count_fn
        self._needed_count_fn = needed_count_fn

    def is_master_for_dispatch(self) -> bool:
        # FSDP 中每个 rank 都独立获取数据，都是自己的 "master"
        return True

    def is_master_for_collect(self) -> bool:
        # FSDP 中每个 rank 都独立写回数据
        return True

    async def dispatch_data(
        self,
        *,
        tq_client: TransferQueueClient,
        fetch_params: dict[str, Any],
    ) -> tuple[Optional[dict], Optional[list[int]]]:
        """各 rank 独立从 TQ 获取数据，然后 barrier 同步。

        TQ 的 consumer 机制保证不同 rank 获取到不同的数据。
        barrier 确保所有 rank 都拿到数据后再开始 FSDP 计算。
        """
        if self._query_usable_count_fn is not None:
            ready = await self._is_usable_cnt_ready()
            if not ready:
                return None, None

        payload, indexes = await tq_client.get_experience_async(**fetch_params)
        return payload, indexes

    async def _is_usable_cnt_ready(self) -> bool:
        """轮询 TQ 可用样本数，跨 rank all_reduce(MIN) 同步，直到数据足够。"""
        needed_global = self._needed_count_fn() if self._needed_count_fn else 0
        dp_world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1

        local_usable_experience_count = await self._query_usable_count_fn()

        # 跨 rank 同步：取 MIN，防止 RPC 抖动导致 rank 间读到不一致的值。
        if dp_world_size > 1:
            device = torch.device(f"{get_device_name()}:{get_device_id()}")
            flag = torch.tensor([local_usable_experience_count], dtype=torch.int64, device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            min_usable_experience_count = int(flag.item())
        else:
            min_usable_experience_count = local_usable_experience_count

        return min_usable_experience_count >= needed_global

    async def collect_data(
        self,
        *,
        tq_client: TransferQueueClient,
        put_params: dict[str, Any],
    ) -> None:
        """各 rank 独立写回 TQ。
        每个 rank 处理不同数据，独立写回自己的结果。
        """
        await tq_client.put_experience_async(**put_params)


class VLLMTPStrategy(DispatchStrategy):
    """vLLM 张量并行策略。

    vLLM 内部管理 TP 通信，只有 TP rank 0 需要与外部交互：
    - TP rank 0 从 TQ 获取数据，调用 vLLM.generate()
    - vLLM 内部自动将输入分发到所有 TP rank
    - TP rank 0 收集输出并写回 TQ
    """

    def __init__(self, tp_rank: int = 0, tp_size: int = 1, tp_group=None, tp_master_global_rank: int = 0):
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.tp_group = tp_group
        self.tp_master_global_rank = tp_master_global_rank
        logger.info(
            f"VLLMTPStrategy initialized: tp_rank={tp_rank}, tp_size={tp_size}, "
            f"master_global_rank={tp_master_global_rank}"
        )

    def is_master_for_dispatch(self) -> bool:
        return self.tp_rank == 0

    def is_master_for_collect(self) -> bool:
        return self.tp_rank == 0

    async def dispatch_data(
        self,
        *,
        tq_client: "TransferQueueClient",
        fetch_params: dict[str, Any],
    ) -> tuple[Optional[dict], Optional[list[int]]]:
        """TP rank 0 获取数据，broadcast 到所有 TP rank。

        确保所有 TP rank 同步获取相同的数据。
        """
        # 单卡模式，直接获取
        if self.tp_size == 1:
            payload, indexes = await tq_client.get_experience_async(**fetch_params)
            return payload, indexes

        import torch.distributed as dist

        # TP rank 0 从 TQ 获取数据
        if self.is_master_for_dispatch():
            payload, indexes = await tq_client.get_experience_async(**fetch_params)
            has_data = payload is not None and indexes is not None and len(indexes) > 0
            # 将 tensor 转移到 CPU 以便序列化 broadcast
            # TODO: 此处存在 GPU -> CPU 的transfer，可能存在性能开销
            if has_data and payload is not None:
                payload = self._tensors_to_cpu(payload)
            objects = [has_data, payload, indexes]
        else:
            objects = [None, None, None]

        # Broadcast 到所有 TP rank
        dist.broadcast_object_list(objects, src=self.tp_master_global_rank, group=self.tp_group)

        has_data, payload, indexes = objects
        if not has_data:
            return None, None
        return payload, indexes

    async def collect_data(
        self,
        *,
        tq_client: "TransferQueueClient",
        put_params: dict[str, Any],
    ) -> None:
        """收集输出并写回 TQ。

        vLLM generate 后所有 TP rank 拥有相同的输出，只需 rank 0 写回。
        需要同步确保所有 rank 完成处理后再继续。
        """
        import torch.distributed as dist

        # 同步所有 TP rank，确保 generate 完成
        if self.tp_size > 1 and self.tp_group is not None:
            dist.barrier(group=self.tp_group)

        # 只有 TP rank 0 写回
        if self.is_master_for_collect():
            await tq_client.put_experience_async(**put_params)

    def _tensors_to_cpu(self, data: dict[str, Any]) -> dict[str, Any]:
        """将 dict 中的 tensor 转移到 CPU（用于序列化）。"""
        result = {}
        for key, value in data.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu()
            elif isinstance(value, list):
                result[key] = [v.cpu() if isinstance(v, torch.Tensor) else v for v in value]
            else:
                result[key] = value
        return result


class MegatronStrategy(DispatchStrategy):
    """Megatron 并行策略。

    TODO: 实现 Megatron TP/PP/CP 的 dispatch/collect 逻辑。
    - Dispatch: TP rank 0 且 PP rank 0 从 TQ 获取数据，broadcast 到所有 TP/PP rank
    - Collect: TP rank 0 且 PP last stage 写回 TQ
    """

    def __init__(self, **kwargs):
        # TODO: 添加 tp_rank, tp_size, pp_rank, pp_size, cp_rank, cp_size 等参数
        self._kwargs = kwargs

    def is_master_for_dispatch(self) -> bool:
        # TODO: return tp_rank == 0 and pp_rank == 0 and cp_rank == 0
        return True

    def is_master_for_collect(self) -> bool:
        # TODO: return tp_rank == 0 and pp_rank == pp_size - 1 and cp_rank == 0
        return True

    async def dispatch_data(
        self,
        *,
        tq_client: "TransferQueueClient",
        fetch_params: dict[str, Any],
    ) -> tuple[Optional[dict], Optional[list[int]]]:
        # TODO: 实现 master 获取数据后 broadcast 到所有 TP/PP/CP rank
        payload, indexes = await tq_client.get_experience_async(**fetch_params)
        return payload, indexes

    async def collect_data(
        self,
        *,
        tq_client: "TransferQueueClient",
        put_params: dict[str, Any],
    ) -> None:
        # TODO: 只有 collect master 写回
        await tq_client.put_experience_async(**put_params)


def create_dispatch_strategy(
    backend: EngineBackend,
    **kwargs,
) -> DispatchStrategy:
    """根据 backend 类型创建对应的并行策略。

    Args:
        backend: 并行后端类型
        **kwargs: 策略特定的参数（如 rank, size, group 等）

    Returns:
        DispatchStrategy 实例
    """
    strategy_map = {
        EngineBackend.NONE: NoneStrategy,
        EngineBackend.FSDP: FSDPStrategy,
        EngineBackend.VLLM: VLLMTPStrategy,
        EngineBackend.MEGATRON: MegatronStrategy,
    }

    strategy_cls = strategy_map.get(backend)
    if strategy_cls is None:
        raise ValueError(f"Unknown parallel backend: {backend}")

    return strategy_cls(**kwargs)
