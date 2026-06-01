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
import random
from abc import ABC, abstractmethod
from typing import Optional


class BaseSampler(ABC):
    @abstractmethod
    def sample(self, indexes: list[int], count: int, **kwargs) -> list[int]:
        """
        采样逻辑接口
        :param indexes: 输入的可选索引池
        :param count: 返回索引的个数
        """
        pass


class RandomSampler(BaseSampler):
    def __init__(self, seed: int = 42, replace: bool = False):
        """
        :param seed: 随机种子
        :param replace: 是否允许重复采样
        """
        # 使用私有的随机实例，避免干扰全局随机状态
        self.seed = seed
        self.replace = replace
        self._random = random.Random(seed)

    def sample(self, indexes: list[int], count: int, **kwargs) -> list[int]:
        if not indexes or not count:
            raise ValueError("Sampler: Must provide indexes and count")
        if self.replace:
            return self._random.choices(indexes, k=count)
        else:
            actual_count = min(count, len(indexes))
            return self._random.sample(sorted(indexes), k=actual_count)


class VersionSampler(BaseSampler):
    def __init__(
        self, n_samples: int = 1, by_group: bool = False, selection_mode: str = "newest", group_ver: str = None
    ):
        """
        :param n_samples: 每组样本的数量
        :param by_group: 是否按组逻辑处理
        :param selection_mode: "newest" (降序, 取version最大) 或 "oldest" (升序, 取version最小)
        :param group_ver: "max" (group_version取组内最大), "min" (group_version取组内最小),
            或 None (根据selection_mode自动推断)
        """
        self.n_samples = max(1, n_samples)
        self.by_group = by_group

        # 排序方向
        if selection_mode == "newest":
            self.descending = True
        elif selection_mode == "oldest":
            self.descending = False
        else:
            raise ValueError("selection_mode must be 'newest' or 'oldest'")

        # 组版本特征提取方式
        if group_ver is None:
            # 向后兼容：未指定时根据 selection_mode 自动推断
            self.group_ver = max if selection_mode == "newest" else min
        elif group_ver == "max":
            self.group_ver = max
        elif group_ver == "min":
            self.group_ver = min
        else:
            raise ValueError("group_ver must be 'max', 'min', or None")

    def sample(self, indexes: list[int], count: int, versions: list[int], **kwargs) -> list[int]:
        if not indexes or not count:
            raise ValueError("Sampler: Must provide indexes and count")
        if versions is None:
            raise ValueError("VersionSampler requires 'versions' passed via kwargs at runtime.")

        # --- 模式 A: 普通采样 ---
        if not self.by_group:
            paired = sorted(zip(indexes, versions, strict=False), key=lambda x: x[1], reverse=self.descending)
            sorted_indexes = [p[0] for p in paired]
            return sorted_indexes if count is None else sorted_indexes[:count]

        # --- 模式 B: 分组模式 ---
        if len(indexes) % self.n_samples != 0:
            raise ValueError(f"Total indexes ({len(indexes)}) must be a multiple of n_samples ({self.n_samples})")
        if count is not None and count % self.n_samples != 0:
            raise ValueError(f"count ({count}) must be a multiple of n_samples ({self.n_samples})")

        groups = []
        for i in range(0, len(indexes), self.n_samples):
            group_indexes = indexes[i : i + self.n_samples]
            group_versions = versions[i : i + self.n_samples]

            # 动态选择 max 或 min 作为组特征
            groups.append({"data": group_indexes, "v_feature": self.group_ver(group_versions)})

        # 根据 descending 动态决定组间排序顺序
        groups.sort(key=lambda x: x["v_feature"], reverse=self.descending)

        num_groups = len(groups) if count is None else min(count // self.n_samples, len(groups))
        result = [idx for g in groups[:num_groups] for idx in g["data"]]

        return result


# TODO
class SeqLenBalSampler(BaseSampler):
    pass


class SameVersionSampler(BaseSampler):
    def __init__(self, target_version: int = None, selection_mode: str = "newest"):
        """
        用于从相同版本中获取所有索引的采样器.

        两种场景:
        1. 如果指定了 target_version: 返回该特定版本的索引
        2. 如果 selection_mode 是 "newest" 或 "oldest": 自动选择最新/最旧版本并返回其索引

        Args:
            target_version: 要采样的特定版本。如果为 None,则使用 selection_mode 自动选择。
            selection_mode: "newest"(最大版本)或 "oldest"(最小版本)。仅在 target_version 为 None 时使用。

        Raises:
            ValueError: 如果 selection_mode 无效。
        """
        self.target_version = target_version
        self.selection_mode = selection_mode

        if target_version is None and selection_mode not in ("newest", "oldest"):
            raise ValueError(f"selection_mode 必须是 'newest' 或 'oldest', 但得到的是 '{selection_mode}'")

    def sample(self, indexes: list[int], count: int, versions: list[int], **kwargs) -> Optional[list[int]]:
        """
        从相同版本中采样索引。

        Args:
            indexes: 可供采样的索引
            count: 返回的最大索引数量
            versions: 每个索引的版本号(与 indexes 对齐)
            **kwargs: 额外参数(用于兼容性)

        Returns:
            List[int]: 从相同版本中选定的索引
            None: 如果版本不存在或索引不足
        """
        if not indexes or not count:
            raise ValueError("SameVersionSampler: 必须提供 indexes 和 count")
        if versions is None:
            raise ValueError("SameVersionSampler 需要在运行时通过 kwargs 传递 'versions'")
        if len(indexes) != len(versions):
            raise ValueError(f"SameVersionSampler: indexes 长度({len(indexes)}) != versions 长度({len(versions)})")

        # 确定选择哪个版本
        if self.target_version is not None:
            selected_version = self.target_version
        else:
            if self.selection_mode == "newest":
                selected_version = max(versions)
            elif self.selection_mode == "oldest":
                selected_version = min(versions)
            else:
                raise ValueError(f"selection_mode 必须是 'newest' 或 'oldest', 但得到的是 '{self.selection_mode}'")

        # 找出选定版本的所有索引
        version_indexes = [idx for idx, ver in zip(indexes, versions, strict=False) if ver == selected_version]

        # 处理边缘情况
        if not version_indexes:
            return None  # 版本未找到

        if len(version_indexes) < count:
            return None  # 索引不足

        return version_indexes[:count]
