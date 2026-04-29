# Dynamo Backend 注册策略对比

**Last updated:** 2026-04-27
**Owner:** sopyang@nvidia.com
**Status:** Draft — 用于在 `recipe/dynamo` 路径与 `docs/design/dynamo_rollout_design.md`（in-tree 方案）之间做明确取舍

---

## 0. TL;DR

| | **方案 A：改 verl 本体** | **方案 B：在 recipe 侧 monkey-patch** |
|---|---|---|
| 触发方式 | `rollout.name=dynamo` 直接生效 | 入口处 `import recipe.dynamo` 触发副作用，再 `rollout.name=dynamo` |
| 主要代价 | 走 verl-project/verl 的 PR 评审，受 CLAUDE.md 的 fail-closed 与 duplicate-work 约束 | 依赖 verl 的私有 `_ROLLOUT_REGISTRY` / `RolloutReplicaRegistry` 接口；verl 内部重构会静默打破 |
| 主要收益 | 上游用户开箱即用；可扩展 `RolloutConfig` schema | 迭代速度不受 verl 评审节奏制约；不污染主仓库 |
| 适用阶段 | P3+：架构稳定、希望被全部 verl 用户消费 | P0–P2：设计验证、benchmark、demo |

**当前选择：B（recipe 侧）**。理由见 §5。

---

## 1. 两种方案的代码落点

### 方案 A — In-tree（原 `docs/design/dynamo_rollout_design.md` §4）

直接编辑 verl 仓库下：

```
verl/workers/rollout/base.py            # _ROLLOUT_REGISTRY 增 ("dynamo","async")
verl/workers/rollout/replica.py         # RolloutReplicaRegistry.register("dynamo", _load_dynamo)
verl/workers/rollout/dynamo_rollout/    # 新增整包
  ├── __init__.py
  ├── dynamo_rollout.py        # ServerAdapter
  ├── dynamo_async_server.py   # DynamoHttpServer + DynamoReplica
  ├── dynamo_frontend_launcher.py
  └── dynamo_worker_extension.py
verl/workers/config/rollout.py          # RolloutConfig 新增 dynamo 子段（可选）
```

发行渠道：`pip install verl>=X.Y.Z`。

### 方案 B — Recipe-side（当前已实现）

只编辑 verl-recipe 仓库下：

```
recipe/dynamo/
├── __init__.py                 # 副作用：写入 _ROLLOUT_REGISTRY 与 RolloutReplicaRegistry
├── dynamo_rollout.py           # ServerAdapter
└── dynamo_async_server.py      # DynamoHttpServer + DynamoReplica（骨架）
```

启用方式：用户的 `main_xxx.py` 顶部添加一行 `import recipe.dynamo`，剩下的 hydra 配置一致。

---

## 2. 对比维度

### 2.1 上游耦合与发行节奏

- **方案 A**：每次改动需上游评审。CLAUDE.md 的 fail-closed 条款要求：
  - 不开 low-value busywork PR；
  - 必须做 `gh issue view` / `gh pr list` duplicate-work 检查；
  - PR 描述需包含 AI 协助声明、本地测试输出、与现有 backend 的差异说明。

  好处是改动 visible 给所有 verl 用户；代价是每个 P 阶段都吃一轮 review 周期。

- **方案 B**：recipe 仓库自治。verl 当前对每个 recipe 已经 pin 了具体 verl 版本（commit `e7f8895`），所以 recipe 侧的 monkey-patch 锁定在已知 verl 版本上，向后兼容由 recipe 自己负责。

### 2.2 接口稳定性

- **方案 A**：所有改动都在 verl 内部，重构时会被一起更新。
- **方案 B**：依赖两个 **private** 标识：
  - `verl.workers.rollout.base._ROLLOUT_REGISTRY`（前缀下划线）；
  - `verl.workers.rollout.replica.RolloutReplicaRegistry`（无下划线，但仍是内部 factory）。

  如果 verl 把注册机制换成 entry-points / setuptools plugin，方案 B 会**静默失效**——dict 写入还是成功，但运行时不再被读到。需要 recipe 侧加 import-time assertion 兜底（例如对 `getattr(base, "_ROLLOUT_REGISTRY", None) is not None` 做检查）。

  当前 recipe 中已有同款做法的先例（`recipe/r1_ascend/megatron_workers.py:37`），说明这是社区可接受的范式，但稳定性保障来自“别人也这么写”，**不是**正式契约。

### 2.3 配置 schema

- **方案 A** 可在 `verl/workers/config/rollout.py:168` 的 `RolloutConfig` 上加：
  ```python
  dynamo: Optional[DynamoConfig] = None  # backend / disagg / frontend / etc.
  ```
  并附带 `Literal[...]` 校验，让 `name=dynamo` 在配置层就过校验。

- **方案 B** 受限于 verl 现有 schema：
  - `RolloutConfig.name` 是 `Optional[str]`（`verl/workers/config/rollout.py:179`），所以 `name=dynamo` 本身可以通过；
  - 但 dynamo-only 字段（`backend`/`disagg`/`frontend.port`/…）没有官方 schema 位置。可选缓解：
    1. 用 hydra 的 `+rollout.dynamo.foo=bar`（OmegaConf 会接受额外键，但不再有类型校验）；
    2. 在 recipe 侧定义独立的 `DynamoExtraConfig` dataclass，在 `ServerAdapter.__init__` 里用 `omega_conf_to_dataclass` 解析 `self.config.get("dynamo")`；
    3. 若必须强类型，可在 `recipe/dynamo/__init__.py` 里 monkey-patch `RolloutConfig`，但这扩大了私有接口耦合面，**不推荐**。

### 2.4 用户启用成本

- **方案 A**：`pip install verl[dynamo]` + `rollout.name=dynamo`，零代码改动。
- **方案 B**：用户的 main 入口需要显式 `import recipe.dynamo`。容易出错的失败模式：
  - 用户忘了 import → `get_rollout_class` 抛 AssertionError，错误信息只说 `Rollout dynamo with mode async not found`，不指向缺失的 import。
  - 用户在 worker 子进程里没继承 import（Ray actor 跨进程边界时常见）→ 主进程能解析，worker 上找不到。

  缓解：在 recipe 的训练脚本入口（`recipe/dynamo/main_dynamo.py`）固定写好 import；并在 `recipe/dynamo/__init__.py` 的注册行附近留一段 docstring 解释这个陷阱。

### 2.5 可测试性

- **方案 A**：测试可加进 verl CI（`tests/workers/rollout/`），跑在 verl matrix 里。
- **方案 B**：测试只能在 verl-recipe CI 跑，且需要先安装匹配版本的 verl。recipe CI 当前已经覆盖了 r1_ascend / collabllm 等带 monkey-patch 的 recipe，所以基础设施是齐的。

### 2.6 可观测性 / debug

- **方案 A**：堆栈直接指向 `verl/workers/rollout/dynamo_rollout/...`。
- **方案 B**：堆栈指向 `recipe/dynamo/...`，但 `_ROLLOUT_REGISTRY` 那一层来自 verl，新加入的同事可能不知道注册发生在哪里。**缓解**：在 `recipe/dynamo/__init__.py` 顶部 docstring 明确说"importing this package mutates verl state"。

### 2.7 锁定 verl 版本的影响

- **方案 A**：dynamo 实现与 verl 一起发版，无版本错配。
- **方案 B**：每条 recipe 已 pin verl 版本（commit `e7f8895`）。如果 dynamo 想跟随 verl 升级，需手动 bump pin 并验证 monkey-patch 还成立——当前正是这种 workflow，所以没有额外成本。

---

## 3. 详细对比表

| 维度 | A: in-tree | B: recipe monkey-patch | 当前权重 |
|---|---|---|---|
| 上游 review 周期影响进度 | 高 | 无 | 高（P1-P2 期间天天改） |
| verl 用户开箱可用 | ✅ | ❌（需 `import recipe.dynamo`） | 低（P1 范围用户即作者） |
| 私有接口耦合风险 | 无 | 中（依赖 `_ROLLOUT_REGISTRY` 名称） | 中 |
| 配置 schema 强类型 | ✅ | ⚠️（需 recipe 侧补 dataclass） | 中 |
| Disagg/etcd 等 dynamo-only 配置位置 | `RolloutConfig.dynamo` | recipe 内部 dataclass | 中 |
| Ray actor 跨进程 import 安全 | ✅ | ⚠️（需在 worker 进程 entrypoint 触发 import） | 中 |
| 出问题时定位路径 | verl 主仓 | recipe 仓 | 低 |
| CI 覆盖 | verl matrix | recipe matrix | 低（recipe CI 够用） |
| 与现有 recipe（r1_ascend）一致 | ❌ | ✅ | 中 |
| Fail-closed (CLAUDE.md) 合规检查 | 必须 | 不必（不开上游 PR） | 中 |
| 升级到 A 的代价 | n/a | 低（直接搬目录 + 删 monkey-patch） | — |

---

## 4. 各场景下的推荐

| 场景 | 推荐方案 |
|---|---|
| 设计/原型验证（P0–P1）：架构未稳定，每天可能改 ServerAdapter / Replica 形状 | **B**。上游 PR 节奏吃不消。 |
| FP8 / disagg benchmark（P2–P3）：需要 dynamo-only 字段进配置 | **B**，但补 `recipe/dynamo/config.py` 给强类型；disagg 字段稳定后再考虑 A。 |
| 多机扩展（P4）：需要 etcd service discovery 等基础设施 | 评估二选一：若 etcd 配置只对 dynamo 用户可见，留 B；若希望复用 verl 的现有 service-discovery 抽象，转 A。 |
| 正式发布给所有 verl 用户（P5）| **A**。届时已有稳定接口、配置、测试，PR 走完整 review。 |
| 内部 fork 临时支持，不打算上游 | **B** 永久。配套写好 import 触发约束与 verl 版本 pin。 |

---

## 5. 当前选择（B）的代价清单与缓解

| 风险 | 缓解措施 | 实际生效位置 |
|---|---|---|
| 用户忘了 `import recipe.dynamo` | recipe 配套 `main_dynamo.py` 顶部固定 import；写到 README 第一行 | TODO（P1 落地时） |
| Ray actor 子进程不继承父进程 import | `recipe/dynamo/__init__.py` 的注册行需在 worker actor 的 module 级别也被触发——通常通过 `runtime_env.py_modules` 或 worker 文件顶部 import | TODO（在写 DynamoReplica.launch_servers 时确认） |
| verl 把 `_ROLLOUT_REGISTRY` 重命名 / 改成 entry-points | recipe import 时加 `assert hasattr(base, "_ROLLOUT_REGISTRY"), "verl rollout registry contract changed; update recipe/dynamo/__init__.py"` | `recipe/dynamo/__init__.py` 可在下次迭代加 |
| dynamo-only 字段无强类型 | recipe 侧加 `DynamoExtraConfig` dataclass + `omega_conf_to_dataclass` 解析 | TODO（P1） |
| 升级 verl pin 时 monkey-patch 失效不被发现 | recipe CI 跑一个 smoke test：`import recipe.dynamo; assert ("dynamo","async") in _ROLLOUT_REGISTRY` | TODO（P1 加入 pytest） |

---

## 6. 升级路径：B → A

当方案 B 通过 P1–P3 验证、配置稳定时，搬到 A 的步骤是机械的：

1. 把 `recipe/dynamo/dynamo_rollout.py` / `dynamo_async_server.py` 整体复制到 `verl/workers/rollout/dynamo_rollout/`。
2. 把 `recipe/dynamo/__init__.py` 里的 monkey-patch 行搬到 `verl/workers/rollout/base.py:83` 的 dict 字面量与 `verl/workers/rollout/replica.py:391` 的 `RolloutReplicaRegistry.register` 调用。
3. 删除 `recipe/dynamo/__init__.py` 的副作用注册（保留作为兼容入口 `from verl.workers.rollout import dynamo_rollout`）。
4. 走 verl-project/verl 的 PR 流程，附 CLAUDE.md 要求的：
   - duplicate-work 检查输出；
   - benchmark 数据（与 vllm/trtllm backend 的精度/吞吐对比）；
   - AI 协助声明；
   - "submitting human reviewed every changed line" 声明。

迁移本身**不会**破坏现有 recipe 用户的脚本（注册键不变；类位置变了但 recipe 入口仍可 re-export）。

---

## 7. 与 `docs/design/dynamo_rollout_design.md` 的关系

`docs/design/dynamo_rollout_design.md` 描述的是**方案 A 的最终目标形态**（in-tree、`RolloutConfig` 扩展、完整的 disagg + KVBM 路径）。本文档不替代它，而是说明：

- P0–P3 阶段在 **recipe 侧** 落地（方案 B），代码骨架在 `recipe/dynamo/`；
- P4+ 视稳定度迁移到 `docs/design/dynamo_rollout_design.md` 描述的 in-tree 形态；
- 两份文档共同定义 dynamo backend 的 long-running roadmap。

---

## 8. 参考

- 既有 monkey-patch 先例：`recipe/r1_ascend/megatron_workers.py:37`
- verl 注册中心：`verl/workers/rollout/base.py:83`、`verl/workers/rollout/replica.py:306`
- recipe 版本 pin commit：`e7f8895 [refactor] pin verl versions of each recipe`
- verl 仓库贡献规则：`CLAUDE.md`（duplicate-work / fail-closed / accountability 段落）
