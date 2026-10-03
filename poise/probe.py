# Copyright 2026 POISE authors
# SPDX-License-Identifier: Apache-2.0
"""Cross-rollout advantages and online PCA/Ridge probes, independent of veRL."""

from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeConfig:
    bootstrap_steps: int = 16
    warmup_steps: int = 8
    layer: int = 19
    pool_tokens: int = 10
    prompt_pca_dim: int = 32
    response_pca_dim: int = 64
    ridge_alpha: float = 100.0
    seed: int = 42
    buffer_rows: dict[str, int] = field(default_factory=lambda: {"math": 4096, "code": 3072, "other": 3072})

    def __post_init__(self):
        if min(self.bootstrap_steps, self.warmup_steps, self.pool_tokens) <= 0 or self.layer < 0:
            raise ValueError("Bootstrap, warmup and pooling must be positive; layer must be nonnegative")
        if min(self.prompt_pca_dim, self.response_pca_dim) < 1 or self.ridge_alpha <= 0:
            raise ValueError("PCA dimensions and ridge_alpha must be positive")
        if not self.buffer_rows or set(self.buffer_rows) - {"math", "code", "other"}:
            raise ValueError("buffer_rows must contain one or more of math, code, other")
        if any(limit < 2 for limit in self.buffer_rows.values()):
            raise ValueError("Each domain buffer must hold at least one pair")


def domain_for(source: str) -> str:
    source = str(source).strip().lower()
    if source.startswith("math") or source in {"openai/gsm8k", "lighteval/math", "huggingfaceh4/math-500"}:
        return "math"
    if source.startswith("codegen") or source in {"codecontests", "apps", "codeforces", "taco"}:
        return "code"
    return "other"


def sibling_indices(uids, domains):
    """Find siblings by prompt ID, including after token balancing reorders rows."""
    if len(uids) != len(domains) or not len(uids):
        raise ValueError("Expected nonempty, aligned prompt IDs and domains")
    groups = defaultdict(list)
    for index, uid in enumerate(uids):
        groups[str(uid)].append(index)
    siblings = np.empty(len(uids), dtype=np.int64)
    for uid, indices in groups.items():
        if len(indices) != 2:
            raise ValueError(f"POISE requires exactly two rollouts per prompt; {uid!r} has {len(indices)}")
        i, j = indices
        if domains[i] != domains[j]:
            raise ValueError(f"Siblings for {uid!r} belong to different domains")
        siblings[i], siblings[j] = j, i
    return siblings


class ProbeRow(NamedTuple):
    prompt: np.ndarray
    response: np.ndarray
    scalars: np.ndarray
    target: float


class RidgeProbe:
    def __init__(self, config: ProbeConfig):
        self.config = config

    def fit(self, rows: list[ProbeRow]):
        prompt = np.stack([row.prompt for row in rows])
        response = np.stack([row.response for row in rows])
        scalars = np.stack([row.scalars for row in rows])
        targets = np.asarray([row.target for row in rows], dtype=np.float32)
        self.prompt_pca = PCA(
            n_components=min(self.config.prompt_pca_dim, *prompt.shape),
            svd_solver="randomized",
            random_state=self.config.seed,
        ).fit(prompt)
        self.response_pca = PCA(
            n_components=min(self.config.response_pca_dim, *response.shape),
            svd_solver="randomized",
            random_state=self.config.seed,
        ).fit(response)
        self.regressor = make_pipeline(StandardScaler(), Ridge(alpha=self.config.ridge_alpha))
        self.regressor.fit(self._features(prompt, response, scalars), targets)
        return self

    def _features(self, prompt, response, scalars):
        return np.concatenate(
            [self.prompt_pca.transform(prompt), self.response_pca.transform(response), scalars], axis=1
        ).astype(np.float32)

    def predict(self, prompt, response, scalars):
        return np.clip(self.regressor.predict(self._features(prompt, response, scalars)), 0.0, 1.0)


class ProbeBank:
    """Predict first; append and refit only after the actor update succeeds."""

    def __init__(self, config: ProbeConfig):
        self.config = config
        self.completed_steps = 0
        self.buffers = {domain: deque() for domain in config.buffer_rows}
        self.observed_steps = dict.fromkeys(config.buffer_rows, 0)
        self.probes = {}
        self._pending = None

    def advantages(self, *, uids, domains, rewards, prompt, response, scalars):
        if self._pending is not None:
            raise RuntimeError("Commit the previous actor update before preparing another batch")
        siblings = sibling_indices(uids, domains)
        count = len(uids)
        rewards = np.asarray(rewards, dtype=np.float32)
        prompt, response, scalars = (np.asarray(x, dtype=np.float32) for x in (prompt, response, scalars))
        if rewards.shape != (count,) or any(x.ndim != 2 or len(x) != count for x in (prompt, response, scalars)):
            raise ValueError("Rewards and feature matrices must have one row per rollout")
        if scalars.shape[1] != 3 or not all(np.isfinite(x).all() for x in (rewards, prompt, response, scalars)):
            raise ValueError("Expected finite features and exactly three entropy scalars")
        if ((rewards < 0) | (rewards > 1)).any():
            raise ValueError("POISE correctness rewards must lie in [0, 1]")
        if set(domains) - set(self.buffers):
            raise ValueError(f"Unconfigured domains: {set(domains) - set(self.buffers)}")

        bootstrap = self.completed_steps < self.config.bootstrap_steps
        predictions = rewards.copy()
        if not bootstrap:
            for domain in set(domains):
                indices = np.flatnonzero(np.asarray(domains) == domain)
                predictions[indices] = self.probes[domain].predict(prompt[indices], response[indices], scalars[indices])
        baselines = predictions[siblings]
        advantages = rewards - baselines
        targets = rewards[siblings]
        pending = defaultdict(list)
        for i, j in enumerate(siblings):
            if i < j:
                pair = [ProbeRow(prompt[k].copy(), response[k].copy(), scalars[k].copy(), targets[k]) for k in (i, j)]
                pending[domains[i]].append(pair)
        self._pending = pending
        metrics = {
            "poise/bootstrap": float(bootstrap),
            "poise/baseline_mean": float(baselines.mean()),
            "poise/advantage_std": float(advantages.std()),
        }
        return advantages, metrics

    def commit(self):
        """Complete one actor update, retaining complete pairs in each FIFO."""
        if self._pending is None:
            raise RuntimeError("No actor update is pending")
        for domain, pairs in self._pending.items():
            buffer = self.buffers[domain]
            buffer.extend(pairs)
            while 2 * len(buffer) > self.config.buffer_rows[domain]:
                buffer.popleft()
            self.observed_steps[domain] += 1
        next_step = self.completed_steps + 1
        transition = next_step == self.config.bootstrap_steps
        if transition:
            missing = [domain for domain, buffer in self.buffers.items() if not buffer]
            if missing:
                raise ValueError(f"Bootstrap received no examples for domains {missing}; adjust poise.buffer_rows")
            # Bootstrap initializes every domain at the same training step, as in the reference implementation.
            self.observed_steps = dict.fromkeys(self.buffers, next_step)
        metrics = {}
        for domain, buffer in self.buffers.items():
            refit = transition or (
                next_step > self.config.bootstrap_steps
                and domain in self._pending
                and self.observed_steps[domain] >= self.config.warmup_steps
            )
            if refit:
                self.probes[domain] = RidgeProbe(self.config).fit([row for pair in buffer for row in pair])
            metrics[f"poise/{domain}/buffer_rows"] = float(2 * len(buffer))
            metrics[f"poise/{domain}/refit"] = float(refit)
        self.completed_steps = next_step
        self._pending = None
        return metrics

    def save(self, directory):
        if self._pending is not None:
            raise RuntimeError("Cannot checkpoint an uncommitted actor update")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1,
            "config": asdict(self.config),
            "completed_steps": self.completed_steps,
            "buffers": self.buffers,
            "observed_steps": self.observed_steps,
            "probes": self.probes,
        }
        temporary = directory / "poise.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(directory / "poise.pt")

    def load(self, directory, expected_step):
        # Checkpoints contain sklearn objects; load only checkpoints you trust, like veRL model checkpoints.
        state = torch.load(Path(directory) / "poise.pt", map_location="cpu", weights_only=False)
        if state["version"] != 1 or state["config"] != asdict(self.config):
            raise ValueError("POISE checkpoint version or configuration differs from this run")
        if state["completed_steps"] != expected_step:
            raise ValueError("POISE and actor checkpoint steps differ")
        for name in ("completed_steps", "buffers", "observed_steps", "probes"):
            setattr(self, name, state[name])
