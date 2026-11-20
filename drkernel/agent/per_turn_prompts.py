# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""Per-turn prompt template engine — minimal port of DR.Kernel's design.

DR.Kernel's `multi_turn_kernel.yaml` describes how the user-facing
"feedback message" between assistant turns is constructed. Their full
engine supports arbitrary Python `condition` expressions (e.g.
``current_turn == 0``, ``make_up_tool_response == True``); we keep the
public schema 1:1 but only implement the two cases the kernel-RL config
actually uses, since extending it is additive.

Schema (mirrors `drkernel/kernel/config/prompt_config/multi_turn_kernel.yaml`):

    method: system
    prompt: null
    per_turn_prompts:
      - name: first_turn
        condition: "current_turn == 0"
        ...
      - name: tool_response
        condition: "make_up_tool_response == True"
        template: |
          ... {feedback} ...

The only field we *actually* read is `template` from the entry whose name
is `"tool_response"`. The first_turn entry exists for completeness but
the original prompt itself is what's already in the conversation.

If you later need richer condition logic, swap `_select_template` for a
real expression evaluator.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


_DEFAULT_TOOL_RESPONSE_TEMPLATE = (
    "Now you have received the server feedback for your last "
    "implementation. Based on that and all your previous responses, "
    "improve the implementation.\n\n"
    "Here is the server feedback. Please refer to this feedback to "
    "improve the implementation:\n"
    "Server feedback (status/metrics/errors):\n"
    "{feedback}\n\n"
    "Return an improved Triton implementation named `ModelNew` as a "
    "single ```python``` block. Let's think step by step."
)


class PerTurnPrompts:
    """Holds the loaded per-turn prompt config and renders feedback."""

    def __init__(self, prompts_cfg: Optional[DictConfig] = None):
        self.cfg = prompts_cfg
        self._tool_response_template = _resolve_tool_response_template(prompts_cfg)

    @classmethod
    def load(cls, prompt_config_path: Optional[str]) -> PerTurnPrompts:
        """Load from a Hydra-resolvable path. Returns a default-equipped
        instance when path is None or load fails (so the agent loop can
        still run with the built-in template)."""
        if not prompt_config_path:
            return cls(None)
        try:
            from verl.experimental.agent_loop.utils import resolve_config_path

            resolved = resolve_config_path(prompt_config_path)
            cfg = OmegaConf.load(resolved)
        except Exception as exc:
            logger.warning(
                "[PerTurnPrompts] Failed to load %s (%s); using built-in template",
                prompt_config_path,
                exc,
            )
            return cls(None)
        return cls(cfg)

    def render_feedback_message(self, feedback: str, turn_idx: int) -> str:
        """Render the user-facing feedback message for the next turn.

        `turn_idx` is the just-completed assistant turn index (0-based).
        DR.Kernel's `tool_response` template is reused for every post-turn
        feedback; if a future config adds turn-specific templates we'd
        switch on `turn_idx` here.
        """
        return self._tool_response_template.format(feedback=feedback)


def _resolve_tool_response_template(cfg: Optional[DictConfig]) -> str:
    """Pull the `tool_response` template string from a loaded config, or
    fall back to the DR.Kernel default."""
    if cfg is None:
        return _DEFAULT_TOOL_RESPONSE_TEMPLATE
    try:
        per_turn: Any = cfg.get("per_turn_prompts", None)
        if not per_turn:
            return _DEFAULT_TOOL_RESPONSE_TEMPLATE
        for entry in per_turn:
            name = entry.get("name", None)
            template = entry.get("template", None)
            if name == "tool_response" and template:
                return str(template)
    except Exception as exc:
        logger.warning("[PerTurnPrompts] config parse error: %s", exc)
    return _DEFAULT_TOOL_RESPONSE_TEMPLATE
