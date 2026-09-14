# Copyright (c) 2026, HUAWEI CORPORATION. All rights reserved.

"""Code-block extraction for the DR.Kernel multi-turn agent loop.

Patterns lifted verbatim from
`drkernel/KernelGYM/drkernel/kernel/workers/agent/kernel_agent.py::KernelAgent`
so the agent loop's "did the model emit something to evaluate?" decision
matches DR.Kernel's behavior bit-for-bit.

Two extractors:

- `extract_answer_block` — looks for the explicit ```answer<newline>...```
  fence DR.Kernel uses as a "I'm done, this is the final answer" marker.
- `extract_kernel_code` — kernel-specific markers (``# Kernel
  Implementation``, ``# Your implementation:``, ``# Generated kernel:``,
  ``# Kernel`` fenced) with a fallback to the *last* generic ```python```
  block in the response. Matches `kernel_reward.extract_kernel_code`.
"""

from __future__ import annotations

import re
from typing import Optional

# `answer` block — DR.Kernel's terminal marker. ` ```answer\n...``` `
_ANSWER_BLOCK_RE = re.compile(
    r"""
    (?P<block>
        ```answer[ \t]*(?:\r?\n)?
        (?P<code>.*?)
        (?:\r?\n)?```
    )
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

# Kernel-specific section markers. Same set as
# `recipe.drkernel.rewards.kernel_reward.extract_kernel_code`.
_KERNEL_MARKERS = [
    r"```triton[ \t]*\r?\n(.*?)```",
    r"#\s*Kernel\s+Implementation\s*\n(.*?)(?=\#\s*End\b|$)",
    r"```python\s*#\s*Kernel\s*\n(.*?)```",
    r"#\s*Your\s+implementation:\s*\n(.*?)(?=\#\s*End\b|$)",
    r"#\s*Generated\s+kernel:\s*\n(.*?)(?=\#\s*End\b|$)",
]
_KERNEL_PATTERNS = [re.compile(p, re.IGNORECASE | re.DOTALL) for p in _KERNEL_MARKERS]
_GENERIC_CODE_BLOCK_RE = re.compile(r"```(?:[\w+-]+)?\s*\n?(.*?)```", re.DOTALL)


def extract_answer_block(response: str) -> Optional[str]:
    """Return the contents of the model's ```answer``` block if present.

    DR.Kernel treats this as a terminal signal: any `answer` block ends
    the multi-turn loop regardless of `max_assistant_turns`.
    """
    if not response:
        return None
    m = _ANSWER_BLOCK_RE.search(response)
    return m.group("block") if m else None


def extract_kernel_code(response: str) -> Optional[str]:
    """Return the kernel implementation contained in the model's response.

    Tries the kernel-specific markers first; falls back to the *last*
    generic ```python``` (or unlabeled) fenced block. Returns None if
    nothing parseable was found — used by the agent loop to decide
    whether there's anything to evaluate against KernelGym this turn.
    """
    if not response:
        return None
    for pat in _KERNEL_PATTERNS:
        m = pat.search(response)
        if m:
            return m.group(1).strip()
    blocks = _GENERIC_CODE_BLOCK_RE.findall(response)
    if blocks:
        return blocks[-1].strip()
    return None
