# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
OpenAI message format utilities.

Normalizes the OpenAI-style message dicts sent by SWE-Agent (which may
contain structured ``content`` blocks) into the flat ``{role, content}``
format expected by ``tokenizer.apply_chat_template``.
"""

from __future__ import annotations


def normalize_openai_messages(openai_messages: list[dict]) -> list[dict]:
    """Normalize OpenAI-format messages for ``tokenizer.apply_chat_template``.

    Handles:
    - ``content`` as a list of ``{"type": "text", "text": "..."}`` blocks →
      joined into a single string.
    - ``content`` as ``None`` → empty string.
    - ``content`` as non-string → stringified.

    Args:
        openai_messages: Raw messages from SWE-Agent / ModelProxy.

    Returns:
        List of ``{"role": str, "content": str}`` dicts.
    """
    messages: list[dict] = []
    for msg in openai_messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text" and "text" in part:
                        text_parts.append(part["text"])
                    else:
                        text_parts.append(str(part.get("text", part)))
                else:
                    text_parts.append(str(part))
            content = "\n".join(text_parts)
        elif content is None:
            content = ""
        else:
            content = str(content)

        messages.append({"role": role, "content": content})

    return messages
