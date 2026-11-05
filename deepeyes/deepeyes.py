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
"""DeepEyes dataset and reward for the multi-turn visual-tool GRPO recipe.

``CustomRLHFDatasetQwen35`` adapts the DeepEyes prompt to Qwen3.5, whose chat
template already injects the opening ``<think>`` marker.
"""

import logging
import os
import random
import re
import time
from typing import Any

import requests
import torch

from verl.utils.dataset.rl_dataset import RLHFDataset

logger = logging.getLogger(__name__)

TOOL_NAME = "image_zoom_in_tool"
AGENT_NAME = "tool_agent"

SYSTEM_PROMPT = (
    "You are a helpful assistant. You can call functions to assist with the user query. "
    "Important: You must call only one function at a time. After each function call, "
    "wait for the execution result before making the next function call if needed."
)

# ``<think>...</think>`` inside the DeepEyes user instruction.
_THINK_INSTRUCTION_RE = re.compile(r"\s*<think>\s*\.\.\.\s*</think>\s*")


class CustomRLHFDataset(RLHFDataset):
    """Build the DeepEyes prompt and per-sample ``image_zoom_in_tool`` kwargs."""

    def _deepeyes_messages(self, row_dict: dict[str, Any]) -> list[dict[str, Any]]:
        """Return the DeepEyes system and user messages."""
        prompt = row_dict[self.prompt_key]
        user_content = None
        for message in reversed(prompt):
            if message.get("role") == "user":
                user_content = message["content"]
                break
        if user_content is None:
            extra_info = row_dict.get("extra_info") or {}
            raise ValueError(f"No user message found in prompt of row {extra_info.get('index')}")

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def __getitem__(self, item: int) -> dict[str, Any]:
        row_dict: dict = self.dataframe[item]
        row_dict[self.prompt_key] = self._deepeyes_messages(row_dict)

        # ``_build_messages`` consumes the raw modality columns.
        messages = self._build_messages(row_dict, key=self.prompt_key)
        images = _collect_images(messages)
        _strip_raw_image_payloads(messages)

        row_dict.pop(self.image_key, None)
        row_dict.pop(self.video_key, None)
        row_dict.pop(self.audio_key, None)

        extra_info = row_dict.get("extra_info") or {}
        row_dict["extra_info"] = extra_info

        tools_kwargs = {}
        if images:
            # ``ToolAgentLoop`` passes this image to ``ImageZoomInTool.create``.
            tools_kwargs[TOOL_NAME] = {"create_kwargs": {"image": images[0]}}

        row_dict["raw_prompt"] = messages
        row_dict["index"] = extra_info.get("index", 0)
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = {}
        row_dict["agent_name"] = AGENT_NAME
        # TODO: a dummy tensor keeps DataProto non-empty until DataProto is deprecated.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        return row_dict


class CustomRLHFDatasetQwen35(CustomRLHFDataset):
    """DeepEyes prompt for Qwen3.5.

    The template injects the opening ``<think>``, so the literal
    ``<think>...</think>`` instruction is removed from the user prompt.
    """

    def _deepeyes_messages(self, row_dict: dict[str, Any]) -> list[dict[str, Any]]:
        messages = super()._deepeyes_messages(row_dict)
        content = messages[-1]["content"]
        if isinstance(content, str):
            messages[-1]["content"] = _strip_think_instruction(content)
        return messages


def _strip_think_instruction(text: str) -> str:
    stripped = _THINK_INSTRUCTION_RE.sub(" ", text)
    if stripped == text:
        return text
    return re.sub(r"[ \t]{2,}", " ", stripped)


def _strip_raw_image_payloads(messages: list[dict[str, Any]]) -> None:
    """Drop the encoded ``bytes``/``path`` next to the decoded image."""
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image":
                block.pop("bytes", None)
                block.pop("path", None)


def _collect_images(messages: list[dict[str, Any]]) -> list[Any]:
    images = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image" and block.get("image") is not None:
                images.append(block["image"])
    return images


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

_JUDGE_BASE_URL = ""
_JUDGE_MODEL = ""
_JUDGE_READY = False
_JUDGE_NEXT_PROBE = 0.0
_JUDGE_TEMPLATE_KWARGS = True
_JUDGE_PROBE_INTERVAL = 60.0


def _judge_headers() -> dict[str, str]:
    api_key = os.environ.get("LLM_AS_A_JUDGE_API_KEY", "").strip()
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def _get_judge() -> tuple[str, str]:
    """Resolve the judge endpoint and model name.

    The endpoint may run on another host. ``LLM_AS_A_JUDGE_MODEL`` skips the
    ``/models`` query when the server does not expose it.
    """
    global _JUDGE_BASE_URL, _JUDGE_MODEL, _JUDGE_READY, _JUDGE_NEXT_PROBE
    if _JUDGE_READY:
        return _JUDGE_BASE_URL, _JUDGE_MODEL
    if time.monotonic() < _JUDGE_NEXT_PROBE:
        return "", ""

    base_url = os.environ.get("LLM_AS_A_JUDGE_BASE", "").strip().rstrip("/")
    if not base_url:
        logger.warning("LLM_AS_A_JUDGE_BASE is not set; DeepEyes accuracy reward is disabled.")
        _JUDGE_READY = True
        return "", ""

    model_name = os.environ.get("LLM_AS_A_JUDGE_MODEL", "").strip()
    if not model_name:
        try:
            response = requests.get(f"{base_url}/models", headers=_judge_headers(), timeout=10.0)
            response.raise_for_status()
            models = response.json()
            model_name = models["data"][0]["id"] if models.get("data") else ""
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            _JUDGE_NEXT_PROBE = time.monotonic() + _JUDGE_PROBE_INTERVAL
            logger.warning(
                f"Failed to query models at {base_url}: {e}. Accuracy reward stays 0 and the query is "
                f"retried in {_JUDGE_PROBE_INTERVAL:.0f}s; set LLM_AS_A_JUDGE_MODEL to skip model discovery."
            )
            return "", ""

    if not model_name:
        _JUDGE_NEXT_PROBE = time.monotonic() + _JUDGE_PROBE_INTERVAL
        logger.warning(f"No models found at {base_url}; set LLM_AS_A_JUDGE_MODEL to skip model discovery.")
        return "", ""

    _JUDGE_BASE_URL, _JUDGE_MODEL = base_url, model_name
    _JUDGE_READY = True
    logger.info(f"DeepEyes judge: {model_name} @ {base_url}")
    return _JUDGE_BASE_URL, _JUDGE_MODEL


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs,
) -> dict[str, Any]:
    """Return ``0.8 * acc + 0.2 * format + 1.2 * tool`` for a DeepEyes rollout."""
    extra_info = extra_info or {}
    is_format_error = False

    # The Qwen3.5 template injects the opening ``<think>``; only extra opens are an error.
    if solution_str.count("<think>") > solution_str.count("</think>"):
        is_format_error = True

    predict_no_think = (
        solution_str.split("</think>")[-1].strip() if "</think>" in solution_str else solution_str.strip()
    )

    if predict_no_think.count("<answer>") != predict_no_think.count("</answer>"):
        is_format_error = True

    answer_match = re.search(r"<answer>(.*?)</answer>", predict_no_think, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1).strip()
    else:
        # Fall back to the text after the last tool response or thinking block.
        is_format_error = True
        tool_response_match = re.search(
            r"</tool_response>\s*assistant\s*\n(.*?)$", predict_no_think, re.DOTALL | re.MULTILINE
        )
        if tool_response_match:
            answer_text = tool_response_match.group(1).strip()
        elif "</think>" in solution_str:
            remaining_content = predict_no_think
            remaining_content = re.sub(r"<tool_call>.*?</tool_call>", "", remaining_content, flags=re.DOTALL)
            remaining_content = re.sub(r"<tool_response>.*?</tool_response>", "", remaining_content, flags=re.DOTALL)
            remaining_content = re.sub(r"\b(user|assistant)\b", "", remaining_content)
            answer_text = remaining_content.strip()
        else:
            answer_text = solution_str.strip()

    if not answer_text:
        is_format_error = True
        answer_text = solution_str.strip()

    # Accuracy: LLM judge, strict semantic equivalence with the ground truth.
    acc_reward = _judge_accuracy(extra_info.get("question", ""), ground_truth, answer_text)

    # Penalise excessively long answers (potential judge hacking).
    if len(answer_text) >= 1000:
        acc_reward = 0.0
        is_format_error = True

    # Tool usage: works for both the JSON (Qwen2.5-VL) and XML (Qwen3.5)
    # <tool_call> formats.
    has_tool_usage = bool(
        re.search(r"<tool_call>.*?</tool_call>", solution_str, re.DOTALL)
        or re.search(r"<tool_response>.*?</tool_response>", solution_str, re.DOTALL)
    )
    tool_reward = 1.0 if has_tool_usage and acc_reward > 0.5 else 0.0
    format_reward = -1.0 if is_format_error else 0.0

    final_score = 0.8 * acc_reward + 0.2 * format_reward + 1.2 * tool_reward
    return {
        "score": final_score,
        "acc": acc_reward,
        "format": format_reward,
        "tool_used": float(has_tool_usage),
        "format_error": float(is_format_error),
        "answer_length": len(answer_text),
    }


def _post_judge(base_url: str, payload: dict[str, Any]) -> requests.Response:
    """POST to the judge, retrying without ``chat_template_kwargs`` if rejected."""
    global _JUDGE_TEMPLATE_KWARGS
    request = dict(payload)
    if not _JUDGE_TEMPLATE_KWARGS:
        request.pop("chat_template_kwargs", None)

    response = requests.post(f"{base_url}/chat/completions", json=request, headers=_judge_headers(), timeout=60.0)
    if response.status_code in (400, 422) and "chat_template_kwargs" in request:
        logger.info(f"Judge rejected chat_template_kwargs (HTTP {response.status_code}); retrying without it.")
        _JUDGE_TEMPLATE_KWARGS = False
        request.pop("chat_template_kwargs", None)
        response = requests.post(f"{base_url}/chat/completions", json=request, headers=_judge_headers(), timeout=60.0)
    return response


def _judge_accuracy(question_text: str, ground_truth: str, answer_text: str) -> float:
    base_url, model_name = _get_judge()
    if not base_url or not model_name:
        return 0.0

    system_prompt = (
        "You are an expert evaluator. Your task is to determine if a model's answer is semantically equivalent to a "
        "provided standard answer, given a specific question.\n"
        "Your evaluation must be strict. The model's answer is only correct if it fully matches the meaning of the "
        "standard answer.\n"
        'You must provide your final judgement as a single word: either "CORRECT" or "INCORRECT". Do not provide '
        "any explanation or other text."
    )
    user_prompt = (
        "I will provide a question, a standard answer, and a model's answer. You must evaluate if the model's "
        "answer is correct.\n\n"
        "---\n"
        "**Example 1:**\n"
        "[Question]: Is the countertop tan or blue?\n"
        "[Standard Answer]: The countertop is tan.\n"
        "[Model's Answer]: tan\n"
        "[Your Judgement]: CORRECT\n"
        "---\n"
        "**Example 2:**\n"
        "[Question]: Is the man phone both blue and closed?\n"
        "[Standard Answer]: Yes, the man phone is both blue and closed.\n"
        "[Model's Answer]: No.\n"
        "[Your Judgement]: INCORRECT\n"
        "---\n"
        "**Task:**\n"
        f"[Question]: {question_text}\n"
        f"[Standard Answer]: {ground_truth}\n"
        f"[Model's Answer]: {answer_text}\n"
        "[Your Judgement]:"
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "seed": random.randint(0, 1000000),
        "temperature": 0.1,
        # vLLM / SGLang accept chat-template kwargs as a top-level field.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    response = None
    for attempt in range(3):
        try:
            http_response = _post_judge(base_url, payload)
            http_response.raise_for_status()
            response = (http_response.json()["choices"][0]["message"]["content"] or "").strip()
            break
        except Exception as e:
            if attempt == 2:
                logger.warning(f"Chat completion request failed: {e}")
                return 0.0
            time.sleep(1.0 + attempt)

    if response is None:
        return 0.0
    if re.search(r"\bCORRECT\b", response, re.IGNORECASE):
        return 1.0
    if re.search(r"\bINCORRECT\b", response, re.IGNORECASE):
        return 0.0

    logger.warning(
        "Judgement format error. Expected 'CORRECT' or 'INCORRECT'.\n"
        f"Response: '{response}'\nModel Answer: '{answer_text}'\nGround Truth: '{ground_truth}'"
    )
    return 0.0


if __name__ == "__main__":
    # Format checks; the judge stays disabled when LLM_AS_A_JUDGE_BASE is unset.
    cases = {
        "qwen35_implicit_think": (
            "I should zoom in.\n</think>\n"
            "<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n[226, 399, 265, 464]\n"
            "</parameter>\n</function>\n</tool_call>\n<answer>yes</answer>"
        ),
        "qwen25_explicit_think": (
            "<think>I need a closer look.</think>"
            '<tool_call>{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [226, 399, 265, 464]}}</tool_call>'
            "<answer>yes</answer>"
        ),
    }
    for name, text in cases.items():
        print(name, compute_score("vstar", text, "yes", {"question": "is it?"}))
