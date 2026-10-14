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
"""Binary LLM judge for open-ended browser tasks.

The `interactive_browser` environment verifies deterministic specs
(`final_url` / `url_contains` / `dom_contains` / `answer_equals`). WebVoyager
tasks are open-ended, so this recipe scores them with a judge until the
environment grows a judge-backed verifier of its own.

An unusable verdict — transport error, unparsable reply — is reported as
`usable=False` rather than as a zero, so the caller can mark the sample
environment-invalid instead of teaching the policy that it failed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

_PROMPT = """You are grading whether a web agent completed a task.

TASK: {question}

AGENT TRAJECTORY (tool calls and page observations):
{transcript}

AGENT'S FINAL ANSWER: {final_answer}
ANSWER STATUS: {answer_status}

Answer YES only if the trajectory shows the task was actually completed and the
final answer is consistent with what the pages showed. Answer NO otherwise.
Reply with YES or NO on the first line, then one sentence of justification."""

_TRANSCRIPT_CHARS = int(os.environ.get("NEMO_GYM_BROWSER_JUDGE_TRANSCRIPT_CHARS", "60000"))


@dataclass(frozen=True)
class JudgeResult:
    reward: float
    reason: str
    usable: bool


def render_transcript(events: list[dict[str, str]], budget: int = _TRANSCRIPT_CHARS) -> str:
    """Render tool calls and observations, truncating the middle of long results."""
    if not events:
        return "(no tool calls)"
    per_event = max(256, budget // max(1, len(events)))
    lines = []
    for event in events:
        observation = event.get("observation", "")
        if len(observation) > per_event:
            half = per_event // 2
            observation = f"{observation[:half]}...(truncated)...{observation[-half:]}"
        lines.append(f"CALL {event.get('action', '')}\nRESULT {observation}")
    rendered = "\n".join(lines)
    if len(rendered) > budget:
        half = budget // 2
        rendered = f"{rendered[:half]}...(truncated)...{rendered[-half:]}"
    return rendered


async def judge_rollout(
    question: str,
    events: list[dict[str, str]],
    final_answer: str,
    answer_status: str,
) -> JudgeResult:
    """Grade one rollout. Requires JUDGE_BASE_URL / JUDGE_API_KEY / JUDGE_MODEL."""
    base_url = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("JUDGE_API_KEY") or os.environ.get("OPENAI_API_KEY")
    model = os.environ.get("JUDGE_MODEL")
    if not (base_url and api_key and model):
        raise ValueError(
            "open-ended tasks need a judge: set JUDGE_BASE_URL, JUDGE_API_KEY and JUDGE_MODEL "
            "(see config.env.example), or give every task a deterministic `verifier_metadata`."
        )

    try:
        from openai import AsyncOpenAI
    except ImportError as exc:  # pragma: no cover - openai ships with verl
        raise ImportError("the judge needs the `openai` package: pip install openai") from exc

    client = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        timeout=float(os.environ.get("NEMO_GYM_BROWSER_JUDGE_TIMEOUT_S", "45")),
    )
    prompt = _PROMPT.format(
        question=question,
        transcript=render_transcript(events),
        final_answer=final_answer or "(none)",
        answer_status=answer_status,
    )
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as exc:
        return JudgeResult(reward=0.0, reason=f"judge_error:{type(exc).__name__}", usable=False)

    verdict = (response.choices[0].message.content or "").strip().lower()
    if verdict.startswith("yes"):
        return JudgeResult(reward=1.0, reason="judge_yes", usable=True)
    if verdict.startswith("no"):
        return JudgeResult(reward=0.0, reason="judge_no", usable=True)
    return JudgeResult(reward=0.0, reason=f"judge_unparsable:{verdict[:60]}", usable=False)
