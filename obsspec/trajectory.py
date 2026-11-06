from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from verl.utils.tokenizer import normalize_token_ids


@dataclass(frozen=True)
class Snapshot:
    length: int


class Trajectory:
    def __init__(
        self,
        tokenizer: Any,
        raw_prompt: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]],
        enable_thinking: bool = False,
        spec: bool = False,
        apply_chat_template_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._spec = spec
        self._im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._turn_end = self._encode("<|im_end|>\n")
        self._newline_id = self._encode("\n")[0]
        self._double_newline_id = self._encode("\n\n")[0]
        self._tool_close_id = self._encode("</tool_response>")[0]
        self._observation_open = self._encode("<|im_start|>user\n<tool_response>\n")
        self._observation_close = self._encode("\n</tool_response><|im_end|>\n")
        generation_prompt = (
            "<|im_start|>assistant\n<think>\n" if enable_thinking else "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        self._generation_prompt = self._encode(generation_prompt)

        prompt_ids = tokenizer.apply_chat_template(
            raw_prompt,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            **(apply_chat_template_kwargs or {}),
        )
        self.prompt_ids = normalize_token_ids(prompt_ids) + self._generation_prompt
        self.ids: list[int] = []
        self.mask: list[int] = []
        self.world_mask: list[int] = []
        self.logprobs: list[float] = []

    @property
    def tokens(self) -> list[int]:
        return self.prompt_ids + self.ids

    def snapshot(self) -> Snapshot:
        return Snapshot(len(self.ids))

    def restore(self, snapshot: Snapshot) -> None:
        for sequence in (self.ids, self.mask, self.world_mask, self.logprobs):
            del sequence[snapshot.length :]

    def add_generation(self, ids: list[int], logprobs: list[float] | None) -> None:
        ids = list(ids)
        self._extend(ids, policy=1, logprobs=list(logprobs) if logprobs else [0.0] * len(ids))

    def add_observation(self, body: str, *, is_env: bool) -> None:
        self._close_turn()
        body_ids = self.observation_body_ids(body)
        chunk = self._observation_open + body_ids + self._observation_close + self._generation_prompt
        world_mask = [0] * len(chunk)
        if is_env:
            start = len(self._observation_open)
            end = start + len(body_ids)
            if self._spec:
                end += len(self._observation_close) - len(self._turn_end)
            world_mask[start:end] = [1] * (end - start)
        self._extend(chunk, policy=0, world_mask=world_mask)

    def get_spec_prompt(self) -> list[int]:
        self._close_turn()
        return self.tokens + self._observation_open

    def get_policy_prompt_after_observation(self, body: str) -> list[int]:
        self._close_turn()
        body_ids = self.observation_body_ids(body)
        return self.tokens + self._observation_open + body_ids + self._observation_close + self._generation_prompt

    def observation_body_ids(self, body: str) -> list[int]:
        return self._strip_terminator(self._encode(body.strip()))

    def _strip_terminator(self, ids: list[int]) -> list[int]:
        terminators = (self._newline_id, self._double_newline_id, self._tool_close_id, self._im_end_id)
        while ids and ids[-1] in terminators:
            ids.pop()
        return ids

    def _close_turn(self) -> None:
        if self.ids[-len(self._turn_end) :] == self._turn_end:
            return
        addition = self._turn_end[1:] if self.ids and self.ids[-1] == self._im_end_id else self._turn_end
        self._extend(addition, policy=0)

    def _extend(
        self,
        ids: list[int],
        *,
        policy: int,
        world_mask: list[int] | None = None,
        logprobs: list[float] | None = None,
    ) -> None:
        if world_mask is not None and len(world_mask) != len(ids):
            raise ValueError("world_mask length must match ids")
        if logprobs is not None and len(logprobs) != len(ids):
            raise ValueError("logprobs length must match ids")
        self.ids += ids
        self.mask += [policy] * len(ids)
        self.world_mask += world_mask if world_mask is not None else [0] * len(ids)
        self.logprobs += logprobs if logprobs is not None else [0.0] * len(ids)

    def _encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)
