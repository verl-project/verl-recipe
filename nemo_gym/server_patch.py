# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

import logging

logger = logging.getLogger(__name__)


def _replace_prefix_tokens(model_prefix, template_prefix, template_ids, tok):
    # matches NeMo-RL implementation
    if not model_prefix:
        return template_ids
    eos = tok.eos_token_id
    if eos is None:
        raise ValueError("tokenizer must have eos_token_id")
    cut_model = len(model_prefix)
    if model_prefix[-1] == eos:
        cut_model -= 1
    if len(template_ids) <= len(template_prefix):
        raise ValueError(
            f"non-monotonically increasing trajectory: "
            f"template_ids={len(template_ids)} template_prefix={len(template_prefix)}"
        )
    cut = -1
    for pos in reversed(range(len(template_prefix))):
        if template_ids[pos] == eos:
            cut = pos
            break
    if cut < 0:
        raise ValueError("no EOS token found in chat-templated messages")
    return model_prefix[:cut_model] + template_ids[cut:]


def _make_patched_preprocess_chat(original):
    async def _patched(
        self,
        request,
        messages,
        default_template,
        default_template_content_format,
        default_template_kwargs,
        tool_dicts=None,
        tool_parser=None,
    ):
        required_prefix = getattr(request, "required_prefix_token_ids", None)
        if required_prefix is None:
            for msg in reversed(messages):
                if isinstance(msg, dict) and "prompt_token_ids" in msg:
                    required_prefix = list(msg["prompt_token_ids"]) + list(msg["generation_token_ids"])
                    break
                elif not isinstance(msg, dict) and getattr(msg, "prompt_token_ids", None):
                    required_prefix = list(msg.prompt_token_ids) + list(msg.generation_token_ids)
                    break

        try:
            res = await original(
                self,
                request,
                messages,
                default_template,
                default_template_content_format,
                default_template_kwargs,
                tool_dicts=tool_dicts,
                tool_parser=tool_parser,
            )
        except ValueError as e:
            if "maximum context length" in str(e):
                logger.warning("Prompt exceeds max_model_len: %s", e)
            raise

        if required_prefix is None:
            return res

        last_asst = next(
            (
                i
                for i in reversed(range(len(messages)))
                if (messages[i].get("role") if isinstance(messages[i], dict) else getattr(messages[i], "role", None))
                == "assistant"
            ),
            None,
        )
        prefix_msgs = messages[: last_asst + 1] if last_asst is not None else messages
        prefix_res = await original(
            self,
            request,
            prefix_msgs,
            default_template,
            default_template_content_format,
            {**(default_template_kwargs or {}), "add_generation_prompt": False},
            tool_dicts=tool_dicts,
            tool_parser=tool_parser,
        )
        # tested on vLLM 0.17.0. other versions may error
        template_prefix_ids = prefix_res[1][0]["prompt_token_ids"]

        tok = self.renderer.get_tokenizer()
        engine_prompt = res[1][0]
        engine_prompt["prompt_token_ids"] = _replace_prefix_tokens(
            required_prefix,
            template_prefix_ids,
            engine_prompt["prompt_token_ids"],
            tok,
        )
        return res

    return _patched


def patch_serving_chat_for_nemo_gym() -> None:
    # vLLM 0.17.0 module paths
    from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
    from vllm.entrypoints.serve.tokenize.serving import OpenAIServingTokenization

    for cls in (OpenAIServingChat, OpenAIServingTokenization):
        cls._preprocess_chat = _make_patched_preprocess_chat(cls._preprocess_chat)
        logger.warning(f"[nemo-gym] applied retokenization patch to {cls.__name__}.")


def patch_hermes_tool_parser_thread_safety() -> None:
    """Cache tokenizer encode/decode results in Hermes2ProToolParser.__init__
    so concurrent instantiations don't race the Rust tokenizer and crash with
    ``RuntimeError: Already borrowed``. First call under lock runs the original
    __init__; subsequent calls for the same tokenizer reuse the cached token IDs.
    Based on prime-rl PR #1837.
    Related vLLM PRs: #37169, #37382, #40059, or #35034 and vLLM issue: #34932
    """
    import re
    import threading

    # vLLM 0.17.0 module paths
    from vllm.tool_parsers.abstract_tool_parser import ToolParser
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    _original_init = Hermes2ProToolParser.__init__
    _cache: dict[int, dict] = {}
    _lock = threading.Lock()

    def _patched_init(self, tokenizer):
        from vllm.tokenizers.mistral import MistralTokenizer

        actual_tokenizer = tokenizer.tokenizer if isinstance(tokenizer, MistralTokenizer) else tokenizer
        key = id(actual_tokenizer)
        if key in _cache:
            ToolParser.__init__(self, tokenizer)
            if isinstance(tokenizer, MistralTokenizer):
                self.model_tokenizer = tokenizer.tokenizer
            self.current_tool_name_sent = False
            self.prev_tool_call_arr = []
            self.current_tool_id = -1
            self.streamed_args_for_tool = []
            self.tool_call_start_token = "<tool_call>"
            self.tool_call_end_token = "</tool_call>"
            self.tool_call_regex = re.compile(r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL)
            self.scratch_pad_regex = re.compile(r"<scratch_pad>(.*?)</scratch_pad>", re.DOTALL)
            cached = _cache[key]
            self.tool_call_start_token_ids = cached["start_ids"]
            self.tool_call_end_token_ids = cached["end_ids"]
            self.tool_call_start_token_array = cached["start_array"]
            self.tool_call_end_token_array = cached["end_array"]
            self.buffered_delta_text = ""
            return

        with _lock:
            if key in _cache:
                _patched_init(self, tokenizer)
                return
            _original_init(self, tokenizer)
            _cache[key] = {
                "start_ids": self.tool_call_start_token_ids,
                "end_ids": self.tool_call_end_token_ids,
                "start_array": self.tool_call_start_token_array,
                "end_array": self.tool_call_end_token_array,
            }

    Hermes2ProToolParser.__init__ = _patched_init
    logger.warning("[nemo-gym] applied hermes tool parser thread-safety patch.")
