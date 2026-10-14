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

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from recipe.langgraph_agent.chat_model import ChatModel
from transformers import BatchEncoding


class _BatchEncodingTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        del args, kwargs
        return BatchEncoding({"input_ids": [101, 102]})


def _make_chat_model() -> ChatModel:
    return ChatModel.model_construct(
        model_name="dummy-model",
        client=None,
        tokenizer=_BatchEncodingTokenizer(),
        max_tokens=16,
    )


@pytest.mark.asyncio
async def test_preprocess_normalizes_batch_encoding_to_token_ids():
    model = _make_chat_model()

    _, prompt_ids, response_mask = await model._preprocess([HumanMessage(content="hello")])

    assert prompt_ids == [101, 102]
    assert response_mask == []


@pytest.mark.asyncio
async def test_preprocess_normalizes_batch_encoding_for_tool_response():
    model = _make_chat_model()
    messages = [
        HumanMessage(content="hello"),
        AIMessage(
            content="",
            response_metadata={"request_id": "request-1", "prompt_ids": [1, 2], "response_mask": [1]},
        ),
        ToolMessage(content="tool result", tool_call_id="tool-call-1"),
    ]

    request_id, prompt_ids, response_mask = await model._preprocess(messages, system_prompt=[101])

    assert request_id == "request-1"
    assert prompt_ids == [1, 2, 102]
    assert response_mask == [1, 0]


def test_bind_tools_normalizes_system_prompt_to_token_ids():
    model = _make_chat_model()

    bound = model.bind_tools([])

    assert bound.kwargs["system_prompt"] == [101, 102]
