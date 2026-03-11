# Copyright 2026 HUMANLM team and/or its affiliates
# Copyright 2026 Bytedance Ltd. and/or its affiliates

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

import copy
import os
from typing import Any
from uuid import uuid4

import torch
from transformers import LogitsProcessor
from transformers.generation.logits_process import _calc_banned_ngram_tokens

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer


class NoRepeatNGramLogitsProcessor(LogitsProcessor):
    """
    Logits processor to prevent n-gram repetitions during generation.

    This processor identifies n-grams that have already appeared in the generated sequence
    and sets their logits to negative infinity, preventing the model from repeating them.

    Args:
        ngram_size (int): Size of n-grams to track. Must be a positive integer.
                         For example, ngram_size=3 prevents any 3-word sequence from repeating.

    Reference:
        - https://github.com/vllm-project/vllm/issues/757
        - https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py
    """

    def __init__(self, ngram_size: int):
        if not isinstance(ngram_size, int) or ngram_size <= 0:
            raise ValueError(f"`ngram_size` has to be a strictly positive integer, but is {ngram_size}")
        self.ngram_size = ngram_size

    def __call__(
        self, prompt_tokens_ids: tuple, past_tokens_ids: tuple, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Process logits to prevent n-gram repetitions.

        Args:
            prompt_tokens_ids (tuple): Original prompt token IDs
            past_tokens_ids (tuple): Previously generated token IDs
            scores (torch.FloatTensor): Current logits scores [batch_size, vocab_size]

        Returns:
            torch.FloatTensor: Modified scores with banned n-gram tokens set to -inf

        Reference:
            https://github.com/vllm-project/vllm/blob/911c8eb0000b1f9d1fef99ac9e209f83d801bd0a/vllm/model_executor/layers/logits_processor.py#L186
        """
        # Combine prompt and generated tokens
        input_ids = prompt_tokens_ids + past_tokens_ids

        if len(input_ids) < self.ngram_size:
            return scores

        # Ensure scores have batch dimension
        if len(scores.shape) == 1:
            scores = scores.reshape(1, -1)

        num_batch_hypotheses = scores.shape[0]
        input_ids = torch.LongTensor(input_ids).reshape(num_batch_hypotheses, -1)
        cur_len = input_ids.shape[-1]

        scores_processed = scores.clone()

        # Calculate which tokens would create banned n-grams
        banned_batch_tokens = _calc_banned_ngram_tokens(self.ngram_size, input_ids, num_batch_hypotheses, cur_len)

        # Set banned tokens to -inf
        for i, banned_tokens in enumerate(banned_batch_tokens):
            scores_processed[i, banned_tokens] = -float("inf")

        return scores_processed


@register("humanlm_agent")
class HumanLMAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prompt_length = self.config.actor_rollout_ref.rollout.prompt_length
        self.response_length = self.config.actor_rollout_ref.rollout.response_length

        custom_template = self.config.actor_rollout_ref.model.get("custom_chat_template", None)
        if custom_template and os.path.isfile(custom_template):
            with open(custom_template) as f:
                self.tokenizer.chat_template = f.read()
            print(f"[HumanLMAgentLoop] Loaded chat template from: {custom_template}", flush=True)
        elif custom_template:
            print(
                f"[HumanLMAgentLoop] WARNING: custom_chat_template is not a file path: {custom_template[:100]}",
                flush=True,
            )
        self.no_repeat_ngram_size = self.config.actor_rollout_ref.get("no_repeat_ngram_size", 0)

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        extra_info = kwargs.get("extra_info", {})
        speak_as = extra_info.get("name", None)
        state_name = extra_info.get("state_name", "response")

        global_steps = kwargs.get("meta_info", {}).get("global_steps", 0)

        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        engine_kwargs = self.config.actor_rollout_ref.rollout.get("engine_kwargs", {})
        stop = engine_kwargs.get("stop", None)
        is_val = extra_info.get("is_val", False)

        if stop:
            if isinstance(stop, str):
                stop = [stop]
            else:
                stop = list(stop)
            if is_val:
                stop = ["</response>"]
            sampling_params = {**sampling_params, "stop": stop}

        apply_kwargs = copy.deepcopy(dict(self.apply_chat_template_kwargs))

        # Handle hetero thinking- disable thinking for non-response states during training
        enable_hetero_think = self.config.data.get("enable_hetero_think", False)
        if enable_hetero_think and state_name != "response":
            apply_kwargs["enable_thinking"] = False

        if speak_as:
            apply_kwargs["speak_as"] = speak_as

        prompt_ids = await self.loop.run_in_executor(
            None,
            lambda: self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                **apply_kwargs,
            ),
        )

        no_repeat_ngram_size = self.config.actor_rollout_ref.rollout.get("no_repeat_ngram_size", 0)
        if no_repeat_ngram_size > 0:
            from recipe.humanlm.no_repeat_ngram import NoRepeatNGramLogitsProcessor

            sampling_params = {
                **sampling_params,
                "logits_processors": [NoRepeatNGramLogitsProcessor(no_repeat_ngram_size)],
            }

        metrics = {}
        with simple_timer("generate_sequences", metrics):
            output = await self.server_manager.generate(
                request_id=uuid4().hex,
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=images,
                video_data=videos,
            )

        if metrics.get("num_preempted") is None:
            metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1

        response_mask = [1] * len(output.token_ids)

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=output.token_ids[: self.response_length],
            response_mask=response_mask[: self.response_length],
            response_logprobs=output.log_probs[: self.response_length] if output.log_probs else None,
            multi_modal_data=multi_modal_data,
            num_turns=2,
            metrics=metrics,
        )

        output.extra_fields.update(
            {
                "turn_scores": [],
                "tool_rewards": [],
                "global_steps": global_steps,
            }
        )

        output.extra_fields.update({"turn_scores": [], "tool_rewards": []})
        return output
