# Copyright 2025 Individual Contributor: furunding
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

import argparse
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple

import torch
from codetiming import Timer
from transformers import AutoConfig
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from vllm.v1.engine.logprobs import LogprobsProcessor


def _update_prompt_logprobs(
  self,
  prompt_logprobs_tensors,
) -> None:
    """Update with prompt logprobs from EngineCore.

    Args:
        prompt_logprobs_tensors: tuple containing the prompt logprobs
                                tensors.

    """

    # Prompt logprobs are enabled.
    assert self.num_prompt_logprobs is not None
    assert self.prompt_logprobs is not None

    self.prompt_logprobs.append(prompt_logprobs_tensors)


def _update_sample_logprobs(self, logprobs_lists) -> None:
    """Update with sample logprobs from EngineCore.

    Outer lists are only of len > 1 if EngineCore made
    >1 tokens in prior step (e.g. in spec decoding).

    Args:
        logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

    """

    assert self.num_logprobs is not None
    assert self.logprobs is not None
    assert self.cumulative_logprob is not None

    self.logprobs.append(logprobs_lists)


LogprobsProcessor._update_prompt_logprobs = _update_prompt_logprobs
LogprobsProcessor._update_sample_logprobs = _update_sample_logprobs


class LogprobsTensors(NamedTuple):
    # [num_reqs, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs]
    selected_token_ranks: torch.Tensor

    def tolists(self):
        return LogprobsTensors(
            logprob_token_ids=self.logprob_token_ids.cpu(),
            logprobs=self.logprobs.cpu(),
            selected_token_ranks=self.selected_token_ranks.cpu(),
        )

    @staticmethod
    def empty_cpu(num_positions: int, num_tokens_per_position: int) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""

        logprob_token_ids = torch.empty((num_positions, num_tokens_per_position), dtype=torch.int32, device="cpu")
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(num_positions, dtype=torch.int32, device="cpu")
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )

    def slice(self, start: int, end: int):
        return LogprobsTensors(
            self.logprob_token_ids[start:end],
            self.logprobs[start:end],
            self.selected_token_ranks[start:end],
        )


class VLLMEngine:
    def __init__(self, ckpt_path, n_logprobs=0, tp_size=1, max_model_len=8192, dtype="bfloat16"):
        self.n_logprobs = n_logprobs

        self.llm = LLM(
            ckpt_path,
            tensor_parallel_size=tp_size,
            trust_remote_code=True,
            enable_chunked_prefill=False,
            max_logprobs=n_logprobs,
            gpu_memory_utilization=0.6,
            max_model_len=max_model_len,
            dtype=dtype,
            async_scheduling=True,
            compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"},
        )

    def get_topk_logprobs(self, prompt_token_ids, temperature=0.8, max_new_tokens=1, only_response=False):
        def make_sampling_params(max_new_token):
            return SamplingParams(
                temperature=temperature,
                top_p=0.95,
                detokenize=False,
                logprobs=self.n_logprobs,
                prompt_logprobs=None if only_response else self.n_logprobs,
                max_tokens=max_new_token,
            )

        prompts = [TokensPrompt(prompt_token_ids=item_prompt_token_ids) for item_prompt_token_ids in prompt_token_ids]
        if isinstance(max_new_tokens, list):
            assert len(prompt_token_ids) == len(max_new_tokens)
        else:
            max_new_tokens = [max_new_tokens] * len(prompt_token_ids)

        sampling_params = [make_sampling_params(item_max_new_tokens) for item_max_new_tokens in max_new_tokens]

        results = self.llm.generate(prompts, sampling_params=sampling_params)

        responses, teacher_topk_logprobs, teacher_topk_indices = [], [], []

        for output in results:
            response_tensor = torch.tensor(output.outputs[0].token_ids, dtype=torch.int32).cpu()

            responses.append(response_tensor)

            if self.n_logprobs > 0:
                response_topk_logprobs = torch.tensor(
                    [x.logprobs[0] for x in output.outputs[0].logprobs],
                    dtype=torch.float32,
                )[:, 1:]
                response_topk_indices = torch.tensor(
                    [x.logprob_token_ids[0] for x in output.outputs[0].logprobs],
                    dtype=torch.int32,
                )[:, 1:]
                if only_response:
                    teacher_topk_logprobs.append(response_topk_logprobs.cpu())
                    teacher_topk_indices.append(response_topk_indices.cpu())
                else:
                    prompt_topk_logprobs = output.prompt_logprobs[1].logprobs[:, 1:].to(torch.float32)
                    prompt_topk_indices = output.prompt_logprobs[1].logprob_token_ids[:, 1:].to(torch.int32)
                    teacher_topk_logprobs.append(torch.vstack([prompt_topk_logprobs, response_topk_logprobs]).cpu())
                    teacher_topk_indices.append(torch.vstack([prompt_topk_indices, response_topk_indices]).cpu())

            torch.npu.empty_cache()

        return responses, teacher_topk_logprobs, teacher_topk_indices


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test vLLM logprob")
    parser.add_argument("model_dir", help="Model directory")
    parser.add_argument("--tp-size", type=int, default=1, help="TP size")
    parser.add_argument("--batch-size", "-b", type=int, default=64, help="Test batch size")
    parser.add_argument("--seq-len", "-s", type=int, default=3840, help="Test sequence length")
    args = parser.parse_args()

    config = AutoConfig.from_pretrained(args.model_dir)
    print(f"Reading configs from {args.model_dir}: {config.vocab_size=}")

    prompt_token_ids = []
    if args.token_file:
        # Init input with tokenid file
        from get_batch import get_batch

        prompt_token_ids = get_batch()
    else:
        # Init input randomly
        prompt_lens = args.batch_size * [args.seq_len]
        for pl in prompt_lens:
            prompt_token_ids.append([random.randint(1, config.vocab_size - 1000) for j in range(pl)])

    engine = VLLMEngine(ckpt_path=args.model_dir, n_logprobs=256, tp_size=args.tp_size)

    with Timer(name="get_topk_logprobs", initial_text=True):
        responses, teacher_topk_logprobs, teacher_topk_indices = engine.get_topk_logprobs(
            prompt_token_ids, temperature=0.7, max_new_tokens=1, only_response=True
        )
    # debug
    import ipdb

    ipdb.set_trace()
