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

"""
Backend for connecting to an existing vLLM serve API server.

This allows using a separate vLLM inference server (started via `vllm serve`)
as the teacher model backend, instead of embedding the vLLM engine in the worker.

Usage:
    # Start vLLM server separately:
    vllm serve Qwen/Qwen2.5-72B --tensor-parallel-size 8 --port 8000

    # Start teacher worker connecting to the server:
    python worker.py --backend vllm_serve --api-base http://localhost:8000 --n-logprobs 256
"""

from typing import List, Optional, Union
from urllib.parse import urljoin

import requests
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed


class VLLMAPIBackend:
    """
    Backend that connects to an existing vLLM serve API server.

    This backend uses the OpenAI-compatible API provided by vLLM serve
    to generate tokens and retrieve logprobs for knowledge distillation.

    Args:
        api_base: Base URL of the vLLM serve API (e.g., "http://localhost:8000")
        n_logprobs: Number of top-k logprobs to return (default: 256)
        max_batch_size: Maximum batch size per API call (default: None, auto-compute)
        timeout: Request timeout in seconds (default: 600)
        max_retries: Maximum number of retries on failure (default: 3)
    """

    def __init__(
      self,
      api_base: str,
      n_logprobs: int = 256,
      timeout: int = 1800,
      max_retries: int = 3,
      model: str = "default"
    ):
        self.api_base = api_base.rstrip("/")
        self.n_logprobs = n_logprobs
        self.timeout = timeout
        self.max_retries = max_retries
        self.model = model
        # Verify server is reachable
        self._health_check()

        print(f"Connected to vLLM serve at {api_base}, n_logprobs={n_logprobs}")

    def _health_check(self):
        """Check if the vLLM server is healthy."""
        url = urljoin(self.api_base, "/health")
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    return
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to connect to vLLM server at {self.api_base}: {e}")
                time.sleep(1)

        raise RuntimeError(f"vLLM server at {self.api_base} is not healthy")

    def _call_completions_api(
      self,
      prompt_token_ids: List[List[int]],
      temperature: float,
      max_tokens: Union[int, List[int]],
      only_response: bool,
    ) -> dict:
        """Call the vLLM completions API with logprobs."""
        url = urljoin(self.api_base, "/v1/completions")

        # Build the request payload
        # vLLM supports passing token IDs directly via the 'prompt' field as a list
        if len(prompt_token_ids) == 1:
            # Single prompt
            payload = {
                "model": self.model,  # vLLM serve uses the loaded model
                "prompt": prompt_token_ids[0],
                "temperature": temperature,
                "max_tokens": max_tokens if isinstance(max_tokens, int) else max_tokens[0],
                "logprobs": self.n_logprobs,
                "prompt_logprobs": None if only_response else self.n_logprobs,  # Get prompt logprobs if needed
                "echo": not only_response,
                "return_tokens_as_token_ids": True,
                "return_token_ids": True,
            }
        else:
            # Batch of prompts
            if isinstance(max_tokens, list):
                # vLLM doesn't support per-prompt max_tokens in batch mode
                # Use the maximum
                max_tokens = max(max_tokens)

            payload = {
                "model": self.model,
                "prompt": prompt_token_ids,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "logprobs": self.n_logprobs,
                "prompt_logprobs": None if only_response else self.n_logprobs,  # Get prompt logprobs if needed
                "echo": not only_response,
                "return_tokens_as_token_ids": True,
                "return_token_ids": True,
            }

        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, timeout=self.timeout)
                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = response.text
                    if attempt == self.max_retries - 1:
                        raise RuntimeError(f"vLLM API error: {response.status_code} - {error_msg}")
                    time.sleep(1)
            except requests.exceptions.RequestException as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"vLLM API request failed: {e}")
                time.sleep(1)

        raise RuntimeError("Failed to get response from vLLM API")

    def _process_logprobs(self, logprobs_data: dict, token_idx: int = -1):
        """
        Extract top-k logprobs from vLLM API response.

        Args:
            logprobs_data: The logprobs dict from API response
            token_idx: Index of the token to extract logprobs for (-1 for all tokens)

        Returns:
            Tuple of (topk_logprobs tensor, topk_indices tensor)
        """
        if logprobs_data is None or "top_logprobs" not in logprobs_data:
            return None, None

        top_logprobs = logprobs_data["top_logprobs"]

        # top_logprobs is a list of dicts: [{token_id: logprob, ...}, ...]
        all_logprobs = []
        all_indices = []

        for token_pos in top_logprobs[1:]:

            token_dict = token_pos  # {token_id_str: logprob, ...}
            # Convert to sorted list by logprob (descending)
            items = sorted(token_dict.items(), key=lambda x: x[1], reverse=True)

            # Take top-k
            topk_items = items[: self.n_logprobs]

            logprobs = []
            indices = []
            for token_id_str, logprob in topk_items:
                if token_id_str.startswith("token_id:"):
                    token_id_str = token_id_str.split(":")[1]
                indices.append(int(token_id_str))
                logprobs.append(logprob)

            # Pad if necessary
            while len(logprobs) < self.n_logprobs:
                logprobs.append(0.0)
                indices.append(0)

            all_logprobs.append(logprobs)
            all_indices.append(indices)

        return (
            torch.tensor(all_logprobs, dtype=torch.float32),
            torch.tensor(all_indices, dtype=torch.int32),
        )

    def get_topk_logprobs(
      self,
      prompt_token_ids: List[List[int]],
      temperature: float = 0.8,
      max_new_tokens: Union[int, List[int]] = 1,
      only_response: bool = False,
    ):
        """
        Get top-k logprobs for given prompts using vLLM serve API.

        When n_logprobs is large, storing logprobs for all positions across a large batch
        can cause OOM on the vLLM server side. This method splits the batch into smaller
        chunks and processes them sequentially, freeing memory between chunks.

        Memory estimation: batch_size * seq_len * n_logprobs * 4 bytes * 2 (logprobs + indices)
        Example: 32 * 4096 * 10000 * 8 ≈ 10.5 GB

        Args:
            prompt_token_ids: List of token ID lists
            temperature: Sampling temperature
            max_new_tokens: Maximum new tokens to generate (int or list of ints)
            only_response: If True, only return logprobs for response tokens

        Returns:
            responses: List of response token tensors
            teacher_topk_logprobs: List of logprob tensors
            teacher_topk_indices: List of index tensors
        """

        batch_size = len(prompt_token_ids)

        def process_single(idx):
            single_prompt = [prompt_token_ids[idx]]
            single_max_new_tokens = max_new_tokens[idx] if isinstance(max_new_tokens, list) else max_new_tokens
            # Call API
            api_response = self._call_completions_api(
                single_prompt, temperature, single_max_new_tokens, only_response
            )
            return idx, single_prompt, api_response

        results = [None] * batch_size

        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            futures = {executor.submit(process_single, i): i for i in range(batch_size)}
            for future in as_completed(futures):
                idx, single_prompt, api_response = future.result()
                results[idx] = (single_prompt, api_response)

        responses, teacher_topk_logprobs, teacher_topk_indices = [], [], []

        for single_prompt, api_response in results:

            # Process response
            choices = api_response.get("choices", [])
            choice = choices[0] if choices else {}

            response_token_ids = choice.get("token_ids")

            if response_token_ids:
                response_tensor = torch.tensor(response_token_ids, dtype=torch.int32)
            else:
                response_tensor = torch.tensor([], dtype=torch.int32)

            responses.append(response_tensor)

            # Extract logprobs
            logprobs_data = choice.get("logprobs")
            if logprobs_data and self.n_logprobs > 0:
                # Get response logprobs
                response_topk_logps, response_topk_indices = self._process_logprobs(logprobs_data)

                if response_topk_logps is not None:
                    if only_response:
                        teacher_topk_logprobs.append(response_topk_logps)
                        teacher_topk_indices.append(response_topk_indices)
                    else:
                        # Get prompt logprobs from separate field
                        prompt_logprobs_data = choice.get("prompt_logprobs")
                        if prompt_logprobs_data:
                            prompt_topk_logps, prompt_topk_indices = self._process_logprobs(prompt_logprobs_data)
                            if prompt_topk_logps is not None:
                                combined_logps = torch.vstack([prompt_topk_logps, response_topk_logps])
                                combined_indices = torch.vstack([prompt_topk_indices, response_topk_indices])
                                teacher_topk_logprobs.append(combined_logps)
                                teacher_topk_indices.append(combined_indices)
                            else:
                                # Fallback: only response logprobs available
                                teacher_topk_logprobs.append(response_topk_logps)
                                teacher_topk_indices.append(response_topk_indices)
                        else:
                            # No prompt logprobs available, use response only
                            teacher_topk_logprobs.append(response_topk_logps)
                            teacher_topk_indices.append(response_topk_indices)
                else:
                    teacher_topk_logprobs.append(None)
                    teacher_topk_indices.append(None)
            else:
                teacher_topk_logprobs.append(None)
                teacher_topk_indices.append(None)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test vLLM serve backend")
    parser.add_argument("--api-base", type=str, default="http://localhost:8000", help="vLLM serve API base URL")
    parser.add_argument("--model", type=str, default="default", help="vLLM serve model name")
    parser.add_argument("--n-logprobs", type=int, default=256, help="Number of logprobs")
    parser.add_argument("--batch-size", "-b", type=int, default=4, help="Test batch size")
    parser.add_argument("--seq-len", "-s", type=int, default=128, help="Test sequence length")
    args = parser.parse_args()

    import random

    # Generate random prompts
    prompt_token_ids = []
    for _ in range(args.batch_size):
        prompt_token_ids.append([random.randint(1, 10000) for _ in range(args.seq_len)])

    backend = VLLMAPIBackend(api_base=args.api_base, n_logprobs=args.n_logprobs,
                             model=args.model)

    print("Testing get_topk_logprobs...")
    import time

    start = time.time()
    responses, logps, indices = backend.get_topk_logprobs(
        prompt_token_ids, temperature=0.7, max_new_tokens=1, only_response=False
    )
    elapsed = time.time() - start

    print(f"Completed in {elapsed:.2f}s")
    print(f"Responses: {len(responses)}")
    if logps[0] is not None:
        print(f"responses shape: {responses[0].shape}")
        print(f"Logprobs:{logps}")
        print(f"Logprobs shape: {logps[0].shape}")
        print(f"indices shape: {indices[0].shape}")
