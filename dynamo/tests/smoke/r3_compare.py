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
"""Compare R3 (routed experts) from Dynamo's /generate against stock SGLang.

The reference is a plain ``sglang.launch_server --enable-return-routed-experts``:
the same TokenizerManager.generate_request path verl's native SGLang server
calls. The candidate is the Dynamo frontend's native ``/generate`` in front of
the SGLang sidecar, launched the way the recipe launches it
(``launch_mode=sidecar``, ``generate_api=generate``). Both get identical greedy
requests, one at a time, so batching cannot change the numerics.

Checked per prompt:

* ``output_ids`` identical (first divergence reported);
* ``output_token_logprobs`` token ids identical, values within ``--logprob-atol``;
* ``routed_experts`` decoded exactly as verl does (base64 -> int32 ->
  ``(-1, num_hidden_layers, num_experts_per_tok)``), both covering
  ``prompt + output - 1`` rows, and equal on every row up to the first token
  divergence. Rows are compared as top-k *sets* as well as in order, so an
  engine that only reorders the top-k is reported separately from one that
  routes differently.

Exit status is 0 only if every prompt matches. Needs only ``requests`` and
``numpy``.

Example (one H100 node, two GPUs)::

    # reference: GPU 0
    CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server --model-path $MODEL \\
        --port 30000 --enable-return-routed-experts --attention-backend flashinfer \\
        --context-length 8192 &
    # candidate: the recipe's sidecar stack on GPU 1 (or any running recipe
    # rollout with rollout.enable_rollout_routing_replay=True), frontend on 8000
    python recipe/dynamo/tests/smoke/r3_compare.py --model $MODEL \\
        --reference http://127.0.0.1:30000 --candidate http://127.0.0.1:8000 \\
        --prompt-lens 128 1024 4096 7000 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import requests


def load_moe_shape(model_path: str) -> tuple[int, int, int]:
    with open(os.path.join(model_path, "config.json")) as f:
        cfg = json.load(f)
    text = cfg.get("text_config") or cfg
    layers, topk, vocab = text.get("num_hidden_layers"), text.get("num_experts_per_tok"), text.get("vocab_size")
    if not layers or not topk:
        sys.exit(f"{model_path}/config.json has no num_hidden_layers/num_experts_per_tok: not an MoE model")
    return int(layers), int(topk), int(vocab or cfg.get("vocab_size") or 32000)


def make_prompts(args, vocab: int) -> list[list[int]]:
    if args.prompt_ids_file:
        with open(args.prompt_ids_file) as f:
            return [json.loads(line) for line in f if line.strip()]
    rng = np.random.default_rng(args.seed)
    # Skip the low ids (special/control tokens in most vocabularies).
    return [rng.integers(1000, vocab - 1000, size=n).tolist() for n in args.prompt_lens]


def generate(base_url: str, input_ids: list[int], max_new_tokens: int, timeout: float) -> dict:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0.0, "ignore_eos": True},
        "return_logprob": True,
        "return_routed_experts": True,
        "stream": False,
    }
    resp = requests.post(f"{base_url.rstrip('/')}/generate", json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"{base_url}/generate -> {resp.status_code}: {resp.text[:1000]}")
    data = resp.json()
    return data[0] if isinstance(data, list) else data


def decode_routing(data: dict, layers: int, topk: int) -> np.ndarray:
    encoded = (data.get("meta_info") or {}).get("routed_experts")
    if not encoded:
        raise RuntimeError("response has no meta_info.routed_experts")
    flat = np.frombuffer(base64.b64decode(encoded.encode("utf-8")), dtype=np.int32)
    if flat.size % (layers * topk):
        raise RuntimeError(f"routed_experts size {flat.size} is not a multiple of {layers}*{topk}")
    return flat.reshape(-1, layers, topk)


@dataclass
class Result:
    prompt_len: int
    ok: bool = True
    notes: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.ok = False
        self.notes.append("FAIL " + msg)

    def info(self, msg: str) -> None:
        self.notes.append(msg)


def compare_one(ref: dict, cand: dict, prompt_len: int, layers: int, topk: int, atol: float) -> Result:
    res = Result(prompt_len)
    ref_ids, cand_ids = list(ref["output_ids"]), list(cand["output_ids"])
    common = min(len(ref_ids), len(cand_ids))
    diverge = next((i for i in range(common) if ref_ids[i] != cand_ids[i]), None)
    if diverge is None and len(ref_ids) == len(cand_ids):
        res.info(f"output_ids identical ({len(ref_ids)} tokens)")
        agree = len(ref_ids)
    else:
        agree = diverge if diverge is not None else common
        res.fail(f"output_ids diverge at {agree} (ref {len(ref_ids)} tokens, cand {len(cand_ids)})")

    ref_lp = (ref.get("meta_info") or {}).get("output_token_logprobs") or []
    cand_lp = (cand.get("meta_info") or {}).get("output_token_logprobs") or []
    if len(ref_lp) != len(ref_ids) or len(cand_lp) != len(cand_ids):
        res.fail(f"logprob length ref {len(ref_lp)}/{len(ref_ids)} cand {len(cand_lp)}/{len(cand_ids)}")
    elif agree:
        if any(cand_lp[i][1] != cand_ids[i] for i in range(len(cand_ids))):
            res.fail("candidate logprob token ids do not match its output_ids")
        diff = np.abs(np.array([r[0] for r in ref_lp[:agree]]) - np.array([c[0] for c in cand_lp[:agree]]))
        msg = f"logprob |diff| max {diff.max():.2e} mean {diff.mean():.2e} over {agree} tokens"
        if diff.max() > atol:
            res.fail(msg + f" (atol {atol})")
        else:
            res.info(msg)

    try:
        ref_r, cand_r = decode_routing(ref, layers, topk), decode_routing(cand, layers, topk)
    except RuntimeError as exc:
        res.fail(f"routed_experts: {exc}")
        return res
    for name, routing, ids in (("ref", ref_r, ref_ids), ("cand", cand_r, cand_ids)):
        expected = prompt_len + len(ids) - 1
        if routing.shape[0] != expected:
            res.fail(f"{name} routed_experts rows {routing.shape[0]} != prompt+output-1 = {expected}")
    # Rows [0, prompt_len + agree - 1) were computed from identical inputs on both sides.
    rows = min(prompt_len + agree - 1, ref_r.shape[0], cand_r.shape[0])
    if rows <= 0:
        return res
    a, b = ref_r[:rows], cand_r[:rows]
    exact = np.all(a == b, axis=-1)  # (rows, layers)
    as_set = np.all(np.sort(a, axis=-1) == np.sort(b, axis=-1), axis=-1)
    msg = f"routed_experts {rows} rows x {layers} layers: exact {exact.mean():.4%}, same top-k set {as_set.mean():.4%}"
    if not as_set.all():
        pos, layer = np.argwhere(~as_set)[0]
        res.fail(msg + f"; first set mismatch at position {pos} layer {layer}: {a[pos, layer]} vs {b[pos, layer]}")
    elif not exact.all():
        res.fail(msg + " (same experts, different order)")
    else:
        res.info(msg)
    return res


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="model directory (reads config.json for layers/top-k)")
    p.add_argument("--reference", required=True, help="stock SGLang server base URL")
    p.add_argument("--candidate", required=True, help="Dynamo frontend base URL (serves /generate)")
    p.add_argument("--prompt-lens", type=int, nargs="+", default=[128, 1024, 4096, 7000])
    p.add_argument("--prompt-ids-file", help="jsonl of token-id lists instead of random prompts")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logprob-atol", type=float, default=1e-2)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--dump", help="write both raw responses per prompt to this jsonl")
    args = p.parse_args()

    layers, topk, vocab = load_moe_shape(args.model)
    prompts = make_prompts(args, vocab)
    print(f"model: {layers} layers, top-{topk}; {len(prompts)} prompts; max_new_tokens={args.max_new_tokens}")
    dump = open(args.dump, "w") if args.dump else None
    all_ok = True
    for i, ids in enumerate(prompts):
        ref = generate(args.reference, ids, args.max_new_tokens, args.timeout)
        cand = generate(args.candidate, ids, args.max_new_tokens, args.timeout)
        if dump:
            dump.write(json.dumps({"prompt_ids": ids, "reference": ref, "candidate": cand}) + "\n")
        res = compare_one(ref, cand, len(ids), layers, topk, args.logprob_atol)
        all_ok &= res.ok
        print(f"[{i}] prompt_len={res.prompt_len} {'OK' if res.ok else 'MISMATCH'}")
        for note in res.notes:
            print("    " + note)
    if dump:
        dump.close()
    print("ALL MATCH" if all_ok else "MISMATCH FOUND")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
