"""CPU-only HTTP proxy between atropos environments and verl's internal vLLM.

Handles three request formats used across all atropos environments:
- /generate: legacy atropos format (token IDs + logprob dicts) — translates to /v1/completions
- /v1/completions: OpenAI completions — forwards with model name override
- /v1/chat/completions: OpenAI chat — forwards with model name override

Drain support: the trainer calls POST /pause before sleeping vLLM and POST /resume
after waking it. /pause stops accepting new requests and waits for in-flight ones
to complete, preventing CUDA errors from freed GPU memory during active generation.
Requests arriving while paused wait for the resume signal instead of failing
immediately, allowing eval and async generation to survive training pauses.

With multiple backend URLs (multi-GPU DP mode), requests are round-robined across
all vLLM servers so every GPU participates in generation.

Usage:
    python -m recipe.atropos.generate_proxy \
        --backend-url http://192.168.1.1:8000,http://192.168.1.1:8001 \
        --model Qwen/Qwen3-1.7B \
        --port 9004
"""

import argparse
import asyncio
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

_backend_urls: list[str] = []
_rr_counter: int = 0
_model: str = ""
_client: httpx.AsyncClient | None = None

# drain state — trainer calls /pause before sleep_replicas() and /resume after update_weights()
_paused: bool = False
_in_flight: int = 0
_DRAIN_TIMEOUT: float = 300.0
_GENERATION_TIMEOUT: float = 300.0
_resume_event: asyncio.Event | None = None


def _next_backend() -> str:
    """Round-robin across backend vLLM servers."""
    global _rr_counter
    url = _backend_urls[_rr_counter % len(_backend_urls)]
    _rr_counter += 1
    return url


@asynccontextmanager
async def lifespan(app):
    global _client, _resume_event
    _client = httpx.AsyncClient(timeout=httpx.Timeout(_GENERATION_TIMEOUT, connect=10))
    _resume_event = asyncio.Event()
    _resume_event.set()  # start unpaused
    yield
    await _client.aclose()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    """Passthrough to backend /health. Returns 503 when paused or vLLM is sleeping.

    All backends sleep/wake together (coordinated by checkpoint_manager),
    so checking any single backend reflects the state of all.
    """
    if _paused:
        return Response(status_code=503)
    try:
        resp = await _client.get(f"{_backend_urls[0]}/health", timeout=5)
        return Response(status_code=resp.status_code)
    except Exception:
        return Response(status_code=503)


@app.post("/generate")
async def generate(request: Request):
    """Translate atropos /generate -> /v1/completions, translate response back."""
    global _in_flight
    if _paused:
        await _resume_event.wait()
    _in_flight += 1
    try:
        data = await request.json()

        # extract prompt — pass token IDs directly to avoid decode/re-encode round-trip.
        # vLLM's /v1/completions accepts List[int] as token IDs natively.
        prompt = data.get("prompt")
        if isinstance(prompt, dict):
            prompt = prompt["prompt_token_ids"]

        # logprobs: atropos sends 0 ("return sampled token logprob"), OpenAI API needs >= 1
        logprobs_raw = data.get("logprobs")
        logprobs_val = max(1, int(logprobs_raw)) if logprobs_raw is not None else 1

        completions_req = {
            "model": _model,
            "prompt": prompt,
            "n": data.get("n", 1),
            "max_tokens": data.get("max_tokens", 16),
            "temperature": data.get("temperature", 1.0),
            "top_p": data.get("top_p", 1.0),
            "logprobs": logprobs_val,
            "return_tokens_as_token_ids": True,
        }
        if data.get("stop"):
            completions_req["stop"] = data["stop"]

        try:
            resp = await _client.post(f"{_next_backend()}/v1/completions", json=completions_req)
            if resp.status_code != 200:
                return Response(status_code=resp.status_code, content=resp.content)
            result = resp.json()
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)

        # translate response: choices[] -> text[], logprobs[], finish_reasons[]
        texts = []
        logprobs_out = []
        finish_reasons = []

        for choice in result.get("choices", []):
            texts.append(choice["text"])
            finish_reasons.append(choice.get("finish_reason") or "length")

            lp = choice.get("logprobs")
            if lp and lp.get("tokens"):
                seq = []
                for tok_str, tok_lp in zip(lp["tokens"], lp["token_logprobs"], strict=True):
                    # vLLM returns "token_id:123" format with return_tokens_as_token_ids=True
                    tid = int(tok_str.split(":")[1])
                    seq.append([{tid: tok_lp if tok_lp is not None else 0.0}])
                logprobs_out.append(seq)
            else:
                logprobs_out.append([])

        return JSONResponse({"text": texts, "logprobs": logprobs_out, "finish_reasons": finish_reasons})
    finally:
        _in_flight -= 1


@app.post("/v1/completions")
async def v1_completions(request: Request):
    """Forward OpenAI-format completions requests with model name override."""
    global _in_flight
    if _paused:
        await _resume_event.wait()
    _in_flight += 1
    try:
        data = await request.json()
        # override model name — env's ServerBaseline config uses a default
        # that doesn't match the actual served model
        data["model"] = _model
        try:
            resp = await _client.post(f"{_next_backend()}/v1/completions", json=data)
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)
    finally:
        _in_flight -= 1


@app.post("/v1/chat/completions")
async def v1_chat_completions(request: Request):
    """Forward OpenAI chat-format requests with model name override.

    Many atropos environments (text_reversal, gsm8k, mcqa, rlaif, tool_calling,
    etc.) use server.chat_completion() which hits this endpoint.
    """
    global _in_flight
    if _paused:
        await _resume_event.wait()
    _in_flight += 1
    try:
        data = await request.json()
        data["model"] = _model
        try:
            resp = await _client.post(f"{_next_backend()}/v1/chat/completions", json=data)
            return Response(content=resp.content, status_code=resp.status_code, media_type="application/json")
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=503)
    finally:
        _in_flight -= 1


@app.post("/pause")
async def pause():
    """Stop accepting new requests and wait for in-flight ones to complete.

    Called by the trainer before sleep_replicas() to ensure no active
    generation requests hit vLLM while GPU memory is being freed.
    """
    global _paused
    _paused = True
    _resume_event.clear()
    deadline = asyncio.get_running_loop().time() + _DRAIN_TIMEOUT
    while _in_flight > 0:
        if asyncio.get_running_loop().time() > deadline:
            # self-recover so the proxy doesn't stay permanently paused
            # if the trainer crashes after receiving the 504
            _paused = False
            _resume_event.set()
            return JSONResponse(
                {"status": "timeout", "in_flight": _in_flight},
                status_code=504,
            )
        await asyncio.sleep(0.1)
    return JSONResponse({"status": "paused", "drained": True})


@app.post("/resume")
async def resume():
    """Resume accepting requests. Called by the trainer after update_weights()."""
    global _paused
    _paused = False
    _resume_event.set()
    return JSONResponse({"status": "resumed"})


def main():
    parser = argparse.ArgumentParser(description="atropos /generate -> vLLM /v1/completions proxy")
    parser.add_argument(
        "--backend-url",
        required=True,
        help="comma-separated vLLM backend URLs (e.g. http://ip:8000,http://ip:8001)",
    )
    parser.add_argument("--model", required=True, help="Model name for /v1/completions model field")
    parser.add_argument("--port", type=int, default=9004)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--drain-timeout", type=float, default=300.0, help="seconds to wait for in-flight requests to drain on /pause"
    )
    parser.add_argument(
        "--generation-timeout",
        type=float,
        default=300.0,
        help="seconds to wait for a single generation request to vLLM",
    )
    args = parser.parse_args()

    global _backend_urls, _model, _DRAIN_TIMEOUT, _GENERATION_TIMEOUT
    _backend_urls = [url.rstrip("/") for url in args.backend_url.split(",")]
    _model = args.model
    _DRAIN_TIMEOUT = args.drain_timeout
    _GENERATION_TIMEOUT = args.generation_timeout

    print(f"generate proxy: {args.host}:{args.port} -> {_backend_urls} (model={_model})")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
