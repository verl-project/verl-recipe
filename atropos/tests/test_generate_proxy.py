"""Tests for the generate proxy (recipe.atropos.generate_proxy).

Uses httpx.ASGITransport to drive the FastAPI app in-process and patches the
outgoing _client to mock backend vLLM responses.
"""

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest
import pytest_asyncio
import recipe.atropos.generate_proxy as proxy


@pytest.fixture(autouse=True)
def _reset_proxy_state():
    """Reset module-level state before each test."""
    proxy._backend_urls = ["http://test-backend:8000"]
    proxy._model = "test-model"
    proxy._paused = False
    proxy._in_flight = 0
    proxy._rr_counter = 0
    yield
    proxy._paused = False
    proxy._in_flight = 0
    proxy._rr_counter = 0


@pytest_asyncio.fixture
async def client():
    """httpx async client wired to the FastAPI app via ASGITransport."""
    transport = httpx.ASGITransport(app=proxy.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _mock_client():
    """Create a mock httpx.AsyncClient with async post/get methods."""
    mock = AsyncMock(spec=httpx.AsyncClient)
    return mock


# ---------------------------------------------------------------------------
# health
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_returns_200(client):
    mock = _mock_client()
    mock.get.return_value = httpx.Response(200)
    with patch.object(proxy, "_client", mock):
        resp = await client.get("/health")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_health_returns_503_when_paused(client):
    proxy._paused = True
    resp = await client.get("/health")
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_health_returns_503_on_backend_error(client):
    mock = _mock_client()
    mock.get.side_effect = httpx.ConnectError("connection refused")
    with patch.object(proxy, "_client", mock):
        resp = await client.get("/health")
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /generate — format translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_format_translation(client):
    """Verify atropos /generate request is translated to /v1/completions and back."""
    vllm_response = {
        "choices": [
            {
                "text": "hello world",
                "finish_reason": "stop",
                "logprobs": {
                    "tokens": ["token_id:42", "token_id:99"],
                    "token_logprobs": [-0.5, -1.2],
                },
            }
        ]
    }
    mock = _mock_client()
    mock.post.return_value = httpx.Response(200, json=vllm_response)

    with patch.object(proxy, "_client", mock):
        resp = await client.post(
            "/generate",
            json={
                "prompt": {"prompt_token_ids": [1, 2, 3]},
                "n": 1,
                "max_tokens": 128,
                "temperature": 0.7,
                "logprobs": 0,
                "stop": ["</answer>"],
            },
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == ["hello world"]
    assert data["finish_reasons"] == ["stop"]
    # logprobs: list of [{token_id: logprob}] — JSON keys are always strings
    assert data["logprobs"][0] == [[{"42": -0.5}], [{"99": -1.2}]]

    # verify the outgoing request to vLLM
    call_kwargs = mock.post.call_args
    sent_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert sent_json["model"] == "test-model"
    assert sent_json["prompt"] == [1, 2, 3]
    assert sent_json["return_tokens_as_token_ids"] is True
    assert sent_json["stop"] == ["</answer>"]
    # logprobs 0 from atropos → clamped to 1 for OpenAI API
    assert sent_json["logprobs"] == 1


@pytest.mark.asyncio
async def test_generate_string_prompt(client):
    """Verify string prompts are passed through directly."""
    vllm_response = {"choices": [{"text": "ok", "finish_reason": "stop", "logprobs": None}]}
    mock = _mock_client()
    mock.post.return_value = httpx.Response(200, json=vllm_response)

    with patch.object(proxy, "_client", mock):
        resp = await client.post(
            "/generate",
            json={
                "prompt": "hello",
                "max_tokens": 10,
            },
        )

    assert resp.status_code == 200
    call_kwargs = mock.post.call_args
    sent_json = call_kwargs.kwargs.get("json") or call_kwargs[1].get("json")
    assert sent_json["prompt"] == "hello"


@pytest.mark.asyncio
async def test_generate_returns_503_when_paused(client):
    proxy._paused = True
    resp = await client.post("/generate", json={"prompt": "hi", "max_tokens": 10})
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_generate_no_include_stop_str(client):
    """Verify include_stop_str_in_output is NOT injected."""
    vllm_response = {"choices": [{"text": "ok", "finish_reason": "stop", "logprobs": None}]}
    mock = _mock_client()
    mock.post.return_value = httpx.Response(200, json=vllm_response)

    with patch.object(proxy, "_client", mock):
        await client.post("/generate", json={"prompt": "hi", "max_tokens": 10})

    sent_json = mock.post.call_args.kwargs.get("json") or mock.post.call_args[1].get("json")
    assert "include_stop_str_in_output" not in sent_json


# ---------------------------------------------------------------------------
# /v1/completions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_completions_model_override(client):
    """Verify model name is overridden in forwarded request."""
    mock = _mock_client()
    mock.post.return_value = httpx.Response(200, json={"choices": []})

    with patch.object(proxy, "_client", mock):
        resp = await client.post(
            "/v1/completions",
            json={
                "model": "wrong-model",
                "prompt": "test",
                "max_tokens": 10,
            },
        )

    assert resp.status_code == 200
    sent_json = mock.post.call_args.kwargs.get("json") or mock.post.call_args[1].get("json")
    assert sent_json["model"] == "test-model"


@pytest.mark.asyncio
async def test_completions_no_include_stop_str(client):
    """Verify include_stop_str_in_output is NOT injected."""
    mock = _mock_client()
    mock.post.return_value = httpx.Response(200, json={"choices": []})

    with patch.object(proxy, "_client", mock):
        await client.post(
            "/v1/completions",
            json={
                "model": "x",
                "prompt": "test",
                "max_tokens": 10,
            },
        )

    sent_json = mock.post.call_args.kwargs.get("json") or mock.post.call_args[1].get("json")
    assert "include_stop_str_in_output" not in sent_json


@pytest.mark.asyncio
async def test_completions_returns_503_when_paused(client):
    proxy._paused = True
    resp = await client.post("/v1/completions", json={"model": "x", "prompt": "t", "max_tokens": 1})
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# /v1/chat/completions
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_completions_forwards(client):
    """Verify chat completions are forwarded with model override."""
    mock = _mock_client()
    mock.post.return_value = httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}]})

    with patch.object(proxy, "_client", mock):
        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "wrong",
                "messages": [{"role": "user", "content": "hello"}],
            },
        )

    assert resp.status_code == 200
    sent_json = mock.post.call_args.kwargs.get("json") or mock.post.call_args[1].get("json")
    assert sent_json["model"] == "test-model"
    assert "include_stop_str_in_output" not in sent_json


@pytest.mark.asyncio
async def test_chat_completions_returns_503_when_paused(client):
    proxy._paused = True
    resp = await client.post(
        "/v1/chat/completions",
        json={
            "model": "x",
            "messages": [{"role": "user", "content": "t"}],
        },
    )
    assert resp.status_code == 503


# ---------------------------------------------------------------------------
# pause / resume / drain
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pause_blocks_new_requests(client):
    """After POST /pause, new generation requests get 503."""
    resp = await client.post("/pause")
    assert resp.status_code == 200
    assert resp.json()["status"] == "paused"

    resp = await client.post("/generate", json={"prompt": "hi", "max_tokens": 10})
    assert resp.status_code == 503


@pytest.mark.asyncio
async def test_resume_allows_requests(client):
    """After POST /resume, requests work again."""
    proxy._paused = True
    resp = await client.post("/resume")
    assert resp.status_code == 200
    assert resp.json()["status"] == "resumed"
    assert proxy._paused is False


@pytest.mark.asyncio
async def test_drain_waits_for_in_flight(client):
    """POST /pause waits until active requests complete before returning."""
    # simulate an in-flight request that completes after 0.2s
    proxy._in_flight = 1

    async def _decrement():
        await asyncio.sleep(0.2)
        proxy._in_flight = 0

    task = asyncio.create_task(_decrement())
    resp = await client.post("/pause")
    assert resp.status_code == 200
    assert resp.json()["drained"] is True
    assert proxy._in_flight == 0
    await task


# ---------------------------------------------------------------------------
# round-robin
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_round_robin(client):
    """Requests cycle through multiple backend URLs."""
    proxy._backend_urls = ["http://backend-0:8000", "http://backend-1:8000", "http://backend-2:8000"]
    proxy._rr_counter = 0

    mock = _mock_client()
    mock.post.return_value = httpx.Response(200, json={"choices": []})

    urls_hit = []

    async def _capture_post(url, **kwargs):
        urls_hit.append(url)
        return httpx.Response(200, json={"choices": []})

    mock.post = AsyncMock(side_effect=_capture_post)

    with patch.object(proxy, "_client", mock):
        for _ in range(6):
            await client.post("/v1/completions", json={"model": "x", "prompt": "t", "max_tokens": 1})

    # should cycle: 0, 1, 2, 0, 1, 2
    expected_backends = [
        "http://backend-0:8000/v1/completions",
        "http://backend-1:8000/v1/completions",
        "http://backend-2:8000/v1/completions",
    ] * 2
    assert urls_hit == expected_backends
