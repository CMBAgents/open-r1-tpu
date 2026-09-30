"""Tests for `open_r1_tpu.evaluation.generate`, against a stub
OpenAI-compatible HTTP server: the retry policy, prompt rendering, and the
task function's circuit breaker.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from typing import cast

import pytest

# The evaluation client needs the `eval` extra; a training-only install skips.
pytest.importorskip("openai")

import openai

from open_r1_tpu.evaluation import generate
from open_r1_tpu.evaluation.generate import GenerationFailed, GenerationRefused

# --- a stub OpenAI-compatible server -----------------------------------------


def _success_payload(text="the answer", finish_reason="stop"):
    return {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "stub",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": finish_reason,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        server = cast(StubServer, self.server)
        server.request_count += 1
        length = int(self.headers.get("Content-Length", 0))
        server.bodies.append(json.loads(self.rfile.read(length)))
        status, payload = server.behavior(server.request_count)
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class StubServer(http.server.ThreadingHTTPServer):
    """Answers the nth request with `behavior(n)`'s `(status, payload)`."""

    def __init__(self, behavior):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.behavior = behavior
        self.request_count = 0
        self.bodies = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


@pytest.fixture
def stub_server():
    servers = []

    def factory(behavior):
        server = StubServer(behavior)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield factory
    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture
def fast_backoff(monkeypatch):
    monkeypatch.setattr(generate, "BACKOFF_BASE_SECS", 0.001)
    monkeypatch.setattr(generate, "BACKOFF_MAX_SECS", 0.01)


def _client_for(server) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key="local", base_url=server.base_url, max_retries=0)


async def _generate(client):
    return await generate.generate_one(
        client,
        served_model_name="stub",
        messages=[{"role": "user", "content": "hi"}],
        temperature=0.0,
        top_p=1.0,
        max_tokens=16,
    )


# --- generate_one: retry policy ---------------------------------------------


def test_generate_one_returns_usage_and_finish_reason(stub_server):
    server = stub_server(lambda n: (200, _success_payload()))
    outcome = asyncio.run(_generate(_client_for(server)))

    assert outcome.text == "the answer"
    assert outcome.finish_reason == "stop"
    assert outcome.prompt_tokens == 10
    assert outcome.completion_tokens == 5
    assert outcome.attempts == 1
    assert server.request_count == 1


def test_generate_one_never_retries_a_4xx(stub_server):
    server = stub_server(lambda n: (400, {"error": {"message": "bad sampling params"}}))
    with pytest.raises(GenerationRefused, match="bad sampling params"):
        asyncio.run(_generate(_client_for(server)))
    assert server.request_count == 1


def test_generate_one_retries_a_5xx_then_succeeds(stub_server, fast_backoff):
    def behavior(n):
        if n < 3:
            return 500, {"error": {"message": "boom"}}
        return 200, _success_payload()

    server = stub_server(behavior)
    outcome = asyncio.run(_generate(_client_for(server)))

    assert outcome.attempts == 3
    assert server.request_count == 3


def test_generate_one_exhausts_retries_and_raises(
    stub_server, fast_backoff, monkeypatch
):
    monkeypatch.setattr(generate, "MAX_ATTEMPTS", 3)
    server = stub_server(lambda n: (500, {"error": {"message": "boom"}}))
    with pytest.raises(GenerationFailed):
        asyncio.run(_generate(_client_for(server)))
    assert server.request_count == 3


# --- render_messages ---------------------------------------------------------


class _StubDoc:
    def __init__(self, fewshot_samples=None):
        self.query = "question"
        self.instruction = None
        self.fewshot_samples = fewshot_samples or []
        self.task_name = "stub_task"


def test_render_messages_prepends_a_system_prompt_when_set():
    messages = generate.render_messages(_StubDoc(), "be nice")
    assert messages == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "question"},
    ]


def test_render_messages_omits_the_system_turn_when_none():
    messages = generate.render_messages(_StubDoc(), None)
    assert messages == [{"role": "user", "content": "question"}]


def test_render_messages_rejects_fewshot_documents():
    with pytest.raises(NotImplementedError):
        generate.render_messages(_StubDoc(fewshot_samples=["something"]), None)


# --- make_task ---------------------------------------------------------------


class _StubItem:
    def __init__(self, messages):
        self.input = messages


def _task_for(server, fail_fast_after=3):
    settings = {
        "served_model_name": "stub",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 16,
        "fail_fast_after": fail_fast_after,
    }
    return generate.make_task(settings, client=_client_for(server))


def _item():
    return _StubItem([{"role": "user", "content": "hi"}])


def test_make_task_returns_a_dict_with_text_and_usage(stub_server):
    server = stub_server(lambda n: (200, _success_payload()))

    output = asyncio.run(_task_for(server)(item=_item()))

    assert output == {
        "text": "the answer",
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "latency_s": pytest.approx(output["latency_s"]),
        "attempts": 1,
    }


def test_make_task_never_sends_a_seed(stub_server):
    # The TPU backend rejects a per-request seed whenever temperature > 0.
    server = stub_server(lambda n: (200, _success_payload()))

    asyncio.run(_task_for(server)(item=_item()))

    assert server.bodies and "seed" not in server.bodies[0]


def test_a_refusal_trips_the_breaker_for_every_later_call(stub_server):
    server = stub_server(lambda n: (400, {"error": {"message": "bad sampling params"}}))
    task = _task_for(server)

    async def run():
        with pytest.raises(GenerationRefused):
            await task(item=_item())
        # A second item fails too, without a second request.
        with pytest.raises(GenerationRefused):
            await task(item=_item())

    asyncio.run(run())
    assert server.request_count == 1


def test_fail_fast_after_trips_after_n_consecutive_failures(stub_server, monkeypatch):
    monkeypatch.setattr(generate, "MAX_ATTEMPTS", 1)
    server = stub_server(lambda n: (500, {"error": {"message": "boom"}}))
    task = _task_for(server, fail_fast_after=2)

    async def run():
        for _ in range(3):
            with pytest.raises(GenerationFailed):
                await task(item=_item())

    asyncio.run(run())
    # The third call fails without reaching the server.
    assert server.request_count == 2


def test_a_success_resets_the_consecutive_failure_count(stub_server, monkeypatch):
    monkeypatch.setattr(generate, "MAX_ATTEMPTS", 1)

    def behavior(n):
        if n in (1, 3):
            return 500, {"error": {"message": "boom"}}
        return 200, _success_payload()

    server = stub_server(behavior)
    task = _task_for(server, fail_fast_after=2)

    async def run():
        with pytest.raises(GenerationFailed):
            await task(item=_item())
        await task(item=_item())
        with pytest.raises(GenerationFailed):
            await task(item=_item())
        # One consecutive failure since the reset, so this still reaches the
        # server.
        result = await task(item=_item())
        assert result["text"] == "the answer"

    asyncio.run(run())
    assert server.request_count == 4
