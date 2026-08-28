"""Tests for `open_r1_tpu.evaluation.task_fn`, against a stub OpenAI-compatible
HTTP server (stdlib `http.server`, no network, no Docker, no real vLLM) --
the same style `test_runner.py` uses for `generate_one`, since `make_task` is
a thin wrapper over exactly that function.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from typing import cast

import openai
import pytest

from open_r1_tpu.evaluation import task_fn
from open_r1_tpu.evaluation.runner import GenerationFailed, GenerationRefused

# --- a stub OpenAI-compatible server, matching test_runner.py's -------------


def _success_payload(text="the answer", finish_reason="stop", completion_tokens=5):
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
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": completion_tokens,
            "total_tokens": 10 + completion_tokens,
        },
    }


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        server = cast(StubServer, self.server)
        server.request_count += 1
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        status, payload = server.behavior(server.request_count)
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class StubServer(http.server.ThreadingHTTPServer):
    def __init__(self, behavior):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.behavior = behavior
        self.request_count = 0

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


def _client_for(server) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key="local", base_url=server.base_url, max_retries=0)


class _StubItem:
    def __init__(self, messages):
        self.input = messages


def _settings(**overrides):
    settings = {
        "served_model_name": "stub",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 16,
        "fail_fast_after": 3,
    }
    settings.update(overrides)
    return settings


async def _call(task, n=1):
    outputs = []
    for _ in range(n):
        outputs.append(await task(item=_StubItem([{"role": "user", "content": "hi"}])))
    return outputs


# --- make_task: happy path ---------------------------------------------


def test_make_task_returns_a_dict_with_text_and_usage(stub_server):
    server = stub_server(lambda n: (200, _success_payload()))
    task = task_fn.make_task(_settings(), client=_client_for(server))

    (output,) = asyncio.run(_call(task))

    assert output == {
        "text": "the answer",
        "finish_reason": "stop",
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "latency_s": pytest.approx(output["latency_s"]),
        "attempts": 1,
    }


def test_make_task_never_sends_a_seed(stub_server):
    seen_bodies = []

    class _CapturingHandler(_Handler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            seen_bodies.append(json.loads(body))
            payload = _success_payload()
            data = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = StubServer(lambda n: (200, _success_payload()))
    server.RequestHandlerClass = _CapturingHandler
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        task = task_fn.make_task(_settings(), client=_client_for(server))
        asyncio.run(_call(task))
    finally:
        server.shutdown()
        server.server_close()

    assert seen_bodies and "seed" not in seen_bodies[0]


# --- circuit breaker: refusal is sticky ---------------------------------


def test_a_refusal_trips_the_breaker_for_every_later_call(stub_server):
    server = stub_server(lambda n: (400, {"error": {"message": "bad sampling params"}}))
    task = task_fn.make_task(_settings(), client=_client_for(server))

    async def run():
        with pytest.raises(GenerationRefused):
            await task(item=_StubItem([]))
        # A second, independent item must fail immediately too -- without
        # this task function making a second request.
        with pytest.raises(GenerationRefused):
            await task(item=_StubItem([]))

    asyncio.run(run())
    assert server.request_count == 1


def test_a_refusal_never_retries(stub_server):
    server = stub_server(lambda n: (400, {"error": {"message": "bad sampling params"}}))
    task = task_fn.make_task(_settings(), client=_client_for(server))

    with pytest.raises(GenerationRefused):
        asyncio.run(_call(task))
    assert server.request_count == 1


# --- circuit breaker: fail_fast_after consecutive GenerationFailed ------


def test_fail_fast_after_trips_after_n_consecutive_failures(stub_server, monkeypatch):
    monkeypatch.setattr("open_r1_tpu.evaluation.runner.MAX_ATTEMPTS", 1)
    server = stub_server(lambda n: (500, {"error": {"message": "boom"}}))
    task = task_fn.make_task(_settings(fail_fast_after=2), client=_client_for(server))

    async def run():
        with pytest.raises(GenerationFailed):
            await task(item=_StubItem([]))
        with pytest.raises(GenerationFailed):
            await task(item=_StubItem([]))
        # Budget (2) is now exhausted: a third call must fail without a
        # third request ever reaching the server.
        with pytest.raises(GenerationFailed):
            await task(item=_StubItem([]))

    asyncio.run(run())
    assert server.request_count == 2


def test_a_success_resets_the_consecutive_failure_count(stub_server, monkeypatch):
    monkeypatch.setattr("open_r1_tpu.evaluation.runner.MAX_ATTEMPTS", 1)
    behavior_calls = []

    def behavior(n):
        behavior_calls.append(n)
        if n in (1, 3):
            return 500, {"error": {"message": "boom"}}
        return 200, _success_payload()

    server = stub_server(behavior)
    task = task_fn.make_task(_settings(fail_fast_after=2), client=_client_for(server))

    async def run():
        with pytest.raises(GenerationFailed):
            await task(item=_StubItem([]))
        # A success in between must reset the counter.
        await task(item=_StubItem([]))
        with pytest.raises(GenerationFailed):
            await task(item=_StubItem([]))
        # Budget is not yet exhausted (only 1 consecutive failure since the
        # reset), so a fourth call still reaches the server.
        result = await task(item=_StubItem([]))
        assert result["text"] == "the answer"

    asyncio.run(run())
    assert server.request_count == 4
