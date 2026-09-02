"""Tests for `open_r1_tpu.evaluation.runner`'s generation primitives, against
a stub OpenAI-compatible HTTP server (stdlib `http.server`, no network, no
Docker, no real vLLM): `generate_one`'s retry policy and `render_messages`'s
prompt construction. `runner` no longer owns the generation loop itself
(`evaluation.task_fn`/`evaluation.experiment` do, driven by
`dataset.run_experiment()`, which owns concurrency) -- the circuit-breaker
and failure-containment logic that replaced the old error budget is covered
by `test_task_fn.py`, and `LangfuseGuard`'s
`create_dataset`/`create_dataset_item`/`flush` by `test_dataset_sync.py`.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import threading
from typing import cast

import openai
import pytest

from open_r1_tpu.evaluation import runner

# --- a stub OpenAI-compatible server -----------------------------------------


def _success_payload(
    text="the answer", finish_reason="stop", prompt_tokens=10, completion_tokens=5
):
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
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


class _Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # matches the base class's own signature
        pass  # silence stderr spam

    def do_POST(self):
        # `self.server` is typed as the stdlib's plain `BaseServer`; this
        # handler is only ever registered on a `StubServer` (see
        # `StubServer.__init__`), which owns `request_count` and friends.
        server = cast(StubServer, self.server)
        server.request_count += 1
        with server.concurrency_lock:
            server.in_flight += 1
            server.max_in_flight = max(server.max_in_flight, server.in_flight)
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            if server.per_request_delay:
                import time

                time.sleep(server.per_request_delay)
            status, payload = server.behavior(server.request_count, body)
            data = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            with server.concurrency_lock:
                server.in_flight -= 1


class StubServer(http.server.ThreadingHTTPServer):
    def __init__(self, behavior, per_request_delay=0.0):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.behavior = behavior
        self.per_request_delay = per_request_delay
        self.request_count = 0
        self.in_flight = 0
        self.max_in_flight = 0
        self.concurrency_lock = threading.Lock()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


@pytest.fixture
def stub_server():
    servers = []

    def factory(behavior, per_request_delay=0.0):
        server = StubServer(behavior, per_request_delay)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield factory
    for server in servers:
        server.shutdown()
        server.server_close()


def _client_for(server) -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key="local", base_url=server.base_url, max_retries=0)


async def _generate(client, **overrides):
    kwargs = {
        "served_model_name": "stub",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": 16,
    }
    kwargs.update(overrides)
    return await runner.generate_one(client, **kwargs)


# --- generate_one: retry policy ---------------------------------------------


def test_generate_one_returns_usage_and_finish_reason(stub_server):
    server = stub_server(lambda n, body: (200, _success_payload()))
    outcome = asyncio.run(_generate(_client_for(server)))

    assert outcome.text == "the answer"
    assert outcome.finish_reason == "stop"
    assert outcome.prompt_tokens == 10
    assert outcome.completion_tokens == 5
    assert outcome.attempts == 1
    assert server.request_count == 1


def test_generate_one_never_retries_a_4xx(stub_server):
    server = stub_server(
        lambda n, body: (400, {"error": {"message": "bad sampling params"}})
    )
    with pytest.raises(runner.GenerationRefused, match="bad sampling params"):
        asyncio.run(_generate(_client_for(server)))
    assert server.request_count == 1


def test_generate_one_retries_a_5xx_then_succeeds(stub_server, monkeypatch):
    monkeypatch.setattr(runner, "BACKOFF_BASE_SECS", 0.001)
    monkeypatch.setattr(runner, "BACKOFF_MAX_SECS", 0.01)

    def behavior(n, body):
        if n < 3:
            return 500, {"error": {"message": "boom"}}
        return 200, _success_payload()

    server = stub_server(behavior)
    outcome = asyncio.run(_generate(_client_for(server)))

    assert outcome.attempts == 3
    assert server.request_count == 3


def test_generate_one_exhausts_retries_and_raises(stub_server, monkeypatch):
    monkeypatch.setattr(runner, "MAX_ATTEMPTS", 3)
    monkeypatch.setattr(runner, "BACKOFF_BASE_SECS", 0.001)
    monkeypatch.setattr(runner, "BACKOFF_MAX_SECS", 0.01)

    server = stub_server(lambda n, body: (500, {"error": {"message": "boom"}}))
    with pytest.raises(runner.GenerationFailed):
        asyncio.run(_generate(_client_for(server)))
    assert server.request_count == 3


# --- render_messages ---------------------------------------------------------


class _StubDoc:
    def __init__(self, doc_id, instruction=None, fewshot_samples=None):
        self.query = f"question {doc_id}"
        self.choices = ["answer"]
        self.gold_index = 0
        self.instruction = instruction
        self.fewshot_samples = fewshot_samples or []
        self.specific = None
        self.task_name = "stub_task"


def test_render_messages_prepends_a_system_prompt_when_set():
    doc = _StubDoc("x")
    messages = runner.render_messages(doc, "be nice")
    assert messages[0] == {"role": "system", "content": "be nice"}
    assert messages[-1] == {"role": "user", "content": doc.query}


def test_render_messages_omits_the_system_turn_when_none():
    doc = _StubDoc("x")
    messages = runner.render_messages(doc, None)
    assert messages == [{"role": "user", "content": doc.query}]


def test_render_messages_rejects_fewshot_documents():
    doc = _StubDoc("x", fewshot_samples=["something"])
    with pytest.raises(NotImplementedError):
        runner.render_messages(doc, None)
