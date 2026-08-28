"""Tests for `open_r1_tpu.evaluation.runner`, against a stub OpenAI-compatible
HTTP server (stdlib `http.server`, no network, no Docker, no real vLLM) and a
fake Langfuse client. `iter_documents` and `resolve_task_configs` are always
monkeypatched or bypassed here -- these tests exercise the runner's own
concurrency, retry, error-budget, and failure-containment logic, none of
which needs LightEval or a dataset download.
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


# --- run_seed_task: concurrency, error budget, Langfuse containment ---------


class _StubConfig:
    def __init__(self, prompt_function, metrics=()):
        self.prompt_function = prompt_function
        self.metrics = list(metrics)
        self.hf_repo = "stub"
        self.hf_subset = "stub"
        self.hf_revision = None
        self.evaluation_splits = ["test"]
        self.hf_avail_splits = ["test"]


class FakeLangfuseClient:
    def __init__(self):
        self.observation_calls = []
        self.score_calls = []

    def create_trace_id(self, *, seed):
        return f"trace-{seed}"

    def start_observation(self, **kwargs):
        self.observation_calls.append(kwargs)

        class _Obs:
            def end(self):
                pass

        return _Obs()

    def create_score(self, **kwargs):
        self.score_calls.append(kwargs)

    def _create_trace_tags_via_ingestion(self, **kwargs):
        pass

    def flush(self):
        pass


class _AlwaysRaisingLangfuseClient:
    def create_trace_id(self, **kwargs):
        raise RuntimeError("langfuse is down")

    def start_observation(self, **kwargs):
        raise RuntimeError("langfuse is down")

    def create_score(self, **kwargs):
        raise RuntimeError("langfuse is down")

    def flush(self):
        raise RuntimeError("langfuse is down")


def _base_settings():
    return {
        "tier": "t",
        "served_model_name": "stub",
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 16,
        "system_prompt": None,
        "max_samples": None,
    }


def _fake_documents(count):
    return [(str(i), {"id": i}) for i in range(count)]


def _leaf_exceptions(exc):
    if isinstance(exc, ExceptionGroup):
        leaves = []
        for sub in exc.exceptions:
            leaves.extend(_leaf_exceptions(sub))
        return leaves
    return [exc]


def test_concurrency_never_exceeds_the_configured_width(
    stub_server, monkeypatch, tmp_path
):
    server = stub_server(
        lambda n, body: (200, _success_payload()), per_request_delay=0.05
    )
    monkeypatch.setattr(
        runner, "iter_documents", lambda config, *, max_samples: _fake_documents(20)
    )
    config = _StubConfig(prompt_function=lambda row, task_name: _StubDoc(row["id"]))

    asyncio.run(
        runner.run_seed_task(
            client=_client_for(server),
            settings=_base_settings(),
            task="stub_task|0",
            config=config,
            seed=0,
            semaphore=asyncio.Semaphore(3),
            error_budget=runner.ErrorBudget(fail_fast_after=100),
            langfuse=runner.LangfuseGuard(FakeLangfuseClient()),
            output_path=tmp_path / "out.jsonl",
        )
    )

    assert server.max_in_flight <= 3
    assert server.request_count == 20
    records = [
        json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()
    ]
    assert len(records) == 20
    assert all(record["status"] == "ok" for record in records)


def test_error_budget_stops_the_run_early(stub_server, monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "MAX_ATTEMPTS", 2)
    monkeypatch.setattr(runner, "BACKOFF_BASE_SECS", 0.001)
    monkeypatch.setattr(runner, "BACKOFF_MAX_SECS", 0.01)
    server = stub_server(lambda n, body: (500, {"error": {"message": "boom"}}))
    monkeypatch.setattr(
        runner, "iter_documents", lambda config, *, max_samples: _fake_documents(20)
    )
    config = _StubConfig(prompt_function=lambda row, task_name: _StubDoc(row["id"]))

    with pytest.raises(ExceptionGroup) as excinfo:
        asyncio.run(
            runner.run_seed_task(
                client=_client_for(server),
                settings=_base_settings(),
                task="stub_task|0",
                config=config,
                seed=0,
                semaphore=asyncio.Semaphore(5),
                error_budget=runner.ErrorBudget(fail_fast_after=3),
                langfuse=runner.LangfuseGuard(FakeLangfuseClient()),
                output_path=tmp_path / "out.jsonl",
            )
        )

    leaves = _leaf_exceptions(excinfo.value)
    assert any(isinstance(error, runner.ErrorBudgetExceeded) for error in leaves)
    # Stopped well short of every document retrying to exhaustion.
    assert server.request_count < 20 * runner.MAX_ATTEMPTS


def test_a_refused_request_stops_the_run_without_retrying(
    stub_server, monkeypatch, tmp_path
):
    server = stub_server(lambda n, body: (400, {"error": {"message": "bad params"}}))
    monkeypatch.setattr(
        runner, "iter_documents", lambda config, *, max_samples: _fake_documents(20)
    )
    config = _StubConfig(prompt_function=lambda row, task_name: _StubDoc(row["id"]))

    with pytest.raises(ExceptionGroup) as excinfo:
        asyncio.run(
            runner.run_seed_task(
                client=_client_for(server),
                settings=_base_settings(),
                task="stub_task|0",
                config=config,
                seed=0,
                semaphore=asyncio.Semaphore(5),
                error_budget=runner.ErrorBudget(fail_fast_after=100),
                langfuse=runner.LangfuseGuard(FakeLangfuseClient()),
                output_path=tmp_path / "out.jsonl",
            )
        )

    leaves = _leaf_exceptions(excinfo.value)
    assert any(isinstance(error, runner.GenerationRefused) for error in leaves)


def test_a_dead_langfuse_does_not_fail_the_run(stub_server, monkeypatch, tmp_path):
    server = stub_server(lambda n, body: (200, _success_payload()))
    monkeypatch.setattr(
        runner, "iter_documents", lambda config, *, max_samples: _fake_documents(5)
    )
    config = _StubConfig(prompt_function=lambda row, task_name: _StubDoc(row["id"]))
    langfuse = runner.LangfuseGuard(_AlwaysRaisingLangfuseClient())

    asyncio.run(
        runner.run_seed_task(
            client=_client_for(server),
            settings=_base_settings(),
            task="stub_task|0",
            config=config,
            seed=0,
            semaphore=asyncio.Semaphore(3),
            error_budget=runner.ErrorBudget(fail_fast_after=100),
            langfuse=langfuse,
            output_path=tmp_path / "out.jsonl",
        )
    )

    records = [
        json.loads(line) for line in (tmp_path / "out.jsonl").read_text().splitlines()
    ]
    assert len(records) == 5
    assert all(record["status"] == "ok" for record in records)
    assert all(record["trace_id"] is None for record in records)
    assert langfuse.failures >= 5


def test_rerunning_the_same_documents_reuses_the_same_trace_and_score_ids(
    stub_server, monkeypatch, tmp_path
):
    server = stub_server(lambda n, body: (200, _success_payload()))
    monkeypatch.setattr(
        runner, "iter_documents", lambda config, *, max_samples: _fake_documents(3)
    )
    config = _StubConfig(prompt_function=lambda row, task_name: _StubDoc(row["id"]))

    def run_once(output_path):
        client = FakeLangfuseClient()
        asyncio.run(
            runner.run_seed_task(
                client=_client_for(server),
                settings=_base_settings(),
                task="stub_task|0",
                config=config,
                seed=0,
                semaphore=asyncio.Semaphore(3),
                error_budget=runner.ErrorBudget(fail_fast_after=100),
                langfuse=runner.LangfuseGuard(client),
                output_path=output_path,
            )
        )
        return client

    first = run_once(tmp_path / "run1.jsonl")
    second = run_once(tmp_path / "run2.jsonl")

    first_trace_ids = {c["trace_context"]["trace_id"] for c in first.observation_calls}
    second_trace_ids = {
        c["trace_context"]["trace_id"] for c in second.observation_calls
    }
    assert first_trace_ids and first_trace_ids == second_trace_ids

    first_score_ids = {c["score_id"] for c in first.score_calls}
    second_score_ids = {c["score_id"] for c in second.score_calls}
    assert first_score_ids and first_score_ids == second_score_ids
