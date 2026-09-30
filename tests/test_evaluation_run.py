"""Tests for `open_r1_tpu.evaluation.run`: the Langfuse path with the sync and
each experiment stubbed out, and the local path end to end against a stub
OpenAI-compatible server and stub task configs.
"""

from __future__ import annotations

import http.server
import json
import threading
from typing import cast

import pytest

# The evaluation client needs the `eval` extra; a training-only install skips.
pytest.importorskip("openai")

from open_r1_tpu.evaluation import run as eval_run
from open_r1_tpu.evaluation import summary as eval_summary
from open_r1_tpu.evaluation import traced


def _settings(**overrides):
    settings = {
        "served_model_name": "stub-model",
        "tier": "tier0",
        "tasks": ["stub_task|0"],
        "seeds": [0],
        "max_concurrency": 4,
        "fail_fast_after": 10,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_new_tokens": 16,
        "base_url": "http://127.0.0.1:0/v1",
        "output_dir": "unused",
    }
    settings.update(overrides)
    return settings


# --- the Langfuse path, with the sync and each experiment stubbed out --------


class _FakeAsyncClient:
    def __init__(self, close_error: Exception | None = None):
        self.close_error = close_error

    async def close(self):
        if self.close_error is not None:
            raise self.close_error


def _run_langfuse_path(monkeypatch, client, experiment_error=None, **overrides):
    """Run `run()` down the Langfuse path. Returns the `openai.AsyncOpenAI`
    kwargs, the sync and experiment calls in order, and `run()`'s result.
    """
    built = {}
    events = []

    def fake_sync(langfuse_client, settings):
        events.append("sync")
        return {task: f"{task}@fp" for task in settings["tasks"]}

    def fake_experiment(langfuse_client, *, task, seed, dataset_name, **kwargs):
        events.append((task, seed, dataset_name))
        if experiment_error is not None:
            raise experiment_error

    def fake_async_openai(**kwargs):
        built.update(kwargs)
        return client

    monkeypatch.setattr(traced, "sync_datasets", fake_sync)
    monkeypatch.setattr(traced, "run_experiment_for_task_seed", fake_experiment)
    monkeypatch.setattr(eval_run.openai, "AsyncOpenAI", fake_async_openai)
    output_dir = eval_run.run(
        _settings(**overrides), langfuse_client=object(), recipe_path="r.yaml"
    )
    return built, events, output_dir


def test_the_langfuse_path_syncs_then_runs_every_seed_and_task(tmp_path, monkeypatch):
    _, events, _ = _run_langfuse_path(
        monkeypatch,
        _FakeAsyncClient(),
        tasks=["gsm8k|0", "math_500|0"],
        seeds=[0, 1],
        output_dir=str(tmp_path),
    )

    assert events == [
        "sync",
        ("gsm8k|0", 0, "gsm8k|0@fp"),
        ("math_500|0", 0, "math_500|0@fp"),
        ("gsm8k|0", 1, "gsm8k|0@fp"),
        ("math_500|0", 1, "math_500|0@fp"),
    ]


def test_run_never_lets_a_connection_outlive_its_event_loop(tmp_path, monkeypatch):
    built, _, _ = _run_langfuse_path(
        monkeypatch, _FakeAsyncClient(), output_dir=str(tmp_path)
    )

    # Each (task, seed) runs on its own event loop, and a pooled keep-alive
    # connection cannot outlive the loop that opened it.
    assert built["default_headers"]["Connection"] == "close"


def test_run_reaches_its_summary_even_when_closing_the_client_fails(
    tmp_path, monkeypatch, caplog
):
    _, events, output_dir = _run_langfuse_path(
        monkeypatch,
        _FakeAsyncClient(close_error=RuntimeError("Event loop is closed")),
        tasks=["gsm8k|0", "math_500|0"],
        seeds=[0, 1],
        output_dir=str(tmp_path),
    )

    # Every record is written before the client is closed, so a teardown
    # failure costs a warning, never the summary.
    assert output_dir == tmp_path
    assert len(events) == 5
    assert "closing the HTTP client failed" in caplog.text


def test_run_teardown_failure_never_masks_the_real_one(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="the server refused this request"):
        _run_langfuse_path(
            monkeypatch,
            _FakeAsyncClient(close_error=RuntimeError("Event loop is closed")),
            experiment_error=RuntimeError("the server refused this request"),
            output_dir=str(tmp_path),
        )


# --- the local path: generate against a stub server, score, reduce --------


class _AnswerHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_POST(self):
        server = cast(_AnswerServer, self.server)
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        server.requests.append(body)
        question = body["messages"][-1]["content"]
        status = server.status
        payload = {
            "id": "cmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": body["model"],
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": server.answers.get(question, ""),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        }
        if status != 200:
            payload = {"error": {"message": "refused", "type": "BadRequestError"}}
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _AnswerServer(http.server.ThreadingHTTPServer):
    def __init__(self, answers, status=200):
        super().__init__(("127.0.0.1", 0), _AnswerHandler)
        self.answers = answers
        self.status = status
        self.requests = []

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"


@pytest.fixture
def answer_server():
    servers = []

    def factory(answers, status=200):
        server = _AnswerServer(answers, status)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server

    yield factory
    for server in servers:
        server.shutdown()
        server.server_close()


class _ExactMatch:
    """A stand-in LightEval metric: 1.0 when the stripped completion equals
    the gold. Records the thread it scored on, since LightEval's own maths
    metrics arm a `signal.alarm` timeout that only works on the main thread.
    """

    metric_name = "em"
    batched_compute = False

    def __init__(self):
        self.threads = set()

    def compute_sample(self, *, model_response, doc):
        self.threads.add(threading.current_thread())
        return {"em": model_response.text_post_processed[0] == doc.get_golds()[0]}

    def get_corpus_aggregations(self):
        return {"em": lambda values: sum(values) / len(values)}


class _LocalConfig:
    def __init__(self):
        self.metrics = [_ExactMatch()]

    @staticmethod
    def prompt_function(row, task_name):
        from lighteval.tasks.requests import Doc

        return Doc(
            query=f"What is {row['q']}?",
            choices=[row["a"]],
            gold_index=0,
            task_name=task_name,
        )


ROWS = [{"q": "2+2", "a": "4"}, {"q": "3+3", "a": "6"}, {"q": "4+4", "a": "8"}]


def _local_settings(server, tmp_path, **overrides):
    settings = _settings(
        base_url=server.base_url,
        output_dir=str(tmp_path),
        tasks=["arith|0"],
        seeds=[0, 1],
        max_samples=None,
        system_prompt="Answer briefly.",
        max_concurrency=2,
        fail_fast_after=2,
        reasoning_start="<think>",
        reasoning_end="</think>",
        answer_marker="\\boxed{",
    )
    settings.update(overrides)
    return settings


def _patch_local_task(monkeypatch, config):
    monkeypatch.setattr(
        eval_run, "resolve_task_configs", lambda tasks: dict.fromkeys(tasks, config)
    )
    monkeypatch.setattr(
        eval_run,
        "iter_documents",
        lambda config, *, max_samples: [
            (str(i), row) for i, row in enumerate(ROWS[:max_samples])
        ],
    )


def test_the_local_path_generates_scores_and_writes_what_summary_reduces(
    tmp_path, monkeypatch, answer_server
):
    pytest.importorskip("lighteval")
    server = answer_server(
        {"What is 2+2?": "4", "What is 3+3?": "6", "What is 4+4?": "9"}
    )
    config = _LocalConfig()
    _patch_local_task(monkeypatch, config)
    settings = _local_settings(server, tmp_path)

    output_dir = eval_run.run(settings)

    records = [
        json.loads(line)
        for line in (output_dir / "seed-1" / "arith-0.jsonl").read_text().splitlines()
    ]
    assert [(r["doc_id"], r["status"], r["scores"]) for r in records] == [
        ("0", "ok", {"em": 1.0}),
        ("1", "ok", {"em": 1.0}),
        ("2", "ok", {"em": 0.0}),
    ]
    assert records[0]["gold"] == "4"
    assert records[0]["query"] == "What is 2+2?"
    assert records[0]["completion_tokens"] == 2
    assert records[0]["trace_id"] is None
    # The recipe's system prompt leads every request.
    assert server.requests[0]["messages"][0] == {
        "role": "system",
        "content": "Answer briefly.",
    }
    assert len(server.requests) == 6  # 3 documents x 2 seeds
    assert config.metrics[0].threads == {threading.main_thread()}

    metrics, stats = eval_summary.reduce_seed(
        settings, 0, {"arith|0": config}, output_dir
    )
    assert metrics == {"arith|0": {"em": pytest.approx(2 / 3)}}
    assert stats["documents"] == 3


def test_the_local_path_honours_max_samples(tmp_path, monkeypatch, answer_server):
    pytest.importorskip("lighteval")
    server = answer_server({})
    _patch_local_task(monkeypatch, _LocalConfig())

    eval_run.run(_local_settings(server, tmp_path, seeds=[0], max_samples=2))

    lines = (tmp_path / "seed-0" / "arith-0.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_the_local_path_records_a_refused_request_as_failed(
    tmp_path, monkeypatch, answer_server
):
    pytest.importorskip("lighteval")
    server = answer_server({}, status=400)
    _patch_local_task(monkeypatch, _LocalConfig())

    eval_run.run(_local_settings(server, tmp_path, seeds=[0]))

    records = [
        json.loads(line)
        for line in (tmp_path / "seed-0" / "arith-0.jsonl").read_text().splitlines()
    ]
    assert [record["status"] for record in records] == ["failed"] * 3
    assert all("GenerationRefused" in record["error"] for record in records)


def test_the_local_path_rejects_a_multi_gold_document_before_generating(
    tmp_path, monkeypatch, answer_server
):
    pytest.importorskip("lighteval")
    from lighteval.tasks.requests import Doc

    server = answer_server({})
    config = _LocalConfig()
    config.prompt_function = lambda row, task_name: Doc(
        query=row["q"], choices=["a", "b"], gold_index=[0, 1], task_name=task_name
    )
    _patch_local_task(monkeypatch, config)

    with pytest.raises(ValueError, match="single gold"):
        eval_run.run(_local_settings(server, tmp_path, seeds=[0]))
    assert server.requests == []
