"""Tests for `open_r1_tpu.evaluation.experiment`.

The Langfuse path runs against a faked Langfuse client/dataset -- no real
`run_experiment`, no network, no lighteval registry: it only ever calls
`dataset.run_experiment()` and reads back its return value, so faking that
one call exercises the run name, the evaluator/task wiring, and the JSONL
record shape `evaluation.reduce` reads. The local path runs end to end
against a stub OpenAI-compatible server and stub task configs.
"""

from __future__ import annotations

import http.server
import json
import threading
from typing import cast

import pytest

# The evaluation client needs the `eval` extra; a training-only install skips.
pytest.importorskip("openai")

from open_r1_tpu.evaluation import experiment, reduce


class _FakeEvaluation:
    def __init__(self, name, value, metadata=None):
        self.name = name
        self.value = value
        self.metadata = metadata


class _FakeDatasetItem:
    def __init__(self, item_id, metadata, expected_output="gold"):
        self.id = item_id
        self.metadata = metadata
        # The gold `evaluation.dataset_sync` stored. Carried into every JSONL
        # record so `evaluation.consensus` can judge a cons@n winner without
        # reading anything back out of Langfuse.
        self.expected_output = expected_output


class _FakeItemResult:
    def __init__(self, item, output, evaluations, trace_id="trace-1"):
        self.item = item
        self.output = output
        self.evaluations = evaluations
        self.trace_id = trace_id


class _FakeExperimentResult:
    def __init__(self, item_results):
        self.item_results = item_results


class _FakeDataset:
    def __init__(self, items, result):
        self.items = items
        self._result = result
        self.run_experiment_calls = []

    def run_experiment(self, **kwargs):
        self.run_experiment_calls.append(kwargs)
        return self._result


class _FakeLangfuseClient:
    def __init__(self, dataset):
        self._dataset = dataset
        self.requested_names = []

    def get_dataset(self, name):
        self.requested_names.append(name)
        return self._dataset


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


# --- write_experiment_jsonl --------------------------------------------


def test_record_from_item_result_matches_reduce_expected_shape(tmp_path):
    item = _FakeDatasetItem(
        "item-1",
        {"doc_id": "3", "task": "stub_task|0", "query": "Question: ...\nAnswer:"},
        expected_output="18",
    )
    result = _FakeExperimentResult(
        [
            _FakeItemResult(
                item,
                output={
                    "text": "18",
                    "finish_reason": "stop",
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "latency_s": 0.5,
                    "attempts": 1,
                },
                evaluations=[
                    _FakeEvaluation("extractive_match", 1.0),
                    _FakeEvaluation("completion_tokens", 2.0),
                    _FakeEvaluation("truncated", 0.0),
                ],
            )
        ]
    )
    dataset = _FakeDataset(items=[item], result=result)

    output_path = tmp_path / "seed-0" / "stub_task-0.jsonl"
    experiment.write_experiment_jsonl(
        result, dataset, task="stub_task|0", seed=0, output_path=output_path
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert records == [
        {
            "status": "ok",
            "doc_id": "3",
            "task": "stub_task|0",
            "seed": 0,
            # The gold and the prompt travel with the record so a reduction
            # is self-contained -- `evaluation.consensus` rebuilds a scoring
            # Doc from exactly these two, never from Langfuse.
            "gold": "18",
            "query": "Question: ...\nAnswer:",
            "completion": "18",
            "finish_reason": "stop",
            "prompt_tokens": 7,
            "completion_tokens": 2,
            "latency_s": 0.5,
            "attempts": 1,
            "scores": {"extractive_match": 1.0},
            "failed_metrics": [],
            "scoring_errors": {},
            "trace_id": "trace-1",
        }
    ]


def test_write_experiment_jsonl_writes_a_dropped_record_for_a_missing_item(tmp_path):
    present = _FakeDatasetItem("item-1", {"doc_id": "0"})
    missing = _FakeDatasetItem("item-2", {"doc_id": "1"})
    result = _FakeExperimentResult(
        [
            _FakeItemResult(
                present, output={"text": "x", "finish_reason": "stop"}, evaluations=[]
            )
        ]
    )
    dataset = _FakeDataset(items=[present, missing], result=result)

    output_path = tmp_path / "out.jsonl"
    experiment.write_experiment_jsonl(
        result, dataset, task="stub_task|0", seed=1, output_path=output_path
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 2
    by_doc_id = {r["doc_id"]: r for r in records}
    assert by_doc_id["0"]["status"] == "ok"
    assert by_doc_id["1"]["status"] == "dropped"
    assert by_doc_id["1"]["seed"] == 1
    assert by_doc_id["1"]["task"] == "stub_task|0"


def test_write_experiment_jsonl_reconstructs_scoring_failed_metadata(tmp_path):
    item = _FakeDatasetItem("item-1", {"doc_id": "0"})
    result = _FakeExperimentResult(
        [
            _FakeItemResult(
                item,
                output={"text": "x", "finish_reason": "stop", "completion_tokens": 1},
                evaluations=[
                    _FakeEvaluation("completion_tokens", 1.0),
                    _FakeEvaluation(
                        "scoring_failed",
                        1.0,
                        metadata={
                            "failed_metrics": ["boom"],
                            "errors": {"boom": "kaboom"},
                        },
                    ),
                ],
            )
        ]
    )
    dataset = _FakeDataset(items=[item], result=result)

    output_path = tmp_path / "out.jsonl"
    experiment.write_experiment_jsonl(
        result, dataset, task="stub_task|0", seed=0, output_path=output_path
    )

    (record,) = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert record["failed_metrics"] == ["boom"]
    assert record["scoring_errors"] == {"boom": "kaboom"}
    assert record["scores"] == {}


# --- run_experiment_for_task_seed: one call per (task, seed) -----------


def test_run_experiment_for_task_seed_calls_run_experiment_once_with_expected_args(
    tmp_path, monkeypatch
):
    item = _FakeDatasetItem("item-1", {"doc_id": "0"})
    result = _FakeExperimentResult(
        [
            _FakeItemResult(
                item, output={"text": "x", "finish_reason": "stop"}, evaluations=[]
            )
        ]
    )
    dataset = _FakeDataset(items=[item], result=result)
    client = _FakeLangfuseClient(dataset)

    captured_evaluator_task_name = {}

    def fake_lighteval_evaluator(task_name):
        captured_evaluator_task_name["task"] = task_name
        return lambda **kwargs: []

    def fake_make_task(settings, *, client):
        return lambda **kwargs: None

    monkeypatch.setattr(
        experiment.scoring, "lighteval_evaluator", fake_lighteval_evaluator
    )
    monkeypatch.setattr(experiment, "make_task", fake_make_task)

    settings = _settings()
    experiment.run_experiment_for_task_seed(
        client,
        task="stub_task|0",
        seed=2,
        dataset_name="stub_task|0@fixedprint",
        settings=settings,
        client=object(),
        output_dir=tmp_path,
        run_metadata={"recipe_path": "r.yaml", "git_commit": "abc"},
    )

    assert captured_evaluator_task_name["task"] == "stub_task|0"
    assert len(dataset.run_experiment_calls) == 1
    call = dataset.run_experiment_calls[0]
    assert call["name"] == "stub-model-tier0-seed2"
    assert call["max_concurrency"] == 4
    assert call["metadata"]["recipe_path"] == "r.yaml"
    assert call["metadata"]["git_commit"] == "abc"
    assert call["metadata"]["dataset_fingerprint"] == "fixedprint"
    assert client.requested_names == ["stub_task|0@fixedprint"]

    output_path = tmp_path / "seed-2" / "stub_task-0.jsonl"
    assert output_path.is_file()
    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["seed"] == 2


def test_run_calls_run_experiment_for_every_task_seed_pair(tmp_path, monkeypatch):
    calls = []

    def fake_run_experiment_for_task_seed(langfuse_client, *, task, seed, **kwargs):
        calls.append((task, seed))

    class _FakeAsyncClient:
        async def close(self):
            pass

    monkeypatch.setattr(
        experiment, "run_experiment_for_task_seed", fake_run_experiment_for_task_seed
    )
    monkeypatch.setattr(
        experiment.openai, "AsyncOpenAI", lambda **kwargs: _FakeAsyncClient()
    )
    monkeypatch.setattr(
        experiment,
        "sync_datasets",
        lambda langfuse_client, settings: {t: f"{t}@fp" for t in settings["tasks"]},
    )

    settings = _settings(
        tasks=["gsm8k|0", "math_500|0"], seeds=[0, 1], output_dir=str(tmp_path)
    )
    experiment.run(settings, langfuse_client=object(), recipe_path="r.yaml")

    assert set(calls) == {
        ("gsm8k|0", 0),
        ("math_500|0", 0),
        ("gsm8k|0", 1),
        ("math_500|0", 1),
    }


# --- run: nothing asyncio-bound outlives one run_experiment call -------


def _run_with_client(monkeypatch, client, **settings_overrides):
    """Drive `run()` with `run_experiment_for_task_seed` stubbed out, so the
    only thing under test is what `run()` does with the HTTP client itself.
    Returns the `openai.AsyncOpenAI` kwargs and the `(task, seed)` pairs run.
    """
    built = {}
    calls = []

    def fake_run_experiment_for_task_seed(langfuse_client, *, task, seed, **kwargs):
        calls.append((task, seed))

    def fake_async_openai(**kwargs):
        built.update(kwargs)
        return client

    monkeypatch.setattr(
        experiment, "run_experiment_for_task_seed", fake_run_experiment_for_task_seed
    )
    monkeypatch.setattr(experiment.openai, "AsyncOpenAI", fake_async_openai)
    monkeypatch.setattr(
        experiment,
        "sync_datasets",
        lambda langfuse_client, settings: {t: f"{t}@fp" for t in settings["tasks"]},
    )
    settings = _settings(**settings_overrides)
    output_dir = experiment.run(
        settings, langfuse_client=object(), recipe_path="r.yaml"
    )
    return built, calls, output_dir


def test_run_never_lets_a_connection_outlive_its_event_loop(tmp_path, monkeypatch):
    class _FakeAsyncClient:
        async def close(self):
            pass

    built, _, _ = _run_with_client(
        monkeypatch, _FakeAsyncClient(), output_dir=str(tmp_path)
    )

    # run_experiment runs each (task, seed) on an event loop of its own, and a
    # pooled keep-alive connection cannot outlive the loop that opened it --
    # reusing one fails the next seed's requests, and closing one afterwards
    # raises RuntimeError: Event loop is closed.
    assert built["default_headers"]["Connection"] == "close"


def test_run_reaches_its_summary_even_when_closing_the_client_fails(
    tmp_path, monkeypatch, caplog
):
    class _FakeAsyncClient:
        async def close(self):
            raise RuntimeError("Event loop is closed")

    _, calls, output_dir = _run_with_client(
        monkeypatch,
        _FakeAsyncClient(),
        tasks=["gsm8k|0", "math_500|0"],
        seeds=[0, 1],
        output_dir=str(tmp_path),
    )

    # Every seed's records are already written by the time the client is
    # closed, so a teardown failure must cost a warning, never the summary.
    assert output_dir == tmp_path
    assert len(calls) == 4
    assert "closing the HTTP client failed" in caplog.text


def test_run_teardown_failure_never_masks_the_real_one(tmp_path, monkeypatch):
    class _FakeAsyncClient:
        async def close(self):
            raise RuntimeError("Event loop is closed")

    def fake_run_experiment_for_task_seed(langfuse_client, *, task, seed, **kwargs):
        raise RuntimeError("the server refused this request")

    monkeypatch.setattr(
        experiment, "run_experiment_for_task_seed", fake_run_experiment_for_task_seed
    )
    monkeypatch.setattr(
        experiment.openai, "AsyncOpenAI", lambda **kwargs: _FakeAsyncClient()
    )
    monkeypatch.setattr(
        experiment,
        "sync_datasets",
        lambda langfuse_client, settings: {t: f"{t}@fp" for t in settings["tasks"]},
    )

    settings = _settings(output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="the server refused this request"):
        experiment.run(settings, langfuse_client=object(), recipe_path="r.yaml")


def test_the_langfuse_path_syncs_before_it_runs_and_uses_the_synced_names(
    tmp_path, monkeypatch
):
    events = []

    class _FakeAsyncClient:
        async def close(self):
            pass

    def fake_sync(langfuse_client, settings):
        events.append("sync")
        return {task: f"{task}@fp[:5]" for task in settings["tasks"]}

    def fake_run_experiment_for_task_seed(langfuse_client, *, task, seed, **kwargs):
        events.append((task, seed, kwargs["dataset_name"]))

    monkeypatch.setattr(experiment, "sync_datasets", fake_sync)
    monkeypatch.setattr(
        experiment, "run_experiment_for_task_seed", fake_run_experiment_for_task_seed
    )
    monkeypatch.setattr(
        experiment.openai, "AsyncOpenAI", lambda **kwargs: _FakeAsyncClient()
    )

    settings = _settings(tasks=["gsm8k|0"], seeds=[0, 1], output_dir=str(tmp_path))
    experiment.run(settings, langfuse_client=object(), recipe_path="r.yaml")

    assert events == [
        "sync",
        ("gsm8k|0", 0, "gsm8k|0@fp[:5]"),
        ("gsm8k|0", 1, "gsm8k|0@fp[:5]"),
    ]


def test_sync_datasets_stops_when_langfuse_failed(monkeypatch):
    class _FlushingClient:
        def flush(self):
            pass

    def fake_sync_recipe(guard, settings):
        guard.failures += 1
        return {"gsm8k|0": ("gsm8k|0@fp", 0)}

    monkeypatch.setattr(experiment.dataset_sync, "sync_recipe", fake_sync_recipe)
    with pytest.raises(RuntimeError, match="drop --tracing-config"):
        experiment.sync_datasets(_FlushingClient(), _settings(tasks=["gsm8k|0"]))


def test_sync_datasets_returns_each_tasks_dataset_name(monkeypatch):
    class _FlushingClient:
        def flush(self):
            pass

    monkeypatch.setattr(
        experiment.dataset_sync,
        "sync_recipe",
        lambda guard, settings: {"gsm8k|0": ("gsm8k|0@fp", 3)},
    )
    names = experiment.sync_datasets(_FlushingClient(), _settings(tasks=["gsm8k|0"]))
    assert names == {"gsm8k|0": "gsm8k|0@fp"}


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
        experiment, "resolve_task_configs", lambda tasks: dict.fromkeys(tasks, config)
    )
    monkeypatch.setattr(
        experiment,
        "iter_documents",
        lambda config, *, max_samples: [
            (str(i), row) for i, row in enumerate(ROWS[:max_samples])
        ],
    )


def test_the_local_path_generates_scores_and_writes_what_reduce_reads(
    tmp_path, monkeypatch, answer_server
):
    pytest.importorskip("lighteval")
    server = answer_server(
        {"What is 2+2?": "4", "What is 3+3?": "6", "What is 4+4?": "9"}
    )
    config = _LocalConfig()
    _patch_local_task(monkeypatch, config)
    settings = _local_settings(server, tmp_path)

    output_dir = experiment.run(settings)

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

    metrics, stats = reduce.reduce_seed(settings, 0, {"arith|0": config}, output_dir)
    assert metrics == {"arith|0": {"em": pytest.approx(2 / 3)}}
    assert stats["documents"] == 3


def test_the_local_path_honours_max_samples(tmp_path, monkeypatch, answer_server):
    pytest.importorskip("lighteval")
    server = answer_server({})
    _patch_local_task(monkeypatch, _LocalConfig())

    experiment.run(_local_settings(server, tmp_path, seeds=[0], max_samples=2))

    lines = (tmp_path / "seed-0" / "arith-0.jsonl").read_text().splitlines()
    assert len(lines) == 2


def test_the_local_path_records_a_refused_request_as_failed(
    tmp_path, monkeypatch, answer_server
):
    pytest.importorskip("lighteval")
    server = answer_server({}, status=400)
    _patch_local_task(monkeypatch, _LocalConfig())

    experiment.run(_local_settings(server, tmp_path, seeds=[0]))

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
        experiment.run(_local_settings(server, tmp_path, seeds=[0]))
    assert server.requests == []
