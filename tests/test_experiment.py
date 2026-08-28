"""Tests for `open_r1_tpu.evaluation.experiment`, against a faked Langfuse
client/dataset -- no real `run_experiment`, no network, no lighteval
registry. `run_experiment_for_task_seed`/`run` only ever call
`dataset.run_experiment()` and read back its return value, so faking that
one call is enough to exercise everything this module owns: the run name,
the evaluator/task wiring, and the JSONL record shape `evaluation.reduce`
reads.
"""

from __future__ import annotations

import json

import pytest

from open_r1_tpu.evaluation import experiment


class _FakeEvaluation:
    def __init__(self, name, value, metadata=None):
        self.name = name
        self.value = value
        self.metadata = metadata


class _FakeDatasetItem:
    def __init__(self, item_id, metadata):
        self.id = item_id
        self.metadata = metadata


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


class _StubResolvedConfig:
    """A placeholder resolved config. `derive_task_spec`/`dataset_name` are
    monkeypatched out in every test that reaches `run_experiment_for_task_seed`
    -- naming is `taskpack.dataset_fingerprint`'s job, already covered
    unconditionally in `test_taskpack.py`, and deriving a real spec would
    need a real dataset load for its best-effort `example` field."""


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
    item = _FakeDatasetItem("item-1", {"doc_id": "3", "task": "stub_task|0"})
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
    monkeypatch.setattr(experiment, "derive_task_spec", lambda task, config: None)
    monkeypatch.setattr(
        experiment, "taskpack_dataset_name", lambda task, spec: f"{task}@fixedprint"
    )

    settings = _settings()
    experiment.run_experiment_for_task_seed(
        client,
        task="stub_task|0",
        seed=2,
        settings=settings,
        resolved_config=_StubResolvedConfig(),
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
    assert "dataset_fingerprint" in call["metadata"]

    (name,) = client.requested_names
    assert name.startswith("stub_task|0@")

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
        "resolve_task_configs",
        lambda tasks: {task: _StubResolvedConfig() for task in tasks},
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
        "resolve_task_configs",
        lambda tasks: {task: _StubResolvedConfig() for task in tasks},
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
        "resolve_task_configs",
        lambda tasks: {task: _StubResolvedConfig() for task in tasks},
    )

    settings = _settings(output_dir=str(tmp_path))
    with pytest.raises(RuntimeError, match="the server refused this request"):
        experiment.run(settings, langfuse_client=object(), recipe_path="r.yaml")
