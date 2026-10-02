"""Tests for `open_r1_tpu.evaluation.traced`, against fake Langfuse clients:
the tracing config, the dataset sync, the experiment records, and the
LightEval evaluator.
"""

from __future__ import annotations

import copy
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

# The evaluation client needs the `eval` extra; a training-only install skips.
pytest.importorskip("openai")

from open_r1_tpu.evaluation import scoring, tasks, traced
from open_r1_tpu.evaluation.traced import LangfuseGuard

# --- the tracing config -----------------------------------------------------

EXAMPLE_CONFIG = Path(__file__).parents[1] / "configs" / "tracing.example.yaml"

VALID_CONFIG = {
    "langfuse": {"host": "127.0.0.1", "port": 3000},
}


def write_config(tmp_path: Path, overrides: dict | None = None) -> Path:
    payload = copy.deepcopy(VALID_CONFIG)
    for section, keys in (overrides or {}).items():
        if keys is None:
            payload.pop(section, None)
            continue
        payload.setdefault(section, {}).update(keys)
    path = tmp_path / "tracing.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_the_committed_example_config_loads():
    config = traced.load_tracing_config(EXAMPLE_CONFIG)
    assert config["langfuse"]["host"]
    assert config["langfuse"]["port"] == 3000


def test_a_valid_config_loads(tmp_path):
    config = traced.load_tracing_config(write_config(tmp_path))
    assert config["langfuse"]["host"] == "127.0.0.1"


def test_a_missing_section_names_it(tmp_path):
    path = write_config(tmp_path, {"langfuse": None})
    with pytest.raises(ValueError, match="Missing configuration section: langfuse"):
        traced.load_tracing_config(path)


def test_a_missing_key_within_a_section_names_it(tmp_path):
    payload = copy.deepcopy(VALID_CONFIG)
    del payload["langfuse"]["host"]
    path = tmp_path / "tracing.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"langfuse\.host"):
        traced.load_tracing_config(path)


def test_an_unknown_key_suggests_the_near_match(tmp_path):
    path = write_config(tmp_path, {"langfuse": {"hosst": "typo"}})
    with pytest.raises(ValueError, match=r"langfuse\.hosst.*did you mean 'host'"):
        traced.load_tracing_config(path)


def test_an_unknown_top_level_section_is_rejected(tmp_path):
    path = write_config(tmp_path, {"langfusee": {"host": "x"}})
    with pytest.raises(
        ValueError,
        match="Unknown configuration section langfusee; did you mean 'langfuse'",
    ):
        traced.load_tracing_config(path)


def test_a_typoed_dotted_override_is_caught_after_merging(tmp_path):
    path = write_config(tmp_path)
    with pytest.raises(ValueError, match=r"langfuse\.hosst"):
        traced.load_tracing_config(path, ["langfuse.hosst=typo"])


def test_a_valid_dotted_override_applies(tmp_path):
    path = write_config(tmp_path)
    config = traced.load_tracing_config(path, ["langfuse.port=5000"])
    assert config["langfuse"]["port"] == 5000


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("langfuse", "port", 0),
        ("langfuse", "port", 70000),
        ("langfuse", "port", "not-a-port"),
        ("langfuse", "host", ""),
    ],
)
def test_invalid_values_are_rejected(tmp_path, section, key, value):
    path = write_config(tmp_path, {section: {key: value}})
    with pytest.raises(ValueError):
        traced.load_tracing_config(path)


def test_build_langfuse_client_uses_the_langfuse_section(tmp_path, monkeypatch):
    captured = {}

    class _FakeLangfuse:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_module = types.ModuleType("langfuse")
    # setattr, not `fake_module.Langfuse = ...`: ModuleType has no such
    # attribute to assign to, which pyright rejects outright.
    setattr(fake_module, "Langfuse", _FakeLangfuse)  # noqa: B010
    monkeypatch.setitem(sys.modules, "langfuse", fake_module)

    config = traced.load_tracing_config(write_config(tmp_path))
    traced.build_langfuse_client(config)
    assert captured["base_url"] == "http://127.0.0.1:3000"


# --- dataset sync -----------------------------------------------------------


class _StubDoc:
    def __init__(self, doc_id, choices, gold_index: int | list[int] = 0, specific=None):
        self.query = f"question {doc_id}"
        self.choices = choices
        self.gold_index = gold_index
        self.instruction = None
        self.fewshot_samples = []
        self.specific = specific
        self.task_name = "stub_task"

    def get_golds(self):
        gold_indices = (
            self.gold_index if isinstance(self.gold_index, list) else [self.gold_index]
        )
        golds = []
        for index in gold_indices:
            value = self.choices[index]
            golds.extend(value if isinstance(value, list) else [value])
        return golds


class _StubConfig:
    def __init__(
        self, prompt_function, hf_avail_splits=("test",), evaluation_splits=("test",)
    ):
        self.prompt_function = prompt_function
        self.hf_repo = "stub"
        self.hf_subset = "stub"
        self.hf_revision = None
        self.hf_avail_splits = list(hf_avail_splits)
        self.evaluation_splits = list(evaluation_splits)


class FakeLangfuseClient:
    def __init__(self):
        self.create_calls = []
        self.dataset_calls = []

    def create_dataset(self, **kwargs):
        self.dataset_calls.append(kwargs)
        return kwargs

    def create_dataset_item(self, **kwargs):
        self.create_calls.append(kwargs)
        return kwargs


class _AlwaysRaisingClient:
    def create_dataset(self, **kwargs):
        raise RuntimeError("langfuse is down")

    def create_dataset_item(self, **kwargs):
        raise RuntimeError("langfuse is down")


class _SelectiveDatasetClient:
    """`create_dataset` fails for one specific name (simulating a dead
    Langfuse for just that task's dataset); `create_dataset_item` always
    succeeds. Records every call, in order, in one shared list so call
    ordering across the two methods is observable."""

    def __init__(self, failing_name):
        self._failing_name = failing_name
        self.calls = []

    def create_dataset(self, **kwargs):
        self.calls.append(("create_dataset", kwargs))
        if kwargs["name"] == self._failing_name:
            raise RuntimeError("langfuse is down for this dataset")
        return kwargs

    def create_dataset_item(self, **kwargs):
        self.calls.append(("create_dataset_item", kwargs))
        return kwargs


def _fake_iter_documents(monkeypatch, count):
    monkeypatch.setattr(
        traced,
        "iter_documents",
        lambda config, *, max_samples: [(str(i), {"id": i}) for i in range(count)],
    )


# --- item_id -----------------------------------------------------------


def test_item_id_is_deterministic_on_dataset_and_doc_id():
    first = traced.item_id("gsm8k|0@abcd1234", "3")
    second = traced.item_id("gsm8k|0@abcd1234", "3")
    assert first == second
    assert first != traced.item_id("gsm8k|0@abcd1234", "4")
    assert first != traced.item_id("gsm8k|0@deadbeef", "3")


# --- sync_task -----------------------------------------------------------


def test_sync_task_upserts_one_item_per_document(monkeypatch):
    _fake_iter_documents(monkeypatch, 3)
    client = FakeLangfuseClient()
    guard = LangfuseGuard(client)
    config = _StubConfig(
        prompt_function=lambda row, task_name: _StubDoc(row["id"], choices=["answer"])
    )

    count = traced.sync_task(
        guard,
        "stub_task|0",
        config,
        name="stub_task|0@abcd1234",
        system_prompt=None,
        max_samples=None,
    )

    assert count == 3
    assert len(client.create_calls) == 3
    for i, call in enumerate(client.create_calls):
        assert call["dataset_name"] == "stub_task|0@abcd1234"
        assert call["expected_output"] == "answer"
        assert call["input"] == [{"role": "user", "content": f"question {i}"}]
        assert call["metadata"] == {
            "task": "stub_task|0",
            "doc_id": str(i),
            "specific": None,
            "query": f"question {i}",
        }
        assert call["id"] == traced.item_id("stub_task|0@abcd1234", str(i))


def test_sync_task_stores_lightevals_gold_verbatim_including_prefix(monkeypatch):
    _fake_iter_documents(monkeypatch, 1)
    client = FakeLangfuseClient()
    guard = LangfuseGuard(client)
    # math_500's own convention: choices=["ANSWER: {solution}"], gold_index=0.
    config = _StubConfig(
        prompt_function=lambda row, task_name: _StubDoc(
            row["id"], choices=["ANSWER: 42"]
        )
    )

    traced.sync_task(
        guard,
        "math_500|0",
        config,
        name="math_500|0@abcd1234",
        system_prompt=None,
        max_samples=None,
    )

    assert client.create_calls[0]["expected_output"] == "ANSWER: 42"


def test_sync_task_prepends_the_system_prompt_when_set(monkeypatch):
    _fake_iter_documents(monkeypatch, 1)
    client = FakeLangfuseClient()
    guard = LangfuseGuard(client)
    config = _StubConfig(
        prompt_function=lambda row, task_name: _StubDoc(row["id"], choices=["answer"])
    )

    traced.sync_task(
        guard,
        "stub_task|0",
        config,
        name="ds",
        system_prompt="be nice",
        max_samples=None,
    )

    assert client.create_calls[0]["input"][0] == {
        "role": "system",
        "content": "be nice",
    }


def test_sync_task_is_idempotent_across_reruns(monkeypatch):
    _fake_iter_documents(monkeypatch, 3)
    config = _StubConfig(
        prompt_function=lambda row, task_name: _StubDoc(row["id"], choices=["answer"])
    )

    first_client = FakeLangfuseClient()
    traced.sync_task(
        LangfuseGuard(first_client),
        "stub_task|0",
        config,
        name="ds",
        system_prompt=None,
        max_samples=None,
    )
    second_client = FakeLangfuseClient()
    traced.sync_task(
        LangfuseGuard(second_client),
        "stub_task|0",
        config,
        name="ds",
        system_prompt=None,
        max_samples=None,
    )

    first_ids = [c["id"] for c in first_client.create_calls]
    second_ids = [c["id"] for c in second_client.create_calls]
    assert first_ids == second_ids
    assert len(set(first_ids)) == 3


def test_sync_task_rejects_a_multi_gold_document(monkeypatch):
    _fake_iter_documents(monkeypatch, 1)
    guard = LangfuseGuard(FakeLangfuseClient())
    config = _StubConfig(
        prompt_function=lambda row, task_name: _StubDoc(
            row["id"], choices=["a", "b"], gold_index=[0, 1]
        )
    )

    with pytest.raises(ValueError, match="single gold"):
        traced.sync_task(
            guard,
            "stub_task|0",
            config,
            name="ds",
            system_prompt=None,
            max_samples=None,
        )


def test_sync_task_survives_a_dead_langfuse(monkeypatch):
    _fake_iter_documents(monkeypatch, 5)
    guard = LangfuseGuard(_AlwaysRaisingClient())
    config = _StubConfig(
        prompt_function=lambda row, task_name: _StubDoc(row["id"], choices=["answer"])
    )

    # Must not raise: a dead Langfuse costs missing items, never a crashed sync.
    count = traced.sync_task(
        guard, "stub_task|0", config, name="ds", system_prompt=None, max_samples=None
    )
    assert count == 5
    assert guard.failures == 5


# --- ensure_dataset --------------------------------------------------------


def test_ensure_dataset_creates_the_named_dataset():
    client = FakeLangfuseClient()
    guard = LangfuseGuard(client)

    assert traced.ensure_dataset(guard, "stub_task|0@abcd1234") is True
    assert client.dataset_calls == [{"name": "stub_task|0@abcd1234"}]


def test_ensure_dataset_returns_false_on_a_dead_langfuse():
    guard = LangfuseGuard(_AlwaysRaisingClient())

    assert traced.ensure_dataset(guard, "ds") is False
    assert guard.failures == 1


# --- sync_recipe: dataset creation happens before any item upsert ---------


def _fake_resolve_and_name(monkeypatch, configs_by_task):
    monkeypatch.setattr(traced, "resolve_task_configs", lambda tasks: configs_by_task)
    monkeypatch.setattr(
        traced,
        "dataset_name",
        lambda task, config, max_samples=None: (
            f"{task}@fixed" + (f"[:{max_samples}]" if max_samples else "")
        ),
    )


def test_sync_recipe_creates_the_dataset_before_any_item(monkeypatch):
    _fake_iter_documents(monkeypatch, 3)
    config = _StubConfig(
        prompt_function=lambda row, task_name: _StubDoc(row["id"], choices=["answer"])
    )
    _fake_resolve_and_name(monkeypatch, {"stub_task|0": config})
    client = _SelectiveDatasetClient(failing_name="nothing-fails")
    guard = LangfuseGuard(client)

    results = traced.sync_recipe(guard, {"tasks": ["stub_task|0"]})

    assert results == {"stub_task|0": ("stub_task|0@fixed", 3)}
    kinds = [kind for kind, _ in client.calls]
    assert kinds == [
        "create_dataset",
        "create_dataset_item",
        "create_dataset_item",
        "create_dataset_item",
    ]


def test_sync_recipe_skips_a_tasks_items_when_its_dataset_cannot_be_ensured(
    monkeypatch,
):
    _fake_iter_documents(monkeypatch, 2)
    configs = {
        "broken_task|0": _StubConfig(
            prompt_function=lambda row, task_name: _StubDoc(row["id"], choices=["x"])
        ),
        "ok_task|0": _StubConfig(
            prompt_function=lambda row, task_name: _StubDoc(row["id"], choices=["y"])
        ),
    }
    _fake_resolve_and_name(monkeypatch, configs)
    client = _SelectiveDatasetClient(failing_name="broken_task|0@fixed")
    guard = LangfuseGuard(client)

    results = traced.sync_recipe(guard, {"tasks": ["broken_task|0", "ok_task|0"]})

    assert results["broken_task|0"] == ("broken_task|0@fixed", 0)
    assert results["ok_task|0"] == ("ok_task|0@fixed", 2)
    item_calls = [
        kwargs for kind, kwargs in client.calls if kind == "create_dataset_item"
    ]
    assert all(call["dataset_name"] == "ok_task|0@fixed" for call in item_calls)
    assert len(item_calls) == 2
    assert guard.failures == 1


def test_sync_recipe_names_a_capped_task_by_its_cap(monkeypatch):
    seen = {}

    def fake_iter_documents(config, *, max_samples):
        seen["max_samples"] = max_samples
        return [(str(i), {"id": i}) for i in range(max_samples)]

    monkeypatch.setattr(traced, "iter_documents", fake_iter_documents)
    config = _StubConfig(
        prompt_function=lambda row, task_name: _StubDoc(row["id"], choices=["answer"])
    )
    _fake_resolve_and_name(monkeypatch, {"stub_task|0": config})
    guard = LangfuseGuard(FakeLangfuseClient())

    results = traced.sync_recipe(guard, {"tasks": ["stub_task|0"], "max_samples": 2})

    assert results == {"stub_task|0": ("stub_task|0@fixed[:2]", 2)}
    assert seen["max_samples"] == 2


# --- experiments ------------------------------------------------------------


class _FakeEvaluation:
    def __init__(self, name, value, metadata=None):
        self.name = name
        self.value = value
        self.metadata = metadata


class _FakeDatasetItem:
    def __init__(self, item_id, metadata, expected_output="gold"):
        self.id = item_id
        self.metadata = metadata
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


class _FakeExperimentClient:
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


# --- write_experiment_jsonl -------------------------------------------------


def test_write_experiment_jsonl_writes_the_record_shape_summary_reads(tmp_path):
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
    traced.write_experiment_jsonl(
        result, dataset, task="stub_task|0", seed=0, output_path=output_path
    )

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert records == [
        {
            "status": "ok",
            "doc_id": "3",
            "task": "stub_task|0",
            "seed": 0,
            # What `evaluation.consensus` rebuilds a scoring Doc from.
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
    traced.write_experiment_jsonl(
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
    traced.write_experiment_jsonl(
        result, dataset, task="stub_task|0", seed=0, output_path=output_path
    )

    (record,) = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert record["failed_metrics"] == ["boom"]
    assert record["scoring_errors"] == {"boom": "kaboom"}
    assert record["scores"] == {}


# --- run_experiment_for_task_seed and sync_datasets -------------------------


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
    client = _FakeExperimentClient(dataset)

    captured_evaluator_task_name = {}

    def fake_lighteval_evaluator(task_name):
        captured_evaluator_task_name["task"] = task_name
        return lambda **kwargs: []

    def fake_make_task(settings, *, client):
        return lambda **kwargs: None

    monkeypatch.setattr(traced, "lighteval_evaluator", fake_lighteval_evaluator)
    monkeypatch.setattr(traced, "make_task", fake_make_task)

    settings = _settings()
    traced.run_experiment_for_task_seed(
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


def test_sync_datasets_stops_when_langfuse_failed(monkeypatch):
    class _FlushingClient:
        def flush(self):
            pass

    def fake_sync_recipe(guard, settings):
        guard.failures += 1
        return {"gsm8k|0": ("gsm8k|0@fp", 0)}

    monkeypatch.setattr(traced, "sync_recipe", fake_sync_recipe)
    with pytest.raises(RuntimeError, match="drop --tracing-config"):
        traced.sync_datasets(_FlushingClient(), _settings(tasks=["gsm8k|0"]))


def test_sync_datasets_returns_each_tasks_dataset_name(monkeypatch):
    class _FlushingClient:
        def flush(self):
            pass

    monkeypatch.setattr(
        traced,
        "sync_recipe",
        lambda guard, settings: {"gsm8k|0": ("gsm8k|0@fp", 3)},
    )
    names = traced.sync_datasets(_FlushingClient(), _settings(tasks=["gsm8k|0"]))
    assert names == {"gsm8k|0": "gsm8k|0@fp"}


# --- lighteval_evaluator: needs LightEval and Langfuse ---------------------


def _evaluator_inputs(task: str):
    """The task's evaluator, plus its first document (downloaded from the Hub)
    and that document's item metadata.
    """
    pytest.importorskip("lighteval")
    from datasets import load_dataset

    config = tasks.resolve_task_configs([task])[task]
    row = load_dataset(
        config.hf_repo, config.hf_subset, split=f"{config.evaluation_splits[0]}[:1]"
    )[0]
    doc = scoring.build_doc(config.prompt_function, row, task)
    metadata = {
        "task": task,
        "doc_id": "0",
        "specific": doc.specific,
        "query": doc.query,
    }
    return traced.lighteval_evaluator(task), doc, metadata


@pytest.mark.network
def test_lighteval_evaluator_scores_a_correct_and_an_incorrect_completion():
    evaluator, doc, metadata = _evaluator_inputs("gsm8k|0")
    gold = doc.get_golds()[0]

    correct = evaluator(
        input=doc.query,
        output={"text": gold, "finish_reason": "stop", "completion_tokens": 3},
        expected_output=gold,
        metadata=metadata,
    )
    by_name = {e.name: e for e in correct}
    assert by_name["extractive_match"].value == 1.0
    assert by_name["extractive_match"].data_type == "NUMERIC"
    assert by_name["completion_tokens"].value == 3.0
    assert by_name["truncated"].value == 0.0

    wrong = evaluator(
        input=doc.query,
        output={
            "text": " not the answer at all",
            "finish_reason": "length",
            "completion_tokens": 5,
        },
        expected_output=gold,
        metadata=metadata,
    )
    wrong_by_name = {e.name: e for e in wrong}
    assert wrong_by_name["extractive_match"].value == 0.0
    assert wrong_by_name["truncated"].value == 1.0


@pytest.mark.network
def test_lighteval_evaluator_returns_several_named_scores_for_a_grouping_metric():
    evaluator, doc, metadata = _evaluator_inputs("ifeval|0")

    evaluations = evaluator(
        input=doc.query,
        output={
            "text": "an arbitrary completion",
            "finish_reason": "stop",
            "completion_tokens": 4,
        },
        expected_output=doc.get_golds()[0],
        metadata=metadata,
    )
    names = {e.name for e in evaluations}
    assert {"prompt_level_strict_acc", "prompt_level_loose_acc"} <= names


def test_lighteval_evaluator_marks_a_metric_failure(monkeypatch):
    pytest.importorskip("lighteval")
    pytest.importorskip("langfuse")

    class _RaisingMetric:
        metric_name = "boom"
        batched_compute = False

        def compute_sample(self, **kwargs):
            raise RuntimeError("kaboom")

    class _FakeConfig:
        def __init__(self):
            self.metrics = [_RaisingMetric()]

    monkeypatch.setattr(
        traced, "resolve_task_configs", lambda tasks: {tasks[0]: _FakeConfig()}
    )

    evaluator = traced.lighteval_evaluator("gsm8k|0")
    metadata = {
        "task": "gsm8k|0",
        "doc_id": "0",
        "specific": None,
        "query": "Question: x\nAnswer:",
    }
    evaluations = evaluator(
        input=metadata["query"],
        output={"text": "18", "finish_reason": "stop", "completion_tokens": 2},
        expected_output=" 18",
        metadata=metadata,
    )
    by_name = {e.name: e for e in evaluations}
    assert by_name["scoring_failed"].value == 1.0
    assert by_name["scoring_failed"].metadata["failed_metrics"] == ["boom"]
    assert "kaboom" in by_name["scoring_failed"].metadata["errors"]["boom"]
