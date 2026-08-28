"""Tests for `open_r1_tpu.evaluation.dataset_sync`, against a faked Langfuse
client -- no lighteval registry, no dataset download, no network. `sync_task`
takes its dataset `name` as a parameter (rather than deriving one itself) for
exactly this reason: naming is `taskpack.dataset_fingerprint`'s job, already
covered unconditionally in `test_taskpack.py`.
"""

from __future__ import annotations

import pytest

from open_r1_tpu.evaluation import dataset_sync
from open_r1_tpu.evaluation.runner import LangfuseGuard


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
        dataset_sync,
        "iter_documents",
        lambda config, *, max_samples: [(str(i), {"id": i}) for i in range(count)],
    )


# --- item_id -----------------------------------------------------------


def test_item_id_is_deterministic_on_dataset_and_doc_id():
    first = dataset_sync.item_id("gsm8k|0@abcd1234", "3")
    second = dataset_sync.item_id("gsm8k|0@abcd1234", "3")
    assert first == second
    assert first != dataset_sync.item_id("gsm8k|0@abcd1234", "4")
    assert first != dataset_sync.item_id("gsm8k|0@deadbeef", "3")


# --- sync_task -----------------------------------------------------------


def test_sync_task_upserts_one_item_per_document(monkeypatch):
    _fake_iter_documents(monkeypatch, 3)
    client = FakeLangfuseClient()
    guard = LangfuseGuard(client)
    config = _StubConfig(
        prompt_function=lambda row, task_name: _StubDoc(row["id"], choices=["answer"])
    )

    count = dataset_sync.sync_task(
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
        assert call["id"] == dataset_sync.item_id("stub_task|0@abcd1234", str(i))


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

    dataset_sync.sync_task(
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

    dataset_sync.sync_task(
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
    dataset_sync.sync_task(
        LangfuseGuard(first_client),
        "stub_task|0",
        config,
        name="ds",
        system_prompt=None,
        max_samples=None,
    )
    second_client = FakeLangfuseClient()
    dataset_sync.sync_task(
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
        dataset_sync.sync_task(
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
    count = dataset_sync.sync_task(
        guard, "stub_task|0", config, name="ds", system_prompt=None, max_samples=None
    )
    assert count == 5
    assert guard.failures == 5


# --- ensure_dataset --------------------------------------------------------


def test_ensure_dataset_creates_the_named_dataset():
    client = FakeLangfuseClient()
    guard = LangfuseGuard(client)

    assert dataset_sync.ensure_dataset(guard, "stub_task|0@abcd1234") is True
    assert client.dataset_calls == [{"name": "stub_task|0@abcd1234"}]


def test_ensure_dataset_returns_false_on_a_dead_langfuse():
    guard = LangfuseGuard(_AlwaysRaisingClient())

    assert dataset_sync.ensure_dataset(guard, "ds") is False
    assert guard.failures == 1


# --- sync_recipe: dataset creation happens before any item upsert ---------


def _fake_resolve_and_name(monkeypatch, configs_by_task):
    monkeypatch.setattr(
        dataset_sync, "resolve_task_configs", lambda tasks: configs_by_task
    )
    monkeypatch.setattr(dataset_sync, "derive_task_spec", lambda task, config: None)
    monkeypatch.setattr(
        dataset_sync, "taskpack_dataset_name", lambda task, spec: f"{task}@fixed"
    )


def test_sync_recipe_creates_the_dataset_before_any_item(monkeypatch):
    _fake_iter_documents(monkeypatch, 3)
    config = _StubConfig(
        prompt_function=lambda row, task_name: _StubDoc(row["id"], choices=["answer"])
    )
    _fake_resolve_and_name(monkeypatch, {"stub_task|0": config})
    client = _SelectiveDatasetClient(failing_name="nothing-fails")
    guard = LangfuseGuard(client)

    results = dataset_sync.sync_recipe(guard, {"tasks": ["stub_task|0"]})

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

    results = dataset_sync.sync_recipe(guard, {"tasks": ["broken_task|0", "ok_task|0"]})

    assert results["broken_task|0"] == ("broken_task|0@fixed", 0)
    assert results["ok_task|0"] == ("ok_task|0@fixed", 2)
    item_calls = [
        kwargs for kind, kwargs in client.calls if kind == "create_dataset_item"
    ]
    assert all(call["dataset_name"] == "ok_task|0@fixed" for call in item_calls)
    assert len(item_calls) == 2
    assert guard.failures == 1
