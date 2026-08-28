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

    def create_dataset_item(self, **kwargs):
        self.create_calls.append(kwargs)
        return kwargs


class _AlwaysRaisingClient:
    def create_dataset_item(self, **kwargs):
        raise RuntimeError("langfuse is down")


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
