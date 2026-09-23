import sys
from types import SimpleNamespace
from typing import Any

import pytest

from open_r1_tpu.grpo import data as grpo_data
from open_r1_tpu.grpo.data import build_row_encoder, render_prompt


class FakeTokenizer:
    """Mirrors tests/test_data.py's FakeTokenizer, plus a token-length knob."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt) -> Any:
        assert tokenize in (True, False)
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        if not tokenize:
            return rendered
        return [ord(character) for character in rendered]


# ---------------------------------------------------------------------------
# render_prompt
# ---------------------------------------------------------------------------


def test_render_prompt_includes_system_prompt_and_opens_generation():
    prompt = render_prompt("2+2?", FakeTokenizer(), system_prompt="Be terse.")
    assert prompt == "<system>Be terse.</system><user>2+2?</user><assistant>"


def test_render_prompt_omits_system_message_when_none_configured():
    prompt = render_prompt("2+2?", FakeTokenizer(), system_prompt=None)
    assert prompt == "<user>2+2?</user><assistant>"


def test_render_prompt_rejects_a_non_string_template_result():
    class BadTokenizer(FakeTokenizer):
        def apply_chat_template(self, *args, **kwargs):
            return [1, 2, 3]

    with pytest.raises(ValueError):
        render_prompt("q", BadTokenizer(), system_prompt=None)


# ---------------------------------------------------------------------------
# build_row_encoder (the per-row Grain transform, tested without Grain)
# ---------------------------------------------------------------------------


def test_row_encoder_builds_prompts_question_and_answer():
    encode = build_row_encoder(
        FakeTokenizer(),
        question_column="problem",
        answer_column="answer",
        system_prompt=None,
        max_prompt_length=None,
    )
    record = encode({"problem": "What is 2+2?", "answer": "4"})
    assert record == {
        "prompts": "<user>What is 2+2?</user><assistant>",
        "question": "What is 2+2?",
        "answer": "4",
    }


def test_row_encoder_honours_a_custom_column_mapping():
    encode = build_row_encoder(
        FakeTokenizer(),
        question_column="question",
        answer_column="gold",
        system_prompt=None,
        max_prompt_length=None,
    )
    record = encode({"question": "q", "gold": "g"})
    assert record is not None
    assert record["question"] == "q"
    assert record["answer"] == "g"


def test_row_encoder_drops_rows_over_the_prompt_budget():
    # FakeTokenizer's "tokenization" is one id per character, so a budget
    # smaller than the rendered length is guaranteed to trip.
    encode = build_row_encoder(
        FakeTokenizer(),
        question_column="problem",
        answer_column="answer",
        system_prompt=None,
        max_prompt_length=5,
    )
    assert encode({"problem": "a much longer problem statement", "answer": "4"}) is None


def test_row_encoder_keeps_rows_within_the_prompt_budget():
    encode = build_row_encoder(
        FakeTokenizer(),
        question_column="problem",
        answer_column="answer",
        system_prompt=None,
        max_prompt_length=1000,
    )
    assert encode({"problem": "short", "answer": "4"}) is not None


# ---------------------------------------------------------------------------
# load_grpo_prompts, with datasets and Grain both stubbed
# ---------------------------------------------------------------------------


class FakeHFDataset:
    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def filter(self, fn):
        return FakeHFDataset([row for row in self.rows if fn(row)])

    def select(self, indices):
        indices = list(indices)
        return FakeHFDataset([self.rows[i] for i in indices])

    def train_test_split(self, *, test_size, seed):
        cut = len(self.rows) - max(1, int(len(self.rows) * test_size))
        return {
            "train": FakeHFDataset(self.rows[:cut]),
            "test": FakeHFDataset(self.rows[cut:]),
        }


def test_load_grpo_prompts_forwards_name_and_data_files(monkeypatch):
    captured = {}

    def fake_load_dataset(name, config, **kwargs):
        captured.update(name=name, config=config, **kwargs)
        return FakeHFDataset([{"problem": "q", "answer": "4"}])

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )
    monkeypatch.setattr(
        grpo_data,
        "build_grain_prompt_batches",
        lambda source, *, to_record, batch_size, seed, shuffle, num_epochs: list(
            filter(None, (to_record(row) for row in source.rows))
        ),
    )
    grpo_data.load_grpo_prompts(
        {
            "name": "parquet",
            "config": None,
            "data_files": "data/DAPO-Math-17k-Processed/en/train-*.parquet",
            "train_split": "train",
            "batch_size": 2,
        },
        FakeTokenizer(),
    )
    assert captured["name"] == "parquet"
    assert captured["data_files"] == "data/DAPO-Math-17k-Processed/en/train-*.parquet"
    assert captured["split"] == "train"


def test_load_grpo_prompts_drops_rows_missing_question_or_answer(monkeypatch):
    rows = [
        {"problem": "q1", "answer": "4"},
        {"problem": "", "answer": "4"},
        {"problem": "q3", "answer": None},
    ]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *a, **k: FakeHFDataset(rows)),
    )
    seen_sources = []

    def fake_build(source, *, to_record, batch_size, seed, shuffle, num_epochs):
        seen_sources.append(source)
        return source

    monkeypatch.setattr(grpo_data, "build_grain_prompt_batches", fake_build)
    train_ds, eval_ds = grpo_data.load_grpo_prompts(
        {"name": "parquet", "config": None, "batch_size": 1}, FakeTokenizer()
    )
    assert eval_ds is None
    assert [row["problem"] for row in train_ds.rows] == ["q1"]


def test_load_grpo_prompts_splits_off_an_eval_fraction(monkeypatch):
    rows = [{"problem": f"q{i}", "answer": "4"} for i in range(10)]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *a, **k: FakeHFDataset(rows)),
    )
    calls = []

    def fake_build(source, *, to_record, batch_size, seed, shuffle, num_epochs):
        calls.append(shuffle)
        return source

    monkeypatch.setattr(grpo_data, "build_grain_prompt_batches", fake_build)
    train_ds, eval_ds = grpo_data.load_grpo_prompts(
        {"name": "parquet", "config": None, "batch_size": 1, "eval_fraction": 0.2},
        FakeTokenizer(),
    )
    assert eval_ds is not None
    assert len(train_ds.rows) == 8
    assert len(eval_ds.rows) == 2
    assert calls == [True, False]


def test_load_grpo_prompts_uses_a_separate_eval_batch_size(monkeypatch):
    rows = [{"problem": f"q{i}", "answer": "4"} for i in range(10)]
    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *a, **k: FakeHFDataset(rows)),
    )
    sizes = []

    def fake_build(source, *, to_record, batch_size, seed, shuffle, num_epochs):
        sizes.append((shuffle, batch_size))
        return source

    monkeypatch.setattr(grpo_data, "build_grain_prompt_batches", fake_build)
    config = {"name": "parquet", "config": None, "batch_size": 4, "eval_fraction": 0.2}
    grpo_data.load_grpo_prompts(config, FakeTokenizer())
    grpo_data.load_grpo_prompts(config | {"eval_batch_size": 1}, FakeTokenizer())
    assert sizes == [(True, 4), (False, 4), (True, 4), (False, 1)]
