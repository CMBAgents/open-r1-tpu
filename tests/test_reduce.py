"""Tests for `open_r1_tpu.evaluation.reduce` -- fixture JSONL in, the same
summary shape `evaluation.run.build_summary` has always produced out. No
LightEval needed: `FakeMetric` stands in for a real `Metric`, exposing only
`get_corpus_aggregations()`.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from open_r1_tpu.evaluation import reduce


class FakeMetric:
    def __init__(self, aggregations):
        self._aggregations = aggregations

    def get_corpus_aggregations(self):
        return self._aggregations


class FakeConfig:
    def __init__(self, metrics):
        self.metrics = list(metrics)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


# --- read_jsonl ---------------------------------------------------------


def test_read_jsonl_missing_file_is_a_named_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No runner output"):
        reduce.read_jsonl(tmp_path / "nope.jsonl")


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert reduce.read_jsonl(path) == [{"a": 1}, {"a": 2}]


# --- reduce_task_metrics --------------------------------------------------


def test_reduce_task_metrics_uses_the_metrics_own_corpus_fn():
    metric = FakeMetric({"acc": statistics.fmean})
    records = [
        {"status": "ok", "scores": {"acc": 1.0}},
        {"status": "ok", "scores": {"acc": 0.0}},
    ]
    assert reduce.reduce_task_metrics(records, [metric]) == {"acc": 0.5}


def test_reduce_task_metrics_skips_none_missing_and_failed_documents():
    metric = FakeMetric({"acc": statistics.fmean})
    records = [
        {"status": "ok", "scores": {"acc": 1.0}},
        {"status": "ok", "scores": {"acc": None}},
        {"status": "ok", "scores": {}},
        {"status": "generation_failed"},
    ]
    # Only the first record has a usable value; a mean of one value is itself.
    assert reduce.reduce_task_metrics(records, [metric]) == {"acc": 1.0}


def test_reduce_task_metrics_uses_a_non_mean_corpus_fn_when_the_metric_has_one():
    # ifeval's inst_level_*_acc: one list-of-bools per document, flattened
    # and meaned across the whole corpus -- not a mean of per-document means.
    def flatten_mean(items):
        flat = [x for sublist in items for x in sublist]
        return sum(flat) / len(flat)

    metric = FakeMetric({"inst_level_strict_acc": flatten_mean})
    records = [
        {"status": "ok", "scores": {"inst_level_strict_acc": [True, False]}},
        {"status": "ok", "scores": {"inst_level_strict_acc": [True]}},
    ]
    # 2 of 3 instructions correct overall, not a mean of [0.5, 1.0] (= 0.75).
    assert reduce.reduce_task_metrics(records, [metric]) == pytest.approx(
        {"inst_level_strict_acc": 2 / 3}
    )


def test_reduce_task_metrics_returns_nothing_when_no_document_has_a_value():
    metric = FakeMetric({"acc": statistics.fmean})
    records = [{"status": "generation_failed"}]
    assert reduce.reduce_task_metrics(records, [metric]) == {}


# --- completion_stats_from_records ---------------------------------------


def test_truncation_rate_comes_from_finish_reason_not_token_count():
    records = [
        # Token count alone would call this truncated under the old
        # heuristic (>= a hypothetical max_new_tokens), but finish_reason
        # says the model stopped on its own.
        {
            "status": "ok",
            "completion": "x",
            "completion_tokens": 999,
            "finish_reason": "stop",
        },
        {
            "status": "ok",
            "completion": "x",
            "completion_tokens": 1,
            "finish_reason": "length",
        },
    ]
    stats = reduce.completion_stats_from_records(
        records, reasoning_start=None, reasoning_end="</think>", answer_marker="ANSWER:"
    )
    assert stats["truncation_rate"] == 0.5


def test_completion_stats_reasoning_and_marker_rates():
    records = [
        {
            "status": "ok",
            "completion": "reasoning</think>ANSWER: 4",
            "completion_tokens": 10,
            "finish_reason": "stop",
        },
        {
            "status": "ok",
            "completion": "no marker and not closed",
            "completion_tokens": 10,
            "finish_reason": "stop",
        },
        {"status": "generation_failed"},
    ]
    stats = reduce.completion_stats_from_records(
        records, reasoning_start=None, reasoning_end="</think>", answer_marker="ANSWER:"
    )
    assert stats["documents"] == 3
    assert stats["completions"] == 2
    assert stats["reasoning_closed_rate"] == 0.5
    assert stats["answer_marker_rate"] == 0.5
    assert stats["format_rate"] == 0.5
    assert stats["mean_completion_tokens"] == 10


def test_completion_stats_reports_null_rates_with_no_completions():
    stats = reduce.completion_stats_from_records(
        [{"status": "generation_failed"}],
        reasoning_start=None,
        reasoning_end="</think>",
        answer_marker="ANSWER:",
    )
    assert stats["completions"] == 0
    assert stats["truncation_rate"] is None
    assert stats["mean_completion_tokens"] is None


# --- reduce_seed -----------------------------------------------------------


def _settings(tasks, seeds=(0,)):
    return {
        "tasks": list(tasks),
        "seeds": list(seeds),
        "reasoning_start": None,
        "reasoning_end": "</think>",
        "answer_marker": "ANSWER:",
    }


def test_reduce_seed_raises_naming_the_task_with_no_scored_documents(tmp_path):
    _write_jsonl(
        tmp_path / "seed-0" / "gsm8k-0.jsonl", [{"status": "generation_failed"}]
    )
    settings = _settings(["gsm8k|0"])
    configs = {"gsm8k|0": FakeConfig([FakeMetric({"acc": statistics.fmean})])}

    with pytest.raises(ValueError, match="gsm8k\\|0"):
        reduce.reduce_seed(settings, 0, configs, tmp_path)


def test_reduce_seed_allows_two_tasks_to_report_the_same_metric_name(tmp_path):
    # Unlike the old LightEval-results-key scheme, this is expected: each
    # task's metrics live under its own key in the returned mapping.
    _write_jsonl(
        tmp_path / "seed-0" / "task-a-0.jsonl",
        [{"status": "ok", "completion": "x", "scores": {"dup": 1.0}}],
    )
    _write_jsonl(
        tmp_path / "seed-0" / "task-b-0.jsonl",
        [{"status": "ok", "completion": "x", "scores": {"dup": 0.0}}],
    )
    settings = _settings(["task-a|0", "task-b|0"])
    metric = FakeMetric({"dup": statistics.fmean})
    configs = {"task-a|0": FakeConfig([metric]), "task-b|0": FakeConfig([metric])}

    metrics, _ = reduce.reduce_seed(settings, 0, configs, tmp_path)
    assert metrics == {"task-a|0": {"dup": 1.0}, "task-b|0": {"dup": 0.0}}


# --- build_summary_from_records: end to end -------------------------------


def test_build_summary_from_records_single_seed_has_null_std(tmp_path):
    _write_jsonl(
        tmp_path / "seed-0" / "gsm8k-0.jsonl",
        [
            {
                "status": "ok",
                "completion": "reasoning</think>ANSWER: 4",
                "completion_tokens": 10,
                "finish_reason": "stop",
                "scores": {"extractive_match": 1.0},
            },
            {
                "status": "ok",
                "completion": "reasoning</think>ANSWER: 5",
                "completion_tokens": 10,
                "finish_reason": "stop",
                "scores": {"extractive_match": 0.0},
            },
        ],
    )
    settings = {
        **_settings(["gsm8k|0"]),
        "tier": "tier1-core",
        "model_path": "models/x",
        "served_model_name": "x",
        "max_samples": None,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 16384,
        "server_image": None,
        "serve_command": ["scripts/run_vllm_tpu_container.sh"],
        "host": "127.0.0.1",
        "port": 8000,
    }
    configs = {
        "gsm8k|0": FakeConfig([FakeMetric({"extractive_match": statistics.fmean})])
    }

    summary = reduce.build_summary_from_records(settings, configs, tmp_path)

    assert summary["tasks_metrics"]["gsm8k|0"]["extractive_match"]["mean"] == 0.5
    assert summary["tasks_metrics"]["gsm8k|0"]["extractive_match"]["std"] is None
    assert summary["tasks_metrics"]["gsm8k|0"]["extractive_match"]["n"] == 1
    assert summary["truncation_rate_source"] == "finish_reason"
    assert summary["generation"]["truncation_rate"]["mean"] == 0.0
