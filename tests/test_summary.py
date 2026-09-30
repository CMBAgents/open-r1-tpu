"""Tests for `open_r1_tpu.evaluation.summary`: fixture JSONL in, the summary
out. `FakeMetric` stands in for a LightEval metric, exposing only
`get_corpus_aggregations()`.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from open_r1_tpu.evaluation import summary as eval_summary
from open_r1_tpu.evaluation.config import load_eval_config, resolve_settings
from open_r1_tpu.evaluation.stack import VLLM_TPU_BASE_IMAGE, vllm_tpu_image_tag


class FakeMetric:
    def __init__(self, aggregations):
        self._aggregations = aggregations

    def get_corpus_aggregations(self):
        return self._aggregations


class FakeConfig:
    def __init__(self, metrics):
        self.metrics = list(metrics)


TIER1 = Path(__file__).parents[1] / "recipes/Qwen2.5-Math-1.5B/eval/tier1_core.yaml"


def _settings(tasks=("t",), seeds=(0,), **overrides):
    settings = {
        "tasks": list(tasks),
        "seeds": list(seeds),
        "reasoning_start": None,
        "reasoning_end": "</think>",
        "answer_marker": "ANSWER:",
        "tier": "tier1-core",
        "model_path": "models/x",
        "served_model_name": "x",
        "max_samples": None,
        "temperature": 0.6,
        "top_p": 0.95,
        "max_new_tokens": 16384,
        "serve_command": ["scripts/run_vllm_tpu_container.sh"],
        "server_image": vllm_tpu_image_tag(),
        "host": "127.0.0.1",
        "port": 8000,
    }
    settings.update(overrides)
    return settings


# --- read_jsonl ---------------------------------------------------------


def test_read_jsonl_missing_file_is_a_named_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="No runner output"):
        eval_summary.read_jsonl(tmp_path / "nope.jsonl")


def test_read_jsonl_skips_blank_lines(tmp_path):
    path = tmp_path / "records.jsonl"
    path.write_text('{"a": 1}\n\n{"a": 2}\n', encoding="utf-8")
    assert eval_summary.read_jsonl(path) == [{"a": 1}, {"a": 2}]


# --- reduce_task_metrics --------------------------------------------------


def test_reduce_task_metrics_uses_the_metrics_own_corpus_fn():
    metric = FakeMetric({"acc": statistics.fmean})
    records = [
        {"status": "ok", "scores": {"acc": 1.0}},
        {"status": "ok", "scores": {"acc": 0.0}},
    ]
    assert eval_summary.reduce_task_metrics(records, [metric]) == {"acc": 0.5}


def test_reduce_task_metrics_skips_none_missing_and_failed_documents():
    metric = FakeMetric({"acc": statistics.fmean})
    records = [
        {"status": "ok", "scores": {"acc": 1.0}},
        {"status": "ok", "scores": {"acc": None}},
        {"status": "ok", "scores": {}},
        {"status": "generation_failed"},
    ]
    # Only the first record has a usable value; a mean of one value is itself.
    assert eval_summary.reduce_task_metrics(records, [metric]) == {"acc": 1.0}


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
    assert eval_summary.reduce_task_metrics(records, [metric]) == pytest.approx(
        {"inst_level_strict_acc": 2 / 3}
    )


def test_reduce_task_metrics_returns_nothing_when_no_document_has_a_value():
    metric = FakeMetric({"acc": statistics.fmean})
    records = [{"status": "generation_failed"}]
    assert eval_summary.reduce_task_metrics(records, [metric]) == {}


# --- completion_stats_from_records ---------------------------------------


def test_truncation_rate_comes_from_finish_reason_not_token_count():
    records = [
        # A token count near the budget does not mean truncation;
        # finish_reason says the model stopped on its own.
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
    stats = eval_summary.completion_stats_from_records(
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
    stats = eval_summary.completion_stats_from_records(
        records, reasoning_start=None, reasoning_end="</think>", answer_marker="ANSWER:"
    )
    assert stats["documents"] == 3
    assert stats["completions"] == 2
    assert stats["reasoning_closed_rate"] == 0.5
    assert stats["answer_marker_rate"] == 0.5
    assert stats["format_rate"] == 0.5
    assert stats["mean_completion_tokens"] == 10


def test_completion_stats_reports_null_rates_with_no_completions():
    stats = eval_summary.completion_stats_from_records(
        [{"status": "generation_failed"}],
        reasoning_start=None,
        reasoning_end="</think>",
        answer_marker="ANSWER:",
    )
    assert stats["completions"] == 0
    assert stats["truncation_rate"] is None
    assert stats["mean_completion_tokens"] is None


# --- reduce_seed -----------------------------------------------------------


def test_reduce_seed_raises_naming_the_task_with_no_scored_documents(tmp_path):
    eval_summary.write_jsonl(
        tmp_path / "seed-0" / "gsm8k-0.jsonl", [{"status": "generation_failed"}]
    )
    settings = _settings(["gsm8k|0"])
    configs = {"gsm8k|0": FakeConfig([FakeMetric({"acc": statistics.fmean})])}

    with pytest.raises(ValueError, match="gsm8k\\|0"):
        eval_summary.reduce_seed(settings, 0, configs, tmp_path)


def test_reduce_seed_allows_two_tasks_to_report_the_same_metric_name(tmp_path):
    # Each task's metrics live under its own key.
    eval_summary.write_jsonl(
        tmp_path / "seed-0" / "task-a-0.jsonl",
        [{"status": "ok", "completion": "x", "scores": {"dup": 1.0}}],
    )
    eval_summary.write_jsonl(
        tmp_path / "seed-0" / "task-b-0.jsonl",
        [{"status": "ok", "completion": "x", "scores": {"dup": 0.0}}],
    )
    settings = _settings(["task-a|0", "task-b|0"])
    metric = FakeMetric({"dup": statistics.fmean})
    configs = {"task-a|0": FakeConfig([metric]), "task-b|0": FakeConfig([metric])}

    metrics, _ = eval_summary.reduce_seed(settings, 0, configs, tmp_path)
    assert metrics == {"task-a|0": {"dup": 1.0}, "task-b|0": {"dup": 0.0}}


# --- build_summary_from_records: end to end -------------------------------


def test_build_summary_from_records_single_seed_has_null_std(tmp_path):
    eval_summary.write_jsonl(
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
    settings = _settings(["gsm8k|0"], server_image=None)
    configs = {
        "gsm8k|0": FakeConfig([FakeMetric({"extractive_match": statistics.fmean})])
    }

    summary = eval_summary.build_summary_from_records(settings, configs, tmp_path)

    assert summary["tasks_metrics"]["gsm8k|0"]["extractive_match"]["mean"] == 0.5
    assert summary["tasks_metrics"]["gsm8k|0"]["extractive_match"]["std"] is None
    assert summary["tasks_metrics"]["gsm8k|0"]["extractive_match"]["n"] == 1
    assert summary["truncation_rate_source"] == "finish_reason"
    assert summary["generation"]["truncation_rate"]["mean"] == 0.0


# --- build_summary and its outputs -------------------------------------------


def test_aggregate_reports_mean_and_spread_across_seeds():
    aggregated = eval_summary.aggregate_across_seeds(
        {
            0: {"t": {"acc": 0.40}},
            1: {"t": {"acc": 0.50}},
            2: {"t": {"acc": 0.60}},
        }
    )

    assert aggregated["t"]["acc"]["mean"] == pytest.approx(0.50)
    assert aggregated["t"]["acc"]["std"] == pytest.approx(0.10)
    assert aggregated["t"]["acc"]["n"] == 3


def test_a_single_seed_reports_no_spread_rather_than_zero_spread():
    aggregated = eval_summary.aggregate_across_seeds({0: {"t": {"acc": 0.4}}})

    assert aggregated["t"]["acc"]["std"] is None
    assert aggregated["t"]["acc"]["n"] == 1


def test_build_summary_records_the_stack_and_the_sampling_parameters():
    service_versions = {"vllm-tpu": "0.27.0", "tpu-inference": "0.27.0"}
    # The resolver's own output, so a renamed settings key fails here.
    settings = resolve_settings(load_eval_config(TIER1))

    summary = eval_summary.build_summary(
        settings,
        {0: {"t": {"acc": 0.4}}},
        {0: {"format_rate": 1.0, "truncation_rate": None}},
        {"image_id": "sha256:local-image", "service_versions": service_versions},
    )

    assert summary["sampling"]["temperature"] == settings["temperature"]
    # Replicates are unseeded on this backend, so an archived summary listing
    # `seeds` must not be read as reproducible sample by sample.
    assert summary["seeded_replicates"] is False
    assert set(summary["stack"]) >= {
        "python",
        "lighteval",
        "openai",
        "latex2sympy2-extended",
    }
    # vLLM runs outside this environment, so its image and complete command are
    # recorded rather than a package version.
    assert summary["serve_command"] == ["scripts/run_vllm_tpu_container.sh"]
    assert summary["server_image"] == vllm_tpu_image_tag()
    assert summary["server_command"][:3] == [
        "scripts/run_vllm_tpu_container.sh",
        "--image",
        vllm_tpu_image_tag(),
    ]
    assert summary["server_image_provenance"] == {
        "spec_tag": vllm_tpu_image_tag(),
        "image_id": "sha256:local-image",
        "base_image": VLLM_TPU_BASE_IMAGE,
        "service_versions": service_versions,
    }
    assert summary["tasks_metrics"]["t"]["acc"]["mean"] == pytest.approx(0.4)
    assert summary["generation"]["format_rate"]["mean"] == pytest.approx(1.0)
    # Absent in every seed, so it stays absent rather than becoming 0.0.
    assert summary["generation"]["truncation_rate"]["mean"] is None


def test_summary_rows_flatten_one_row_per_metric():
    summary = {
        "tier": "t1",
        "tasks_metrics": {"task": {"acc": {"mean": 0.5, "std": 0.1, "n": 3}}},
    }

    assert eval_summary.summary_rows(summary) == [["t1", "task", "acc", 0.5, 0.1, 3]]


def test_write_summary_creates_the_parent_directory(tmp_path):
    path = tmp_path / "nested" / "summary.json"

    eval_summary.write_summary(str(path), {"tier": "t", "n": 1})

    assert json.loads(path.read_text(encoding="utf-8")) == {"tier": "t", "n": 1}
