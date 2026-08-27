"""`open_r1_tpu.tracing.scores`, exercised over a fixture LightEval detail
parquet plus a fixture ingester state index -- no network, no Docker, no real
Langfuse (tests/conftest.py stubs the SDK; a `FakeLangfuseClient` stands in
for the client everywhere below).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from open_r1_tpu.tracing import hashing, ingest, scores

BASE_CONFIG = {
    "gcs": {"bucket": "unused", "prefix_template": "traces/{recipe}/{timestamp}"},
    "proxy": {
        "port": 4000,
        "upstream_base_url": "http://127.0.0.1:8000/v1",
        "image": "img@sha256:" + "a" * 64,
    },
    "langfuse": {"host": "127.0.0.1", "port": 3000},
    "ingester": {"poll_secs": 30, "state_dir": ""},
}


class FakeGeneration:
    def end(self):
        pass


class FakeLangfuseClient:
    def __init__(self):
        self.score_calls = []
        self.tag_calls = []
        self.observations = []
        self._trace_ids = 0

    def create_trace_id(self, *, seed):
        return f"trace-{seed}"

    def start_observation(self, **kwargs):
        self.observations.append(kwargs)
        return FakeGeneration()

    def create_score(self, **kwargs):
        self.score_calls.append(kwargs)

    def _create_trace_tags_via_ingestion(self, *, trace_id, tags):
        self.tag_calls.append((trace_id, tags))

    def flush(self):
        pass


def config_for(tmp_path: Path) -> dict:
    config = copy.deepcopy(BASE_CONFIG)
    config["ingester"]["state_dir"] = str(tmp_path / "state")
    return config


def write_tier_output(
    tmp_path: Path, *, matched_text: str, unmatched_text: str
) -> tuple[Path, Path]:
    """A minimal but real tier output directory: one seed, one task, one
    detail parquet shard with two rows and an extra per-document numeric
    metric column, plus the summary JSON the score pass reads its
    seeds/tasks/sampling settings from.
    """
    output_dir = tmp_path / "out"
    task_dir = output_dir / "seed-0" / "gsm8k-0" / "details" / "model" / "2026-01-01"
    task_dir.mkdir(parents=True)

    table = pa.table(
        {
            "model_response": [
                {"text": [matched_text], "output_tokens": [50]},
                {"text": [unmatched_text], "output_tokens": [120]},
            ],
            "doc": [{"gold": "4"}, {"gold": "5"}],
            "extractive_match": [1.0, 0.0],
        }
    )
    pq.write_table(table, task_dir / "details_gsm8k_2026-01-01.parquet")

    summary = {
        "tier": "tier0-smoke",
        "served_model_name": "model",
        "seeds": [0],
        "tasks": ["gsm8k|0"],
        "sampling": {"max_new_tokens": 100},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return output_dir, summary_path


def write_index(tmp_path: Path, *, content: str, trace_id: str) -> None:
    state_dir = Path(config_for(tmp_path)["ingester"]["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    content_hash = hashing.content_sha256(content)
    (state_dir / "run1.json").write_text(
        json.dumps(
            {
                "processed": ["x.json"],
                "content_sha256_to_trace_id": {content_hash: trace_id},
            }
        ),
        encoding="utf-8",
    )


# --- matched / created / failed ---------------------------------------------


def test_a_matched_row_posts_scores_without_creating_a_new_trace(tmp_path):
    output_dir, summary_path = write_tier_output(
        tmp_path,
        matched_text="matched completion",
        unmatched_text="brand new completion",
    )
    write_index(tmp_path, content="matched completion", trace_id="trace-matched-1")
    config = config_for(tmp_path)
    client = FakeLangfuseClient()

    counts = scores.score_pass(
        config,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        client=client,
    )

    assert counts == {"matched": 1, "created": 1, "failed": 0}
    matched_scores = [
        c for c in client.score_calls if c["trace_id"] == "trace-matched-1"
    ]
    assert {c["name"]: c["value"] for c in matched_scores} == {
        "extractive_match": 1.0,
        "completion_tokens": 50.0,
        "truncated": 0.0,
    }
    assert all(c["metadata"]["source"] == "proxy" for c in matched_scores)
    # No new trace was created for the matched row -- only the unmatched one.
    assert len(client.observations) == 1


def test_an_unmatched_row_creates_a_trace_tagged_source_parquet(tmp_path):
    output_dir, summary_path = write_tier_output(
        tmp_path,
        matched_text="matched completion",
        unmatched_text="brand new completion",
    )
    write_index(tmp_path, content="matched completion", trace_id="trace-matched-1")
    config = config_for(tmp_path)
    client = FakeLangfuseClient()

    counts = scores.score_pass(
        config,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        client=client,
    )

    assert counts["created"] == 1
    [created_observation] = client.observations
    assert created_observation["output"] == "brand new completion"
    assert created_observation["metadata"]["source"] == "parquet"
    created_scores = [
        c
        for c in client.score_calls
        if c["trace_id"] == created_observation["trace_context"]["trace_id"]
    ]
    assert {c["name"]: c["value"] for c in created_scores} == {
        "extractive_match": 0.0,
        "completion_tokens": 120.0,
        "truncated": 1.0,
    }
    assert all(c["metadata"]["source"] == "parquet" for c in created_scores)


def test_it_tags_the_trace_with_tier_seed_task(tmp_path):
    output_dir, summary_path = write_tier_output(
        tmp_path,
        matched_text="matched completion",
        unmatched_text="brand new completion",
    )
    write_index(tmp_path, content="matched completion", trace_id="trace-matched-1")
    config = config_for(tmp_path)
    client = FakeLangfuseClient()

    scores.score_pass(
        config,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        client=client,
    )

    _trace_id, tags = next(
        call for call in client.tag_calls if call[0] == "trace-matched-1"
    )
    assert set(tags) == {"tier:tier0-smoke", "seed:0", "task:gsm8k|0", "source:proxy"}


def test_a_row_lighteval_cannot_read_counts_as_failed(tmp_path):
    output_dir, summary_path = write_tier_output(
        tmp_path,
        matched_text="matched completion",
        unmatched_text="brand new completion",
    )
    # Overwrite the shard with one whose response column has no readable text
    # under any known key, so extract_completions raises for both rows.
    task_dir = output_dir / "seed-0" / "gsm8k-0" / "details" / "model" / "2026-01-01"
    table = pa.table(
        {
            "model_response": [{"surprise": "x"}, {"surprise": "y"}],
            "doc": [{"gold": "4"}, {"gold": "5"}],
        }
    )
    pq.write_table(table, task_dir / "details_gsm8k_2026-01-01.parquet")
    config = config_for(tmp_path)
    client = FakeLangfuseClient()

    counts = scores.score_pass(
        config,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        client=client,
    )

    assert counts == {"matched": 0, "created": 0, "failed": 2}


def test_cli_exits_non_zero_on_any_failure(tmp_path, monkeypatch):
    output_dir, summary_path = write_tier_output(
        tmp_path,
        matched_text="matched completion",
        unmatched_text="brand new completion",
    )
    task_dir = output_dir / "seed-0" / "gsm8k-0" / "details" / "model" / "2026-01-01"
    table = pa.table({"model_response": [{"surprise": "x"}], "doc": [{"gold": "4"}]})
    pq.write_table(table, task_dir / "details_gsm8k_2026-01-01.parquet")

    config = config_for(tmp_path)
    monkeypatch.setattr(scores, "_langfuse_client", lambda cfg: FakeLangfuseClient())
    monkeypatch.setattr(scores, "load_tracing_config", lambda path: config)
    monkeypatch.setattr(
        "sys.argv",
        [
            "scores",
            "--config",
            "unused",
            "--output-dir",
            str(output_dir),
            "--summary",
            str(summary_path),
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        scores.main()
    assert excinfo.value.code == 1


# --- shared hash function ----------------------------------------------------


def test_ingest_and_scores_share_one_hash_implementation():
    assert ingest.content_sha256 is hashing.content_sha256
    assert scores.content_sha256 is hashing.content_sha256


def test_content_hash_is_stable_for_the_same_text():
    assert hashing.content_sha256("same text") == hashing.content_sha256("same text")


def test_content_hash_differs_for_different_text():
    assert hashing.content_sha256("a") != hashing.content_sha256("b")


# --- re-run idempotency and hash collisions ---------------------------------


def test_a_rerun_posts_no_new_generations_and_upserts_the_same_score_ids(tmp_path):
    """Re-running over the same output directory (the natural recovery move
    after a partial failure) must not append duplicate generations to
    parquet-created traces, and must re-post every score under the same
    deterministic id so Langfuse upserts instead of accumulating copies.
    """
    output_dir, summary_path = write_tier_output(
        tmp_path,
        matched_text="matched completion",
        unmatched_text="brand new completion",
    )
    write_index(tmp_path, content="matched completion", trace_id="trace-matched-1")
    config = config_for(tmp_path)

    first_client = FakeLangfuseClient()
    first_counts = scores.score_pass(
        config,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        client=first_client,
    )
    second_client = FakeLangfuseClient()
    second_counts = scores.score_pass(
        config,
        output_dir=str(output_dir),
        summary_path=str(summary_path),
        client=second_client,
    )

    assert first_counts == {"matched": 1, "created": 1, "failed": 0}
    assert second_counts == {"matched": 2, "created": 0, "failed": 0}
    assert second_client.observations == []
    assert all("score_id" in c for c in first_client.score_calls)
    first_ids = {c["score_id"] for c in first_client.score_calls}
    second_ids = {c["score_id"] for c in second_client.score_calls}
    assert first_ids == second_ids
    # The re-run still knows which traces were parquet-created, not proxy-matched.
    assert any(c["metadata"]["source"] == "parquet" for c in second_client.score_calls)


def test_identical_completions_across_rows_warn_about_the_shared_trace(
    tmp_path, caplog
):
    import logging

    output_dir, summary_path = write_tier_output(
        tmp_path,
        matched_text="the identical completion",
        unmatched_text="the identical completion",
    )
    config = config_for(tmp_path)
    client = FakeLangfuseClient()

    with caplog.at_level(logging.WARNING, logger="open_r1_tpu.tracing.scores"):
        counts = scores.score_pass(
            config,
            output_dir=str(output_dir),
            summary_path=str(summary_path),
            client=client,
        )

    # Row one creates the trace; row two collides onto it (warned), so only
    # one generation exists and no row is lost.
    assert counts == {"matched": 1, "created": 1, "failed": 0}
    assert len(client.observations) == 1
    assert any("collide" in record.message for record in caplog.records)
