"""`open_r1_tpu.tracing.ingest`, exercised entirely over `file://` fixtures --
no network, no Docker, no real Langfuse (tests/conftest.py stubs the SDK; a
`FakeLangfuseClient` stands in for the client everywhere below).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from open_r1_tpu.tracing import ingest

# --- fixtures: full, minimal, and malformed litellm log payloads -----------

FULL_PAYLOAD = {
    "id": "req-full",
    "model": "served-model",
    "messages": [{"role": "user", "content": "hi"}],
    "response": {"choices": [{"message": {"content": "hello there"}}]},
    "prompt_tokens": 3,
    "completion_tokens": 5,
    "total_tokens": 8,
    "startTime": 1000.0,
    "endTime": 1002.5,
}
MINIMAL_PAYLOAD = {
    "id": "req-minimal",
    "messages": [{"role": "user", "content": "x"}],
    "response": "just text",
}
MALFORMED_PAYLOAD = {"id": "req-malformed", "messages": None, "response": "nope"}

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
    def __init__(self):
        self.ended = False

    def end(self):
        self.ended = True


class FakeLangfuseClient:
    def __init__(self):
        self.calls = []  # list of (kwargs, FakeGeneration)

    def create_trace_id(self, *, seed):
        return f"trace-{seed}"

    def start_observation(self, **kwargs):
        generation = FakeGeneration()
        self.calls.append((kwargs, generation))
        return generation

    def flush(self):
        pass

    @property
    def observations(self):
        return [kwargs for kwargs, _ in self.calls]


def write_fixtures(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("full.json", FULL_PAYLOAD),
        ("minimal.json", MINIMAL_PAYLOAD),
        ("malformed.json", MALFORMED_PAYLOAD),
    ):
        (directory / name).write_text(json.dumps(payload), encoding="utf-8")


def config_for(tmp_path: Path) -> dict:
    config = copy.deepcopy(BASE_CONFIG)
    config["ingester"]["state_dir"] = str(tmp_path / "state")
    return config


# --- parse_payload: full, minimal, malformed --------------------------------


def test_parse_payload_reads_the_full_shape():
    record = ingest.parse_payload(FULL_PAYLOAD, object_name="full.json")
    assert record is not None
    assert record["response_text"] == "hello there"
    assert record["messages"] == [{"role": "user", "content": "hi"}]
    assert record["model"] == "served-model"
    assert record["prompt_tokens"] == 3
    assert record["completion_tokens"] == 5


def test_parse_payload_reads_the_minimal_shape():
    record = ingest.parse_payload(MINIMAL_PAYLOAD, object_name="minimal.json")
    assert record is not None
    assert record["response_text"] == "just text"
    assert record["model"] is None


def test_parse_payload_rejects_the_malformed_shape():
    assert ingest.parse_payload(MALFORMED_PAYLOAD, object_name="malformed.json") is None


# --- ingest_once: the right trace calls, idempotency, determinism ----------


def test_ingest_once_creates_a_trace_per_usable_object(tmp_path):
    fixtures = tmp_path / "fixtures"
    write_fixtures(fixtures)
    config = config_for(tmp_path)
    client = FakeLangfuseClient()

    counts = ingest.ingest_once(config, prefix=f"file://{fixtures}", client=client)

    assert counts == {"ingested": 2, "skipped": 0, "failed": 1}
    assert len(client.calls) == 2
    assert all(generation.ended for _, generation in client.calls)
    outputs = {obs["output"] for obs in client.observations}
    assert outputs == {"hello there", "just text"}


def test_ingest_once_skips_processed_objects_across_restarts(tmp_path):
    fixtures = tmp_path / "fixtures"
    write_fixtures(fixtures)
    config = config_for(tmp_path)

    first_counts = ingest.ingest_once(
        config, prefix=f"file://{fixtures}", client=FakeLangfuseClient()
    )
    # A fresh client stands in for a restarted ingester process; only the
    # on-disk state under ingester.state_dir should decide what is re-done.
    second_client = FakeLangfuseClient()
    second_counts = ingest.ingest_once(
        config, prefix=f"file://{fixtures}", client=second_client
    )

    assert first_counts == {"ingested": 2, "skipped": 0, "failed": 1}
    assert second_counts == {"ingested": 0, "skipped": 3, "failed": 0}
    assert second_client.calls == []


def test_once_over_the_same_prefix_twice_yields_zero_new_ingestions(tmp_path):
    fixtures = tmp_path / "fixtures"
    write_fixtures(fixtures)
    config = config_for(tmp_path)
    client = FakeLangfuseClient()

    ingest.ingest_once(config, prefix=f"file://{fixtures}", client=client)
    before = len(client.calls)
    ingest.ingest_once(config, prefix=f"file://{fixtures}", client=client)

    assert len(client.calls) == before


def test_trace_ids_are_deterministic_across_separate_writes():
    client = FakeLangfuseClient()
    record = ingest.parse_payload(FULL_PAYLOAD, object_name="full.json")
    assert record is not None

    first = ingest.write_trace(client, record, run_prefix="file:///x")
    second = ingest.write_trace(client, record, run_prefix="file:///x")

    assert first == second


def test_trace_ids_differ_for_different_request_ids():
    client = FakeLangfuseClient()
    record_a = ingest.parse_payload(FULL_PAYLOAD, object_name="a.json")
    record_b = ingest.parse_payload(MINIMAL_PAYLOAD, object_name="b.json")
    assert record_a is not None
    assert record_b is not None

    a = ingest.write_trace(client, record_a, run_prefix="file:///x")
    b = ingest.write_trace(client, record_b, run_prefix="file:///x")
    assert a != b


def test_the_index_is_keyed_by_content_hash_after_a_pass(tmp_path):
    fixtures = tmp_path / "fixtures"
    write_fixtures(fixtures)
    config = config_for(tmp_path)
    client = FakeLangfuseClient()

    ingest.ingest_once(config, prefix=f"file://{fixtures}", client=client)

    state_path = ingest._state_path(
        config["ingester"]["state_dir"], f"file://{fixtures}"
    )
    state = ingest.load_state(state_path)
    from open_r1_tpu.tracing.hashing import content_sha256

    assert content_sha256("hello there") in state["content_sha256_to_trace_id"]
    assert content_sha256("just text") in state["content_sha256_to_trace_id"]


# --- prefix_url --------------------------------------------------------------


def test_prefix_url_defaults_to_the_templates_static_root():
    config = {"gcs": {"bucket": "b", "prefix_template": "traces/{recipe}/{timestamp}"}}
    assert ingest.prefix_url(config, None) == "gs://b/traces"


def test_prefix_url_accepts_a_full_url_override():
    config = {"gcs": {"bucket": "b", "prefix_template": "traces/{recipe}/{timestamp}"}}
    assert ingest.prefix_url(config, "file:///tmp/x") == "file:///tmp/x"


def test_prefix_url_accepts_a_bare_sub_prefix():
    config = {"gcs": {"bucket": "b", "prefix_template": "traces/{recipe}/{timestamp}"}}
    assert ingest.prefix_url(config, "traces/tier0/x") == "gs://b/traces/tier0/x"


# --- crash safety and hash collisions ---------------------------------------


def test_a_failed_trace_write_leaves_the_object_unprocessed(tmp_path):
    """If Langfuse errors mid-backlog, the object whose write failed must not
    be recorded as processed -- a restarted ingester retries it rather than
    permanently skipping a trace that was never written.
    """
    fixtures = tmp_path / "fixtures"
    write_fixtures(fixtures)
    config = config_for(tmp_path)

    class RaisingClient(FakeLangfuseClient):
        def start_observation(self, **kwargs):
            raise RuntimeError("langfuse down")

    try:
        ingest.ingest_once(config, prefix=f"file://{fixtures}", client=RaisingClient())
    except RuntimeError:
        pass
    else:  # pragma: no cover - the write must have been attempted
        raise AssertionError("expected the client failure to propagate")

    recovery_client = FakeLangfuseClient()
    counts = ingest.ingest_once(
        config, prefix=f"file://{fixtures}", client=recovery_client
    )

    # Both usable objects are still ingested after the crash; only the
    # malformed one was (correctly) marked processed by the failed pass.
    assert counts["ingested"] == 2
    assert len(recovery_client.calls) == 2


def test_a_duplicate_completion_hash_warns_and_keeps_the_last_trace(tmp_path, caplog):
    import logging

    fixtures = tmp_path / "fixtures"
    fixtures.mkdir(parents=True)
    for name, request_id in (("a.json", "req-a"), ("b.json", "req-b")):
        (fixtures / name).write_text(
            json.dumps(
                {
                    "id": request_id,
                    "messages": [{"role": "user", "content": "q"}],
                    "response": "the identical completion",
                }
            ),
            encoding="utf-8",
        )
    config = config_for(tmp_path)

    with caplog.at_level(logging.WARNING, logger="open_r1_tpu.tracing.ingest"):
        counts = ingest.ingest_once(
            config, prefix=f"file://{fixtures}", client=FakeLangfuseClient()
        )

    assert counts["ingested"] == 2
    assert any("already indexed" in record.message for record in caplog.records)
