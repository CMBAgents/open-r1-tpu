"""`docker/trace-proxy/gcs_logger.py`'s write path, exercised for real.

`STORAGE_EMULATOR_HOST` (the standard Google emulator convention the module
honours) points the upload at a stdlib HTTP server here, so the exact bytes
and URL the proxy container would send to GCS are asserted locally with no
network, no credentials, and no Docker. litellm itself is not a local
dependency; its `CustomLogger` base class is stubbed the same way
tests/conftest.py stubs `langfuse`.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import sys
import threading
import types
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

REPO_ROOT = Path(__file__).parents[1]
LOGGER_PATH = REPO_ROOT / "docker" / "trace-proxy" / "gcs_logger.py"


def _stub_litellm_if_absent() -> None:
    try:
        import litellm.integrations.custom_logger
    except ImportError:
        pass
    else:
        return

    litellm = types.ModuleType("litellm")
    integrations = types.ModuleType("litellm.integrations")
    custom_logger = types.ModuleType("litellm.integrations.custom_logger")

    class CustomLogger:
        pass

    setattr(custom_logger, "CustomLogger", CustomLogger)  # noqa: B010 - ModuleType has no such attribute to set directly
    sys.modules["litellm"] = litellm
    sys.modules["litellm.integrations"] = integrations
    sys.modules["litellm.integrations.custom_logger"] = custom_logger


def _load_gcs_logger():
    _stub_litellm_if_absent()
    spec = importlib.util.spec_from_file_location("gcs_logger", LOGGER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeGCSHandler(BaseHTTPRequestHandler):
    """Records every upload the logger sends; responds like GCS's JSON API."""

    uploads: ClassVar[list[tuple[str, bytes]]] = []
    fail_with: ClassVar[int | None] = None

    def log_message(self, format, *args):
        pass

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        if self.fail_with is not None:
            self.send_response(self.fail_with)
            self.end_headers()
            return
        type(self).uploads.append((self.path, body))
        reply = json.dumps({"name": "recorded"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(reply)))
        self.end_headers()
        self.wfile.write(reply)


@pytest.fixture
def fake_gcs(monkeypatch):
    _FakeGCSHandler.uploads = []
    _FakeGCSHandler.fail_with = None
    server = HTTPServer(("127.0.0.1", 0), _FakeGCSHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv(
        "STORAGE_EMULATOR_HOST", f"http://127.0.0.1:{server.server_port}"
    )
    monkeypatch.setenv("TRACE_GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("TRACE_GCS_PREFIX", "traces/tier0/20260101T000000Z")
    yield _FakeGCSHandler
    server.shutdown()


PAYLOAD = {
    "id": "req-123",
    "model": "test-model",
    "messages": [{"role": "user", "content": "hello"}],
    "response": {"choices": [{"message": {"content": "hi"}}]},
    "prompt_tokens": 3,
    "completion_tokens": 1,
}


def _log(module, payload):
    handler = module.proxy_handler_instance
    asyncio.run(
        handler.async_log_success_event(
            {"standard_logging_object": payload}, None, None, None
        )
    )


def test_it_uploads_the_payload_under_the_prefix_named_by_the_request_id(fake_gcs):
    module = _load_gcs_logger()
    _log(module, PAYLOAD)

    assert len(fake_gcs.uploads) == 1
    path, body = fake_gcs.uploads[0]
    assert path.startswith("/upload/storage/v1/b/test-bucket/o?")
    name = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)["name"][0]
    assert name == "traces/tier0/20260101T000000Z/req-123.json"
    assert json.loads(body) == PAYLOAD


def test_a_payload_without_a_logging_object_is_skipped_without_a_request(fake_gcs):
    module = _load_gcs_logger()
    handler = module.proxy_handler_instance
    asyncio.run(handler.async_log_success_event({}, None, None, None))
    assert fake_gcs.uploads == []


def test_missing_bucket_or_prefix_drops_the_payload_with_a_warning(
    fake_gcs, monkeypatch, caplog
):
    module = _load_gcs_logger()
    monkeypatch.delenv("TRACE_GCS_BUCKET")
    with caplog.at_level(logging.WARNING, logger="gcs_logger"):
        _log(module, PAYLOAD)
    assert fake_gcs.uploads == []
    assert any("TRACE_GCS_BUCKET" in record.message for record in caplog.records)


def test_an_unreachable_endpoint_warns_and_never_raises(monkeypatch, caplog):
    module = _load_gcs_logger()
    monkeypatch.setattr(module, "_UPLOAD_ATTEMPTS", 1)
    # A closed port: connection refused on every attempt.
    monkeypatch.setenv("STORAGE_EMULATOR_HOST", "http://127.0.0.1:9")
    monkeypatch.setenv("TRACE_GCS_BUCKET", "test-bucket")
    monkeypatch.setenv("TRACE_GCS_PREFIX", "traces/x")
    with caplog.at_level(logging.WARNING, logger="gcs_logger"):
        _log(module, PAYLOAD)
    assert any("Dropping trace" in record.message for record in caplog.records)


def test_a_client_error_is_not_retried(fake_gcs, caplog):
    module = _load_gcs_logger()
    fake_gcs.fail_with = 403
    with caplog.at_level(logging.WARNING, logger="gcs_logger"):
        _log(module, PAYLOAD)
    assert fake_gcs.uploads == []
    rejected = [r for r in caplog.records if "rejected" in r.message]
    assert len(rejected) == 1
