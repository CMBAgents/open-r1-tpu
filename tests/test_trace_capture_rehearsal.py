"""End-to-end local rehearsal of the trace-capture pipeline: a stub
OpenAI-compatible server standing in for vLLM, behind the real trace-capture
litellm proxy container (`scripts/run_trace_proxy.sh` +
`docker/trace-proxy/`), writing through the real `gcs_logger.py` custom
callback to a real (disposable) GCS prefix, read back by the real
`open_r1_tpu.tracing.ingest.ingest_once` against a faked Langfuse client.

Needs Docker (to run the proxy container -- a real image pull the first
time) and a real GCS bucket with application-default credentials able to
write and delete objects under it, since `gcs_logger.py` always writes to
real GCS here, exactly as it does in production (see its module docstring,
and `open_r1_tpu.tracing.ingest`'s test suite for the `file://`-only
coverage that needs neither; `tests/test_trace_proxy_logger.py` covers the
logger's write path itself against a local emulator endpoint). Point
TRACE_REHEARSAL_GCS_BUCKET at a bucket you can freely write disposable
objects to and delete afterwards; skipped without it, and skipped without
Docker.

Two environment notes:
- The proxy container runs with --network host (that is how it reaches the
  stub on the host's loopback, and vLLM in production). On Docker Desktop
  for Mac this needs host networking enabled in settings (4.34+); on Linux
  it just works.
- Inside the container, GCS auth is ambient ADC. On a GCE VM the metadata
  server provides it; on a workstation, export GOOGLE_APPLICATION_CREDENTIALS
  (a service-account key or `gcloud auth application-default login`'s JSON)
  so `scripts/run_trace_proxy.sh` mounts it into the container.

The last step -- confirming a scored trace actually renders in the Langfuse
UI -- is not automated here even when this test runs: it needs a live
Langfuse instance and a human to look at a web page. Do that manually per
docker/langfuse/README.md's "Viewing the UI" section after running the
ingester (or this rehearsal) against a real Langfuse; a faked client is
exactly as good as a real one for proving the wire path works end-to-end,
which is what this test actually checks.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest
import yaml

from open_r1_tpu.tracing import ingest

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).parents[1]
REHEARSAL_BUCKET = os.environ.get("TRACE_REHEARSAL_GCS_BUCKET")


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = [
    pytestmark,
    pytest.mark.skipif(not _docker_available(), reason="Docker is not available"),
    pytest.mark.skipif(
        not REHEARSAL_BUCKET,
        reason="set TRACE_REHEARSAL_GCS_BUCKET to a scratch bucket to run this",
    ),
]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _StubVLLMHandler(BaseHTTPRequestHandler):
    """The minimum an OpenAI-compatible chat endpoint needs to answer, so
    the real litellm proxy in front of it can build a valid
    StandardLoggingPayload from the exchange.
    """

    def log_message(self, format, *args):  # matches BaseHTTPRequestHandler's signature
        pass  # quiet; failures still show up as a failed assertion below

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)  # request body is unused by the stub
        body = json.dumps(
            {
                "id": "cmpl-stub-1",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "stub-model",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "stub completion text",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 4,
                    "total_tokens": 9,
                },
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        body = json.dumps({"data": [{"id": "stub-model"}]}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_stub_server_through_proxy_to_gcs_to_ingester(tmp_path):
    import urllib.request

    stub_port = _free_port()
    proxy_port = _free_port()
    stub_server = HTTPServer(("127.0.0.1", stub_port), _StubVLLMHandler)
    stub_thread = threading.Thread(target=stub_server.serve_forever, daemon=True)
    stub_thread.start()

    example = yaml.safe_load(
        (REPO_ROOT / "configs" / "tracing.example.yaml").read_text()
    )
    config = {
        **example,
        "gcs": {
            "bucket": REHEARSAL_BUCKET,
            "prefix_template": "rehearsal/{recipe}/{timestamp}",
        },
        "proxy": {
            **example["proxy"],
            "port": proxy_port,
            "upstream_base_url": f"http://127.0.0.1:{stub_port}/v1",
        },
        "ingester": {"poll_secs": 30, "state_dir": str(tmp_path / "state")},
    }
    tracing_config_path = tmp_path / "tracing.yaml"
    tracing_config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    recipe, timestamp = "rehearsal", f"pytest-{os.getpid()}"
    launch = subprocess.run(
        [
            str(REPO_ROOT / "scripts" / "run_trace_proxy.sh"),
            "--config",
            str(tracing_config_path),
            "--recipe",
            recipe,
            "--timestamp",
            timestamp,
        ],
        capture_output=True,
        text=True,
    )
    try:
        assert launch.returncode == 0, launch.stderr

        # litellm needs a moment to start; poll rather than sleeping a fixed
        # duration for it.
        deadline = time.monotonic() + 60
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{proxy_port}/v1/chat/completions",
                    data=json.dumps(
                        {
                            # The bare served name, exactly what production
                            # sends: the harness's litellm client consumes
                            # the `hosted_vllm/` provider prefix client-side
                            # and puts only the served model name on the
                            # wire. Sending the prefixed form here would
                            # double-prefix through the proxy's wildcard
                            # route and exercise a path production never
                            # takes.
                            "model": "stub-model",
                            "messages": [{"role": "user", "content": "hello"}],
                        }
                    ).encode("utf-8"),
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": "Bearer local",
                    },
                )
                with urllib.request.urlopen(request, timeout=10) as response:
                    assert response.status == 200
                break
            except Exception as error:  # noqa: BLE001 - retried until the deadline
                last_error = error
                time.sleep(2)
        else:
            pytest.fail(f"trace proxy never answered: {last_error}")

        # The callback's GCS write is async (see gcs_logger.py's module
        # docstring); give it a moment to land before the ingester looks.
        time.sleep(5)

        client = _FakeLangfuseClient()
        counts = ingest.ingest_once(
            config,
            prefix=f"gs://{REHEARSAL_BUCKET}/rehearsal/{recipe}/{timestamp}",
            client=client,
        )
        assert counts["ingested"] == 1, counts
        assert client.observations[0]["output"] == "stub completion text"
    finally:
        stub_server.shutdown()
        subprocess.run(
            [
                str(REPO_ROOT / "scripts" / "run_trace_proxy.sh"),
                "--stop",
                "--config",
                str(tracing_config_path),
            ],
            capture_output=True,
        )
        _delete_rehearsal_objects(REHEARSAL_BUCKET, f"rehearsal/{recipe}/{timestamp}")


class _FakeLangfuseClient:
    def __init__(self):
        self.observations = []

    def create_trace_id(self, *, seed):
        return f"trace-{seed}"

    def start_observation(self, **kwargs):
        self.observations.append(kwargs)
        return self

    def end(self):
        pass

    def flush(self):
        pass


def _delete_rehearsal_objects(bucket: str | None, prefix: str) -> None:
    if not bucket:
        return
    try:
        import gcsfs

        fs = gcsfs.GCSFileSystem()
        for path in fs.find(f"gs://{bucket}/{prefix}"):
            fs.rm(path)
    except Exception:  # noqa: BLE001 - best-effort cleanup, never fails the test
        pass
