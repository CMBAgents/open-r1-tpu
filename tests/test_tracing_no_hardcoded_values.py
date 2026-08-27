"""No committed tracing file carries a deployment-specific literal.

Bucket names, non-loopback hosts, and GCS URIs belong exclusively in the
gitignored `configs/tracing.yaml` and the uncommitted `docker/langfuse/.env` /
`docker/trace-proxy` environment -- never hard-coded into a script or a
committed config, per open-r1-tpu/AGENTS.md's "Keep committed defaults
neutral and deployment-independent." `127.0.0.1` is exempt: it is a fixed
security default (every tracing service binds to loopback only), not a
deployment identifier, and is expected in every file this test covers.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]

# Exactly the files the plan names: the proxy config, the compose file, and
# the two shell scripts that wire tracing into an evaluation launch. All four
# are non-Python (YAML/bash), where "no gs://, no bucket-like name, no
# non-loopback URL, only os.environ/ references and 127.0.0.1" is a clean,
# low-false-positive textual property. `docker/trace-proxy/gcs_logger.py` and
# `scripts/run_langfuse_stack.sh` hold the same invariant by inspection --
# every value they touch comes from an environment variable -- but as Python
# f-strings and general shell they are not a good fit for this regex-based
# check.
FILES = [
    REPO_ROOT / "docker" / "trace-proxy" / "config.yaml",
    REPO_ROOT / "docker" / "langfuse" / "docker-compose.yaml",
    REPO_ROOT / "scripts" / "run_trace_proxy.sh",
    REPO_ROOT / "scripts" / "run_eval_tpu.sh",
]

# A literal gs:// URI, excluding a shell/format interpolation
# (`gs://${VAR}`, `gs://{var}`) building one dynamically from config-sourced
# values -- which is how every one of these files actually constructs a real
# path.
GCS_URI = re.compile(r"gs://(?!\$|\{)\S+")

# A URL whose host is neither a loopback address nor an env/format
# placeholder. Matches http(s):// and the bare host:port litellm/Docker use.
NON_LOOPBACK_URL = re.compile(
    r"(?:https?://)(?!127\.0\.0\.1|localhost|\$|\{)[a-zA-Z0-9.-]+"
)

# docker-compose.yaml is legitimately full of `http://<service-name>:<port>`
# references -- Docker Compose's own internal service DNS (clickhouse, minio,
# redis, postgres), resolved from the `services:` block in the same file, not
# a deployment value -- and one citation comment linking to the upstream
# compose it was adapted from. Neither is what this check exists to catch;
# only the proxy config and the two shell scripts, where any non-loopback
# host really would be a hard-coded deployment value, are checked for one.
URL_CHECKED_FILES = [
    path for path in FILES if path.suffix != ".yaml" or "trace-proxy" in path.parts
]


def test_no_gcs_uri_literal():
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        matches = GCS_URI.findall(text)
        assert not matches, f"{path} has a literal gs:// URI: {matches}"


def test_no_non_loopback_url_literal():
    for path in URL_CHECKED_FILES:
        text = path.read_text(encoding="utf-8")
        matches = NON_LOOPBACK_URL.findall(text)
        assert not matches, f"{path} has a non-loopback URL literal: {matches}"


def test_every_file_covered_exists():
    # A typo'd path here would silently exempt a file from the checks above.
    for path in FILES:
        assert path.is_file(), f"expected to exist: {path}"


def test_the_example_tracing_config_is_the_only_config_with_a_real_bucket_value():
    # configs/tracing.example.yaml is allowed a placeholder bucket name -- it
    # is documentation, never read by a script -- but the real, gitignored
    # configs/tracing.yaml must not be committed at all.
    assert not (REPO_ROOT / "configs" / "tracing.yaml").exists()
