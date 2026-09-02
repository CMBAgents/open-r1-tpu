"""No committed tracing file carries a deployment-specific literal.

A non-loopback host belongs exclusively in the gitignored
`configs/tracing.yaml` -- never hard-coded into a script or a committed
config, per open-r1-tpu/AGENTS.md's "Keep committed defaults neutral and
deployment-independent." `127.0.0.1` is exempt: it is a fixed security
default (every tracing service binds to loopback only), not a deployment
identifier, and is expected in every file this test covers.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]

# The two files that wire tracing into an evaluation launch: the Langfuse
# compose file and the eval launch script. Both are non-Python (YAML/bash),
# where "no non-loopback URL, only os.environ/ references and 127.0.0.1" is
# a clean, low-false-positive textual property.
FILES = [
    REPO_ROOT / "docker" / "langfuse" / "docker-compose.yaml",
    REPO_ROOT / "scripts" / "run_eval_tpu.sh",
]

# A URL whose host is neither a loopback address nor an env/format
# placeholder. Matches http(s):// and the bare host:port Docker uses.
NON_LOOPBACK_URL = re.compile(
    r"(?:https?://)(?!127\.0\.0\.1|localhost|\$|\{)[a-zA-Z0-9.-]+"
)

# docker-compose.yaml is legitimately full of `http://<service-name>:<port>`
# references -- Docker Compose's own internal service DNS (clickhouse, minio,
# redis, postgres), resolved from the `services:` block in the same file, not
# a deployment value. Only run_eval_tpu.sh, where any non-loopback host
# really would be a hard-coded deployment value, is checked for one.
URL_CHECKED_FILES = [path for path in FILES if path.suffix != ".yaml"]


def test_no_non_loopback_url_literal():
    for path in URL_CHECKED_FILES:
        text = path.read_text(encoding="utf-8")
        matches = NON_LOOPBACK_URL.findall(text)
        assert not matches, f"{path} has a non-loopback URL literal: {matches}"


def test_every_file_covered_exists():
    # A typo'd path here would silently exempt a file from the checks above.
    for path in FILES:
        assert path.is_file(), f"expected to exist: {path}"


def test_the_real_tracing_config_is_not_committed():
    # configs/tracing.example.yaml is documentation, with a placeholder host
    # -- never read by a script -- but the real, gitignored
    # configs/tracing.yaml must not be committed at all.
    assert not (REPO_ROOT / "configs" / "tracing.yaml").exists()
