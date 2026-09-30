"""No committed tracing file carries a deployment-specific literal.

A non-loopback host belongs only in the gitignored `configs/tracing.yaml` and
`docker/langfuse/.env`, which `scripts/gen_langfuse_env.sh` writes from its
arguments. `127.0.0.1` is exempt: it is the loopback default, not a
deployment identifier.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]

# The files that wire tracing into an evaluation launch, including the
# templates gen_langfuse_env.sh copies into the generated files.
FILES = [
    REPO_ROOT / "docker" / "langfuse" / "docker-compose.yaml",
    REPO_ROOT / "docker" / "langfuse" / ".env.example",
    REPO_ROOT / "configs" / "tracing.example.yaml",
    REPO_ROOT / "scripts" / "run_eval_tpu.sh",
    REPO_ROOT / "scripts" / "gen_langfuse_env.sh",
]

# An http(s) URL whose host is neither loopback nor a variable placeholder.
NON_LOOPBACK_URL = re.compile(
    r"(?:https?://)(?!127\.0\.0\.1|localhost|\$|\{)[a-zA-Z0-9.-]+"
)

# An IPv4 literal that is not loopback, such as an address baked into
# gen_langfuse_env.sh as a default for --web-bind or --langfuse-host.
NON_LOOPBACK_IPV4 = re.compile(
    r"(?<![\w.])(?!127\.0\.0\.1|0\.0\.0\.0)\d{1,3}(?:\.\d{1,3}){3}(?![\w.])"
)

# The Compose file, the .env template and gen_langfuse_env.sh legitimately
# reference Compose service names (`http://clickhouse:8123`), so only
# run_eval_tpu.sh is checked for a non-loopback URL. Every file is checked for
# an IP literal.
URL_CHECKED_FILES = [REPO_ROOT / "scripts" / "run_eval_tpu.sh"]


def test_no_non_loopback_url_literal():
    for path in URL_CHECKED_FILES:
        text = path.read_text(encoding="utf-8")
        matches = NON_LOOPBACK_URL.findall(text)
        assert not matches, f"{path} has a non-loopback URL literal: {matches}"


def test_no_non_loopback_ipv4_literal():
    for path in FILES:
        text = path.read_text(encoding="utf-8")
        matches = NON_LOOPBACK_IPV4.findall(text)
        assert not matches, f"{path} has an IP literal: {matches}"


def test_every_file_covered_exists():
    # A typo'd path here would silently exempt a file from the checks above.
    for path in FILES:
        assert path.is_file(), f"expected to exist: {path}"


def test_the_real_tracing_config_is_not_committed():
    # scripts/gen_langfuse_env.sh writes the real file; it may exist on disk
    # but must be neither tracked nor trackable.
    path = "configs/tracing.yaml"
    tracked = subprocess.run(
        ["git", "ls-files", "--", path],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert not tracked.strip()
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", "--", path], cwd=REPO_ROOT
    )
    assert ignored.returncode == 0
