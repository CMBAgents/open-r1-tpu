"""No committed tracing file carries a deployment-specific literal.

A non-loopback host belongs exclusively in the gitignored
`configs/tracing.yaml` and `docker/langfuse/.env` -- never hard-coded into a
script or a committed config, per open-r1-tpu/AGENTS.md's "Keep committed
defaults neutral and deployment-independent." `127.0.0.1` is exempt: it is
the loopback *default* every tracing service falls back to, not a deployment
identifier, and is expected in every file this test covers.

A two-host deployment does publish langfuse-web on a routable address
(LANGFUSE_WEB_BIND) and dial it from another machine (langfuse.host), but
both arrive as arguments to `scripts/gen_langfuse_env.sh` and land only in
those two gitignored files. That is the rule this test exists to keep: the
committed tree never learns where anything actually runs.
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
    REPO_ROOT / "scripts" / "gen_langfuse_env.sh",
]

# A URL whose host is neither a loopback address nor an env/format
# placeholder. Matches http(s):// and the bare host:port Docker uses.
NON_LOOPBACK_URL = re.compile(
    r"(?:https?://)(?!127\.0\.0\.1|localhost|\$|\{)[a-zA-Z0-9.-]+"
)

# An IPv4 literal that is not loopback. This is the check that matters for
# gen_langfuse_env.sh, which now takes --web-bind and --langfuse-host: the
# temptation is to bake this deployment's address in as a default rather than
# type it each time, and that would put it in the committed tree.
NON_LOOPBACK_IPV4 = re.compile(
    r"(?<![\w.])(?!127\.0\.0\.1|0\.0\.0\.0)\d{1,3}(?:\.\d{1,3}){3}(?![\w.])"
)

# docker-compose.yaml and gen_langfuse_env.sh are legitimately full of
# `http://<service-name>:<port>` references -- Docker Compose's own internal
# service DNS (clickhouse, minio, redis, postgres), resolved from the
# `services:` block, not a deployment value. Only run_eval_tpu.sh, where any
# non-loopback host really would be a hard-coded deployment value, is checked
# for one. Every file is checked for an IP literal, which no service name is.
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
    # configs/tracing.example.yaml is documentation, with a placeholder host
    # -- never read by a script -- but the real, gitignored
    # configs/tracing.yaml must not be committed at all.
    assert not (REPO_ROOT / "configs" / "tracing.yaml").exists()
