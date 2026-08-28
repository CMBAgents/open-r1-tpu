"""Importing the two tracing modules that need `langfuse` fails helpfully
without it -- run in a subprocess with this same interpreter, since
tests/conftest.py stubs `langfuse` for every other test in this file's own
process. `langfuse` is part of the `eval` extra (no longer a separate
optional one), so this guard only has something to demonstrate in an
environment without that extra installed at all; skipped otherwise.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "module", ["open_r1_tpu.tracing.ingest", "open_r1_tpu.tracing.scores"]
)
def test_importing_without_the_extra_fails_helpfully(module):
    completed = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": "src", "PATH": "/usr/bin:/bin:/opt/homebrew/bin"},
        cwd=str(REPO_ROOT),
    )
    if completed.returncode == 0:
        pytest.skip("langfuse is installed in this environment; guard not exercised")

    assert "tracing" in completed.stderr
    assert "uv pip install" in completed.stderr
