"""Shared pytest setup.

Stubs a minimal `langfuse` module into `sys.modules` when the real package is
not installed, so `open_r1_tpu.tracing.ingest`/`.scores` -- guarded to fail
helpfully without it (see their module docstrings; `langfuse` is part of the
`eval` extra) -- can still be imported and exercised. Every test that touches
them passes an explicit fake client (`client=...`); nothing here ever
constructs a real `Langfuse`. A real `langfuse` installation is left
untouched.
"""

from __future__ import annotations

import sys
import types


def _stub_langfuse_if_absent() -> None:
    try:
        import langfuse  # noqa: F401
    except ImportError:
        pass
    else:
        return

    module = types.ModuleType("langfuse")

    class Langfuse:
        """Never instantiated: every test passes `client=` explicitly."""

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "tests/conftest.py's langfuse stub was constructed directly; "
                "pass an explicit fake client to ingest_once()/score_pass() instead."
            )

    setattr(module, "Langfuse", Langfuse)  # noqa: B010 - ModuleType has no such attribute to set directly
    sys.modules["langfuse"] = module


_stub_langfuse_if_absent()
