"""Tracing configuration: one YAML file, one loader, no literal deployment
values anywhere else.

Down to one section now that the trace-capture proxy, the ingester, and the
score pass are gone: `langfuse`, the host and port `evaluation.experiment`/
`evaluation.dataset_sync` build a `Langfuse` client from. Validated the way
`open_r1_tpu.evaluation.run.validate_eval_config` validates an evaluation
recipe: required keys named in a `ValueError`, an unknown key rejected with a
close-match suggestion rather than silently ignored.

Commit only `configs/tracing.example.yaml`, with neutral placeholder values.
The real file -- `configs/tracing.yaml` by convention, but any path works --
is gitignored, since a Langfuse host is a deployment identifier and this
repository's confidentiality rule extends to anything that would identify
the project it runs for.
"""

from __future__ import annotations

import difflib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from open_r1_tpu.core.config import load_config

# The complete key set each section accepts. A key outside this set is either
# a typo or a stale setting from a schema that moved on, exactly as
# `evaluation.run._reject_unknown_keys` treats an eval recipe.
LANGFUSE_KEYS = {"host", "port"}
SECTIONS = {"langfuse": LANGFUSE_KEYS}


def _reject_unknown_keys(
    prefix: str, section: Mapping[str, Any], allowed: set[str]
) -> None:
    """Reject a key outside a section's schema, suggesting the nearest match."""
    for key in section:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), sorted(allowed), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"Unknown key {prefix}.{key}{hint}")


def _require_port(field: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 65535:
        raise ValueError(f"{field} must be a TCP port number")


def _require_nonempty_str(field: str, value: Any) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")


def validate_tracing_config(config: dict[str, Any]) -> None:
    """Fail early for a tracing config mistake, before anything is launched."""
    for section in config:
        if section not in SECTIONS:
            close = difflib.get_close_matches(str(section), sorted(SECTIONS), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"Unknown configuration section {section!r}{hint}")

    for section, allowed in SECTIONS.items():
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
        _reject_unknown_keys(section, config[section], allowed)

    langfuse = config["langfuse"]
    _require_nonempty_str("langfuse.host", langfuse.get("host"))
    _require_port("langfuse.port", langfuse.get("port"))


def load_tracing_config(
    path: str | Path, overrides: list[str] | None = None
) -> dict[str, Any]:
    """Load a tracing config and apply dotted `section.key=value` overrides."""
    return load_config(path, overrides, validator=validate_tracing_config)


def build_langfuse_client(tracing_config: Mapping[str, Any]) -> Any:
    """A `Langfuse` client from this project's own tracing config -- the
    `langfuse` section. Shared by `evaluation.dataset_sync` and
    `evaluation.experiment`.
    """
    from langfuse import Langfuse

    langfuse_section = tracing_config["langfuse"]
    base_url = f"http://{langfuse_section['host']}:{langfuse_section['port']}"
    return Langfuse(base_url=base_url)
