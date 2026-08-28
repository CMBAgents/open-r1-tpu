"""Tracing configuration: one YAML file, one loader, no literal deployment
values anywhere else.

The trace-capture proxy, the Langfuse compose stack, the ingester, and the
score pass are all deployment-specific in the same handful of ways -- a GCS
bucket, a proxy port, a Langfuse host. Every one of those values lives here,
validated the way `open_r1_tpu.evaluation.run.validate_eval_config` validates
an evaluation recipe: required keys named in a `ValueError`, an unknown key
rejected with a close-match suggestion rather than silently ignored.

Nothing that reads this config parses YAML itself. Shell scripts call this
module's `--export-env` mode and `eval "$(...)"` the result, so a value only
ever passes through one parser on its way from the YAML file to a shell
variable or a Python dict.

Commit only `configs/tracing.example.yaml`, with neutral placeholder values.
The real file -- `configs/tracing.yaml` by convention, but any path works --
is gitignored, since a bucket name or a Langfuse host is a deployment
identifier and this repository's confidentiality rule extends to anything
that would identify the project it runs for.
"""

from __future__ import annotations

import argparse
import difflib
import shlex
import string
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from open_r1_tpu.core.config import load_config

# The complete key set each section accepts. A key outside this set is either
# a typo or a stale setting from a schema that moved on, exactly as
# `evaluation.run._reject_unknown_keys` treats an eval recipe.
GCS_KEYS = {"bucket", "prefix_template"}
PROXY_KEYS = {"port", "upstream_base_url", "image"}
LANGFUSE_KEYS = {"host", "port"}
INGESTER_KEYS = {"poll_secs", "state_dir"}
SECTIONS = {
    "gcs": GCS_KEYS,
    "proxy": PROXY_KEYS,
    "langfuse": LANGFUSE_KEYS,
    "ingester": INGESTER_KEYS,
}

# The only substitutions `gcs.prefix_template` may reference. Filled in by
# this module from the recipe name and a launch timestamp -- never by shell
# string-pasting -- so the rendered prefix is deterministic and testable on
# its own.
PREFIX_TEMPLATE_FIELDS = {"recipe", "timestamp"}


def _reject_unknown_keys(
    prefix: str, section: Mapping[str, Any], allowed: set[str]
) -> None:
    """Reject a key outside a section's schema, suggesting the nearest match."""
    for key in section:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), sorted(allowed), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"Unknown key {prefix}.{key}{hint}")


def _validate_image_digest(field: str, image: str) -> None:
    """Require an immutable `@sha256:<64 hex>` suffix, matching the repo's
    existing convention for `server.image` in an evaluation recipe. Unlike
    that field, a tracing proxy image has no "derived local tag" escape
    hatch: it is always a container pulled from a registry.
    """
    prefix, marker, digest = image.rpartition("@sha256:")
    has_digest = (
        bool(marker)
        and bool(prefix)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest)
    )
    if not has_digest:
        raise ValueError(f"{field} must include an immutable @sha256:<64 hex> digest")


def _validate_prefix_template(template: str) -> None:
    try:
        fields = {
            name
            for _, name, _, _ in string.Formatter().parse(template)
            if name is not None
        }
    except ValueError as error:
        raise ValueError(
            f"gcs.prefix_template is not a valid format string: {error}"
        ) from error
    unknown = fields - PREFIX_TEMPLATE_FIELDS
    if unknown:
        raise ValueError(
            f"gcs.prefix_template references unknown field(s) {sorted(unknown)}; "
            f"allowed fields are {sorted(PREFIX_TEMPLATE_FIELDS)}"
        )


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

    gcs = config["gcs"]
    _require_nonempty_str("gcs.bucket", gcs.get("bucket"))
    _require_nonempty_str("gcs.prefix_template", gcs.get("prefix_template"))
    _validate_prefix_template(gcs["prefix_template"])

    proxy = config["proxy"]
    _require_port("proxy.port", proxy.get("port"))
    _require_nonempty_str("proxy.upstream_base_url", proxy.get("upstream_base_url"))
    _require_nonempty_str("proxy.image", proxy.get("image"))
    _validate_image_digest("proxy.image", proxy["image"])

    langfuse = config["langfuse"]
    _require_nonempty_str("langfuse.host", langfuse.get("host"))
    _require_port("langfuse.port", langfuse.get("port"))

    ingester = config["ingester"]
    poll_secs = ingester.get("poll_secs")
    if not isinstance(poll_secs, int) or isinstance(poll_secs, bool) or poll_secs <= 0:
        raise ValueError("ingester.poll_secs must be a positive integer")
    _require_nonempty_str("ingester.state_dir", ingester.get("state_dir"))


def load_tracing_config(
    path: str | Path, overrides: list[str] | None = None
) -> dict[str, Any]:
    """Load a tracing config and apply dotted `section.key=value` overrides."""
    return load_config(path, overrides, validator=validate_tracing_config)


def build_langfuse_client(tracing_config: Mapping[str, Any]) -> Any:
    """A `Langfuse` client from this project's own tracing config -- the
    `langfuse` section only. Shared by `evaluation.dataset_sync` and
    `evaluation.experiment`; `evaluation.runner` keeps its own private copy
    until Tasks 5/6 of `eval-langfuse-native-plan.md` retire it.
    """
    from langfuse import Langfuse

    langfuse_section = tracing_config["langfuse"]
    base_url = f"http://{langfuse_section['host']}:{langfuse_section['port']}"
    return Langfuse(base_url=base_url)


def render_prefix(prefix_template: str, *, recipe: str, timestamp: str) -> str:
    """Render the run-scoped GCS prefix deterministically from its inputs.

    A pure function on purpose: the same three inputs always yield the same
    prefix, whether called from the launch script, a test, or the ingester
    working out where a run's objects live.
    """
    return prefix_template.format(recipe=recipe, timestamp=timestamp)


def export_env_lines(
    config: Mapping[str, Any],
    *,
    recipe: str | None = None,
    timestamp: str | None = None,
) -> list[str]:
    """Render this config as `export KEY=VALUE` shell lines.

    Every deployment-specific value a shell script needs is emitted here so
    scripts never parse YAML or embed a literal port, bucket, image, or host.
    Values are shell-quoted, so `eval "$(... --export-env)"` is safe whatever
    a recipe's values contain.

    `recipe` and `timestamp`, given together, additionally render the
    run-scoped prefix as `TRACE_GCS_PREFIX` -- the one value that is not
    static configuration but depends on this particular launch.
    """
    if bool(recipe) != bool(timestamp):
        raise ValueError("recipe and timestamp must be given together")

    pairs: dict[str, Any] = {
        "TRACE_GCS_BUCKET": config["gcs"]["bucket"],
        "TRACE_GCS_PREFIX_TEMPLATE": config["gcs"]["prefix_template"],
        "TRACE_PROXY_PORT": config["proxy"]["port"],
        "TRACE_PROXY_UPSTREAM_BASE_URL": config["proxy"]["upstream_base_url"],
        "TRACE_PROXY_IMAGE": config["proxy"]["image"],
        "TRACE_LANGFUSE_HOST": config["langfuse"]["host"],
        "TRACE_LANGFUSE_PORT": config["langfuse"]["port"],
        "TRACE_INGESTER_POLL_SECS": config["ingester"]["poll_secs"],
        "TRACE_INGESTER_STATE_DIR": config["ingester"]["state_dir"],
    }
    if recipe:
        # The check above guarantees timestamp is set whenever recipe is.
        assert timestamp is not None
        pairs["TRACE_GCS_PREFIX"] = render_prefix(
            config["gcs"]["prefix_template"], recipe=recipe, timestamp=timestamp
        )
    return [f"export {key}={shlex.quote(str(value))}" for key, value in pairs.items()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML tracing config")
    parser.add_argument(
        "--export-env",
        action="store_true",
        help="Print `export KEY=VALUE` lines for a shell script to eval",
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help="Recipe name to render gcs.prefix_template with (needs --timestamp)",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Timestamp to render gcs.prefix_template with (needs --recipe)",
    )
    parser.add_argument("overrides", nargs="*", help="section.key=value overrides")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_tracing_config(args.config, args.overrides)
    if not args.export_env:
        raise SystemExit("Nothing to do: pass --export-env")
    for line in export_env_lines(config, recipe=args.recipe, timestamp=args.timestamp):
        print(line)


if __name__ == "__main__":
    main()
