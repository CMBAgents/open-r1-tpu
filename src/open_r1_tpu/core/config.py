"""YAML recipe loading shared by training and evaluation: `extends`, dotted
overrides, and the schema checks each stage's validator builds on."""

from __future__ import annotations

import copy
import difflib
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

import yaml


def _set_dotted(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    if not all(parts):
        raise ValueError(f"Invalid override key: {key!r}")

    current = config
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise ValueError(
                f"Cannot set {key!r}: {part!r} is not a configuration mapping"
            )
        current = child
    current[parts[-1]] = value


def parse_override(raw: str) -> tuple[str, Any]:
    """Parse a Tunix-style ``section.key=value`` command-line override."""
    if "=" not in raw:
        raise ValueError(f"Invalid override {raw!r}; expected section.key=value")
    key, raw_value = raw.split("=", 1)
    return key, yaml.safe_load(raw_value)


def _deep_merge(base: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Merge `child` onto `base`. Mappings merge recursively; lists and scalars
    are replaced wholesale, since a list of tasks or seeds has no sound
    element-wise merge.
    """
    merged = dict(base)
    for key, value in child.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration at {path} must contain a mapping")
    return loaded


def _resolve_extends(config: dict[str, Any], declaring_path: Path) -> dict[str, Any]:
    """Resolve a top-level `extends: <path>` key into a merged mapping.

    The base path is relative to the declaring file. One level only: a base
    recipe with its own `extends` raises, so the merge order is obvious from
    reading a single file.
    """
    extends = config.pop("extends", None)
    if extends is None:
        return config
    base_path = (declaring_path.parent / str(extends)).resolve()
    base = _load_yaml_mapping(base_path)
    if "extends" in base:
        raise ValueError(
            f"{base_path} is a base recipe and cannot itself set 'extends'"
        )
    return _deep_merge(base, config)


def read_prompt_file(path: str | Path | None) -> str | None:
    """Read a system-prompt text file shared between training and evaluation.

    `None` means the recipe wants no system prompt and returns `None`. A
    trailing newline is stripped so an editor-added one cannot make two
    otherwise identical prompt files diverge, and a missing file fails with
    its path named.
    """
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"system prompt file not found: {file_path}")
    return file_path.read_text(encoding="utf-8").rstrip("\n")


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    *,
    validator: Callable[[dict[str, Any]], None],
) -> dict[str, Any]:
    """Load a YAML recipe, apply dotted command-line overrides, and validate it.

    `extends` is merged first, then the overrides, so `validator` sees the
    final recipe without the `extends` key. Each stage passes its own
    validator (`validate_sft_config`, `validate_grpo_config`, ...).
    """
    recipe_path = Path(path)
    config = copy.deepcopy(
        _resolve_extends(_load_yaml_mapping(recipe_path), recipe_path)
    )
    for raw_override in overrides or []:
        key, value = parse_override(raw_override)
        _set_dotted(config, key, value)
    validator(config)
    return config


def reject_unknown_keys(
    prefix: str, section: Mapping[str, Any], allowed: Collection[str]
) -> None:
    """Reject a key outside a section's schema, suggesting the nearest match.

    An empty `prefix` names the recipe's top-level sections.
    """
    for key in section:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), sorted(allowed), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            name = f"key {prefix}.{key}" if prefix else f"configuration section {key}"
            raise ValueError(f"Unknown {name}{hint}")


def check_sections(
    config: Mapping[str, Any],
    required: Collection[str],
    optional: Collection[str] = (),
) -> None:
    """Require each `required` section to be a mapping, allow `optional` ones
    (also mappings when present), and reject every other top-level key."""
    reject_unknown_keys("", config, {*required, *optional})
    for section in required:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")
    for section in optional:
        if section in config and not isinstance(config[section], dict):
            raise ValueError(f"{section} must be a configuration mapping")
