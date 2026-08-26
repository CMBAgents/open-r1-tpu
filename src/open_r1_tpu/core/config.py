"""Shared configuration loading and validation."""

from __future__ import annotations

import copy
from collections.abc import Callable
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
    """Merge `child` onto `base`. Mappings merge recursively; everything else
    -- lists and scalars alike -- is replaced wholesale by the child's value,
    since there is no sound way to merge a list of tasks or seeds element-wise.
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

    The base path is relative to the file that declares it, matching how a
    recipe is normally read from the repository root regardless of which
    directory `extends` it. One level only: a base recipe with its own
    `extends` raises rather than chasing a chain, which keeps the merge order
    obvious from reading a single file.
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


def read_prompt_file(path: str | Path) -> str:
    """Read a system-prompt text file shared between training and evaluation.

    Strips a trailing newline (`.rstrip("\\n")`) so an editor-added final
    newline cannot make two otherwise-identical prompt files diverge. Missing
    files fail with the path named, rather than a bare `FileNotFoundError`
    surfacing deep inside a recipe loader -- a recipe that wants no system
    prompt spells that as an explicit `null` instead of a missing file.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise ValueError(f"system prompt file not found: {file_path}")
    return file_path.read_text(encoding="utf-8").rstrip("\n")


def load_config(
    path: str | Path,
    overrides: list[str] | None = None,
    validator: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Load a YAML recipe and apply dotted command-line overrides.

    `validator` defaults to `validate_config`, which checks a training recipe.
    Evaluation recipes are a different shape entirely -- no optimizer, no
    dataset -- so they pass their own validator rather than being forced into
    the training schema.

    A top-level `extends: <path>` key merges onto a base recipe before
    overrides and validation run, so the validator never sees the key itself.
    """
    recipe_path = Path(path)
    loaded = _load_yaml_mapping(recipe_path)
    merged = _resolve_extends(loaded, recipe_path)

    config = copy.deepcopy(merged)
    for raw_override in overrides or []:
        key, value = parse_override(raw_override)
        _set_dotted(config, key, value)
    (validator or validate_config)(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail early for recipe mistakes that would otherwise waste TPU time."""
    for section in ("model", "tokenizer", "dataset", "optimizer", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")

    mesh = config["model"].get("mesh", {})
    shape = mesh.get("shape")
    axis_names = mesh.get("axis_names")
    if (
        not isinstance(shape, list)
        or not shape
        or not all(isinstance(size, int) and size > 0 for size in shape)
    ):
        raise ValueError("model.mesh.shape must be a non-empty list of integers")
    if not isinstance(axis_names, list) or len(axis_names) != len(shape):
        raise ValueError("model.mesh.axis_names must have one name per mesh dimension")

    batch_size = config["dataset"].get("batch_size")
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("dataset.batch_size must be a positive integer")

    max_length = config["dataset"].get("max_length")
    if not isinstance(max_length, int) or max_length < 2:
        raise ValueError("dataset.max_length must be at least 2")

    accumulation = config["training"].get("gradient_accumulation_steps", 1)
    if not isinstance(accumulation, int) or accumulation <= 0:
        raise ValueError(
            "training.gradient_accumulation_steps must be a positive integer"
        )

    wandb = config["training"].get("wandb", {})
    if not isinstance(wandb, dict):
        raise ValueError("training.wandb must be a configuration mapping")
    if not isinstance(wandb.get("enabled", False), bool):
        raise ValueError("training.wandb.enabled must be a boolean")
    mode = wandb.get("mode", "online")
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("training.wandb.mode must be online, offline, or disabled")
    tags = wandb.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError("training.wandb.tags must be a list of strings")

    # Imported here so the data module stays out of this module's import
    # graph; both checks are cheap and catch a recipe that would otherwise
    # filter every example and train on an empty dataset.
    from open_r1_tpu.training.data import (
        OVERLENGTH_POLICIES,
        message_schema_from_config,
    )

    overlength_policy = config["dataset"].get("overlength_policy", "drop")
    if overlength_policy not in OVERLENGTH_POLICIES:
        raise ValueError(
            "dataset.overlength_policy must be one of " + ", ".join(OVERLENGTH_POLICIES)
        )
    message_schema_from_config(config["dataset"].get("message_schema"))

    eval_fraction = config["dataset"].get("eval_fraction", 0.0)
    if not isinstance(eval_fraction, (int, float)) or not 0.0 <= eval_fraction < 1.0:
        raise ValueError("dataset.eval_fraction must be in [0.0, 1.0)")
    eval_max_examples = config["dataset"].get("eval_max_examples")
    if eval_max_examples is not None and (
        not isinstance(eval_max_examples, int) or eval_max_examples <= 0
    ):
        raise ValueError("dataset.eval_max_examples must be a positive integer or null")

    transcripts = config["training"].get("transcripts", {})
    if not isinstance(transcripts, dict):
        raise ValueError("training.transcripts must be a configuration mapping")
    if not isinstance(transcripts.get("enabled", False), bool):
        raise ValueError("training.transcripts.enabled must be a boolean")
    for key in ("every_n_steps", "max_new_tokens"):
        value = transcripts.get(key, 1)
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"training.transcripts.{key} must be a positive integer")
    prompts = transcripts.get("prompts")
    if prompts is not None and (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(prompt, str) for prompt in prompts)
    ):
        raise ValueError(
            "training.transcripts.prompts must be a non-empty list of strings or null"
        )
