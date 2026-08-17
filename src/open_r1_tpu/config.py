"""Configuration loading for the TPU training recipes."""

from __future__ import annotations

import copy
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
        raise ValueError(
            f"Invalid override {raw!r}; expected section.key=value"
        )
    key, raw_value = raw.split("=", 1)
    return key, yaml.safe_load(raw_value)


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    """Load a YAML recipe and apply dotted command-line overrides."""
    with Path(path).open(encoding="utf-8") as config_file:
        loaded = yaml.safe_load(config_file)
    if not isinstance(loaded, dict):
        raise ValueError(f"Configuration at {path} must contain a mapping")

    config = copy.deepcopy(loaded)
    for raw_override in overrides or []:
        key, value = parse_override(raw_override)
        _set_dotted(config, key, value)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Fail early for recipe mistakes that would otherwise waste TPU time."""
    for section in ("model", "tokenizer", "dataset", "optimizer", "training"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")

    mesh = config["model"].get("mesh", {})
    shape = mesh.get("shape")
    axis_names = mesh.get("axis_names")
    if not isinstance(shape, list) or not shape or not all(
        isinstance(size, int) and size > 0 for size in shape
    ):
        raise ValueError("model.mesh.shape must be a non-empty list of integers")
    if not isinstance(axis_names, list) or len(axis_names) != len(shape):
        raise ValueError(
            "model.mesh.axis_names must have one name per mesh dimension"
        )

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
        raise ValueError(
            "training.wandb.mode must be online, offline, or disabled"
        )
    tags = wandb.get("tags", [])
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise ValueError("training.wandb.tags must be a list of strings")

    eval_fraction = config["dataset"].get("eval_fraction", 0.0)
    if not isinstance(eval_fraction, (int, float)) or not 0.0 <= eval_fraction < 1.0:
        raise ValueError("dataset.eval_fraction must be in [0.0, 1.0)")
    eval_max_examples = config["dataset"].get("eval_max_examples")
    if eval_max_examples is not None and (
        not isinstance(eval_max_examples, int) or eval_max_examples <= 0
    ):
        raise ValueError(
            "dataset.eval_max_examples must be a positive integer or null"
        )

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
