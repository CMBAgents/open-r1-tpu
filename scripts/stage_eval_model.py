#!/usr/bin/env python3
"""Stage a copy of a model directory that serves a GRPO recipe's prompt format.

vLLM renders a chat request with the model directory's chat template and stops
on the ``eos_token_id`` list in its ``generation_config.json``. A recipe that
trains under its own ``tokenizer.chat_template`` and extra
``rollout.eos_token_ids`` (recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml) needs both
in whatever directory the evaluation serves, or the benchmark measures a
prompt the model never trained on. That holds for the untouched base and a
published checkpoint as much as for this project's own export, so every model
in a comparison is staged the same way.

The copy hard-links the weight files (copies them where the filesystem
refuses; a symlink would dangle inside the vLLM container's bind mount),
copies everything else, and rewrites two files:

- ``tokenizer_config.json``: ``chat_template`` becomes the recipe's, and a
  ``chat_template.jinja`` beside it, which newer Transformers read in
  preference to the JSON field, is rewritten to match;
- ``generation_config.json``: ``eos_token_id`` becomes the recipe's
  ``rollout.eos_token_ids``, and the sampling defaults are dropped. vLLM
  applies a model's ``top_k``, ``repetition_penalty`` and the like to any
  request that does not set them, and the evaluation sends only temperature,
  top-p and the token budget, so a published checkpoint's own defaults would
  otherwise make it the one model sampled differently.

Usage, from the repo root:
    python scripts/stage_eval_model.py \
        --recipe recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml \
        --source models/Qwen2.5-1.5B \
        --output models/Qwen2.5-1.5B-abel
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

WEIGHT_SUFFIXES = (".safetensors", ".bin")
# generation_config.json keys vLLM turns into default sampling parameters.
SAMPLING_DEFAULTS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="the GRPO recipe")
    parser.add_argument("--source", required=True, help="model directory to copy")
    parser.add_argument("--output", required=True, help="staged directory to write")
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing output"
    )
    return parser.parse_args(argv)


def serving_settings(recipe: dict[str, Any]) -> tuple[str, list[int]]:
    """The recipe's chat template and stop-token ids, both required."""
    template = recipe.get("tokenizer", {}).get("chat_template")
    if not isinstance(template, str) or not template:
        raise ValueError("the recipe sets no tokenizer.chat_template to stage")
    eos_ids = recipe.get("rollout", {}).get("eos_token_ids")
    if (
        not isinstance(eos_ids, list)
        or not eos_ids
        or not all(isinstance(token, int) for token in eos_ids)
    ):
        raise ValueError("the recipe sets no rollout.eos_token_ids to stage")
    return template, eos_ids


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text("utf-8")) if path.is_file() else {}


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", "utf-8")


def stage(
    source: Path, output: Path, template: str, eos_ids: list[int], overwrite: bool
) -> None:
    source, output = source.resolve(), output.resolve()
    if not (source / "config.json").is_file():
        raise FileNotFoundError(f"{source} has no config.json")
    if output == source or source.is_relative_to(output):
        raise ValueError(f"refusing to stage {source} into {output}")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"{output} exists; pass --overwrite to replace it")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    for item in sorted(source.iterdir()):
        if not item.is_file():
            continue
        target = output / item.name
        if item.name.endswith(WEIGHT_SUFFIXES):
            try:
                os.link(item, target)
                continue
            except OSError:
                pass
        shutil.copy2(item, target)

    tokenizer_config = _read_json(output / "tokenizer_config.json")
    tokenizer_config["chat_template"] = template
    _write_json(output / "tokenizer_config.json", tokenizer_config)
    if (output / "chat_template.jinja").is_file():
        (output / "chat_template.jinja").write_text(template, "utf-8")

    generation_config = _read_json(output / "generation_config.json")
    generation_config["eos_token_id"] = list(eos_ids)
    for key in SAMPLING_DEFAULTS:
        generation_config.pop(key, None)
    _write_json(output / "generation_config.json", generation_config)


def main(argv: list[str] | None = None) -> None:
    from open_r1_tpu.core.config import load_config
    from open_r1_tpu.grpo.run import validate_grpo_config

    args = parse_args(argv)
    recipe = load_config(args.recipe, [], validator=validate_grpo_config)
    template, eos_ids = serving_settings(recipe)
    stage(Path(args.source), Path(args.output), template, eos_ids, args.overwrite)
    print(f"staged {args.source} -> {args.output} (eos_token_id {eos_ids})")


if __name__ == "__main__":
    main()
