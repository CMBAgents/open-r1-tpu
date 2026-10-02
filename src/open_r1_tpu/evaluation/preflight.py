"""Preflight the evaluation stack before committing TPU time to a benchmark.

Checks the serving side: the pinned LightEval dependency stack, the vLLM
container image and its service versions, the exported checkpoint, and that
every task the recipe names resolves in LightEval's registry, so a mistake
fails here, before the server spends minutes loading weights. The TPU is not
touched: vLLM holds it while serving, so initialising JAX here would fail
exactly when the server is up.

An export must end every turn on the recipe's `server.turn_end_token`
(`<|im_end|>` for Qwen3, the end-of-sentence token for DeepSeek's distills).
A stop string cannot do this, because vLLM matches stop strings against text
with special tokens stripped; only the export's `generation_config.json`
`eos_token_id` can. An export without it runs past the end of every reply and
invalidates every benchmark number, so `check_export_dir` fails on it, as it
does on a missing chat template.

Run from the repository root::

    python -m open_r1_tpu.evaluation.preflight \
      --config recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from open_r1_tpu.core.cli import parse_recipe_args, recipe_parser
from open_r1_tpu.core.packages import installed_version
from open_r1_tpu.evaluation.config import (
    load_eval_config,
    resolve_settings,
    uses_container_wrapper,
)
from open_r1_tpu.evaluation.server import container_image_provenance
from open_r1_tpu.evaluation.stack import (
    EVALUATION_PACKAGE_VERSIONS,
    EVALUATION_PYTHON_VERSION,
    VLLM_TPU_SERVICE_VERSIONS,
)
from open_r1_tpu.evaluation.tasks import resolve_task_configs

# Files a merged export needs before vLLM can serve it as a chat model.
REQUIRED_FILES = ("config.json", "tokenizer_config.json")
# Any one of these carries the weights.
WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json")


def check_dependency_versions(
    installed: Mapping[str, str] | None = None,
    *,
    python_version: str | None = None,
) -> list[str]:
    """Reject an evaluation stack that differs from the validated lock."""
    actual = (
        dict(installed)
        if installed is not None
        else {name: installed_version(name) for name in EVALUATION_PACKAGE_VERSIONS}
    )
    actual_python = python_version or platform.python_version()
    errors: list[str] = []
    if actual_python != EVALUATION_PYTHON_VERSION:
        errors.append(
            f"Python is {actual_python}, expected {EVALUATION_PYTHON_VERSION}; "
            "run `uv sync --frozen --extra eval --extra test`"
        )
    for name, expected in EVALUATION_PACKAGE_VERSIONS.items():
        found = actual.get(name, "unknown")
        if found != expected:
            errors.append(
                f"{name} is {found}, expected {expected}; run "
                "`uv sync --frozen --extra eval --extra test`"
            )
    return errors


def check_server_runtime(settings: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    """Check the container wrapper's image and its service versions. Returns
    (errors, warnings); a server that is not the wrapper's image is a warning.
    """
    image = settings.get("server_image")
    serve_command = [str(part) for part in settings.get("serve_command", [])]
    if image is None:
        return (
            [],
            [
                "server.image is null; the external inference environment is "
                "not reproducibility-checked"
            ],
        )
    if not uses_container_wrapper(serve_command):
        return (
            [],
            [
                "custom server command is not the supported container wrapper; "
                "its runtime was not checked"
            ],
        )

    command = [*serve_command, "--image", str(image), "--check"]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return ([f"could not check vLLM container runtime: {error}"], [])
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        return (
            ["vLLM container preflight failed" + (f": {detail}" if detail else "")],
            [],
        )

    try:
        provenance = container_image_provenance(settings)
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        return ([f"could not read vLLM container provenance: {error}"], [])
    if provenance is None:
        return (["vLLM container preflight returned no provenance"], [])
    versions = provenance.get("service_versions")
    if not isinstance(versions, Mapping):
        return (["vLLM container provenance omitted service_versions"], [])
    errors = [
        f"vLLM container has {name} {versions.get(name, 'unknown')}, "
        f"expected {expected}"
        for name, expected in VLLM_TPU_SERVICE_VERSIONS.items()
        if versions.get(name) != expected
    ]
    return (errors, [])


def _turn_end_token_id(
    tokenizer_config: Mapping[str, Any], directory: Path, turn_end_token: str
) -> int | None:
    """The turn-end token's id from `tokenizer_config.json`'s
    `added_tokens_decoder`, else `tokenizer.json`'s `added_tokens`; None when
    neither names it.
    """
    added_tokens_decoder = tokenizer_config.get("added_tokens_decoder")
    if isinstance(added_tokens_decoder, Mapping):
        for token_id, spec in added_tokens_decoder.items():
            if isinstance(spec, Mapping) and spec.get("content") == turn_end_token:
                try:
                    return int(token_id)
                except (TypeError, ValueError):
                    continue

    tokenizer_json_path = directory / "tokenizer.json"
    if not tokenizer_json_path.is_file():
        return None
    try:
        tokenizer_json = json.loads(tokenizer_json_path.read_text("utf-8"))
    except ValueError:
        return None
    for spec in tokenizer_json.get("added_tokens") or []:
        if isinstance(spec, Mapping) and spec.get("content") == turn_end_token:
            token_id = spec.get("id")
            if isinstance(token_id, int):
                return token_id
    return None


def check_export_dir(model_path: str, turn_end_token: str) -> list[str]:
    """Errors in an export that vLLM must serve as a chat model. Without a chat
    template the server falls back to raw completion, and without the
    turn-end token in `generation_config.json` it runs past every reply (see
    the module docstring).
    """
    errors: list[str] = []
    directory = Path(model_path).expanduser()
    if not directory.is_dir():
        return [f"server.model_path is not a directory: {directory}"]

    for name in REQUIRED_FILES:
        if not (directory / name).is_file():
            errors.append(f"export is missing {name}")
    if not any((directory / name).is_file() for name in WEIGHT_FILES):
        errors.append("export has no model.safetensors or model.safetensors.index.json")

    tokenizer_config_path = directory / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        return errors

    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text("utf-8"))
    except ValueError as error:
        errors.append(f"tokenizer_config.json is not valid JSON: {error}")
        return errors

    has_template = (
        bool(tokenizer_config.get("chat_template"))
        or (directory / "chat_template.jinja").is_file()
    )
    if not has_template:
        errors.append(
            "export carries no chat template, so served prompts would not "
            "match the format training used"
        )

    turn_end_id = _turn_end_token_id(tokenizer_config, directory, turn_end_token)
    if turn_end_id is None:
        errors.append(
            f"export's tokenizer files do not name a token id for "
            f"{turn_end_token!r} (checked tokenizer_config.json's "
            "added_tokens_decoder and tokenizer.json's added_tokens); cannot "
            "verify the export stops at turn boundaries"
        )
        return errors

    generation_config_path = directory / "generation_config.json"
    if not generation_config_path.is_file():
        errors.append(
            "export has no generation_config.json, so vLLM falls back to the "
            f"tokenizer's own EOS rather than {turn_end_token!r}; every "
            "benchmark number from it would run past the turn boundary"
        )
        return errors
    try:
        generation_config = json.loads(generation_config_path.read_text("utf-8"))
    except ValueError as error:
        errors.append(f"generation_config.json is not valid JSON: {error}")
        return errors

    eos_token_id = generation_config.get("eos_token_id")
    eos_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
    if turn_end_id not in eos_ids:
        errors.append(
            f"generation_config.json's eos_token_id ({eos_token_id!r}) does "
            f"not include {turn_end_token!r}'s id ({turn_end_id}); the export "
            "will not stop at turn boundaries and every benchmark number from "
            "it would be invalid"
        )
    return errors


def main() -> None:
    args = parse_recipe_args(recipe_parser(__doc__))
    settings = resolve_settings(load_eval_config(args.config, args.overrides))

    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(check_dependency_versions())
    errors.extend(check_export_dir(settings["model_path"], settings["turn_end_token"]))

    try:
        resolve_task_configs(settings["tasks"])
    except (ImportError, ValueError) as error:
        errors.append(f"could not resolve the recipe's tasks: {error}")

    runtime_errors, runtime_warnings = check_server_runtime(settings)
    errors.extend(runtime_errors)
    warnings.extend(runtime_warnings)

    if str(settings["summary_path"]).startswith("gs://"):
        try:
            import gcsfs  # noqa: F401
        except ImportError:
            errors.append("writing the summary to GCS requires the gcsfs package")

    print(
        f"Evaluation stack: Python {platform.python_version()}, "
        f"LightEval {installed_version('lighteval')}"
    )
    if settings.get("server_image"):
        print(f"vLLM image: {settings['server_image']}")
    print(f"Export: {settings['model_path']}")
    print(
        f"Tier {settings['tier']}: {len(settings['tasks'])} tasks x "
        f"{len(settings['seeds'])} seeds"
    )
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        raise SystemExit("Evaluation preflight failed:\n- " + "\n- ".join(errors))
    print("Evaluation preflight passed.")


if __name__ == "__main__":
    main()
