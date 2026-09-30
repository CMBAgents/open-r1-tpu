"""Preflight the evaluation stack before committing TPU time to a benchmark.

The training preflight in `open_r1_tpu.sft.preflight` validates the
Tunix/JAX stack. This
validates the serving side: the exact LightEval dependency stack, the pinned
vLLM container image, the exported checkpoint it will be pointed at, and the
recipe's task names. The TPU itself is deliberately not initialized. vLLM
holds the chip while it serves, so a preflight that called `jax.devices()`
would fail precisely when the server was up and working.

It exists because the failures worth catching here are silent, expensive, or
both. A merged export missing its tokenizer files or its chat
template loads far enough to serve requests and then answers off-distribution,
producing a benchmark number that measures the wrong thing. And Qwen3-Base
names `<|endoftext|>` as its EOS while the chat template closes turns with
`<|im_end|>`, so a server left to the tokenizer's own EOS runs past the end of
every reply and writes the user's next turn as well -- which under a benchmark
looks like a model that cannot stop reasoning. vLLM does not stop at
`<|im_end|>` because a stop *string* names it -- vLLM matches stop strings
against decoded text with special tokens stripped, so one can never fire on the
real token -- it stops because the export's `generation_config.json` names the
token's id as an `eos_token_id`. `check_export_dir` therefore verifies that
setting directly and fails the preflight, rather than warning, when it is
missing or wrong: every benchmark number from such an export would be invalid. The token
itself differs per model -- Qwen3 closes turns with `<|im_end|>`, DeepSeek's
distills with their end-of-sentence token -- so the recipe names it via
`server.turn_end_token`.
The recipe's tasks are resolved against the installed LightEval and diffed
against the frozen task pack (`configs/taskpack.yaml`): a task that no longer
exists, or a LightEval upgrade that quietly moves a prompt template, a
generation parameter or a metric's configuration, fails here, naming what
moved, before the server spends fifteen minutes loading weights.

Run from the repository root::

    python -m open_r1_tpu.evaluation.preflight \
      --config recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml
"""

from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from open_r1_tpu.core.cli import parse_recipe_args, recipe_parser
from open_r1_tpu.evaluation.config import load_eval_config, resolve_settings
from open_r1_tpu.evaluation.server import container_image_provenance
from open_r1_tpu.evaluation.stack import (
    EVALUATION_PACKAGE_VERSIONS,
    EVALUATION_PYTHON_VERSION,
    VLLM_TPU_SERVICE_VERSIONS,
)
from open_r1_tpu.evaluation.taskpack import DEFAULT_TASKPACK_PATH, verify_task_specs

# Files a merged export needs before vLLM can serve it as a chat model.
REQUIRED_FILES = ("config.json", "tokenizer_config.json")
# Any one of these carries the weights.
WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json")


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def check_dependency_versions(
    installed: Mapping[str, str] | None = None,
    *,
    python_version: str | None = None,
) -> list[str]:
    """Reject an evaluation stack that differs from the validated lock."""
    actual = (
        dict(installed)
        if installed is not None
        else {name: _version(name) for name in EVALUATION_PACKAGE_VERSIONS}
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


def check_server_runtime(settings: Mapping[str, object]) -> tuple[list[str], list[str]]:
    """Verify the supported wrapper, local image, and service versions."""
    image = settings.get("server_image")
    raw_serve_command = settings.get("serve_command", [])
    serve_command = (
        [str(part) for part in raw_serve_command]
        if isinstance(raw_serve_command, Sequence)
        and not isinstance(raw_serve_command, str)
        else []
    )
    if image is None:
        return (
            [],
            [
                "server.image is null; the external inference environment is "
                "not reproducibility-checked"
            ],
        )
    if not serve_command or not serve_command[0].endswith("run_vllm_tpu_container.sh"):
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
    if errors:
        return (errors, [])
    return ([], [])


def _turn_end_token_id(
    tokenizer_config: Mapping[str, Any], directory: Path, turn_end_token: str
) -> int | None:
    """Find the turn-end token's id from the export's tokenizer files.

    Checked in `tokenizer_config.json`'s `added_tokens_decoder` first, which is
    where a merged Qwen3 export carries it. `tokenizer.json`'s `added_tokens`
    is the fallback for an export that omits it there. Returns None -- rather
    than guessing an id -- when neither file names the token.
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


def check_export_dir(
    model_path: str, turn_end_token: str
) -> tuple[list[str], list[str]]:
    """Check an exported checkpoint for what vLLM needs to serve it as chat.

    Returns (errors, warnings). A missing chat template is an error rather than
    a warning: without it the server falls back to raw completion, and every
    prompt then reaches the model in a format it was never trained on. Turn
    termination is checked the same way: vLLM never stops on a stop *string*
    matching the turn-end token -- it matches decoded text with special tokens
    stripped, so the string can never fire on the real token -- so the setting
    that actually governs it, the export's `generation_config.json`, is
    checked directly and any problem with it is an error rather than a
    warning.
    """
    errors: list[str] = []
    warnings: list[str] = []
    directory = Path(model_path).expanduser()
    if not directory.is_dir():
        return ([f"server.model_path is not a directory: {directory}"], warnings)

    for name in REQUIRED_FILES:
        if not (directory / name).is_file():
            errors.append(f"export is missing {name}")
    if not any((directory / name).is_file() for name in WEIGHT_FILES):
        errors.append("export has no model.safetensors or model.safetensors.index.json")

    tokenizer_config_path = directory / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        return (errors, warnings)

    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text("utf-8"))
    except ValueError as error:
        errors.append(f"tokenizer_config.json is not valid JSON: {error}")
        return (errors, warnings)

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
        return (errors, warnings)

    generation_config_path = directory / "generation_config.json"
    if not generation_config_path.is_file():
        errors.append(
            "export has no generation_config.json, so vLLM falls back to the "
            f"tokenizer's own EOS rather than {turn_end_token!r}; every "
            "benchmark number from it would run past the turn boundary"
        )
        return (errors, warnings)
    try:
        generation_config = json.loads(generation_config_path.read_text("utf-8"))
    except ValueError as error:
        errors.append(f"generation_config.json is not valid JSON: {error}")
        return (errors, warnings)

    eos_token_id = generation_config.get("eos_token_id")
    eos_ids = eos_token_id if isinstance(eos_token_id, list) else [eos_token_id]
    if turn_end_id not in eos_ids:
        errors.append(
            f"generation_config.json's eos_token_id ({eos_token_id!r}) does "
            f"not include {turn_end_token!r}'s id ({turn_end_id}); the export "
            "will not stop at turn boundaries and every benchmark number from "
            "it would be invalid"
        )
    return (errors, warnings)


def main() -> None:
    args = parse_recipe_args(recipe_parser(__doc__))
    settings = resolve_settings(load_eval_config(args.config, args.overrides))

    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(check_dependency_versions())

    export_errors, export_warnings = check_export_dir(
        settings["model_path"], settings["turn_end_token"]
    )
    errors.extend(export_errors)
    warnings.extend(export_warnings)

    pack_errors, pack_warnings = verify_task_specs(
        DEFAULT_TASKPACK_PATH, settings["tasks"]
    )
    errors.extend(pack_errors)
    warnings.extend(pack_warnings)

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
        f"LightEval {_version('lighteval')}"
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
