"""Preflight the evaluation stack before committing TPU time to a benchmark.

The training preflight in `check_env.py` validates the Tunix/JAX stack. This
validates the serving side: the LightEval harness, and the exported checkpoint
it will be pointed at. vLLM itself is not checked for here, because it runs
outside this environment -- tpu-inference does not support this project's
Python -- and whether it is reachable is a question for the server, not an
import.

It exists because the failures worth catching here are silent, expensive, or
both. A merged export missing its tokenizer files or its chat
template loads far enough to serve requests and then answers off-distribution,
producing a benchmark number that measures the wrong thing. And Qwen3-Base
names `<|endoftext|>` as its EOS while the chat template closes turns with
`<|im_end|>`, so a server left to the tokenizer's own EOS runs past the end of
every reply and writes the user's next turn as well -- which under a benchmark
looks like a model that cannot stop reasoning. Task names are checked here
too. That failure is loud rather than silent, but LightEval moves tasks between
suites and releases and a recipe naming one that no longer exists is worth
knowing before the server spends fifteen minutes loading weights.

Run from the repository root::

    python -m open_r1_tpu.check_eval_env \
      --config recipes/Qwen3-1.7B-Math/eval/tier0_smoke.yaml
"""

from __future__ import annotations

import argparse
import difflib
import json
from importlib import metadata
from pathlib import Path
from typing import Any

from open_r1_tpu.evaluate import load_eval_config, resolve_settings

DEFAULT_CONFIG = "recipes/Qwen3-1.7B-Math/eval/tier0_smoke.yaml"

# Files a merged export needs before vLLM can serve it as a chat model.
REQUIRED_FILES = ("config.json", "tokenizer_config.json")
# Any one of these carries the weights.
WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json")
# Qwen3 closes a chat turn with this; see the module docstring.
TURN_END_TOKEN = "<|im_end|>"


def registry_task_names() -> set[str] | None:
    """Every task LightEval can resolve here, as ``suite|name``.

    Returns None when the registry cannot be read. `_task_registry` is private
    to an upstream class, and `task_to_configs` -- the public-looking mapping
    beside it -- is empty, so there is no supported way to ask this question.
    A rename upstream must therefore degrade to "unchecked" rather than to
    "every task is missing", which would be a preflight that fails the moment
    LightEval is upgraded.

    Multilingual tasks are left unloaded on purpose: that tree calls
    `langcodes.language_name()` at import, which needs an optional package we
    do not install, and no recipe uses the suite.
    """
    try:
        from lighteval.tasks.registry import Registry
    except ImportError:
        return None
    for load_community in (True, False):
        try:
            registry = Registry(
                custom_tasks=None,
                load_community=load_community,
                load_extended=True,
                load_multilingual=False,
            )
        except Exception:  # noqa: BLE001 - any import in any task tree
            continue
        names = getattr(registry, "_task_registry", None)
        if isinstance(names, dict) and names:
            return set(names)
    return None


def check_task_names(
    tasks: list[str], known: set[str] | None = None
) -> tuple[list[str], list[str]]:
    """Check recipe task strings against what LightEval can actually resolve."""
    if known is None:
        known = registry_task_names()
    if known is None:
        return ([], ["could not read LightEval's task registry; task names unchecked"])

    errors: list[str] = []
    warnings: list[str] = []
    suites = {name.split("|", 1)[0] for name in known}
    for task in tasks:
        parts = str(task).split("|")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            errors.append(f"task {task!r} is not in suite|name|few_shot|truncate form")
            continue
        suite, name = parts[0], parts[1]
        if f"{suite}|{name}" in known:
            continue
        if suite not in suites:
            warnings.append(
                f"task {task!r}: suite {suite!r} is not loaded here, so the "
                "name is unchecked"
            )
            continue
        errors.append(
            f"task {task!r} is not in LightEval's registry{_hint(name, known)}"
        )
    return (errors, warnings)


def _hint(name: str, known: set[str]) -> str:
    """Suggest what a missing task was probably meant to be.

    Same name in another suite first: LightEval splits tasks across `lighteval`
    and `extended` with no rule you can infer from the name, so naming the
    right task in the wrong suite is the likeliest mistake by some margin.
    """
    elsewhere = sorted(other for other in known if other.split("|", 1)[1] == name)
    close = difflib.get_close_matches(name, [k.split("|", 1)[1] for k in known], n=3)
    suggestions = (
        elsewhere
        or sorted({other for other in known if other.split("|", 1)[1] in close})[:3]
    )
    return f"; did you mean {', '.join(suggestions)}?" if suggestions else ""


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def check_export_dir(model_path: str) -> tuple[list[str], list[str]]:
    """Check an exported checkpoint for what vLLM needs to serve it as chat.

    Returns (errors, warnings). A missing chat template is an error rather than
    a warning: without it the server falls back to raw completion, and every
    prompt then reaches the model in a format it was never trained on.
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

    eos_token = tokenizer_config.get("eos_token")
    if isinstance(eos_token, dict):
        eos_token = eos_token.get("content")
    if eos_token and eos_token != TURN_END_TOKEN:
        warnings.append(
            f"tokenizer EOS is {eos_token!r}, not {TURN_END_TOKEN!r}: pass "
            f"--generation-config or a stop token so generation ends at the "
            "turn boundary rather than running into the next turn"
        )
    return (errors, warnings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    settings = resolve_settings(load_eval_config(args.config, args.overrides))

    errors: list[str] = []
    warnings: list[str] = []

    if _version("lighteval") == "unknown":
        errors.append(
            "lighteval is not installed; install the evaluation extra with "
            "`pip install -e '.[eval]'`"
        )
    # vLLM is not expected in this environment -- it cannot be, on this Python
    # -- so its absence here is not an error. Whether the server is reachable
    # is answered by the server itself, not by an import.
    if _version("math-verify") == "unknown":
        warnings.append(
            "math-verify is not installed; maths tasks fall back to string "
            "matching, which understates accuracy substantially"
        )

    try:
        import jax

        devices: list[Any] = jax.devices()
    except ImportError:
        errors.append("JAX is not installed; vLLM cannot reach the TPU without it")
        devices = []
    else:
        non_tpu = [str(device) for device in devices if device.platform != "tpu"]
        if not devices:
            errors.append("JAX sees no devices at all")
        elif non_tpu:
            errors.append(f"non-TPU JAX devices detected: {non_tpu}")

    export_errors, export_warnings = check_export_dir(settings["model_path"])
    errors.extend(export_errors)
    warnings.extend(export_warnings)

    task_errors, task_warnings = check_task_names(settings["tasks"])
    errors.extend(task_errors)
    warnings.extend(task_warnings)

    if str(settings["summary_path"]).startswith("gs://"):
        try:
            import gcsfs  # noqa: F401
        except ImportError:
            errors.append("writing the summary to GCS requires the gcsfs package")

    print(f"LightEval {_version('lighteval')}")
    print(f"Devices ({len(devices)}): {devices}")
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
