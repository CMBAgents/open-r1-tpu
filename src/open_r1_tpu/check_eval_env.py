"""Preflight the evaluation stack before committing TPU time to a benchmark.

The training preflight in `check_env.py` validates the Tunix/JAX stack. This
validates the serving side: the LightEval harness, the exported checkpoint it
will be pointed at, and the recipe's task names. Neither vLLM nor the TPU is
checked for here. Both live on the other side of the HTTP boundary -- vLLM runs
in its own environment because tpu-inference does not support this project's
Python, and it holds the chip while it serves, so a preflight that called
`jax.devices()` would fail precisely when the server was up and working.

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
import inspect
import json
from importlib import metadata
from pathlib import Path

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

    The constructor's keywords move between releases: 0.11 took
    `load_community` and `load_extended`, 0.13 takes neither. Only the ones
    this installation accepts are passed, so the check survives the next
    change rather than needing one of its own.

    Multilingual tasks stay unloaded on purpose: that tree calls
    `langcodes.language_name()` at import, which needs an optional package we
    do not install, and no recipe uses the suite.
    """
    try:
        from lighteval.tasks.registry import Registry
    except ImportError:
        return None

    wanted = {
        "custom_tasks": None,
        "load_multilingual": False,
        "load_community": False,
        "load_extended": True,
    }
    try:
        accepted = inspect.signature(Registry).parameters
        registry = Registry(**{k: v for k, v in wanted.items() if k in accepted})
    except Exception:  # noqa: BLE001 - any import in any task tree
        return None

    names = getattr(registry, "_task_registry", None)
    if not isinstance(names, dict) or not names:
        return None
    return set(names)


def check_task_names(
    tasks: list[str], known: set[str] | None = None
) -> tuple[list[str], list[str]]:
    """Check recipe task strings against what LightEval can actually resolve.

    A task name is ``name|num_fewshot``. Both older spellings are reported
    rather than passed through: 0.11 deprecated the trailing truncate_fewshot
    field and 0.13 removed it, and 0.13 also stopped keying its registry by
    suite, so a leading ``lighteval|`` is now discarded in silence. Neither
    surfaces until the harness has already started.
    """
    if known is None:
        known = registry_task_names()
    if known is None:
        return ([], ["could not read LightEval's task registry; task names unchecked"])

    errors: list[str] = []
    warnings: list[str] = []
    for task in tasks:
        parts = str(task).split("|")
        if len(parts) > 3:
            errors.append(
                f"task {task!r} has a trailing field LightEval removed in "
                f"0.13; use {'|'.join(parts[1:3])!r}"
            )
            continue
        if len(parts) == 3:
            warnings.append(
                f"task {task!r} carries a suite prefix LightEval 0.13 ignores; "
                f"use {'|'.join(parts[1:])!r}"
            )
            name = parts[1]
        elif len(parts) == 2:
            name = parts[0]
        else:
            errors.append(f"task {task!r} is not in name|num_fewshot form")
            continue

        if not name:
            errors.append(f"task {task!r} is not in name|num_fewshot form")
        elif name not in known:
            errors.append(
                f"task {task!r} is not in LightEval's registry{_hint(name, known)}"
            )
    return (errors, warnings)


def _hint(name: str, known: set[str]) -> str:
    """Suggest what a missing task was probably meant to be."""
    close = difflib.get_close_matches(name, sorted(known), n=3)
    return f"; did you mean {', '.join(close)}?" if close else ""


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
    if _version("latex2sympy2-extended") == "unknown":
        warnings.append(
            "latex2sympy2-extended is not installed; maths tasks fall back to "
            "string matching, which understates accuracy substantially"
        )

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
