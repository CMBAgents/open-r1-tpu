"""Freeze LightEval's own task definitions into a committed, diffable spec.

The comparability spine for the Langfuse-native runner (`evaluation.runner`)
and scorer bridge (`evaluation.scoring`): everything that decides *what the
model is asked* and *how the answer is judged* is read once out of LightEval's
own `LightevalTaskConfig` -- the prompt function, the dataset coordinates, the
generation defaults, and the metric objects -- and never re-specified by hand.
`derive_taskpack` performs that read; `configs/taskpack.yaml` is its committed
output; `verify_task_specs` re-derives and diffs against that file at
preflight, so a LightEval upgrade that moves a prompt or a metric fails loudly
there rather than silently moving a headline number.

What this module does **not** do: reimplement prompt rendering or scoring. The
prompt function and metric objects a task pack entry names are *references* --
`derive_task_spec` records enough to describe them for review and drift
detection, but `resolve_task_configs` is what callers actually use to get live
objects, by asking the installed LightEval's own registry the same question
again. The committed file is documentation and a tripwire, not a serialization
format for Python callables.

Two things are deliberately excluded from the strict diff `verify_task_specs`
enforces:

- The rendered `example` block (one real dataset row's prompt, for human
  review) is best-effort. Some datasets this project evaluates against are
  gated on the Hub (`gpqa`), so deriving or verifying a task pack must still
  succeed for every other task when running unauthenticated or offline.  A
  fetch failure records `{"unavailable": <reason>}` instead of raising, and a
  mismatch here is a warning, never an error.
- Generation size and stop sequence are recorded for visibility but are never
  authoritative: the recipe's `sampling.max_new_tokens` always wins over a
  task's upstream `generation_size` (see `math_500`'s note below), and this
  project sends no stop sequences at all (`evaluation.run.vllm_serve_command`'s
  docstring explains why a stop *string* can never match the real EOS token).

Known upstream/recipe divergences, recorded rather than papered over:

- `math_500` carries `generation_size: 32768` upstream; every recipe using it
  sets a smaller `sampling.max_new_tokens` (the deliberate token budget for
  this project's TPU). The recipe wins.
- `math_500`'s metric is `pass@k:k=1&n=1` (`Metrics.pass_at_k_math` with
  `sample_params={"k": 1, "n": 1}`), not a plain extractive match -- do not
  substitute one for the other when reading or extending this pack.
- `olympiad_bench` is not in `KNOWN_TASKS`. Its `specific` struct is empty for
  every row, which broke LightEval's own Parquet detail write on this stack,
  and it is not part of the Open-R1 headline set.

Run from the repository root::

    python -m open_r1_tpu.evaluation.taskpack --derive
    python -m open_r1_tpu.evaluation.taskpack --verify
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from importlib import import_module
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import yaml

LOGGER = logging.getLogger(__name__)

# The complete task set this project evaluates against, gathered from every
# recipe's `eval.tasks` (see `git grep '^\s*- "' recipes/*/eval/*.yaml`).
# `--derive` with no `--tasks` freezes exactly this list, and
# `evaluation.preflight` verifies a recipe's own subset of it -- so a task
# used by a recipe but missing here is caught at derive time, not at
# preflight, by `derive_taskpack`'s own completeness check against a recipe
# (there is none built in; adding a task to a recipe means adding it here
# too, and CI has nothing to catch a forgotten one yet).
KNOWN_TASKS: tuple[str, ...] = (
    "gsm8k|0",
    "math_500|0",
    "aime24|0",
    "aime25|0",
    "ifeval|0",
    "gpqa:diamond|0",
)

DEFAULT_TASKPACK_PATH = "configs/taskpack.yaml"

# Fields diffed strictly: a mismatch here is a preflight error naming the key.
_STRICT_FIELDS = (
    "hf_repo",
    "hf_subset",
    "hf_revision",
    "hf_avail_splits",
    "evaluation_splits",
    "few_shots_split",
    "few_shots_select",
    "num_fewshots",
    "generation_size",
    "stop_sequence",
    "version",
    "prompt_function_ref",
    "metrics",
)


def _bare_name(task: str) -> str:
    """`"gpqa:diamond|0"` -> `"gpqa:diamond"`; the registry's `task_to_configs`
    key, which keeps a suite-style `subset` colon but drops the trailing
    `|num_fewshots`.
    """
    name, _, _ = task.partition("|")
    if not name:
        raise ValueError(f"task {task!r} is not in name|num_fewshot form")
    return name


@dataclass(frozen=True)
class MetricSpec:
    """A LightEval `Metric`, described well enough to review and detect drift
    -- never enough to reconstruct. Scoring always calls the live object
    `resolve_task_configs` returns, imported fresh from the installed
    `lighteval`; see the module docstring.
    """

    metric_name: Any  # str, or list[str] for a SampleLevelMetricGrouping
    category: str
    batched_compute: bool
    higher_is_better: Any  # bool, or dict[str, bool] for a grouping
    sample_level_fn_class: str
    corpus_level_fn: dict[str, str]


@dataclass(frozen=True)
class TaskSpec:
    """One task's frozen definition. See the module docstring for what is and
    is not authoritative here.
    """

    name: str
    hf_repo: str
    hf_subset: str
    hf_revision: str | None
    hf_avail_splits: list[str]
    evaluation_splits: list[str]
    few_shots_split: str | None
    few_shots_select: str | None
    num_fewshots: int
    generation_size: int | None
    stop_sequence: list[str]
    version: Any  # int for most tasks, but not declared as one upstream
    prompt_function_ref: str
    metrics: list[MetricSpec]
    example: dict[str, Any] = field(default_factory=dict)


def resolve_task_configs(tasks: Sequence[str]) -> dict[str, Any]:
    """Resolve every task string to its live `LightevalTaskConfig`, in one
    `Registry` construction. Raises naming the task on anything that does not
    resolve to exactly one config -- LightEval's own `task_to_configs` is a
    `defaultdict`, which would otherwise let a typo through as a silently
    empty list rather than a `KeyError`.
    """
    from lighteval.tasks.registry import Registry

    registry = Registry(tasks=",".join(tasks), load_multilingual=False)
    configs = dict(registry.task_to_configs)

    resolved: dict[str, Any] = {}
    for task in tasks:
        name = _bare_name(task)
        matches = configs.get(name, [])
        if not matches:
            raise ValueError(
                f"task {task!r} did not resolve in LightEval's registry "
                f"(known: {sorted(configs)})"
            )
        if len(matches) > 1:
            raise ValueError(
                f"task {task!r} resolved to {len(matches)} configs "
                f"({name!r} is ambiguous); name a fully-qualified subset"
            )
        resolved[task] = matches[0]
    return resolved


def _metric_spec(metric: Any) -> MetricSpec:
    cls = type(metric.sample_level_fn)
    return MetricSpec(
        metric_name=metric.metric_name,
        category=metric.category.value,
        batched_compute=bool(metric.batched_compute),
        higher_is_better=metric.higher_is_better,
        sample_level_fn_class=f"{cls.__module__}.{cls.__qualname__}",
        corpus_level_fn={
            name: getattr(fn, "__name__", repr(fn))
            for name, fn in metric.get_corpus_aggregations().items()
        },
    )


def _render_example(config: Any) -> dict[str, Any]:
    """Render one real dataset row's prompt, for human review and drift
    detection. Best-effort: see the module docstring for why a fetch failure
    here must not fail the whole derive.
    """
    split = (config.evaluation_splits or config.hf_avail_splits or (None,))[0]
    if split is None:
        return {"unavailable": "task declares no evaluation or available split"}
    try:
        from datasets import load_dataset

        dataset = load_dataset(
            config.hf_repo,
            config.hf_subset,
            split=f"{split}[:1]",
            revision=config.hf_revision,
        )
        row = dataset[0]
        doc = config.prompt_function(row, config.name)
    except Exception as error:  # noqa: BLE001 - any dataset/network/auth failure
        LOGGER.warning("Could not render an example for %s: %s", config.name, error)
        return {"unavailable": str(error)}
    return {
        "query": doc.query,
        "choices": list(doc.choices),
        "gold_index": doc.gold_index,
        "specific": doc.specific,
    }


def derive_task_spec(task: str, config: Any) -> TaskSpec:
    """Read one task's frozen fields off its live `LightevalTaskConfig`."""
    fn = config.prompt_function
    return TaskSpec(
        name=task,
        hf_repo=str(config.hf_repo),
        hf_subset=str(config.hf_subset),
        hf_revision=config.hf_revision,
        hf_avail_splits=list(config.hf_avail_splits),
        evaluation_splits=list(config.evaluation_splits),
        few_shots_split=config.few_shots_split,
        few_shots_select=config.few_shots_select,
        num_fewshots=int(config.num_fewshots),
        generation_size=config.generation_size,
        stop_sequence=list(config.stop_sequence or []),
        version=config.version,
        prompt_function_ref=f"{fn.__module__}:{fn.__qualname__}",
        metrics=[_metric_spec(metric) for metric in config.metrics],
        example=_render_example(config),
    )


def _spec_to_dict(spec: TaskSpec) -> dict[str, Any]:
    return asdict(spec)


def derive_taskpack(tasks: Sequence[str] = KNOWN_TASKS) -> dict[str, Any]:
    """Derive the complete task pack from the installed LightEval."""
    resolved = resolve_task_configs(tasks)
    return {
        "lighteval_version": _lighteval_version(),
        "tasks": {
            task: _spec_to_dict(derive_task_spec(task, config))
            for task, config in resolved.items()
        },
    }


def _lighteval_version() -> str:
    try:
        return importlib_metadata.version("lighteval")
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def write_taskpack(path: str | Path, pack: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(pack), handle, sort_keys=True, default_flow_style=False)


def load_taskpack(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Task pack at {path} must contain a mapping")
    return loaded


def _diff_strict(
    task: str, committed: Mapping[str, Any], derived: Mapping[str, Any]
) -> list[str]:
    errors = []
    for key in _STRICT_FIELDS:
        committed_value = committed.get(key)
        derived_value = derived.get(key)
        if committed_value != derived_value:
            errors.append(
                f"tasks.{task}.{key} moved: committed={committed_value!r} "
                f"derived={derived_value!r}"
            )
    return errors


def verify_task_specs(
    pack_path: str | Path, tasks: Sequence[str]
) -> tuple[list[str], list[str]]:
    """Re-derive `tasks` from the installed LightEval and diff each against
    the committed pack at `pack_path`. Returns `(errors, warnings)`: a
    structural mismatch (dataset coordinates, generation parameters, the
    prompt function reference, the metric references) is an error naming the
    exact key that moved; an `example` mismatch, or one side being unable to
    render its example at all, is a warning only -- see the module docstring.
    """
    try:
        committed = load_taskpack(pack_path)
    except (OSError, ValueError) as error:
        return ([f"could not read task pack at {pack_path}: {error}"], [])

    committed_tasks = committed.get("tasks", {})
    if not isinstance(committed_tasks, Mapping):
        return ([f"task pack at {pack_path} has no 'tasks' mapping"], [])

    errors: list[str] = []
    warnings: list[str] = []

    missing = [task for task in tasks if task not in committed_tasks]
    if missing:
        errors.append(
            f"task pack at {pack_path} does not cover {sorted(missing)}; "
            "run `python -m open_r1_tpu.evaluation.taskpack --derive`"
        )
        tasks = [task for task in tasks if task not in missing]
    if not tasks:
        return (errors, warnings)

    try:
        resolved = resolve_task_configs(tasks)
    except (ImportError, ValueError) as error:
        return ([*errors, f"could not re-derive task specs: {error}"], warnings)

    for task in tasks:
        derived = _spec_to_dict(derive_task_spec(task, resolved[task]))
        committed_spec = committed_tasks[task]
        if not isinstance(committed_spec, Mapping):
            errors.append(f"tasks.{task} in {pack_path} is not a mapping")
            continue
        errors.extend(_diff_strict(task, committed_spec, derived))

        committed_example = committed_spec.get("example", {})
        derived_example = derived.get("example", {})
        both_rendered = "unavailable" not in committed_example and (
            "unavailable" not in derived_example
        )
        if both_rendered and committed_example != derived_example:
            warnings.append(f"tasks.{task}.example moved (non-authoritative)")
        elif ("unavailable" in committed_example) != ("unavailable" in derived_example):
            warnings.append(
                f"tasks.{task}.example could not be compared "
                f"(committed unavailable={('unavailable' in committed_example)}, "
                f"derived unavailable={('unavailable' in derived_example)})"
            )

    committed_version = committed.get("lighteval_version")
    if committed_version != _lighteval_version():
        warnings.append(
            f"lighteval_version moved: committed={committed_version!r} "
            f"installed={_lighteval_version()!r} (dependency pin should catch "
            "this too; see evaluation.stack)"
        )
    return (errors, warnings)


def import_prompt_function(ref: str) -> Callable[[Mapping[str, Any], str], Any]:
    """Import a `"module:qualname"` prompt function reference.

    Used at generation time by `evaluation.runner`, never by this module's own
    derive/verify path, which reads `prompt_function` straight off the live
    `LightevalTaskConfig` instead.
    """
    module_name, sep, qualname = ref.partition(":")
    if not sep:
        raise ValueError(f"prompt function reference {ref!r} is not module:qualname")
    module = import_module(module_name)
    target: Any = module
    for part in qualname.split("."):
        target = getattr(target, part)
    return target


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--derive", action="store_true", help="Write the task pack")
    mode.add_argument(
        "--verify", action="store_true", help="Diff against the committed task pack"
    )
    parser.add_argument("--pack", default=DEFAULT_TASKPACK_PATH, help="Task pack path")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=list(KNOWN_TASKS),
        help="Task strings (default: every task this project evaluates)",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    if args.derive:
        pack = derive_taskpack(args.tasks)
        write_taskpack(args.pack, pack)
        print(f"Wrote {len(pack['tasks'])} task(s) to {args.pack}")
        return

    errors, warnings = verify_task_specs(args.pack, args.tasks)
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        raise SystemExit("Task pack verification failed:\n- " + "\n- ".join(errors))
    print(f"Task pack at {args.pack} matches the installed LightEval.")


if __name__ == "__main__":
    main()
