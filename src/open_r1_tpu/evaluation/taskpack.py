"""Freeze LightEval's own task definitions into a committed, diffable spec.

What the model is asked and how its answer is judged -- the prompt function,
dataset coordinates, generation defaults and metrics -- is read from
LightEval's `LightevalTaskConfig`, never re-specified by hand.
`derive_taskpack` performs that read and `configs/taskpack.yaml` is its
committed output; `verify_task_specs` re-derives and diffs against it at
preflight, so a LightEval upgrade that moves a prompt or a metric fails loudly
instead of silently moving a headline number. The file is for review and drift
detection only: callers get live objects from `resolve_task_configs`.

Two things are not strictly diffed:

- The rendered `example` (one real row's prompt, for review) is best-effort.
  Some datasets are gated on the Hub (gpqa), so a fetch failure records
  `{"unavailable": <reason>}` and an `example` mismatch is only a warning.
- `generation_size` and `stop_sequence` are recorded but never used: the
  recipe's `sampling.max_new_tokens` always wins, and no stop sequences are
  sent (a stop string cannot end a turn; see `evaluation.preflight`).

Known divergences from upstream:

- `math_500` declares `generation_size: 32768`; recipes set a smaller
  `sampling.max_new_tokens`, and the recipe wins. Its metric is
  `pass@k:k=1&n=1`, not a plain extractive match.
- `lcb:codegeneration` is LightEval's `v4_v5` subset (problems from 2024-08 to
  2025-01, the window of DeepSeek's published LiveCodeBench number). Upstream
  names every other subset `lcb:codegeneration_{subset}`, so the bare name
  cannot quietly move to another window without a strict diff.

Run from the repository root::

    python -m open_r1_tpu.evaluation.taskpack --derive
    python -m open_r1_tpu.evaluation.taskpack --verify
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

import yaml

from open_r1_tpu.core.logging import LOG_LEVELS, configure_logging
from open_r1_tpu.core.packages import installed_version

LOGGER = logging.getLogger(__name__)

# Every task the committed recipes evaluate; `--derive` with no `--tasks`
# freezes exactly these. A task added to a recipe must be added here too, or
# preflight fails on a task the pack does not cover.
KNOWN_TASKS: tuple[str, ...] = (
    "gsm8k|0",
    "math_500|0",
    "aime24|0",
    "aime25|0",
    "ifeval|0",
    "gpqa:diamond|0",
    "lcb:codegeneration|0",
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
    """`"gpqa:diamond|0"` -> `"gpqa:diamond"`, the registry's
    `task_to_configs` key.
    """
    name, _, _ = task.partition("|")
    if not name:
        raise ValueError(f"task {task!r} is not in name|num_fewshot form")
    return name


@dataclass(frozen=True)
class MetricSpec:
    """A LightEval `Metric`, described for review and drift detection, not
    for reconstruction.
    """

    metric_name: Any  # str, or list[str] for a SampleLevelMetricGrouping
    category: str
    batched_compute: bool
    higher_is_better: Any  # bool, or dict[str, bool] for a grouping
    sample_level_fn_class: str
    corpus_level_fn: dict[str, str]


@dataclass(frozen=True)
class TaskSpec:
    """One task's frozen definition."""

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
    """Resolve every task string to its live `LightevalTaskConfig`. Raises
    naming a task that does not resolve to exactly one config (the registry's
    `defaultdict` would let a typo through as an empty list).
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


# A rendered example's `specific` above this size is summarised rather than
# committed: `lcb:codegeneration`'s holds every test case for the problem,
# tens of megabytes for one row.
_EXAMPLE_SPECIFIC_MAX_CHARS = 2_000


def _example_specific(specific: Any) -> Any:
    size = len(json.dumps(specific, sort_keys=True, default=str))
    if size <= _EXAMPLE_SPECIFIC_MAX_CHARS:
        return specific
    keys = sorted(specific) if isinstance(specific, Mapping) else None
    return {"omitted_chars": size, "keys": keys}


def _render_example(config: Any) -> dict[str, Any]:
    """One real dataset row's prompt, for review. Best-effort: a fetch
    failure is recorded, not raised.
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
        "specific": _example_specific(doc.specific),
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


# What a document asks and how its answer is judged. The recipe overrides
# generation size and stop sequence, and no task varies its splits, few-shot
# settings or version, so those stay out of the fingerprint.
_FINGERPRINT_FIELDS = (
    "hf_repo",
    "hf_subset",
    "hf_revision",
    "prompt_function_ref",
    "metrics",
)


def dataset_fingerprint(spec: TaskSpec) -> str:
    """First 8 hex characters of a sha256 over `spec`'s `_FINGERPRINT_FIELDS`.
    A change to one of them starts a new Langfuse dataset; a change to
    anything else keeps comparing against the old one.
    """
    as_dict = asdict(spec)
    payload = {field_name: as_dict[field_name] for field_name in _FINGERPRINT_FIELDS}
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return sha256(canonical.encode("utf-8")).hexdigest()[:8]


def dataset_name(task: str, spec: TaskSpec, max_samples: int | None = None) -> str:
    """The Langfuse dataset name: `{task}@{fingerprint}`, plus `[:N]` when the
    recipe caps the task at `N` documents, since `run_experiment` scores
    every item a dataset holds.
    """
    name = f"{task}@{dataset_fingerprint(spec)}"
    return f"{name}[:{max_samples}]" if max_samples else name


def derive_taskpack(tasks: Sequence[str] = KNOWN_TASKS) -> dict[str, Any]:
    """Derive the complete task pack from the installed LightEval."""
    resolved = resolve_task_configs(tasks)
    return {
        "lighteval_version": installed_version("lighteval"),
        "tasks": {
            task: asdict(derive_task_spec(task, config))
            for task, config in resolved.items()
        },
    }


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
    the pack at `pack_path`. Returns `(errors, warnings)`: a strict field that
    moved is an error naming the key; an `example` that moved, or rendered on
    only one side, is a warning.
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
        derived = asdict(derive_task_spec(task, resolved[task]))
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
    installed = installed_version("lighteval")
    if committed_version != installed:
        warnings.append(
            f"lighteval_version moved: committed={committed_version!r} "
            f"installed={installed!r} (dependency pin should catch "
            "this too; see evaluation.stack)"
        )
    return (errors, warnings)


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
    parser.add_argument(
        "--log-level", default="info", choices=sorted(LOG_LEVELS), type=str.lower
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging(LOG_LEVELS[args.log_level])
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
