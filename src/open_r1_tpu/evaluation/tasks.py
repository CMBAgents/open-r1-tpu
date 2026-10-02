"""Resolve a recipe's task strings to LightEval's own task definitions.

What the model is asked and how its answer is judged -- the dataset, prompt
function and metrics -- come from the installed LightEval, never re-specified
here. LightEval is pinned exactly (`evaluation.stack`), so those definitions
change only when the pin does. Two of their fields are ignored: the recipe's
`sampling.max_new_tokens` replaces `generation_size`, and `stop_sequence` is
never sent, since turns end on the export's EOS ids (`evaluation.preflight`).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from hashlib import sha256
from typing import Any, cast


def _bare_name(task: str) -> str:
    """`"gpqa:diamond|0"` -> `"gpqa:diamond"`, the registry's
    `task_to_configs` key.
    """
    name, _, _ = task.partition("|")
    if not name:
        raise ValueError(f"task {task!r} is not in name|num_fewshot form")
    return name


def resolve_task_configs(tasks: Sequence[str]) -> dict[str, Any]:
    """Resolve every task string to its live `LightevalTaskConfig`. Raises
    naming a task that does not resolve to exactly one config (the registry's
    `defaultdict` would let a typo through as an empty list).
    """
    from lighteval.tasks.registry import Registry

    registry = Registry(tasks=",".join(tasks), load_multilingual=False)
    # Annotated as one config per task, but built as a list per task.
    configs = cast("dict[str, list[Any]]", dict(registry.task_to_configs))

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


def _metric_fields(metric: Any) -> dict[str, Any]:
    cls = type(metric.sample_level_fn)
    return {
        "metric_name": metric.metric_name,
        "category": metric.category.value,
        "batched_compute": bool(metric.batched_compute),
        "higher_is_better": metric.higher_is_better,
        "sample_level_fn_class": f"{cls.__module__}.{cls.__qualname__}",
        "corpus_level_fn": {
            name: getattr(fn, "__name__", repr(fn))
            for name, fn in metric.get_corpus_aggregations().items()
        },
    }


def scoring_fields(config: Any) -> dict[str, Any]:
    """What a task's documents ask and how their answers are judged: the
    fields a Langfuse dataset is fingerprinted by.
    """
    fn = config.prompt_function
    return {
        "hf_repo": str(config.hf_repo),
        "hf_subset": str(config.hf_subset),
        "hf_revision": config.hf_revision,
        "prompt_function_ref": f"{fn.__module__}:{fn.__qualname__}",
        "metrics": [_metric_fields(metric) for metric in config.metrics],
    }


def dataset_fingerprint(fields: dict[str, Any]) -> str:
    """First 8 hex characters of a sha256 over `scoring_fields`. A change to
    one of them starts a new Langfuse dataset.
    """
    canonical = json.dumps(fields, sort_keys=True, default=str)
    return sha256(canonical.encode("utf-8")).hexdigest()[:8]


def dataset_name(task: str, config: Any, max_samples: int | None = None) -> str:
    """The Langfuse dataset name: `{task}@{fingerprint}`, plus `[:N]` when the
    recipe caps the task at `N` documents, since `run_experiment` scores
    every item a dataset holds.
    """
    name = f"{task}@{dataset_fingerprint(scoring_fields(config))}"
    return f"{name}[:{max_samples}]" if max_samples else name
