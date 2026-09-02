"""Point `evaluation.run.build_summary` at `evaluation.experiment`'s own
JSONL output.

`evaluation.run`'s reduction half -- `build_summary`, `aggregate_across_seeds`,
`log_summary_to_wandb` -- is harness-agnostic and unchanged. This module
produces the `(metrics, stats)` per seed that half expects, from
`evaluation.experiment`'s own JSONL records (one file per `(seed, task)`,
written straight from `dataset.run_experiment()`'s `ExperimentResult`)
instead of LightEval's own results JSON and Parquet detail shards.

Two deliberate improvements over how the pre-Langfuse pipeline reduced this:

- **Corpus-level aggregation reuses each metric's own `corpus_level_fn`**,
  read straight off the live `LightevalTaskConfig` (`resolve_task_configs`),
  rather than assuming every metric aggregates by a plain mean. Most do
  (`np.mean`), but `ifeval`'s `inst_level_*_acc` do not -- they flatten a
  list of per-document instruction results across the whole corpus first
  (`agg_inst_level_acc`) -- and calling LightEval's own function is what
  keeps this a bridge rather than a second implementation of its scoring.
- **cons@n is computed here too** (`evaluation.consensus`), and only here:
  it is the one metric that cannot be produced per replicate and averaged,
  because the vote is between replicates. `build_summary` files it under
  `summary["consensus"]`, apart from `tasks_metrics`, since it has no spread
  across seeds to report.
- **`truncation_rate` comes from `finish_reason`**, a fact the server states
  (`finish_reason == "length"`), not a `token_count >= max_new_tokens`
  inference -- the tier-0 gate's former "unmeasurable" result, now a real
  number.
"""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from open_r1_tpu.evaluation.consensus import consensus_metrics
from open_r1_tpu.evaluation.run import build_summary, task_slug

LOGGER = logging.getLogger(__name__)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read one seed/task's records. Missing or empty is reported by name --
    the run probably failed before writing anything for this task, or was
    killed before its first document completed."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"No runner output at {file_path}; the run probably failed "
            "before writing anything for this seed/task"
        )
    records = []
    with file_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def reduce_task_metrics(
    records: Sequence[Mapping[str, Any]], metrics: Sequence[Any]
) -> dict[str, float]:
    """One seed's one task, reduced to `{metric_name: corpus_value}`, the
    same shape LightEval's own results JSON produced per task.

    For each name a task's metrics declare, collects every document's raw
    value (skipping a document where that metric is absent or `None` --
    failed or legitimately unscored, per `evaluation.scoring.compute_scores`'s
    "never coerce absence to zero" rule) and reduces with that metric's own
    `corpus_level_fn`. Never a hand-rolled mean: see the module docstring.
    """
    reduced: dict[str, float] = {}
    for metric in metrics:
        for name, corpus_fn in metric.get_corpus_aggregations().items():
            values = [
                record["scores"][name]
                for record in records
                if record.get("status") == "ok"
                and record.get("scores", {}).get(name) is not None
            ]
            if not values:
                continue
            reduced[name] = float(corpus_fn(values))
    return reduced


def completion_stats_from_records(
    records: Iterable[Mapping[str, Any]],
    *,
    reasoning_start: str | None,
    reasoning_end: str,
    answer_marker: str,
) -> dict[str, Any]:
    """Generation-level statistics computed directly from `evaluation.experiment`'s
    JSONL records rather than a LightEval detail-Parquet cell, in the shape
    `build_summary` expects. The one substantive difference from how the
    pre-Langfuse pipeline computed these is `truncation_rate`; see the module
    docstring.
    """
    documents = 0
    completions = 0
    closed = 0
    marked = 0
    formatted = 0
    chars = 0
    token_counts: list[int] = []
    truncated = 0
    have_finish_reason = 0

    for record in records:
        documents += 1
        if record.get("status") != "ok":
            continue
        completions += 1
        text = str(record["completion"])
        chars += len(text)
        end = text.find(reasoning_end)
        if reasoning_start is None:
            # The chat template opened the block inside the prompt, so the
            # completion can only ever carry the closing tag.
            is_closed = end != -1
        else:
            start = text.find(reasoning_start)
            is_closed = start != -1 and end != -1 and end > start
        has_marker = answer_marker in text
        closed += int(is_closed)
        marked += int(has_marker)
        formatted += int(is_closed and has_marker)

        completion_tokens = record.get("completion_tokens")
        if completion_tokens is not None:
            token_counts.append(int(completion_tokens))

        finish_reason = record.get("finish_reason")
        if finish_reason is not None:
            have_finish_reason += 1
            if finish_reason == "length":
                truncated += 1

    def rate(count: int) -> float | None:
        return count / completions if completions else None

    return {
        "documents": documents,
        "completions": completions,
        "format_rate": rate(formatted),
        "reasoning_closed_rate": rate(closed),
        "answer_marker_rate": rate(marked),
        "mean_completion_chars": (chars / completions) if completions else None,
        "mean_completion_tokens": (
            statistics.fmean(token_counts) if token_counts else None
        ),
        "truncation_rate": (
            truncated / have_finish_reason if have_finish_reason else None
        ),
    }


def reduce_seed(
    settings: Mapping[str, Any],
    seed: int,
    resolved_configs: Mapping[str, Any],
    output_dir: Path,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """One seed's `(metrics, stats)`, in exactly the shape `build_summary`
    expects.
    """
    metrics: dict[str, dict[str, float]] = {}
    all_records: list[dict[str, Any]] = []

    for task in settings["tasks"]:
        path = output_dir / f"seed-{seed}" / f"{task_slug(task)}.jsonl"
        records = read_jsonl(path)
        all_records.extend(records)

        task_metrics = reduce_task_metrics(records, resolved_configs[task].metrics)
        if not task_metrics:
            raise ValueError(
                f"seed {seed} task {task!r} produced no scored documents "
                f"(read {len(records)} record(s) from {path})"
            )
        # Keyed by the recipe's own task string, one entry per task -- unlike
        # LightEval's own results JSON, which keys by whatever key the
        # harness happened to use and so needs a guard against two different
        # keys colliding. That indirection does not exist here: two tasks
        # reporting the same metric name (e.g. two maths tasks both
        # producing `extractive_match`) is expected and fine, since each
        # lands under its own task here.
        metrics[task] = task_metrics

    stats = completion_stats_from_records(
        all_records,
        reasoning_start=settings["reasoning_start"],
        reasoning_end=settings["reasoning_end"],
        answer_marker=settings["answer_marker"],
    )
    return metrics, stats


def build_summary_from_records(
    settings: Mapping[str, Any],
    resolved_configs: Mapping[str, Any],
    output_dir: str | Path,
    server_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read every seed's runner output and assemble the same durable summary
    `evaluation.run.build_summary` always has -- unchanged itself, only fed
    from a different source. `resolved_configs` is
    `evaluation.taskpack.resolve_task_configs(settings["tasks"])`'s result.
    """
    output_path = Path(output_dir)
    per_seed_metrics: dict[int, dict[str, dict[str, float]]] = {}
    per_seed_stats: dict[int, dict[str, Any]] = {}
    for seed in settings["seeds"]:
        metrics, stats = reduce_seed(settings, seed, resolved_configs, output_path)
        per_seed_metrics[seed] = metrics
        per_seed_stats[seed] = stats

    summary = build_summary(
        settings,
        per_seed_metrics,
        per_seed_stats,
        server_provenance,
        # Computed here rather than during the run: a consensus needs every
        # replicate of a document at once, which only exists once the last
        # seed has finished. It reads the same JSONL files the loop above
        # does, so a killed-and-resumed tier reduces to the same number.
        consensus=consensus_metrics(settings, resolved_configs, output_path),
    )
    # `evaluation.run.build_summary` has no provenance field of its own for
    # this; recorded here so a reader of the summary JSON does not have to
    # already know which code path produced truncation_rate to trust it.
    summary["truncation_rate_source"] = "finish_reason"
    return summary
