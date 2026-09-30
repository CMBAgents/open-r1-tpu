"""Per-document records and the summary reduced from them.

Every evaluation writes one JSONL file per `(seed, task)`,
`output_dir/seed-{seed}/{task}.jsonl`, one record per document (`ok_record`).
This module owns that format and reduces it: each task's metrics with the
metric's own LightEval `corpus_level_fn` (never a hand-rolled mean), the
generation statistics that diagnose what accuracy alone cannot (truncation,
closed reasoning, answer markers, completion length), mean and standard
deviation across seeds, and any cons@n the recipe asks for
(`evaluation.consensus`). The summary goes to JSON, locally or on GCS, and to
W&B.

WHY SEEDS ARE MANDATORY. Seed variance alone moves small reasoning benchmarks
by 5-15 points (arXiv 2504.07086), which is more than most recipe changes are
worth, so every task runs once per seed and is reported as mean and standard
deviation. One seed reports a null standard deviation, not a reassuring 0.0.
"""

from __future__ import annotations

import json
import logging
import platform
import statistics
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from open_r1_tpu.evaluation.config import DEFAULT_SERVE_COMMAND, task_slug
from open_r1_tpu.evaluation.consensus import consensus_metrics
from open_r1_tpu.evaluation.server import vllm_serve_command
from open_r1_tpu.evaluation.stack import (
    EVALUATION_PACKAGE_VERSIONS,
    VLLM_TPU_BASE_IMAGE,
    vllm_tpu_image_tag,
)

LOGGER = logging.getLogger(__name__)


def jsonl_path(output_dir: Path, seed: int, task: str) -> Path:
    return output_dir / f"seed-{seed}" / f"{task_slug(task)}.jsonl"


def write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str))
            handle.write("\n")


def ok_record(
    *,
    doc_id: str,
    task: str,
    seed: int,
    gold: Any,
    query: Any,
    output: Mapping[str, Any],
    scores: Mapping[str, Any],
    failed_metrics: Sequence[str],
    scoring_errors: Mapping[str, str],
    trace_id: str | None,
) -> dict[str, Any]:
    """One scored document, in exactly the shape `evaluation.reduce` reads.

    `gold` and `query` make a reduction self-contained: they are what
    `evaluation.consensus` rebuilds a scoring `Doc` from for a cons@n winner.
    `specific` is deliberately not carried; see
    `consensus._score_consensus_document`.
    """
    return {
        "status": "ok",
        "doc_id": doc_id,
        "task": task,
        "seed": seed,
        "gold": gold,
        "query": query,
        "completion": output.get("text"),
        "finish_reason": output.get("finish_reason"),
        "prompt_tokens": output.get("prompt_tokens"),
        "completion_tokens": output.get("completion_tokens"),
        "latency_s": output.get("latency_s"),
        "attempts": output.get("attempts"),
        "scores": dict(scores),
        "failed_metrics": list(failed_metrics),
        "scoring_errors": dict(scoring_errors),
        "trace_id": trace_id,
    }


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


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def stack_versions() -> dict[str, str]:
    """Record the versions that give the numbers their meaning."""
    return {
        "python": platform.python_version(),
        **{name: _version(name) for name in EVALUATION_PACKAGE_VERSIONS},
    }


def aggregate_across_seeds(
    per_seed: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Reduce per-seed metrics to mean and standard deviation.

    The standard deviation is None at a single seed rather than 0.0. Reporting
    zero spread from one sample is the exact overclaim this pipeline exists to
    prevent.
    """
    tasks: dict[str, dict[str, list[float]]] = {}
    for seed in sorted(per_seed):
        for task, metrics in per_seed[seed].items():
            for name, value in metrics.items():
                tasks.setdefault(task, {}).setdefault(name, []).append(float(value))

    aggregated: dict[str, dict[str, dict[str, Any]]] = {}
    for task, metrics in tasks.items():
        aggregated[task] = {
            name: {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "n": len(values),
                "values": values,
            }
            for name, values in metrics.items()
        }
    return aggregated


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file from a local path or a GCS URI."""
    return json.loads(_read_text(str(path)))


def _read_text(path: str) -> str:
    if path.startswith("gs://"):
        import gcsfs

        with gcsfs.GCSFileSystem().open(path, "rt") as handle:
            return str(handle.read())
    return Path(path).expanduser().read_text(encoding="utf-8")


def write_summary(path: str, summary: Mapping[str, Any]) -> None:
    """Write the summary as JSON, locally or to GCS beside the checkpoint."""
    payload = json.dumps(summary, indent=2, sort_keys=True, default=str)
    if path.startswith("gs://"):
        import gcsfs

        with gcsfs.GCSFileSystem().open(path, "wt") as handle:
            handle.write(payload)
        return
    local = Path(path).expanduser()
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(payload, encoding="utf-8")


def build_summary(
    settings: Mapping[str, Any],
    per_seed_metrics: Mapping[int, Mapping[str, Mapping[str, float]]],
    per_seed_stats: Mapping[int, Mapping[str, Any]],
    server_provenance: Mapping[str, Any] | None = None,
    consensus: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the durable record of one evaluation.

    `consensus` is kept out of `tasks_metrics` on purpose. Every entry there
    is a mean and standard deviation *across* replicates; a cons@n number is
    a single value computed *from* all of them jointly and has no spread to
    report, so filing it alongside would invite reading a null standard
    deviation as one-replicate noise rather than as a category difference.
    `summary_rows` flattens both, so W&B still receives it.
    """
    generation: dict[str, Any] = {}
    for name in (
        "format_rate",
        "reasoning_closed_rate",
        "answer_marker_rate",
        "truncation_rate",
        "mean_completion_tokens",
        "mean_completion_chars",
    ):
        values = [
            float(stats[name])
            for stats in per_seed_stats.values()
            if stats.get(name) is not None
        ]
        generation[name] = {
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values),
        }

    local_image = vllm_tpu_image_tag()
    server_image = settings.get("server_image")
    image_provenance = (
        {
            "spec_tag": local_image,
            "image_id": (
                server_provenance.get("image_id") if server_provenance else None
            ),
            "base_image": VLLM_TPU_BASE_IMAGE,
            "service_versions": (
                server_provenance.get("service_versions") if server_provenance else None
            ),
        }
        if server_image == local_image
        else None
    )

    return {
        "tier": settings["tier"],
        "model_path": settings["model_path"],
        "served_model_name": settings["served_model_name"],
        "tasks": list(settings["tasks"]),
        "seeds": list(settings["seeds"]),
        # `seeds` indexes replicates; the backend rejects per-request seeds, so
        # an archived summary must not be read as reproducible sample-by-sample.
        "seeded_replicates": False,
        "sampling": {
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "max_new_tokens": settings["max_new_tokens"],
        },
        "max_samples": settings.get("max_samples"),
        "stack": stack_versions(),
        "serve_command": list(settings.get("serve_command", DEFAULT_SERVE_COMMAND)),
        "server_image": server_image,
        "server_image_provenance": image_provenance,
        "server_command": vllm_serve_command(settings),
        "tasks_metrics": aggregate_across_seeds(per_seed_metrics),
        "consensus": {task: dict(result) for task, result in (consensus or {}).items()},
        "generation": generation,
        "per_seed_generation": {str(k): v for k, v in per_seed_stats.items()},
    }


def summary_rows(summary: Mapping[str, Any]) -> list[list[Any]]:
    """Flatten a summary into one row per task and metric, for tabular logs."""
    rows: list[list[Any]] = []
    for task, metrics in sorted(summary.get("tasks_metrics", {}).items()):
        for name, stats in sorted(metrics.items()):
            rows.append(
                [
                    summary.get("tier"),
                    task,
                    name,
                    stats.get("mean"),
                    stats.get("std"),
                    stats.get("n"),
                ]
            )
    # Consensus rows carry no standard deviation (there is one value, not one
    # per replicate); the count column holds the vote width instead of a
    # replicate count, which is the number that makes `cons@64` mean what it
    # says.
    for task, result in sorted(summary.get("consensus", {}).items()):
        rows.append(
            [
                summary.get("tier"),
                task,
                result.get("name"),
                result.get("value"),
                None,
                result.get("n"),
            ]
        )
    return rows


def log_summary_to_wandb(
    summary: Mapping[str, Any], settings: Mapping[str, Any]
) -> None:
    """Attach the summary to the training run, or a standalone run.

    Passing `reporting.wandb.run_id` puts eval numbers on the same run as the
    loss curves, which is the only way to read them together. W&B resumes by
    id, not by name, so without an id this starts a separate run rather than
    silently appending to whichever run happens to share the name.
    """
    wandb_config = dict(settings.get("wandb", {}))
    if not wandb_config.get("enabled", False):
        return
    try:
        import wandb
    except ImportError:
        LOGGER.warning("wandb is not installed; skipping W&B logging")
        return

    run_id = wandb_config.get("run_id")
    run_name = wandb_config.get("run_name") or f"{settings['tier']}-eval"
    # No fallback for project_name or mode: validate_eval_config requires both
    # whenever reporting.wandb.enabled is true, so logging to the wrong project
    # or mode is a recipe mistake worth catching at load time, not a default
    # worth guessing here.
    init_kwargs: dict[str, Any] = {
        "project": wandb_config["project_name"],
        "mode": wandb_config["mode"],
        "job_type": wandb_config.get("job_type", "eval"),
    }
    for key in ("entity", "group", "tags"):
        if wandb_config.get(key) is not None:
            init_kwargs[key] = wandb_config[key]
    if run_id:
        init_kwargs["id"] = str(run_id)
        init_kwargs["resume"] = wandb_config.get("resume", "allow")
    else:
        init_kwargs["name"] = run_name
        LOGGER.info(
            "No reporting.wandb.run_id set; logging to a standalone run named "
            "%s rather than the training run",
            run_name,
        )

    run = wandb.init(**init_kwargs)
    try:
        flat = {
            f"eval/{summary['tier']}/{task}/{name}": stats["mean"]
            for task, metrics in summary.get("tasks_metrics", {}).items()
            for name, stats in metrics.items()
            if stats.get("mean") is not None
        }
        flat.update(
            {
                f"eval/{summary['tier']}/generation/{name}": stats["mean"]
                for name, stats in summary.get("generation", {}).items()
                if stats.get("mean") is not None
            }
        )
        # Summary rather than a stepped log: evaluation happens after the last
        # optimizer step, so it has no step of its own, and a stepped write
        # after resume would land on an arbitrary one.
        run.summary.update(flat)
        table = wandb.Table(
            columns=["tier", "task", "metric", "mean", "std", "seeds"],  # pyright: ignore[reportArgumentType]
            data=summary_rows(summary),
        )
        run.log({f"eval/{summary['tier']}/table": table})
    finally:
        run.finish()
