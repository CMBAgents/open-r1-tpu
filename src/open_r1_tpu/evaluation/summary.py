"""Per-document records and the summary reduced from them.

Each (seed, task) writes `output_dir/seed-{seed}/{task_slug}.jsonl`: one
`ok_record` per scored document, or a `failed` or `dropped` record carrying
its error. The reduction takes each task's metrics through the metric's own
LightEval `corpus_level_fn` (never a hand-rolled mean), adds generation
statistics that accuracy alone cannot diagnose (truncation, closed reasoning,
answer markers, completion length), reports mean and standard deviation
across seeds, and adds any cons@n the recipe asks for (`evaluation.consensus`).
The summary goes to JSON, locally or on GCS, and optionally to W&B.

Every task runs once per seed because seed variance alone moves small
reasoning benchmarks by 5-15 points (arXiv 2504.07086). A single seed reports
a null standard deviation, not 0.0.
"""

from __future__ import annotations

import json
import logging
import platform
import statistics
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from open_r1_tpu.core.packages import installed_version
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
    """One scored document's record. `gold` and `query` let
    `evaluation.consensus` rebuild a scoring `Doc` without Langfuse; `specific`
    is left out because lcb:codegeneration's holds every test case.
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
    """Read one (seed, task)'s records; a missing file is a named error."""
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
    """One seed's one task as `{metric_name: corpus_value}`: each metric's own
    `corpus_level_fn` over the documents that have a value for it. A failed
    document or a `None` score is skipped, not counted as zero.
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
    """Generation statistics over one seed's records. Rates are over the
    completions; `truncation_rate` is the share of those reporting a
    `finish_reason` that report `"length"`.
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
    """One seed's `(metrics, stats)`, in the shape `build_summary` takes."""
    metrics: dict[str, dict[str, float]] = {}
    all_records: list[dict[str, Any]] = []

    for task in settings["tasks"]:
        path = jsonl_path(output_dir, seed, task)
        records = read_jsonl(path)
        all_records.extend(records)

        task_metrics = reduce_task_metrics(records, resolved_configs[task].metrics)
        if not task_metrics:
            raise ValueError(
                f"seed {seed} task {task!r} produced no scored documents "
                f"(read {len(records)} record(s) from {path})"
            )
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
    """Reduce every seed's records into the summary. `resolved_configs` is
    `tasks.resolve_task_configs(settings["tasks"])`.
    """
    output_path = Path(output_dir)
    per_seed_metrics: dict[int, dict[str, dict[str, float]]] = {}
    per_seed_stats: dict[int, dict[str, Any]] = {}
    for seed in settings["seeds"]:
        metrics, stats = reduce_seed(settings, seed, resolved_configs, output_path)
        per_seed_metrics[seed] = metrics
        per_seed_stats[seed] = stats

    return build_summary(
        settings,
        per_seed_metrics,
        per_seed_stats,
        server_provenance,
        # A vote needs every replicate of a document, so it runs after the
        # last seed, over the same records.
        consensus=consensus_metrics(settings, resolved_configs, output_path),
    )


def stack_versions() -> dict[str, str]:
    """Record the versions that give the numbers their meaning."""
    return {
        "python": platform.python_version(),
        **{name: installed_version(name) for name in EVALUATION_PACKAGE_VERSIONS},
    }


def aggregate_across_seeds(
    per_seed: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Reduce per-seed metrics to mean and standard deviation, which is None
    at a single seed rather than a misleading 0.0.
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

    `consensus` is filed apart from `tasks_metrics`: those are means across
    replicates, while a cons@n is one value computed from all of them, with no
    spread to report.
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
        "truncation_rate_source": "finish_reason",
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
    # A consensus row has no standard deviation; its count is the vote width.
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
    # validate_eval_config requires project_name and mode when enabled.
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
        # Summary rather than a stepped log: evaluation has no optimizer step of
        # its own, and a stepped write after resume would land on an arbitrary
        # one.
        run.summary.update(flat)
        table = wandb.Table(
            columns=["tier", "task", "metric", "mean", "std", "seeds"],  # pyright: ignore[reportArgumentType]
            data=summary_rows(summary),  # pyright: ignore[reportArgumentType]
        )
        run.log({f"eval/{summary['tier']}/table": table})
    finally:
        run.finish()
