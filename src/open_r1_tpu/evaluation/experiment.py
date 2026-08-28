"""Task 4's experiment entry point: `dataset.run_experiment()` per
`(task, seed)`, replacing `evaluation.runner.run_async`/`run_seed_task`.

Langfuse drives the loop from here down: iteration, concurrency, and
per-item tracing are `dataset.run_experiment()`'s
(`evaluation.task_fn.make_task`, `evaluation.scoring.lighteval_evaluator`).
This module is the thin layer above that -- one dataset per task
(`evaluation.dataset_sync` must have already synced it), one experiment run
per `(task, seed)` so seed variance stays visible in Langfuse's comparison
view rather than being averaged away before it arrives, and one JSONL file
per `(seed, task)` written straight from the returned `ExperimentResult` --
never by reading anything back from Langfuse, so `evaluation.reduce`'s
summary and the W&B log stay independent of whether Langfuse is even
reachable afterwards.

**Call `dataset.run_experiment()` only from plain synchronous code, never
from inside an already-running event loop.** Its own
`run_async_safely` (`langfuse._client.utils`) takes `asyncio.run()`'s fast
path -- executing on the calling thread -- only when no loop is already
running; otherwise it spins up a second thread and runs there instead. This
module's own `run()` is synchronous for exactly this reason: it keeps every
evaluator call (`evaluation.scoring.lighteval_evaluator`, and so
`compute_scores`) on the main thread of the main interpreter, which is where
a LightEval metric's own `signal.alarm`-based timeout can arm at all (see
`evaluation.scoring.compute_scores`'s docstring). Task 8's tier-0 checklist
is where this gets a real smoke test; nothing here can prove it without a
live Langfuse and a live server.

**`server.fail_fast_after` is kept, enforced inside the task function.**
`run_experiment` has no run-abort hook a task or evaluator can reach (see
`evaluation.task_fn`'s module docstring) -- there is no direct replacement
for the old runner's `asyncio.TaskGroup` cancellation. `evaluation.task_fn`'s
circuit breaker is the closest available substitute: not a cancellation, but
every item after the budget trips fails before attempting a request. Kept
rather than dropped, because silently ignoring a required recipe key is
worse than an honest, weaker substitute.

**`ExperimentResult.item_results` omits any item whose task function
raised** (`asyncio.gather(..., return_exceptions=True)` drops it, logging
only to Langfuse's own logger) -- there is no per-item error message
available from here. `write_experiment_jsonl` diffs the dataset's full item
list against what came back and writes a `status: "dropped"` record for the
difference, so `evaluation.reduce`'s document counts still add up; it just
cannot say *why* a dropped item failed. `evaluation.task_fn`'s own tests
cover why (`GenerationRefused`/`GenerationFailed`) at the task-function level
directly, where the real exception is still available.
"""

from __future__ import annotations

import json
import logging
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import openai

from open_r1_tpu.evaluation import scoring
from open_r1_tpu.evaluation.run import task_slug
from open_r1_tpu.evaluation.task_fn import make_task
from open_r1_tpu.evaluation.taskpack import (
    dataset_name as taskpack_dataset_name,
)
from open_r1_tpu.evaluation.taskpack import (
    derive_task_spec,
    resolve_task_configs,
)

LOGGER = logging.getLogger(__name__)

# Evaluations `lighteval_evaluator` posts that are not a LightEval metric
# name -- run-level facts and the failure marker -- excluded from a JSONL
# record's `scores` dict, matching `runner._score_and_post_document`'s own
# separation of `result.scores` from `run_level_fields`.
_RUN_LEVEL_EVALUATION_NAMES = frozenset(
    {"completion_tokens", "truncated", "scoring_failed"}
)


def _git_commit() -> str | None:
    """Best-effort provenance: `None` outside a git checkout or without
    `git` on `PATH`, never a hard failure -- this is metadata, not a
    correctness input.
    """
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _record_from_item_result(
    item_result: Any, *, task: str, seed: int
) -> dict[str, Any]:
    """One JSONL record in exactly the shape `evaluation.reduce` reads --
    reproducing `runner._score_and_post_document`'s return from an
    `ExperimentItemResult` instead of from a live generation.
    """
    output = item_result.output if isinstance(item_result.output, Mapping) else {}
    evaluations_by_name = {
        evaluation.name: evaluation for evaluation in item_result.evaluations
    }
    scores = {
        name: evaluation.value
        for name, evaluation in evaluations_by_name.items()
        if name not in _RUN_LEVEL_EVALUATION_NAMES
    }

    scoring_failed = evaluations_by_name.get("scoring_failed")
    failed_metrics: list[str] = []
    scoring_errors: dict[str, str] = {}
    if scoring_failed is not None and scoring_failed.metadata:
        failed_metrics = list(scoring_failed.metadata.get("failed_metrics", []))
        scoring_errors = dict(scoring_failed.metadata.get("errors", {}))

    return {
        "status": "ok",
        "doc_id": item_result.item.metadata["doc_id"],
        "task": task,
        "seed": seed,
        "completion": output.get("text"),
        "finish_reason": output.get("finish_reason"),
        "prompt_tokens": output.get("prompt_tokens"),
        "completion_tokens": output.get("completion_tokens"),
        "latency_s": output.get("latency_s"),
        "attempts": output.get("attempts"),
        "scores": scores,
        "failed_metrics": failed_metrics,
        "scoring_errors": scoring_errors,
        "trace_id": item_result.trace_id,
    }


def _dropped_record(item: Any, *, task: str, seed: int) -> dict[str, Any]:
    """A document `run_experiment` returned no result for at all -- see the
    module docstring's note on why no error message survives to here.
    """
    return {
        "status": "dropped",
        "doc_id": item.metadata["doc_id"],
        "task": task,
        "seed": seed,
        "error": (
            "run_experiment did not return a result for this item; see "
            "Langfuse's own log for the raised exception"
        ),
    }


def write_experiment_jsonl(
    result: Any, dataset: Any, *, task: str, seed: int, output_path: Path
) -> None:
    """Persist one `(seed, task)`'s `ExperimentResult` as the JSONL
    `evaluation.reduce.reduce_seed` reads -- one line per dataset item,
    whether `run_experiment` returned a result for it or dropped it.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = {item_result.item.id for item_result in result.item_results}
    with output_path.open("w", encoding="utf-8") as handle:
        for item_result in result.item_results:
            record = _record_from_item_result(item_result, task=task, seed=seed)
            handle.write(json.dumps(record, default=str))
            handle.write("\n")
        for item in dataset.items:
            if item.id not in seen_ids:
                record = _dropped_record(item, task=task, seed=seed)
                handle.write(json.dumps(record, default=str))
                handle.write("\n")


def run_experiment_for_task_seed(
    langfuse_client: Any,
    *,
    task: str,
    seed: int,
    settings: Mapping[str, Any],
    resolved_config: Any,
    client: Any,
    output_dir: Path,
    run_metadata: Mapping[str, Any],
) -> Any:
    """Run and persist one `(task, seed)`'s experiment against the dataset
    `evaluation.dataset_sync` already synced this task into.
    """
    spec = derive_task_spec(task, resolved_config)
    name = taskpack_dataset_name(task, spec)
    dataset = langfuse_client.get_dataset(name)

    result = dataset.run_experiment(
        name=f"{settings['served_model_name']}-{settings['tier']}-seed{seed}",
        task=make_task(settings, client=client),
        evaluators=[scoring.lighteval_evaluator(task)],
        max_concurrency=int(settings["max_concurrency"]),
        metadata={**run_metadata, "dataset_fingerprint": name.rsplit("@", 1)[-1]},
    )

    output_path = output_dir / f"seed-{seed}" / f"{task_slug(task)}.jsonl"
    write_experiment_jsonl(
        result, dataset, task=task, seed=seed, output_path=output_path
    )
    LOGGER.info(
        "seed %d %s -> %s (%d/%d item(s) scored)",
        seed,
        task,
        output_path,
        len(result.item_results),
        len(dataset.items),
    )
    return result


def run(
    settings: Mapping[str, Any], *, langfuse_client: Any, recipe_path: str | None = None
) -> Path:
    """Run every `(task, seed)` in `settings`, writing
    `output_dir/seed-{seed}/{task_slug}.jsonl`. `langfuse_client` is
    required, matching `evaluation.runner.run_async`: no default, so
    silently constructing one from ambient environment never hides which
    deployment's Langfuse a run actually talks to.
    """
    output_dir = Path(settings["output_dir"]).expanduser()
    client = openai.AsyncOpenAI(
        api_key="local",
        base_url=settings["base_url"],
        max_retries=0,  # evaluation.task_fn/runner own retries; see generate_one
    )
    resolved = resolve_task_configs(list(settings["tasks"]))
    run_metadata = {
        "recipe_path": recipe_path,
        "git_commit": _git_commit(),
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "max_new_tokens": settings["max_new_tokens"],
    }

    try:
        for seed in settings["seeds"]:
            for task in settings["tasks"]:
                run_experiment_for_task_seed(
                    langfuse_client,
                    task=task,
                    seed=seed,
                    settings=settings,
                    resolved_config=resolved[task],
                    client=client,
                    output_dir=output_dir,
                    run_metadata=run_metadata,
                )
    finally:
        import asyncio

        # No loop is running here -- see the module docstring -- so this
        # opens and closes its own, purely for cleanup, after every
        # run_experiment() call has already returned.
        asyncio.run(client.close())

    return output_dir


def _parse_args() -> Any:
    import argparse

    from open_r1_tpu.core.logging import LOG_LEVELS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML evaluation recipe")
    parser.add_argument(
        "--tracing-config", required=True, help="YAML tracing config (Langfuse section)"
    )
    parser.add_argument("--log-level", default="info", choices=sorted(LOG_LEVELS))
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    from open_r1_tpu.core.logging import LOG_LEVELS, configure_logging
    from open_r1_tpu.evaluation.reduce import build_summary_from_records
    from open_r1_tpu.evaluation.run import (
        container_image_provenance,
        load_eval_config,
        log_summary_to_wandb,
        resolve_settings,
        wait_for_server,
        write_summary,
    )
    from open_r1_tpu.tracing.config import build_langfuse_client, load_tracing_config

    args = _parse_args()
    configure_logging(LOG_LEVELS[args.log_level])
    settings = resolve_settings(load_eval_config(args.config, args.overrides))
    tracing_config = load_tracing_config(args.tracing_config)

    server_provenance = container_image_provenance(settings)
    wait_for_server(settings["base_url"], int(settings["startup_timeout_secs"]))

    langfuse_client = build_langfuse_client(tracing_config)
    output_dir = run(settings, langfuse_client=langfuse_client, recipe_path=args.config)

    resolved_configs = resolve_task_configs(list(settings["tasks"]))
    summary = build_summary_from_records(
        settings, resolved_configs, output_dir, server_provenance
    )
    write_summary(settings["summary_path"], summary)
    LOGGER.info("Wrote evaluation summary to %s", settings["summary_path"])

    for task, metrics in sorted(summary["tasks_metrics"].items()):
        for name, stats in sorted(metrics.items()):
            std = stats["std"]
            spread = f" +/- {std:.4f}" if std is not None else " (1 seed, no spread)"
            LOGGER.info("%s %s: %.4f%s", task, name, stats["mean"], spread)

    log_summary_to_wandb(summary, settings)


if __name__ == "__main__":
    main()
