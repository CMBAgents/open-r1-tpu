"""Evaluate a served model on a recipe's tasks: generate, score, write JSONL.

Every `(task, seed)` produces one file, `output_dir/seed-{seed}/{task}.jsonl`,
with one record per document in the shape `evaluation.reduce` turns into the
summary. Two paths write it:

- **Local (the default).** Documents are generated concurrently over the
  `openai` SDK (`server.max_concurrency` in flight), then scored with the
  task's own LightEval metrics. Nothing else needs to be running.
- **Langfuse (`--tracing-config`).** The recipe's tasks are synced into
  Langfuse datasets (`evaluation.dataset_sync`), then
  `dataset.run_experiment()` drives generation and scoring per
  `(task, seed)`, so every document is traced and each seed is its own run in
  Langfuse's comparison view. The JSONL is written from the returned
  `ExperimentResult`, never read back from Langfuse, so the summary does not
  depend on Langfuse staying up.

Both paths use the same prompt rendering (`runner.render_messages`), request
and retry policy (`task_fn.make_task`), and metrics
(`scoring.compute_scores`), so a local number and a traced number agree.

**Scoring runs on the main thread.** A LightEval maths metric guards its
symbolic check with a `signal.alarm` timeout, which can only be armed there.
The local path therefore generates a `(task, seed)` inside `asyncio.run()`
and scores afterwards; the Langfuse path calls `run_experiment()` from plain
synchronous code, which then runs its evaluators on the calling thread.

**No connection outlives its event loop.** Each `(task, seed)` runs on a
fresh loop from `asyncio.run()`, so the shared `openai.AsyncOpenAI` asks for
every connection to be closed after its response (`Connection: close`): a
pooled keep-alive socket belongs to the loop that opened it, and reusing one
on the next seed's loop fails that seed's requests.

**`server.fail_fast_after` is a fail-fast, not a cancellation.** Neither path
can cancel requests already in flight. `task_fn`'s circuit breaker makes every
later document fail before sending a request once the server refuses one or
`fail_fast_after` in a row fail.

**`run_experiment()` drops a document whose task function raised**, with no
error message. `write_experiment_jsonl` writes a `status: "dropped"` record
for each one so document counts still add up; the local path records the
error itself under `status: "failed"`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import types
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai

from open_r1_tpu.evaluation import dataset_sync, scoring
from open_r1_tpu.evaluation.run import task_slug
from open_r1_tpu.evaluation.runner import LangfuseGuard, iter_documents, render_messages
from open_r1_tpu.evaluation.task_fn import make_task
from open_r1_tpu.evaluation.taskpack import resolve_task_configs

LOGGER = logging.getLogger(__name__)

# Evaluations `lighteval_evaluator` posts that are not a LightEval metric
# name -- run-level facts and the failure marker -- excluded from a JSONL
# record's `scores` dict, keeping `result.scores` (LightEval's own metric
# names) separate from `evaluation.scoring.run_level_fields`.
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


def jsonl_path(output_dir: Path, seed: int, task: str) -> Path:
    return output_dir / f"seed-{seed}" / f"{task_slug(task)}.jsonl"


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, default=str))
            handle.write("\n")


def _ok_record(
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


# --- the local path ---------------------------------------------------------


@dataclass(frozen=True)
class Document:
    doc_id: str
    doc: Any  # LightEval's Doc
    gold: str
    messages: list[dict[str, str]]


def task_documents(
    task: str, config: Any, settings: Mapping[str, Any]
) -> list[Document]:
    """One task's documents, rendered once and reused for every seed -- as the
    Langfuse path reuses one synced dataset. That matters for `gpqa`, whose
    prompt function shuffles the answer choices per call. A document with
    more than one gold fails here, before anything is generated.
    """
    documents = []
    for doc_id, row in iter_documents(config, max_samples=settings["max_samples"]):
        doc = scoring.build_doc(config.prompt_function, row, task)
        documents.append(
            Document(
                doc_id=doc_id,
                doc=doc,
                gold=scoring.single_gold(doc, task, doc_id),
                messages=render_messages(doc, settings["system_prompt"]),
            )
        )
    return documents


async def _generate(
    documents: Sequence[Document],
    *,
    settings: Mapping[str, Any],
    client: Any,
    label: str,
) -> list[Any]:
    """Every document's task-function output, or the exception it raised, in
    document order, with at most `server.max_concurrency` requests in flight.
    """
    task_fn = make_task(settings, client=client)
    limit = asyncio.Semaphore(int(settings["max_concurrency"]))
    total = len(documents)
    every = max(1, total // 10)
    done = 0

    async def one(messages: list[dict[str, str]]) -> Any:
        nonlocal done
        async with limit:
            try:
                return await task_fn(item=types.SimpleNamespace(input=messages))
            finally:
                done += 1
                if done % every == 0 or done == total:
                    LOGGER.info("%s: %d/%d documents done", label, done, total)

    return await asyncio.gather(
        *(one(document.messages) for document in documents), return_exceptions=True
    )


def _score_output(
    doc: Any, output: Mapping[str, Any], metrics: Sequence[Any]
) -> tuple[dict[str, Any], scoring.ScoringResult]:
    """`scoring.compute_scores` on one completion, with scores coerced exactly
    as the Langfuse path posts them, so both paths write the same values.
    """
    result = scoring.compute_scores(
        doc, scoring.build_model_response(output["text"]), metrics
    )
    scores = {
        name: value for name, (value, _) in scoring.coerce_fields(result.scores).items()
    }
    return scores, result


def run_local_task_seed(
    *,
    task: str,
    seed: int,
    documents: Sequence[Document],
    metrics: Sequence[Any],
    settings: Mapping[str, Any],
    client: Any,
    output_dir: Path,
) -> Path:
    """Generate and score one `(task, seed)` without Langfuse, writing its
    JSONL. Generation runs on its own event loop; scoring then runs here, on
    the main thread (see the module docstring).
    """
    label = f"seed {seed} {task}"
    outputs = asyncio.run(
        _generate(documents, settings=settings, client=client, label=label)
    )

    records = []
    for document, output in zip(documents, outputs, strict=True):
        if isinstance(output, BaseException):
            records.append(
                {
                    "status": "failed",
                    "doc_id": document.doc_id,
                    "task": task,
                    "seed": seed,
                    "error": f"{type(output).__name__}: {output}",
                }
            )
            continue
        scores, result = _score_output(document.doc, output, metrics)
        records.append(
            _ok_record(
                doc_id=document.doc_id,
                task=task,
                seed=seed,
                gold=document.gold,
                query=document.doc.query,
                output=output,
                scores=scores,
                failed_metrics=result.failed_metrics,
                scoring_errors=result.errors,
                trace_id=None,
            )
        )

    path = jsonl_path(output_dir, seed, task)
    _write_jsonl(path, records)
    scored = sum(record["status"] == "ok" for record in records)
    LOGGER.info("%s -> %s (%d/%d scored)", label, path, scored, len(records))
    return path


def _run_local(
    settings: Mapping[str, Any],
    resolved: Mapping[str, Any],
    *,
    client: Any,
    output_dir: Path,
) -> None:
    # Task-major, so only one task's documents are held at a time:
    # `lcb:codegeneration` carries every test case in each document.
    for task in settings["tasks"]:
        config = resolved[task]
        documents = task_documents(task, config, settings)
        for seed in settings["seeds"]:
            run_local_task_seed(
                task=task,
                seed=seed,
                documents=documents,
                metrics=list(config.metrics),
                settings=settings,
                client=client,
                output_dir=output_dir,
            )


# --- the Langfuse path ------------------------------------------------------


def _record_from_item_result(
    item_result: Any, *, task: str, seed: int
) -> dict[str, Any]:
    """One JSONL record built from an `ExperimentItemResult`."""
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

    item = item_result.item
    return _ok_record(
        doc_id=item.metadata["doc_id"],
        task=task,
        seed=seed,
        gold=item.expected_output,
        query=item.metadata.get("query"),
        output=output,
        scores=scores,
        failed_metrics=failed_metrics,
        scoring_errors=scoring_errors,
        trace_id=item_result.trace_id,
    )


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
    seen_ids = {item_result.item.id for item_result in result.item_results}
    records = [
        _record_from_item_result(item_result, task=task, seed=seed)
        for item_result in result.item_results
    ]
    records += [
        _dropped_record(item, task=task, seed=seed)
        for item in dataset.items
        if item.id not in seen_ids
    ]
    _write_jsonl(output_path, records)


def run_experiment_for_task_seed(
    langfuse_client: Any,
    *,
    task: str,
    seed: int,
    dataset_name: str,
    settings: Mapping[str, Any],
    client: Any,
    output_dir: Path,
    run_metadata: Mapping[str, Any],
) -> Any:
    """Run and persist one `(task, seed)`'s experiment against the synced
    Langfuse dataset `dataset_name`.
    """
    dataset = langfuse_client.get_dataset(dataset_name)
    fingerprint = dataset_name.rsplit("@", 1)[-1]
    result = dataset.run_experiment(
        name=f"{settings['served_model_name']}-{settings['tier']}-seed{seed}",
        task=make_task(settings, client=client),
        evaluators=[scoring.lighteval_evaluator(task)],
        max_concurrency=int(settings["max_concurrency"]),
        metadata={**run_metadata, "dataset_fingerprint": fingerprint},
    )

    output_path = jsonl_path(output_dir, seed, task)
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


def sync_datasets(langfuse_client: Any, settings: Mapping[str, Any]) -> dict[str, str]:
    """Sync the recipe's tasks into Langfuse and return `{task: dataset}`.

    Idempotent, so it runs before every traced evaluation rather than as a
    separate step. Stops on any Langfuse failure: an experiment against an
    incomplete dataset would score fewer documents than the recipe asks for.
    """
    guard = LangfuseGuard(langfuse_client)
    synced = dataset_sync.sync_recipe(guard, settings)
    guard.flush()
    if guard.failures:
        raise RuntimeError(
            f"{guard.failures} Langfuse call(s) failed while syncing the "
            "recipe's datasets; fix Langfuse and re-run, or drop "
            "--tracing-config to evaluate without it"
        )
    return {task: name for task, (name, _) in synced.items()}


def _run_langfuse(
    settings: Mapping[str, Any],
    *,
    langfuse_client: Any,
    client: Any,
    output_dir: Path,
    recipe_path: str | None,
) -> None:
    datasets = sync_datasets(langfuse_client, settings)
    run_metadata = {
        "recipe_path": recipe_path,
        "git_commit": _git_commit(),
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "max_new_tokens": settings["max_new_tokens"],
    }
    for seed in settings["seeds"]:
        for task in settings["tasks"]:
            run_experiment_for_task_seed(
                langfuse_client,
                task=task,
                seed=seed,
                dataset_name=datasets[task],
                settings=settings,
                client=client,
                output_dir=output_dir,
                run_metadata=run_metadata,
            )


# --- entry point --------------------------------------------------------------


def _close_client(client: Any) -> None:
    """Release the shared `openai.AsyncOpenAI` once every `(task, seed)` has
    finished, on a loop of its own. Never fatal: every record is already
    written, and a teardown error must not replace the one `run()` is already
    propagating.
    """
    try:
        asyncio.run(client.close())
    except Exception:
        LOGGER.warning(
            "closing the HTTP client failed; every seed's records were "
            "already written, so the run continues to its summary",
            exc_info=True,
        )


def run(
    settings: Mapping[str, Any],
    *,
    langfuse_client: Any = None,
    recipe_path: str | None = None,
) -> Path:
    """Run every `(task, seed)` in `settings`, writing
    `output_dir/seed-{seed}/{task_slug}.jsonl`. Pass a `langfuse_client` to
    trace the run in Langfuse; without one it runs locally.
    """
    output_dir = Path(settings["output_dir"]).expanduser()
    client = openai.AsyncOpenAI(
        api_key="local",
        base_url=settings["base_url"],
        max_retries=0,  # evaluation.runner.generate_one owns retries
        # One connection per request, never a pooled one -- see the module
        # docstring. The cost is a loopback handshake per document, which is
        # nothing beside a multi-second generation.
        default_headers={"Connection": "close"},
    )
    try:
        if langfuse_client is None:
            resolved = resolve_task_configs(list(settings["tasks"]))
            _run_local(settings, resolved, client=client, output_dir=output_dir)
        else:
            _run_langfuse(
                settings,
                langfuse_client=langfuse_client,
                client=client,
                output_dir=output_dir,
                recipe_path=recipe_path,
            )
    finally:
        _close_client(client)
    return output_dir


def _parse_args() -> Any:
    import argparse

    from open_r1_tpu.core.logging import LOG_LEVELS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML evaluation recipe")
    parser.add_argument(
        "--tracing-config",
        help="YAML tracing config; trace the run in Langfuse instead of running "
        "locally",
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

    args = _parse_args()
    configure_logging(LOG_LEVELS[args.log_level])
    settings = resolve_settings(load_eval_config(args.config, args.overrides))

    langfuse_client = None
    if args.tracing_config:
        from open_r1_tpu.tracing.config import (
            build_langfuse_client,
            load_tracing_config,
        )

        tracing_config = load_tracing_config(args.tracing_config)
        langfuse_client = build_langfuse_client(tracing_config)

    server_provenance = container_image_provenance(settings)
    wait_for_server(settings["base_url"], int(settings["startup_timeout_secs"]))

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
