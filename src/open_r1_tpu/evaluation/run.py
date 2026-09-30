"""Evaluate a served model on a recipe's tasks and write the summary.

Each (task, seed) writes `output_dir/seed-{seed}/{task}.jsonl`, one record per
document, and the summary is reduced from those files (`evaluation.summary`).
By default documents are generated concurrently against the server and scored
locally; with `--tracing-config` the run goes through Langfuse instead
(`evaluation.traced`). Both paths share prompt rendering, the request policy
and the metrics, so their numbers agree.

Scoring runs on the main thread, where LightEval's `signal.alarm` timeouts
work, so the local path generates a (task, seed) inside `asyncio.run()` and
scores afterwards. As in `run_experiment()`, each (task, seed) thus gets a
fresh event loop, so the shared HTTP client sends `Connection: close`: a
pooled connection belongs to the loop that opened it, and reusing one on the
next loop fails its requests.
"""

from __future__ import annotations

import asyncio
import logging
import types
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai

from open_r1_tpu.core.cli import parse_recipe_args, recipe_parser
from open_r1_tpu.evaluation import scoring
from open_r1_tpu.evaluation.config import load_eval_config, resolve_settings
from open_r1_tpu.evaluation.generate import iter_documents, make_task, render_messages
from open_r1_tpu.evaluation.server import container_image_provenance, wait_for_server
from open_r1_tpu.evaluation.summary import (
    build_summary_from_records,
    jsonl_path,
    log_summary_to_wandb,
    ok_record,
    write_jsonl,
    write_summary,
)
from open_r1_tpu.evaluation.taskpack import resolve_task_configs

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Document:
    """One rendered document, reused for every seed."""

    doc_id: str
    doc: Any  # LightEval's Doc
    gold: str
    messages: list[dict[str, str]]


def task_documents(
    task: str, config: Any, settings: Mapping[str, Any]
) -> list[Document]:
    """One task's documents, rendered once for every seed because gpqa's
    prompt function shuffles its choices on each call. A multi-gold document
    raises here, before anything is generated.
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
    """Score one completion, with values coerced as the Langfuse path posts
    them so both paths write the same scores.
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
    """Generate and score one (task, seed) without Langfuse and write its
    JSONL. Generation runs on its own event loop, then scoring on the main
    thread. A document whose generation raised is written as `failed`.
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
            ok_record(
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
    write_jsonl(path, records)
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


def _close_client(client: Any) -> None:
    """Close the shared client on a loop of its own. A failure only warns:
    every record is already written, and it must not replace an error `run()`
    is already raising.
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
    """Run every (task, seed) in `settings` and return the output directory.
    Pass a `langfuse_client` to trace the run in Langfuse.
    """
    output_dir = Path(settings["output_dir"]).expanduser()
    client = openai.AsyncOpenAI(
        api_key="local",
        base_url=settings["base_url"],
        max_retries=0,  # generate.generate_one owns retries
        default_headers={"Connection": "close"},  # see the module docstring
    )
    try:
        if langfuse_client is None:
            resolved = resolve_task_configs(list(settings["tasks"]))
            _run_local(settings, resolved, client=client, output_dir=output_dir)
        else:
            from open_r1_tpu.evaluation.traced import run_langfuse

            run_langfuse(
                settings,
                langfuse_client=langfuse_client,
                client=client,
                output_dir=output_dir,
                recipe_path=recipe_path,
            )
    finally:
        _close_client(client)
    return output_dir


def main() -> None:
    parser = recipe_parser(__doc__)
    parser.add_argument(
        "--tracing-config",
        help="YAML tracing config; trace the run in Langfuse instead of running "
        "locally",
    )
    args = parse_recipe_args(parser)
    settings = resolve_settings(load_eval_config(args.config, args.overrides))

    langfuse_client = None
    if args.tracing_config:
        from open_r1_tpu.evaluation.traced import (
            build_langfuse_client,
            load_tracing_config,
        )

        langfuse_client = build_langfuse_client(
            load_tracing_config(args.tracing_config)
        )

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
