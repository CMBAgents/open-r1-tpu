"""Evaluate a served model on a recipe's tasks: generate, score, write JSONL.

Every `(task, seed)` produces one file, `output_dir/seed-{seed}/{task}.jsonl`,
with one record per document in the shape `evaluation.summary` turns into the
summary. Two paths write it:

- **Local (the default).** Documents are generated concurrently over the
  `openai` SDK (`server.max_concurrency` in flight), then scored with the
  task's own LightEval metrics. Nothing else needs to be running.
- **Langfuse (`--tracing-config`).** The recipe's tasks are synced into
  Langfuse datasets (`evaluation.traced`), then
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
        max_retries=0,  # evaluation.generate.generate_one owns retries
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
