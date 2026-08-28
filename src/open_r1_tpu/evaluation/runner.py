"""Langfuse-native generation loop: vLLM direct, no litellm, no proxy.

Replaces `evaluation.run`'s subprocess orchestration (`lighteval_command`,
`run_seed`) for generation. This module owns the layer LightEval's litellm
client was doing badly -- every blocker recorded against this project's eval
pipeline (double generation, seed-1/seed-2 cache replay, a proxy misroute to
`api.openai.com`, a 3.5 hour retry burn on a refused sampling parameter) was
litellm's, or the absence of a callback hook, never the scoring. Scoring
stays LightEval's, called as a library through `evaluation.scoring`; this
module talks to vLLM over the `openai` SDK and to Langfuse in-process.

One document is one `(tier, seed, task, doc_id)`: render its prompt through
the task pack's own prompt function (`evaluation.taskpack.resolve_task_configs`,
never a hand-written guess), request a completion from vLLM, score it against
the task's live LightEval metrics, and post one Langfuse trace plus its
scores before moving on. A bounded `asyncio.Semaphore` caps concurrent
requests at `server.max_concurrency`; a connection error or 5xx retries with
backoff up to a hard cap, a 4xx never retries and stops the whole run
immediately (see `GenerationRefused`), and `server.fail_fast_after`
consecutive post-retry failures stops it too, rather than grinding through
every remaining document against a dead server.

**Scoring runs on the main thread, deliberately un-parallelized.**
`evaluation.scoring.compute_scores` relies on a LightEval metric's own
`signal.alarm`-based timeout, which only arms on the main thread of the main
interpreter -- see that module's docstring. This module's own concurrency is
therefore entirely in the I/O-bound generation requests (`asyncio`, one
event loop, one thread); nothing here ever moves scoring onto
`asyncio.to_thread`, `run_in_executor`, or a worker pool.

Writes one JSONL file per `(seed, task)` under `reporting.output_dir`,
appended to as each document completes -- append-only, so a killed run
leaves every document scored before the kill readable, with no pyarrow
schema inference to trip over. `evaluation.reduce` turns those files back
into the `(metrics, stats)` shape `evaluation.run.build_summary` already
expects, so the reduction, summary, and W&B logging halves of that module
are unchanged.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import openai

from open_r1_tpu.evaluation import scoring
from open_r1_tpu.evaluation.run import task_slug
from open_r1_tpu.evaluation.taskpack import resolve_task_configs
from open_r1_tpu.tracing.scores import post_scores

LOGGER = logging.getLogger(__name__)

# Retry mechanics are not recipe-configurable (unlike `max_concurrency` and
# `fail_fast_after`, which are deployment/policy choices): these are fixed
# implementation constants, the same way `evaluation.run.wait_for_server`'s
# poll interval is.
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECS = 1.0
BACKOFF_MAX_SECS = 30.0
REQUEST_TIMEOUT_SECS = 600.0
# How often (in completed documents) a seed/task flushes Langfuse. Not every
# document: Langfuse's SDK batches and exports on its own schedule already,
# so this is a bound on how stale the UI can get, not a correctness need.
LANGFUSE_FLUSH_EVERY = 50
LANGFUSE_FLUSH_TIMEOUT_SECS = 10.0


class GenerationRefused(RuntimeError):
    """The server rejected a request with a 4xx. Fatal and never retried:
    every other request in the run carries the same sampling parameters, so
    retrying -- or continuing to the next document -- would only reproduce
    the failure at the cost of the whole tier's wall clock. This is the
    3.5-hour retry burn, fixed structurally: see the module docstring.
    """


class GenerationFailed(RuntimeError):
    """A connection error or 5xx survived every retry attempt for one
    document. Not fatal by itself -- `ErrorBudget` decides whether enough of
    these in a row means the server is actually dead.
    """


class ErrorBudgetExceeded(RuntimeError):
    """`server.fail_fast_after` consecutive post-retry failures. Fatal: the
    server is almost certainly dead, and there is nothing to gain from
    reaching document 1,819.
    """


@dataclass(frozen=True)
class GenerationOutcome:
    text: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    attempts: int


class ErrorBudget:
    """Counts consecutive post-retry failures across the concurrent worker
    pool. Approximate under concurrency by construction -- "consecutive" has
    no single meaning once many requests are in flight at once -- but that is
    fine for what this guards: a circuit breaker for a server that has gone
    entirely dead, not a precise ordering claim. Safe without a lock: asyncio
    is single-threaded and cooperative, so a plain increment/reset between
    `await` points cannot race.
    """

    def __init__(self, fail_fast_after: int):
        self._fail_fast_after = fail_fast_after
        self._consecutive = 0

    def record_success(self) -> None:
        self._consecutive = 0

    def record_failure(self) -> None:
        self._consecutive += 1
        if self._consecutive >= self._fail_fast_after:
            raise ErrorBudgetExceeded(
                f"{self._consecutive} consecutive document failures "
                f"(server.fail_fast_after={self._fail_fast_after}); stopping "
                "the run rather than continuing against what is almost "
                "certainly a dead server"
            )


async def generate_one(
    client: Any,
    *,
    served_model_name: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> GenerationOutcome:
    """One chat completion, with this project's own retry policy rather than
    the `openai` SDK's default (the client is constructed with
    `max_retries=0` for exactly this reason): retry only a connection error
    or a 5xx, with bounded exponential backoff; never retry a 4xx, which
    raises `GenerationRefused` on the first attempt.
    """
    last_error: Exception | None = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        start = time.monotonic()
        try:
            response = await client.chat.completions.create(
                model=served_model_name,
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                timeout=REQUEST_TIMEOUT_SECS,
            )
        except openai.APIStatusError as error:
            if 400 <= error.status_code < 500:
                raise GenerationRefused(
                    f"the server refused this request with HTTP "
                    f"{error.status_code}: {error.message}. Evaluating "
                    "would produce nothing for the rest of this tier, so "
                    "the run is stopping instead of retrying."
                ) from error
            last_error = error
        except openai.APIConnectionError as error:  # includes APITimeoutError
            last_error = error
        else:
            choice = response.choices[0]
            usage = response.usage
            return GenerationOutcome(
                text=choice.message.content or "",
                finish_reason=str(choice.finish_reason),
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
                latency_s=time.monotonic() - start,
                attempts=attempt,
            )

        if attempt < MAX_ATTEMPTS:
            backoff = min(BACKOFF_MAX_SECS, BACKOFF_BASE_SECS * (2 ** (attempt - 1)))
            LOGGER.warning(
                "generation attempt %d/%d failed (%s); retrying in %.1fs",
                attempt,
                MAX_ATTEMPTS,
                last_error,
                backoff,
            )
            await asyncio.sleep(backoff)

    raise GenerationFailed(
        f"exhausted {MAX_ATTEMPTS} attempts: {last_error}"
    ) from last_error


def render_messages(doc: Any, system_prompt: str | None) -> list[dict[str, str]]:
    """Build the chat messages array for one document, matching LightEval's
    own `PromptManager.prepare_prompt_api` for the zero-shot, no-instruction
    case every task this project evaluates against actually uses: an
    optional leading system message, then one user turn carrying `doc.query`
    verbatim. Few-shot examples are not supported -- every task pack entry
    this project uses resolves to `num_fewshots: 0` -- and this raises rather
    than silently dropping them if that ever stops being true.
    """
    if doc.fewshot_samples:
        raise NotImplementedError(
            f"{doc.task_name}: this runner does not support few-shot "
            f"documents, but got {len(doc.fewshot_samples)} fewshot_samples"
        )
    query = doc.query
    if doc.instruction:
        query = doc.instruction + query
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    return messages


def iter_documents(config: Any, *, max_samples: int | None) -> list[tuple[str, Any]]:
    """`(doc_id, row)` pairs for one task's evaluation split, capped at
    `max_samples`. `doc_id` is the row's index in that split -- stable across
    runs against the same dataset revision, which is all a trace id needs to
    stay deterministic (`_trace_id`).
    """
    from datasets import load_dataset

    split = (config.evaluation_splits or config.hf_avail_splits)[0]
    slice_suffix = f"[:{max_samples}]" if max_samples else ""
    dataset = load_dataset(
        config.hf_repo,
        config.hf_subset,
        split=f"{split}{slice_suffix}",
        revision=config.hf_revision,
    )
    return [(str(index), dataset[index]) for index in range(len(dataset))]


def _trace_id(client: Any, *, tier: str, seed: int, task: str, doc_id: str) -> str:
    # Deterministic on (tier, seed, task, doc_id): a re-run of the same tier
    # addresses the same traces, and Langfuse upserts rather than
    # accumulating -- see the module docstring and `tracing.scores._score_id`
    # for the matching choice on the score side.
    return client.create_trace_id(seed=f"{tier}:{seed}:{task}:{doc_id}")


class LangfuseGuard:
    """Every Langfuse call funnelled through here, so a dead Langfuse costs
    traces, never generations. The first failure in a run logs a full
    warning; every subsequent one is counted silently, and the total is
    logged once at the end -- a dead Langfuse must not spam the log once per
    document across a 1,819-document tier.
    """

    def __init__(self, client: Any):
        self.client = client
        self.failures = 0
        self._warned = False

    def post_document(
        self,
        *,
        tier: str,
        seed: int,
        task: str,
        doc_id: str,
        prompt: str,
        raw_completion: str,
        model: str,
        finish_reason: str,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        latency_s: float,
        fields: Mapping[str, tuple[Any, str]],
    ) -> str | None:
        """Create one trace with one generation observation, then post
        `fields` (already coerced -- see `evaluation.scoring.coerce_fields`)
        as scores on it. Returns the trace id, or None if Langfuse failed.
        """
        try:
            trace_id = _trace_id(
                self.client, tier=tier, seed=seed, task=task, doc_id=doc_id
            )
            usage_details = (
                {"input": prompt_tokens, "output": completion_tokens}
                if prompt_tokens is not None and completion_tokens is not None
                else None
            )
            generation = self.client.start_observation(
                trace_context={"trace_id": trace_id},
                name="lighteval-generation",
                as_type="generation",
                input=prompt,
                output=raw_completion,
                model=model,
                usage_details=usage_details,
                metadata={
                    "tier": tier,
                    "seed": seed,
                    "task": task,
                    "doc_id": doc_id,
                    "finish_reason": finish_reason,
                    "latency_s": latency_s,
                    "source": "runner",
                },
            )
            generation.end()
            post_scores(
                self.client,
                trace_id,
                fields,
                tier=tier,
                seed=seed,
                task=task,
                source="runner",
            )
        except Exception:
            self.failures += 1
            if not self._warned:
                LOGGER.warning(
                    "Langfuse call failed; continuing without tracing for "
                    "the rest of this run (further failures are counted, "
                    "not logged)",
                    exc_info=True,
                )
                self._warned = True
            return None
        return trace_id

    def flush(self) -> None:
        # The SDK's own flush() has no timeout, and a hung export must not
        # hang the run -- so it is bounded from outside, in a worker thread
        # (safe here: unlike evaluation.scoring.compute_scores, nothing
        # Langfuse does depends on running on the main thread).
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(self.client.flush).result(
                    timeout=LANGFUSE_FLUSH_TIMEOUT_SECS
                )
        except Exception:
            self.failures += 1
            LOGGER.warning("Langfuse flush failed or timed out", exc_info=True)


def _write_record(handle: Any, record: Mapping[str, Any]) -> None:
    # A single synchronous write with no `await` inside it: under asyncio's
    # cooperative single-thread scheduling this cannot interleave with
    # another task's write, so multiple concurrent workers sharing one file
    # handle need no lock.
    handle.write(json.dumps(record, default=str))
    handle.write("\n")
    handle.flush()


async def _score_and_post_document(
    *,
    tier: str,
    seed: int,
    task: str,
    doc_id: str,
    doc: Any,
    metrics: Sequence[Any],
    prompt: str,
    outcome: GenerationOutcome,
    served_model_name: str,
    max_new_tokens: int,
    langfuse: LangfuseGuard,
) -> dict[str, Any]:
    model_response = scoring.build_model_response(outcome.text)
    result = scoring.compute_scores(doc, model_response, metrics)

    fields = dict(result.scores)
    fields.update(
        scoring.run_level_fields(
            completion_tokens=outcome.completion_tokens,
            finish_reason=outcome.finish_reason,
        )
    )
    trace_id = langfuse.post_document(
        tier=tier,
        seed=seed,
        task=task,
        doc_id=doc_id,
        prompt=prompt,
        raw_completion=outcome.text,
        model=served_model_name,
        finish_reason=outcome.finish_reason,
        prompt_tokens=outcome.prompt_tokens,
        completion_tokens=outcome.completion_tokens,
        latency_s=outcome.latency_s,
        fields=scoring.coerce_fields(fields),
    )

    return {
        "status": "ok",
        "doc_id": doc_id,
        "task": task,
        "seed": seed,
        "completion": outcome.text,
        "completion_stripped": model_response.text_post_processed[0],
        "finish_reason": outcome.finish_reason,
        "prompt_tokens": outcome.prompt_tokens,
        "completion_tokens": outcome.completion_tokens,
        "latency_s": outcome.latency_s,
        "attempts": outcome.attempts,
        "scores": result.scores,
        "failed_metrics": list(result.failed_metrics),
        "scoring_errors": result.errors,
        "trace_id": trace_id,
    }


async def run_seed_task(
    *,
    client: Any,
    settings: Mapping[str, Any],
    task: str,
    config: Any,
    seed: int,
    semaphore: asyncio.Semaphore,
    error_budget: ErrorBudget,
    langfuse: LangfuseGuard,
    output_path: Path,
) -> None:
    """Generate, score, trace, and record every document of one task for one
    seed. Raises `GenerationRefused` or `ErrorBudgetExceeded` to stop the
    whole run; every other document-level failure is caught, recorded in the
    JSONL output as `status: generation_failed`, and does not stop its
    siblings.
    """
    metrics = list(config.metrics)
    documents = iter_documents(config, max_samples=settings.get("max_samples"))
    served_model_name = settings["served_model_name"]
    processed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:

        async def worker(doc_id: str, row: Any) -> None:
            nonlocal processed
            doc = scoring.build_doc(config.prompt_function, row, task)
            messages = render_messages(doc, settings.get("system_prompt"))
            async with semaphore:
                try:
                    outcome = await generate_one(
                        client,
                        served_model_name=served_model_name,
                        messages=messages,
                        temperature=settings["temperature"],
                        top_p=settings["top_p"],
                        max_tokens=settings["max_new_tokens"],
                    )
                except GenerationFailed as error:
                    error_budget.record_failure()
                    _write_record(
                        handle,
                        {
                            "status": "generation_failed",
                            "doc_id": doc_id,
                            "task": task,
                            "seed": seed,
                            "error": str(error),
                        },
                    )
                    return

            error_budget.record_success()
            # Scoring runs here, inline, off the main thread's event loop
            # but still on the main *thread* -- see the module docstring.
            record = await _score_and_post_document(
                tier=settings["tier"],
                seed=seed,
                task=task,
                doc_id=doc_id,
                doc=doc,
                metrics=metrics,
                prompt=messages[-1]["content"],
                outcome=outcome,
                served_model_name=served_model_name,
                max_new_tokens=settings["max_new_tokens"],
                langfuse=langfuse,
            )
            _write_record(handle, record)
            processed += 1
            if processed % LANGFUSE_FLUSH_EVERY == 0:
                langfuse.flush()

        async with asyncio.TaskGroup() as group:
            for doc_id, row in documents:
                group.create_task(worker(doc_id, row))

    langfuse.flush()


async def run_async(settings: Mapping[str, Any], *, langfuse_client: Any) -> Path:
    """Run every seed and task in `settings["tasks"]`/`settings["seeds"]`,
    writing `output_dir/seed-{seed}/{task_slug}.jsonl`. Returns the output
    directory. `langfuse_client` is required: pass a real `Langfuse` client
    from `main()`, or an injected fake in tests -- there is no default, since
    silently constructing one from ambient environment would hide which
    deployment's Langfuse a run actually talks to.
    """
    task_names = list(settings["tasks"])
    output_dir = Path(settings["output_dir"]).expanduser()
    client = openai.AsyncOpenAI(
        api_key="local",
        base_url=settings["base_url"],
        max_retries=0,  # this module owns retries; see generate_one
    )
    langfuse = LangfuseGuard(langfuse_client)
    resolved = resolve_task_configs(task_names)
    semaphore = asyncio.Semaphore(int(settings["max_concurrency"]))

    try:
        for seed in settings["seeds"]:
            error_budget = ErrorBudget(int(settings["fail_fast_after"]))
            for task in task_names:
                output_path = output_dir / f"seed-{seed}" / f"{task_slug(task)}.jsonl"
                LOGGER.info("seed %d %s -> %s", seed, task, output_path)
                await run_seed_task(
                    client=client,
                    settings=settings,
                    task=task,
                    config=resolved[task],
                    seed=seed,
                    semaphore=semaphore,
                    error_budget=error_budget,
                    langfuse=langfuse,
                    output_path=output_path,
                )
    finally:
        await client.close()
        langfuse.flush()
        if langfuse.failures:
            LOGGER.warning(
                "Langfuse failed on %d call(s) across this run; traces/scores "
                "for those documents were not recorded, but generation was "
                "unaffected",
                langfuse.failures,
            )

    return output_dir


def run(settings: Mapping[str, Any], *, langfuse_client: Any) -> Path:
    """Synchronous entry point: `asyncio.run(run_async(...))` on the calling
    thread, which must be the main thread of the main interpreter -- see the
    module docstring's note on `evaluation.scoring.compute_scores`.
    """
    return asyncio.run(run_async(settings, langfuse_client=langfuse_client))


def _build_langfuse_client(tracing_config: Mapping[str, Any]) -> Any:
    """A `Langfuse` client from this project's own tracing config -- the
    `langfuse` section only; `evaluation.runner` has no use for `gcs` or
    `proxy`, which belong to the litellm-proxy capture path this module
    replaces (still present in `tracing.config`'s schema until it is
    deleted, but ignored here).
    """
    from langfuse import Langfuse

    langfuse_section = tracing_config["langfuse"]
    base_url = f"http://{langfuse_section['host']}:{langfuse_section['port']}"
    return Langfuse(base_url=base_url)


def _parse_args() -> Any:
    import argparse

    from open_r1_tpu.core.logging import LOG_LEVELS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML evaluation recipe")
    parser.add_argument(
        "--tracing-config", required=True, help="YAML tracing config (Langfuse section)"
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=sorted(LOG_LEVELS),
    )
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
    from open_r1_tpu.tracing.config import load_tracing_config

    args = _parse_args()
    configure_logging(LOG_LEVELS[args.log_level])
    settings = resolve_settings(load_eval_config(args.config, args.overrides))
    tracing_config = load_tracing_config(args.tracing_config)

    server_provenance = container_image_provenance(settings)
    wait_for_server(settings["base_url"], int(settings["startup_timeout_secs"]))

    langfuse_client = _build_langfuse_client(tracing_config)
    output_dir = run(settings, langfuse_client=langfuse_client)

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
