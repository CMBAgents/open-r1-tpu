"""Generation primitives shared by the Langfuse-native evaluation path.

Talks to vLLM directly over the `openai` SDK, no litellm and no proxy in
front of it -- every blocker once recorded against this project's eval
pipeline (double generation, seed-1/seed-2 cache replay, a proxy misroute to
`api.openai.com`, a 3.5 hour retry burn on a refused sampling parameter) was
litellm's, or the absence of a callback hook, never the scoring.

This module used to also own the generation loop itself (`run_async`,
`run_seed_task`, a `main()` CLI): `dataset.run_experiment()` now drives
iteration and concurrency (see `evaluation.experiment`, `evaluation.task_fn`),
so that half was deleted once the tier-1 parity gate passed. What remains is
what `evaluation.task_fn.make_task` and `evaluation.dataset_sync` still
build on: `generate_one` (one chat completion, with this project's own retry
policy), `render_messages`/`iter_documents` (prompt rendering and dataset
iteration, matching LightEval's own zero-shot prompt construction), and
`LangfuseGuard` (every Langfuse call funnelled through one place, so a dead
Langfuse costs a missing trace or dataset item, never a generation).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from dataclasses import dataclass
from typing import Any

import openai

LOGGER = logging.getLogger(__name__)

# Retry mechanics are not recipe-configurable (unlike `max_concurrency` and
# `fail_fast_after`, which are deployment/policy choices): these are fixed
# implementation constants, the same way `evaluation.run.wait_for_server`'s
# poll interval is.
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECS = 1.0
BACKOFF_MAX_SECS = 30.0
REQUEST_TIMEOUT_SECS = 600.0
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
    document. Not fatal by itself -- `evaluation.task_fn._CircuitBreaker`
    decides whether enough of these in a row means the server is actually
    dead.
    """


@dataclass(frozen=True)
class GenerationOutcome:
    text: str
    finish_reason: str
    prompt_tokens: int | None
    completion_tokens: int | None
    latency_s: float
    attempts: int


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
    runs against the same dataset revision.
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


class LangfuseGuard:
    """Every Langfuse call funnelled through here, so a dead Langfuse costs a
    missing trace, score, or dataset item, never a generation. The first
    failure in a run logs a full warning; every subsequent one is counted
    silently, and the total is logged once at the end -- a dead Langfuse must
    not spam the log once per document across a 1,819-document tier.
    """

    def __init__(self, client: Any):
        self.client = client
        self.failures = 0
        self._warned = False

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

    def create_dataset(self, **kwargs: Any) -> Any | None:
        """Ensure one Langfuse dataset exists before any
        `create_dataset_item` call reaches it -- `create_dataset_item` 404s
        against a dataset that was never created, which is exactly what
        happened the first time `evaluation.dataset_sync` ran against a live
        Langfuse without this call. `POST /api/public/v2/datasets`'s
        generated client (checked against the installed `langfuse==4.14.5`)
        has no documented conflict response for an existing name -- every
        status this endpoint's spec models falls through to a 200, so a
        repeat call is expected to return the existing dataset rather than
        error. If that ever turns out wrong in practice, it would show up
        here as `failures` climbing on every routine re-sync, not as a
        silent 404 per item -- a far cheaper failure mode to notice.
        Returns the `Dataset`, or `None` if Langfuse failed.
        """
        try:
            return self.client.create_dataset(**kwargs)
        except Exception:
            self.failures += 1
            if not self._warned:
                LOGGER.warning(
                    "Langfuse call failed; continuing without ensuring "
                    "further datasets exist (further failures are counted, "
                    "not logged)",
                    exc_info=True,
                )
                self._warned = True
            return None

    def create_dataset_item(self, **kwargs: Any) -> Any | None:
        """Every `evaluation.dataset_sync` upsert funnelled through here: a
        dead Langfuse must cost a missing dataset item, never stop the sync
        -- and the run it gates -- from starting. Returns the created
        `DatasetItem`, or `None` if Langfuse failed.
        """
        try:
            return self.client.create_dataset_item(**kwargs)
        except Exception:
            self.failures += 1
            if not self._warned:
                LOGGER.warning(
                    "Langfuse call failed; continuing without syncing "
                    "further dataset items (further failures are counted, "
                    "not logged)",
                    exc_info=True,
                )
                self._warned = True
            return None
