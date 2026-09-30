"""Generation: documents, prompts, and one chat completion per document.

Talks to vLLM directly over the `openai` SDK with this project's own retry
policy (`generate_one`), renders each document's prompt the way LightEval's
zero-shot prompt construction does (`render_messages`), and wraps both in the
task function every evaluation path drives (`make_task`), whose circuit
breaker stops sending requests once the server refuses one or
`server.fail_fast_after` in a row fail.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
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


class _CircuitBreaker:
    """See the module docstring for what this can and cannot do. Scoped to
    one `make_task` call: `evaluation.experiment` builds a fresh task
    function (and so a fresh breaker) per `(task, seed)`
    `dataset.run_experiment()` call -- matching `server.fail_fast_after`'s
    old per-seed scope at worst, and improving on it (per task *and* seed,
    rather than shared across a seed's tasks) at best.

    Safe without a lock: every task function this drives is `async def`
    running under one `run_experiment` call's own `asyncio.gather`, on that
    call's own event loop and thread, and asyncio is single-threaded and
    cooperative, so a plain read/increment between `await` points cannot
    race -- the same reasoning the old `ErrorBudget` relied on.
    """

    def __init__(self, fail_fast_after: int):
        self._fail_fast_after = fail_fast_after
        self._consecutive_failures = 0
        self._tripped: Exception | None = None

    def check(self) -> None:
        if self._tripped is not None:
            raise self._tripped

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_refused(self, error: GenerationRefused) -> None:
        # Sticky: every other request in this (task, seed) carries the same
        # sampling parameters (see GenerationRefused's own docstring), so a
        # refusal here means every other request is expected to be refused
        # too.
        if self._tripped is None:
            self._tripped = error

    def record_failure(self, error: GenerationFailed) -> None:
        self._consecutive_failures += 1
        if (
            self._consecutive_failures >= self._fail_fast_after
            and self._tripped is None
        ):
            self._tripped = GenerationFailed(
                f"{self._consecutive_failures} consecutive document failures "
                f"(server.fail_fast_after={self._fail_fast_after}); every "
                "further item in this (task, seed) fails immediately rather "
                "than attempting a request against what is almost certainly "
                "a dead server"
            )


def make_task(settings: Mapping[str, Any], *, client: Any) -> Callable[..., Any]:
    """Build the `task(*, item, **kwargs)` callable for one `(task, seed)`
    `dataset.run_experiment()` call.

    `settings` is this recipe's resolved settings
    (`evaluation.run.resolve_settings`'s output); `client` is a shared
    `openai.AsyncOpenAI`, built once per CLI invocation by
    `evaluation.experiment` and passed in here so every `(task, seed)` sends
    its requests through one client, configured and torn down in one place.
    Connections themselves are deliberately not reused, there or here: each
    `(task, seed)` runs on its own event loop, and a pooled connection
    cannot outlive the loop that opened it -- see `evaluation.experiment`'s
    module docstring.

    Deliberately no per-request `seed`: vLLM classifies a request carrying
    one as `SamplingType.RANDOM_SEED` whenever `temperature > 0`, and the TPU
    backend refuses that outright (`TpuPlatform.validate_request` raises
    "JAX does not support per-request seed."), reaching the client as an
    empty-body HTTP 500. `eval.seeds` therefore indexes independent
    replicates rather than determining them, and `generate_one` never sends
    one.
    """
    breaker = _CircuitBreaker(int(settings["fail_fast_after"]))
    served_model_name = settings["served_model_name"]
    temperature = settings["temperature"]
    top_p = settings["top_p"]
    max_tokens = settings["max_new_tokens"]

    async def task(*, item: Any, **kwargs: Any) -> dict[str, Any]:
        breaker.check()
        try:
            outcome = await generate_one(
                client,
                served_model_name=served_model_name,
                messages=item.input,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
        except GenerationRefused as error:
            breaker.record_refused(error)
            raise
        except GenerationFailed as error:
            breaker.record_failure(error)
            raise
        breaker.record_success()
        return {
            "text": outcome.text,
            "finish_reason": outcome.finish_reason,
            "prompt_tokens": outcome.prompt_tokens,
            "completion_tokens": outcome.completion_tokens,
            "latency_s": outcome.latency_s,
            "attempts": outcome.attempts,
        }

    return task
