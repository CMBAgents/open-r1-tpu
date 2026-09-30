"""Generation: documents, prompts, and one chat completion per document.

`generate_one` talks to vLLM directly through the `openai` SDK with this
project's retry policy, `render_messages` builds each prompt the way
LightEval's zero-shot prompt construction does, and `make_task` wraps
`generate_one` in the task function both evaluation paths drive. Its circuit
breaker stops a (task, seed) sending requests once the server refuses one or
`server.fail_fast_after` in a row fail; requests already in flight are not
cancelled.
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

# The retry policy is fixed, unlike the recipe's per-deployment
# `server.max_concurrency` and `server.fail_fast_after`.
MAX_ATTEMPTS = 5
BACKOFF_BASE_SECS = 1.0
BACKOFF_MAX_SECS = 30.0
REQUEST_TIMEOUT_SECS = 600.0


class GenerationRefused(RuntimeError):
    """The server rejected a request with a 4xx. Never retried: every request
    of a (task, seed) carries the same sampling parameters, so the rest would
    be refused too.
    """


class GenerationFailed(RuntimeError):
    """A connection error or 5xx outlasted every retry for one document. Not
    fatal by itself: the circuit breaker decides when enough in a row mean the
    server is down.
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
    """One chat completion with this project's retry policy, in place of the
    SDK's (the client is built with `max_retries=0`): a connection error or
    5xx is retried with bounded exponential backoff, and a 4xx raises
    `GenerationRefused` at once.
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
                    f"{error.status_code}: {error.message}. Retrying would "
                    "fail the same way, so the rest of this task and seed "
                    "fails at once."
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
    """The chat messages for one document, as LightEval's
    `PromptManager.prepare_prompt_api` builds them for a zero-shot task: an
    optional system message, then one user turn with the instruction and
    query. A few-shot document raises rather than losing its examples.
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
    `max_samples`. `doc_id` is the row's index in the split, stable for a
    given dataset revision.
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
    """Fails every later document of one (task, seed) before it sends a
    request, once the server has refused one or `fail_fast_after` in a row
    have failed. Needs no lock: its task function runs on one event loop.
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
    """Build the `task(*, item, **kwargs)` callable for one (task, seed): it
    generates `item.input` and returns the completion and its usage as a dict.

    `settings` is `config.resolve_settings`'s output and `client` the run's
    shared `openai.AsyncOpenAI`. Each (task, seed) gets its own circuit breaker.

    No per-request `seed` is sent: vLLM's TPU backend rejects one whenever
    `temperature > 0` (reaching the client as an empty HTTP 500), so
    `eval.seeds` indexes independent replicates rather than seeding them.
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
