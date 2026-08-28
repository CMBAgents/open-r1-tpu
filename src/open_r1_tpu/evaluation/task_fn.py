"""Task 3's task function: the callable `dataset.run_experiment()` drives to
produce one document's completion.

Wraps `evaluation.runner.generate_one` -- the request, its retry policy, and
its two failure types (`GenerationRefused` for a 4xx, `GenerationFailed` once
a 5xx/connection error exhausts its own retries) -- rather than
reimplementing any of it. What is new here is only what `run_experiment`
needs and the old, self-driven `runner.run_seed_task` did not: a task
function with this exact signature (`task(*, item, **kwargs)`), and a
circuit breaker standing in for the old per-seed `ErrorBudget` /
`asyncio.TaskGroup` cancellation, which `run_experiment` has no equivalent
for.

**`run_experiment` cannot cancel a run in progress.** Its
`_run_experiment_async` schedules every item's `process_item()` up front and
collects them with `asyncio.gather(..., return_exceptions=True)` -- one
item's exception fails only that item, never its siblings (checked against
the installed `langfuse==4.14.5`'s own `_client/client.py`). The old runner's
`GenerationRefused`/`ErrorBudgetExceeded` handling relied on exactly the
`asyncio.TaskGroup` cancellation this replaces; `_CircuitBreaker` below is
the closest available substitute: once one `GenerationRefused` is seen, or
`fail_fast_after` consecutive `GenerationFailed`s accumulate, every *later*
call to this task function raises immediately, before making a request -- so
a refusing or dead server costs one wasted request per item already in
flight (bounded by `server.max_concurrency`), not one per remaining document.
Already-scheduled requests still run to their own conclusion; there is no
way to reach into `asyncio.gather`'s already-created tasks from here. This is
the decision `eval-langfuse-native-plan.md`'s Task 4 asked for: keep
`server.fail_fast_after` meaningful rather than silently inert, on the
understanding that it is now a (much cheaper) fail-fast, not a cancellation.

Returns a dict, not a bare completion string:
`{"text", "finish_reason", "completion_tokens", "prompt_tokens", "latency_s",
"attempts"}`. Both `evaluation.scoring.lighteval_evaluator` (scores `"text"`,
computes `run_level_fields` from the rest) and `evaluation.experiment`
(persists the JSONL record `evaluation.reduce` expects) read this same dict
-- the only place either of those facts is available locally, without a
second Langfuse call.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from open_r1_tpu.evaluation.runner import (
    GenerationFailed,
    GenerationRefused,
    generate_one,
)

LOGGER = logging.getLogger(__name__)


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

    Deliberately no per-request `seed`: the TPU backend refuses one whenever
    `temperature > 0` (`evaluation.run.litellm_model_config`'s docstring
    explains why), and `generate_one` never sends one.
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
