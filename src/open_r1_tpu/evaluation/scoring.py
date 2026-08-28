"""Turn a LightEval metric's verdict into a Langfuse score.

The scorer bridge for the Langfuse-native runner (`evaluation.runner`): the
whole job is translating what LightEval's own metric objects return into what
`Langfuse.create_score` accepts, and it is smaller than it sounds because the
two shapes nearly match already. LightEval's sample-level metrics return a
dict of named values (`Metric.compute_sample` wraps a single metric as
`{metric_name: value}`; a `SampleLevelMetricGrouping` carries several names at
once -- see `ifeval`'s four accuracies), and a Langfuse score is a
`(name, value, data_type)` triple. One LightEval metric name becomes one
Langfuse score name; nothing here invents a naming scheme or flattens a
grouping into a single number.

This module never reimplements extraction, normalisation, or symbolic
equivalence. Every scoring call below reaches the installed `lighteval`'s own
metric objects -- resolved fresh per run through
`evaluation.taskpack.resolve_task_configs`, never deserialized from the
committed task pack -- and its own `remove_reasoning_tags`, so a maths answer
judged wrong here is the same judgement the LightEval CLI would have made on
the same text.

`lighteval` is a pinned dependency (`evaluation.stack.EVALUATION_PACKAGE_VERSIONS`),
imported here only for `lighteval.metrics`, `lighteval.models.model_output`,
and `lighteval.tasks.requests` -- library internals with no stability
guarantee across releases. If an upgrade breaks an import this module makes,
the documented fallback is to vendor the affected module under a clearly
named `_vendor/` directory with the upstream commit recorded, rather than
reimplementing its behaviour from scratch.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

LOGGER = logging.getLogger(__name__)

# LightEval's own default (`PipelineParameters.reasoning_tags`, as a Python
# literal): a `<think>...</think>` block is dropped before scoring, leaving
# `text_post_processed` for the metric and the raw text untouched for the
# trace. See the module docstring: this project must score exactly what the
# CLI would have scored, so the tag pair is not configurable here.
REASONING_TAG_PAIRS: tuple[tuple[str, str], ...] = (("<think>", "</think>"),)

_LANGFUSE_NUMERIC = "NUMERIC"
_LANGFUSE_CATEGORICAL = "CATEGORICAL"


def build_doc(prompt_function: Any, row: Mapping[str, Any], task_name: str) -> Any:
    """Render one dataset row through the task pack's own prompt function.

    A thin, single choke point on purpose (see the module docstring): if a
    LightEval upgrade changes what a prompt function needs or returns, this is
    the one place that breaks, not every call site.
    """
    doc = prompt_function(row, task_name)
    if not doc.query or not doc.choices:
        raise ValueError(
            f"{task_name}: prompt function produced a Doc with an empty "
            f"query or choices for row {row!r}"
        )
    return doc


def build_model_response(raw_text: str) -> Any:
    """Construct the `ModelResponse` a LightEval metric expects from one raw
    completion, with the reasoning-tag strip already applied.

    Both `text` (raw) and `text_post_processed` (stripped) are populated:
    `text` is what the trace stores as the generation's output, and
    `text_post_processed` is what a metric actually scores against -- see the
    module docstring's warning about an abandoned candidate answer surviving
    inside an unstripped `<think>` block.
    """
    from lighteval.models.model_output import ModelResponse
    from lighteval.utils.utils import remove_reasoning_tags

    stripped = remove_reasoning_tags(text=raw_text, tag_pairs=list(REASONING_TAG_PAIRS))
    return ModelResponse(text=[raw_text], text_post_processed=[stripped])


@dataclass(frozen=True)
class ScoringResult:
    """One document's scoring outcome.

    `scores` may hold values LightEval itself computed but that have no
    single-document Langfuse posting (see `coerce_score`'s handling of a
    list-valued metric, e.g. `ifeval`'s `inst_level_*_acc`) -- callers that
    need the true corpus number reduce `scores` across documents themselves
    using the metric's own `corpus_level_fn` (`evaluation.reduce`), never a
    hand-rolled mean.
    """

    scores: dict[str, Any] = field(default_factory=dict)
    failed_metrics: tuple[str, ...] = ()
    errors: dict[str, str] = field(default_factory=dict)


def _metric_label(metric_name: Any) -> str:
    return metric_name if isinstance(metric_name, str) else "+".join(metric_name)


def compute_scores(
    doc: Any, model_response: Any, metrics: Sequence[Any]
) -> ScoringResult:
    """Score one document against every metric its task declares.

    Mirrors `lighteval.metrics.apply_metric`'s own batched/non-batched
    dispatch (`Metric.compute_sample`'s two call shapes), but per-metric and
    fault-tolerant: a metric that raises is counted as failed and does not
    stop the others, because the run has already paid for the generation -- a
    scoring failure is a data point, not a crash. A name two metrics both
    produce is not a silent overwrite: it raises, naming both.

    No timeout is added here on top of a metric's own. Every metric this
    project's tasks use (`gsm8k`'s and `math_500`'s, both backed by
    `MultilingualExtractiveMatchMetric`'s sympy-based equivalence check, which
    is what can hang on adversarial output) already carries its own
    `timeout_seconds` via a `signal.alarm`-based guard, because these are the
    live, unmodified objects the installed LightEval constructs -- and
    `signal.alarm` can only be armed on the main thread of the main
    interpreter. Wrapping the call in a second, thread-based timeout here
    would not add protection; it would only break the first one, by moving
    the call off the thread `signal.alarm` requires. **Callers must invoke
    `compute_scores` from the main thread** -- never through
    `asyncio.to_thread`, `run_in_executor`, or a worker pool -- or a metric's
    own timeout stops working silently.
    """
    scores: dict[str, Any] = {}
    failed: list[str] = []
    errors: dict[str, str] = {}
    owners: dict[str, str] = {}

    for metric in metrics:
        label = _metric_label(metric.metric_name)
        try:
            if metric.batched_compute:
                batched = metric.compute_sample(responses=[model_response], docs=[doc])
                raw = {name: values[0] for name, values in batched.items()}
            else:
                raw = metric.compute_sample(model_response=model_response, doc=doc)
        except Exception as error:  # noqa: BLE001 - any metric internal, incl. its own timeout
            failed.append(label)
            errors[label] = f"{type(error).__name__}: {error}"
            continue

        for name, value in raw.items():
            if name in owners:
                raise ValueError(
                    f"metric name collision on {name!r}: both {owners[name]!r} "
                    f"and {label!r} produced it"
                )
            owners[name] = label
            scores[name] = value

    return ScoringResult(scores=scores, failed_metrics=tuple(failed), errors=errors)


def coerce_score(value: Any) -> tuple[Any, str] | None:
    """Map one LightEval metric value to a `(value, langfuse_data_type)` pair
    `tracing.scores.post_scores` can post unchanged.

    | LightEval value | Langfuse `data_type` | Note |
    | --- | --- | --- |
    | `bool` | `NUMERIC` as `0.0`/`1.0` | checked before `int` (`bool` is a subclass) |
    | `int` / `float` | `NUMERIC` | the common case |
    | `str` | `CATEGORICAL` | e.g. an extraction-status label |
    | `None` | *skipped* (returns `None`) | absence is not zero -- never coerce |
    | `list` / `tuple` of scalars | *skipped* (returns `None`) | see below |

    A list or tuple (e.g. `ifeval`'s `inst_level_*_acc`, one bool per
    instruction in the document) has no single-document scalar to post: its
    corpus number needs the metric's own `corpus_level_fn` run over every
    document, not a per-document post -- see `ScoringResult`.

    Anything else raises: an unrecognised shape is a bridge bug, not a value
    to silently drop.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return (1.0 if value else 0.0, _LANGFUSE_NUMERIC)
    if isinstance(value, (int, float)):
        return (float(value), _LANGFUSE_NUMERIC)
    if isinstance(value, str):
        return (value, _LANGFUSE_CATEGORICAL)
    if isinstance(value, (list, tuple)):
        return None
    raise TypeError(f"cannot coerce a {type(value).__name__} score value to Langfuse")


def run_level_fields(
    *, completion_tokens: int | None, finish_reason: str
) -> dict[str, Any]:
    """The two run-level signals posted alongside every document's metric
    scores: `completion_tokens` (from the API response's own `usage`, closing
    the blind `truncation_rate` an empty token count used to cause) and
    `truncated` (`1.0` when `finish_reason == "length"`, a fact the server
    states rather than a token-count inference).
    """
    return {
        "completion_tokens": completion_tokens,
        "truncated": finish_reason == "length",
    }


def coerce_fields(fields: Mapping[str, Any]) -> dict[str, tuple[Any, str]]:
    """Coerce every field in `fields` and drop the ones `coerce_score` skips.

    The convenience wrapper `evaluation.runner` actually calls: run
    `compute_scores`' result and `run_level_fields`' result through this once,
    merged, and pass the result straight to `tracing.scores.post_scores`.
    """
    coerced: dict[str, tuple[Any, str]] = {}
    for name, value in fields.items():
        pair = coerce_score(value)
        if pair is not None:
            coerced[name] = pair
    return coerced


# --- the Langfuse `run_experiment()` adapter --------------------------------
#
# Everything below is additive for the Langfuse-native evaluation plan
# (`eval-langfuse-native-plan.md`, Task 2): `run_experiment()` drives
# iteration and concurrency now, and calls back into this module to score
# each document, through the two functions below. Nothing above this comment
# changes; `runner._score_and_post_document` and `lighteval_evaluator` both
# still call `build_model_response`/`compute_scores`/`run_level_fields`/
# `coerce_fields` the same way, so a document scores identically whichever
# half of the pipeline produced its trace.


def doc_from_item(
    expected_output: Any, metadata: Mapping[str, Any], task_name: str
) -> Any:
    """Rebuild a `Doc` equivalent to the one `evaluation.dataset_sync` built,
    from a Langfuse dataset item -- for scoring only, never by calling the
    task's prompt function a second time.

    That distinction matters for at least one task: `gpqa`'s prompt function
    shuffles its four answer choices with `random.randint` on every call, so
    re-rendering at scoring time would judge a different shuffle than the one
    actually sent to the model. `evaluation.dataset_sync` captures `doc.query`
    and `doc.specific` once, verbatim, into the item's metadata; this rebuilds
    from exactly that, for every task, not just the deterministic ones.

    `choices=[expected_output]`/`gold_index=0` is a deliberate single-choice
    reconstruction. `Doc.get_golds()` only ever reads `choices[gold_index]`,
    so this reproduces the original gold exactly regardless of how many
    choices (or in what order) the original prompt function offered --
    `evaluation.dataset_sync`'s own docstring requires exactly one gold per
    document for the same reason. A test asserts this constructor and
    `build_doc` agree on `get_golds()` for the same document.
    """
    from lighteval.tasks.requests import Doc

    if "query" not in metadata:
        raise ValueError(
            f"{task_name}: dataset item metadata has no 'query' key -- was "
            "this item created by evaluation.dataset_sync?"
        )
    return Doc(
        query=metadata["query"],
        choices=[expected_output],
        gold_index=0,
        specific=metadata.get("specific"),
        task_name=task_name,
    )


def lighteval_evaluator(task_name: str) -> Callable[..., list[Any]]:
    """Factory for the per-document evaluator `dataset.run_experiment()`
    calls: `evaluator(*, input, output, expected_output, metadata, **kwargs)
    -> list[Evaluation]`.

    Closed over `task_name`'s live metric objects, resolved fresh through
    `evaluation.taskpack.resolve_task_configs` on every call -- never
    deserialized from the committed task pack; see that module's docstring
    for why. Scores exactly as `evaluation.runner._score_and_post_document`
    does today: `build_model_response`, `compute_scores`, `run_level_fields`,
    `coerce_fields`, in that order.

    `output` is the dict `evaluation.task_fn.make_task`'s task function
    returns (`{"text", "finish_reason", "completion_tokens", ...}`), not a
    bare completion string -- `"text"` is what gets scored, and
    `"finish_reason"`/`"completion_tokens"` feed `run_level_fields` exactly as
    the generation outcome did before. `evaluation.experiment` reads the same
    `output` dict again, independently, to persist its JSONL record, so
    nothing downstream ever has to read a fact back out of Langfuse.

    Always returns a list, so a `SampleLevelMetricGrouping` (e.g. `ifeval`'s
    four accuracies) produces several named scores rather than being
    flattened into one -- `coerce_fields` already drops the list-valued
    metrics (`inst_level_*_acc`) that have no single-document scalar meaning;
    see `coerce_score`. A `ScoringResult` with `failed_metrics` adds one more:
    `Evaluation(name="scoring_failed", value=1.0, metadata={"failed_metrics":
    ..., "errors": ...})` -- visible in Langfuse rather than only in a log
    line, and readable back locally (its `metadata`) by `evaluation.experiment`
    without another Langfuse round trip.
    """
    from langfuse import Evaluation

    from open_r1_tpu.evaluation.taskpack import resolve_task_configs

    metrics = list(resolve_task_configs([task_name])[task_name].metrics)

    def evaluator(
        *, output: Any, expected_output: Any, metadata: Mapping[str, Any], **kwargs: Any
    ) -> list[Any]:
        text = output["text"] if isinstance(output, Mapping) else output
        doc = doc_from_item(expected_output, metadata, task_name)
        model_response = build_model_response(text)
        result = compute_scores(doc, model_response, metrics)

        fields = dict(result.scores)
        if isinstance(output, Mapping):
            fields.update(
                run_level_fields(
                    completion_tokens=output.get("completion_tokens"),
                    finish_reason=str(output.get("finish_reason", "")),
                )
            )

        evaluations = [
            Evaluation(name=name, value=value, data_type=data_type)
            for name, (value, data_type) in coerce_fields(fields).items()
        ]
        if result.failed_metrics:
            evaluations.append(
                Evaluation(
                    name="scoring_failed",
                    value=1.0,
                    data_type="NUMERIC",
                    metadata={
                        "failed_metrics": list(result.failed_metrics),
                        "errors": dict(result.errors),
                    },
                )
            )
        return evaluations

    return evaluator
