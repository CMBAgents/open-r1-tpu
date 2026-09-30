"""Score one completion with its task's own LightEval metrics.

Extraction, normalisation and equivalence are never reimplemented: every call
reaches the installed LightEval's metric objects and reasoning-tag strip, so a
verdict here is the one the LightEval CLI would give the same text.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

# LightEval's own default (`PipelineParameters.reasoning_tags`): the reasoning
# block is dropped before a metric sees the completion.
REASONING_TAG_PAIRS: tuple[tuple[str, str], ...] = (("<think>", "</think>"),)

_LANGFUSE_NUMERIC = "NUMERIC"
_LANGFUSE_CATEGORICAL = "CATEGORICAL"


def build_doc(prompt_function: Any, row: Mapping[str, Any], task_name: str) -> Any:
    """Render one dataset row through the task's own prompt function."""
    doc = prompt_function(row, task_name)
    if not doc.query or not doc.choices:
        raise ValueError(
            f"{task_name}: prompt function produced a Doc with an empty "
            f"query or choices for row {row!r}"
        )
    return doc


def single_gold(doc: Any, task_name: str, doc_id: str) -> str:
    """The document's one gold answer. Records carry a single gold and
    `doc_from_item` rebuilds a single-choice `Doc` from it, so a multi-gold
    document is rejected rather than guessed.
    """
    golds = doc.get_golds()
    if len(golds) != 1:
        raise ValueError(
            f"{task_name} document {doc_id}: only a single gold per document is "
            f"supported, got {len(golds)} golds"
        )
    return golds[0]


def build_model_response(raw_text: str) -> Any:
    """The `ModelResponse` a LightEval metric expects for one completion.

    `text` keeps the raw completion; `text_post_processed`, which is what the
    metrics score, has the reasoning block stripped, so an answer boxed and
    then abandoned inside `<think>` cannot be extracted.
    """
    from lighteval.models.model_output import ModelResponse
    from lighteval.utils.utils import remove_reasoning_tags

    stripped = remove_reasoning_tags(text=raw_text, tag_pairs=list(REASONING_TAG_PAIRS))
    return ModelResponse(text=[raw_text], text_post_processed=[stripped])


@dataclass(frozen=True)
class ScoringResult:
    """One document's raw LightEval scores (list-valued ones included, such as
    ifeval's `inst_level_*_acc`), and the metrics that raised with their errors.
    """

    scores: dict[str, Any] = field(default_factory=dict)
    failed_metrics: tuple[str, ...] = ()
    errors: dict[str, str] = field(default_factory=dict)


def _metric_label(metric_name: Any) -> str:
    return metric_name if isinstance(metric_name, str) else "+".join(metric_name)


def compute_scores(
    doc: Any, model_response: Any, metrics: Sequence[Any]
) -> ScoringResult:
    """Score one document against every metric its task declares, as
    `lighteval.metrics.apply_metric` does, except that a metric that raises is
    recorded as failed rather than stopping the others.

    **Call this on the main thread.** LightEval's maths metrics guard their
    symbolic check with a `signal.alarm` timeout, which silently stops working
    anywhere else.
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
    """Map one LightEval value to a `(value, Langfuse data_type)` pair.

    A `bool` becomes NUMERIC 1.0 or 0.0, an `int` or `float` NUMERIC, a `str`
    CATEGORICAL. `None` is skipped (absence is not zero), and so is a list or
    tuple, which has no per-document scalar. Anything else raises.
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
    """The generation facts scored beside a document's metrics: its completion
    length and whether the server cut it off (`finish_reason == "length"`).
    """
    return {
        "completion_tokens": completion_tokens,
        "truncated": finish_reason == "length",
    }


def coerce_fields(fields: Mapping[str, Any]) -> dict[str, tuple[Any, str]]:
    """`coerce_score` over a mapping, dropping the values it skips."""
    coerced: dict[str, tuple[Any, str]] = {}
    for name, value in fields.items():
        pair = coerce_score(value)
        if pair is not None:
            coerced[name] = pair
    return coerced


def doc_from_item(
    expected_output: Any, metadata: Mapping[str, Any], task_name: str
) -> Any:
    """Rebuild a scoring `Doc` from a stored gold and the `query` (and
    `specific`) captured when the document was first rendered, never by calling
    the prompt function again: gpqa's shuffles its choices on every call. The
    gold becomes the only choice, and is coerced to `str` because Langfuse
    returns a numeric-looking gold ("204") as a number.
    """
    from lighteval.tasks.requests import Doc

    if "query" not in metadata:
        raise ValueError(
            f"{task_name}: dataset item metadata has no 'query' key -- was "
            "this item created by evaluation.traced?"
        )
    if not isinstance(expected_output, str):
        expected_output = str(expected_output)
    return Doc(
        query=metadata["query"],
        choices=[expected_output],
        gold_index=0,
        specific=metadata.get("specific"),
        task_name=task_name,
    )
