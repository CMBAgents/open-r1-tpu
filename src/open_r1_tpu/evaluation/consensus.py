"""cons@n: a majority vote over each document's replicates.

Every other metric is scored per replicate and averaged; a consensus needs all
n replicates of one document at once. This module joins them from the JSONL
records and returns one value per task for `summary.build_summary`.

The vote is over extracted answers, not raw completions: two long reasoning
traces are never identical, so a vote over text would make every group size
one and the "majority" whichever sample came first. Answers are extracted
with LightEval's `math_normalizer` (the last `\\boxed{}`). LightEval's own
`MajAtN` is not used because it applies that extractor to the gold as well,
and an AIME gold has no `\\boxed{}`, so every answer would be compared with an
empty string.

The winning answer is judged by the task's own metric (`scoring.compute_scores`)
and the documents are reduced by that metric's `corpus_level_fn`. A tie goes to
the answer that appeared first in seed order, so re-reducing the same records
gives the same number. A replicate with no extractable answer does not vote; a
document where none has one scores 0 and is counted separately.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from open_r1_tpu.evaluation import scoring

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsensusResult:
    """One task's cons@n, as filed under `summary["consensus"]`.

    `documents_without_consensus` separates a zero earned because no
    replicate produced an answer (a generation failure) from one earned by a
    wrong majority.
    """

    name: str
    value: float
    n: int
    metric: str
    documents: int
    documents_without_consensus: int


def extract_answer(completion: str) -> str:
    """The answer one completion votes for, extracted after the reasoning
    block is stripped (as for the metrics), so a candidate boxed and then
    rejected inside `<think>` does not vote.
    """
    from lighteval.metrics.normalizations import math_normalizer

    stripped = scoring.build_model_response(completion).text_post_processed[0]
    return math_normalizer(stripped).strip()


def majority_answer(answers: Sequence[str]) -> str | None:
    """The most-voted non-empty answer, ties broken towards the earliest;
    None when no replicate produced one.
    """
    votes = [answer for answer in answers if answer]
    if not votes:
        return None
    counts = Counter(votes)
    best = max(counts.values())
    return next(answer for answer in votes if counts[answer] == best)


def _records_by_document(
    output_dir: Path, task: str, seeds: Sequence[int]
) -> dict[str, list[dict[str, Any]]]:
    """Every `ok` replicate of every document, keyed by `doc_id`, in seed
    order; a failed or dropped generation has no answer to contribute.
    """
    # Imported here: `summary` imports this module.
    from open_r1_tpu.evaluation.summary import jsonl_path, read_jsonl

    by_document: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        for record in read_jsonl(jsonl_path(output_dir, seed, task)):
            if record.get("status") != "ok":
                continue
            by_document.setdefault(str(record["doc_id"]), []).append(record)
    return by_document


def _score_consensus_document(
    records: Sequence[Mapping[str, Any]], *, task: str, metric_name: str, metrics: Any
) -> float | None:
    """One document's consensus score, or None if no replicate answered.

    The first completion to vote for the winning answer is judged afresh by
    the task's metric.
    """
    answers = [extract_answer(str(record["completion"])) for record in records]
    winner = majority_answer(answers)
    if winner is None:
        return None

    record = records[answers.index(winner)]
    # Records carry no `specific` (see `summary.ok_record`), so a task whose
    # metric reads it fails here with the missing-metric error below.
    doc = scoring.doc_from_item(record["gold"], {"query": record["query"]}, task)
    response = scoring.build_model_response(str(record["completion"]))
    result = scoring.compute_scores(doc, response, metrics)
    if metric_name not in result.scores:
        raise ValueError(
            f"{task}: eval.consensus names metric {metric_name!r}, which the "
            f"task did not produce (it produced {sorted(result.scores)})"
        )
    return result.scores[metric_name]


def consensus_for_task(
    *,
    task: str,
    request: Mapping[str, Any],
    config: Any,
    seeds: Sequence[int],
    output_dir: Path,
) -> ConsensusResult:
    """One task's cons@n over the first `request["n"]` seeds, so reducing the
    same records twice gives the same number.
    """
    n = int(request["n"])
    metric_name = str(request["metric"])
    if n > len(seeds):
        raise ValueError(
            f"{task}: cons@{n} needs {n} replicates but only {len(seeds)} ran"
        )
    voting_seeds = list(seeds)[:n]

    # The metric's own corpus reduction, never an assumed mean.
    metrics = list(config.metrics)
    declared: dict[str, Any] = {}
    for metric in metrics:
        declared.update(metric.get_corpus_aggregations())
    if metric_name not in declared:
        raise ValueError(
            f"{task}: eval.consensus names metric {metric_name!r}, which this "
            f"task does not declare (declared: {sorted(declared)})"
        )
    corpus_fn = declared[metric_name]

    by_document = _records_by_document(output_dir, task, voting_seeds)
    if not by_document:
        raise ValueError(
            f"{task}: no scored records under {output_dir} for seeds "
            f"{voting_seeds}, so there is nothing to take a consensus over"
        )

    values: list[float] = []
    without_consensus = 0
    for doc_id, records in sorted(by_document.items()):
        if len(records) < n:
            LOGGER.warning(
                "%s document %s: voting over %d replicate(s), not %d -- the "
                "rest failed or were dropped",
                task,
                doc_id,
                len(records),
                n,
            )
        score = _score_consensus_document(
            records, task=task, metric_name=metric_name, metrics=metrics
        )
        if score is None:
            without_consensus += 1
            LOGGER.warning(
                "%s document %s: no replicate produced an extractable answer; "
                "scoring it 0 for cons@%d",
                task,
                doc_id,
                n,
            )
            values.append(0.0)
        else:
            values.append(float(score))

    return ConsensusResult(
        name=f"cons@{n}",
        value=float(corpus_fn(values)),
        n=n,
        metric=metric_name,
        documents=len(values),
        documents_without_consensus=without_consensus,
    )


def consensus_metrics(
    settings: Mapping[str, Any],
    resolved_configs: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, dict[str, Any]]:
    """Every cons@n `settings["consensus"]` asks for, keyed by task; empty
    when it asks for none.
    """
    requests: Mapping[str, Mapping[str, Any]] = settings.get("consensus") or {}
    if not requests:
        return {}

    path = Path(output_dir)
    seeds = list(settings["seeds"])
    results: dict[str, dict[str, Any]] = {}
    for task, request in requests.items():
        result = consensus_for_task(
            task=task,
            request=request,
            config=resolved_configs[task],
            seeds=seeds,
            output_dir=path,
        )
        LOGGER.info(
            "%s %s (%s): %.4f over %d document(s)%s",
            task,
            result.name,
            result.metric,
            result.value,
            result.documents,
            (
                f", {result.documents_without_consensus} with no extractable answer"
                if result.documents_without_consensus
                else ""
            ),
        )
        results[task] = asdict(result)
    return results
