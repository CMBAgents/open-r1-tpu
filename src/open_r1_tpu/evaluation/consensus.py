"""cons@n: the one number that cannot be read off a per-seed mean.

Every other metric in this pipeline is per-document and per-replicate, so
`evaluation.reduce` can score each replicate on its own and average. A
consensus number is not: it needs all `n` replicates of a *single* document
together, because the vote is between them. This module performs that join,
over the same JSONL records `evaluation.reduce` reads, and returns one value
per task for `evaluation.run.build_summary`.

The vote is over **extracted answers, not raw completions**. LightEval's own
`MajAtN` votes on preprocessed prediction strings, which for a long-CoT model
means voting on thousands of tokens of distinct reasoning: no two samples are
ever equal, every group has size one, and the "majority" is whichever sample
happened to come first. That is pass@1 wearing a different name. Extracting
first -- `lighteval.metrics.normalizations.math_normalizer`, the same
last-`\\boxed{}` extractor LightEval uses -- is what makes the vote a vote.

`MajAtN` is therefore not called directly for a second reason as well: it
applies its `preprocess` to the gold as well as to the predictions, and an
AIME gold is a bare integer with no `\\boxed{}` around it, which
`math_normalizer` maps to the empty string. Passing the extractor through
`MajAtN` would compare every consensus answer against an empty gold and score
the benchmark at zero.

What this module does implement is the vote itself -- a `Counter` -- and
nothing else. Extraction is LightEval's `math_normalizer`; the judgement of
whether the winning answer is right is the task's own metric, run through
`evaluation.scoring.compute_scores` on the winning completion exactly as if
that completion were the only one; the reduction across documents is that
metric's own `corpus_level_fn`. A document judged here gets the same verdict
the CLI would have given the same text.

**Ties break towards the answer that appeared first**, in replicate order.
Any rule is arbitrary at a genuine tie; this one is at least deterministic
across re-reductions of the same records, which "whichever `Counter` returns"
is not guaranteed to be.

**A sample with no extractable answer does not vote.** It is neither evidence
for a wrong answer nor for a right one; counting empty extractions as a
candidate would let a model that mostly fails to answer win its own vote with
silence. A document where *no* replicate extracted an answer has no consensus
at all and is scored 0 for the tier, with a warning naming it.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_r1_tpu.evaluation import scoring
from open_r1_tpu.evaluation.run import task_slug

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConsensusResult:
    """One task's consensus number, in the shape
    `evaluation.run.build_summary` files under `summary["consensus"]`.

    `documents_without_consensus` is reported rather than folded silently
    into `value`: a zero earned because no replicate produced an extractable
    answer is a generation failure, and a zero earned because the majority
    agreed on a wrong answer is a reasoning failure. They read identically in
    the headline number and differently here.
    """

    name: str
    value: float
    n: int
    metric: str
    documents: int
    documents_without_consensus: int


def extract_answer(completion: str) -> str:
    """The answer one completion votes for: LightEval's own last-`\\boxed{}`
    extractor, applied to the completion with its reasoning block stripped.

    The strip matters. A DeepSeek distill boxes candidate answers inside
    `<think>...</think>` while working, so extracting from the raw text would
    sometimes vote for an answer the model went on to reject -- and would do
    it precisely on the documents where the model reconsidered, which are the
    ones a consensus vote exists to get right. `build_model_response` applies
    the same strip the metrics themselves score behind.
    """
    from lighteval.metrics.normalizations import math_normalizer

    stripped = scoring.build_model_response(completion).text_post_processed[0]
    return math_normalizer(stripped).strip()


def majority_answer(answers: Sequence[str]) -> str | None:
    """The most-voted non-empty answer, ties broken towards the earliest.

    `None` when nothing was extractable from any replicate -- see the module
    docstring on why that is reported rather than scored as a wrong answer.
    """
    votes = [answer for answer in answers if answer]
    if not votes:
        return None
    counts = Counter(votes)
    best = max(counts.values())
    for answer in votes:  # first occurrence wins a tie
        if counts[answer] == best:
            return answer
    raise AssertionError("unreachable: a counted answer must be in the votes")


def _records_by_document(
    output_dir: Path, task: str, seeds: Sequence[int]
) -> dict[str, list[dict[str, Any]]]:
    """Every replicate of every document, keyed by `doc_id`, in seed order.

    Only `status: ok` records are collected: a dropped or failed generation
    has no answer to contribute, and `evaluation.reduce` already counts it
    against the tier's own document totals.
    """
    from open_r1_tpu.evaluation.reduce import read_jsonl

    by_document: dict[str, list[dict[str, Any]]] = {}
    for seed in seeds:
        path = output_dir / f"seed-{seed}" / f"{task_slug(task)}.jsonl"
        for record in read_jsonl(path):
            if record.get("status") != "ok":
                continue
            by_document.setdefault(str(record["doc_id"]), []).append(record)
    return by_document


def _score_consensus_document(
    records: Sequence[Mapping[str, Any]], *, task: str, metric_name: str, metrics: Any
) -> float | None:
    """One document's consensus score, or `None` if no replicate answered.

    The winning *completion* is re-scored rather than its already-computed
    per-replicate score being reused. The vote groups by
    `math_normalizer`'s string output, which is a coarser equivalence than
    the metric's own symbolic one -- two completions can share a normalised
    answer and still be judged differently from their full text -- so reusing
    a sibling's score would occasionally report a verdict the metric never
    reached for the text being reported.
    """
    answers = [extract_answer(str(record["completion"])) for record in records]
    winner = majority_answer(answers)
    if winner is None:
        return None

    record = records[answers.index(winner)]
    if "gold" not in record or "query" not in record:
        raise ValueError(
            f"{task} document {record['doc_id']}: this JSONL record carries no "
            "'gold'/'query', so the consensus answer cannot be judged. Records "
            "written before eval.consensus existed lack them; re-run the tier, "
            "or reduce it without eval.consensus."
        )
    # No `specific`: it is the one Doc field this record deliberately does not
    # carry, because `lcb:codegeneration`'s holds every test case for the
    # problem and would dwarf the completions in every JSONL line. A task
    # whose metric reads `doc.specific` therefore cannot be a consensus
    # target -- its metric raises, `compute_scores` records the failure, and
    # the missing metric name below turns that into an error naming the task.
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
    """One task's cons@n over the first `request["n"]` replicates.

    The first `n`, not a sample of them: `eval.seeds` is an ordered list and
    reducing the same records twice must give the same number.
    """
    n = int(request["n"])
    metric_name = str(request["metric"])
    if n > len(seeds):
        raise ValueError(
            f"{task}: cons@{n} needs {n} replicates but only {len(seeds)} ran"
        )
    voting_seeds = list(seeds)[:n]

    # The corpus reduction is the metric's own, looked up by name rather than
    # assumed to be a mean -- the same rule `evaluation.reduce` follows, and
    # for the same reason.
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
    """Every consensus number `settings["consensus"]` asks for, keyed by task.

    Empty when the recipe asks for none, which is every tier but AIME's --
    `build_summary` files the empty mapping rather than omitting the key, so
    a summary always states whether a consensus was requested.
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
        results[task] = {
            "name": result.name,
            "value": result.value,
            "n": result.n,
            "metric": result.metric,
            "documents": result.documents,
            "documents_without_consensus": result.documents_without_consensus,
        }
    return results
