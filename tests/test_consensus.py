"""Tests for `open_r1_tpu.evaluation.consensus`.

The vote, the replicate join, and the corpus reduction are pure logic and are
tested here against small fakes at the LightEval boundary -- the same shape
`tests/test_scoring.py` uses for `compute_scores`. The one function that
genuinely needs a LightEval installation, `extract_answer` (it calls
`lighteval.metrics.normalizations.math_normalizer`), is
`@pytest.mark.integration`: deselected by default, run with
`pytest -m integration` once the `eval` extra is installed.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import pytest

from open_r1_tpu.evaluation import consensus
from open_r1_tpu.evaluation.run import task_slug


class FakeMetric:
    """Only what `consensus_for_task` and `_score_consensus_document` use."""

    def __init__(self, name, verdicts=None):
        self.metric_name = name
        self.batched_compute = False
        self._name = name
        # completion text -> score, for the fake judgement
        self._verdicts = verdicts or {}

    def get_corpus_aggregations(self):
        return {self._name: statistics.fmean}

    def compute_sample(self, *, model_response, doc):
        return {self._name: self._verdicts.get(model_response, 0.0)}


class FakeConfig:
    def __init__(self, *metrics):
        self.metrics = list(metrics)


def write_records(output_dir: Path, task: str, seed: int, records) -> None:
    path = output_dir / f"seed-{seed}" / f"{task_slug(task)}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def ok_record(doc_id: str, completion: str, **extra):
    return {
        "status": "ok",
        "doc_id": doc_id,
        "task": "aime24|0",
        "gold": "204",
        "query": "Solve the problem.",
        "completion": completion,
        "scores": {"pass@k:k=1": 0.0},
        **extra,
    }


# --- the vote --------------------------------------------------------------


def test_the_most_frequent_answer_wins():
    assert consensus.majority_answer(["7", "204", "204", "13", "204"]) == "204"


def test_a_tie_breaks_towards_the_earliest_answer():
    # Any rule is arbitrary at a genuine tie; this one has to be stable, so
    # that reducing the same records twice cannot report two different numbers.
    assert consensus.majority_answer(["13", "204", "204", "13"]) == "13"
    assert consensus.majority_answer(["204", "13", "13", "204"]) == "204"


def test_samples_with_no_extractable_answer_do_not_vote():
    # Four of six samples failed to produce an answer at all. Silence is not
    # evidence, so the two that did answer decide it -- rather than "" winning
    # 4-2 and scoring the document wrong.
    assert consensus.majority_answer(["", "", "204", "", "204", ""]) == "204"


def test_a_document_no_replicate_answered_has_no_consensus():
    assert consensus.majority_answer(["", "", ""]) is None
    assert consensus.majority_answer([]) is None


# --- the replicate join ----------------------------------------------------


def test_replicates_are_joined_by_document_in_seed_order(tmp_path):
    write_records(tmp_path, "aime24|0", 0, [ok_record("0", "a0"), ok_record("1", "b0")])
    write_records(tmp_path, "aime24|0", 1, [ok_record("0", "a1"), ok_record("1", "b1")])

    joined = consensus._records_by_document(tmp_path, "aime24|0", [0, 1])

    assert sorted(joined) == ["0", "1"]
    assert [record["completion"] for record in joined["0"]] == ["a0", "a1"]
    assert [record["completion"] for record in joined["1"]] == ["b0", "b1"]


def test_failed_and_dropped_replicates_contribute_no_vote(tmp_path):
    write_records(
        tmp_path,
        "aime24|0",
        0,
        [
            ok_record("0", "answered"),
            {"status": "generation_failed", "doc_id": "1", "error": "boom"},
            {"status": "dropped", "doc_id": "2", "error": "no result"},
        ],
    )

    joined = consensus._records_by_document(tmp_path, "aime24|0", [0])

    assert list(joined) == ["0"]


# --- the tier number -------------------------------------------------------


@pytest.fixture
def flat_extraction(monkeypatch):
    """Extraction stubbed to the identity, so a test can name a completion
    and the answer it votes for in one string.
    """
    monkeypatch.setattr(consensus, "extract_answer", lambda completion: completion)


@pytest.fixture
def fake_scoring(monkeypatch):
    """The LightEval boundary `_score_consensus_document` crosses."""
    monkeypatch.setattr(
        consensus.scoring, "doc_from_item", lambda gold, metadata, task: gold
    )
    monkeypatch.setattr(consensus.scoring, "build_model_response", lambda text: text)

    def compute_scores(doc, model_response, metrics):
        scores = {}
        for metric in metrics:
            scores.update(metric.compute_sample(model_response=model_response, doc=doc))
        return consensus.scoring.ScoringResult(scores=scores)

    monkeypatch.setattr(consensus.scoring, "compute_scores", compute_scores)


def test_the_consensus_answer_is_judged_not_the_per_replicate_mean(
    tmp_path, flat_extraction, fake_scoring
):
    # Document 0: "204" wins 3-2 and is right. Document 1: "13" wins 3-2 and
    # is wrong. cons@5 is therefore 0.5 -- while the per-replicate pass@1
    # across the same ten generations is 0.5 on document 0 and 0.4 on
    # document 1, i.e. 0.45. The two numbers are genuinely different
    # quantities, which is the whole reason this module exists.
    for seed, (first, second) in enumerate(
        [("204", "13"), ("204", "13"), ("204", "13"), ("7", "204"), ("7", "204")]
    ):
        write_records(
            tmp_path,
            "aime24|0",
            seed,
            [ok_record("0", first), ok_record("1", second)],
        )

    metric = FakeMetric("pass@k:k=1", verdicts={"204": 1.0, "13": 0.0, "7": 0.0})
    result = consensus.consensus_for_task(
        task="aime24|0",
        request={"n": 5, "metric": "pass@k:k=1"},
        config=FakeConfig(metric),
        seeds=[0, 1, 2, 3, 4],
        output_dir=tmp_path,
    )

    assert result.name == "cons@5"
    assert result.value == pytest.approx(0.5)
    assert result.documents == 2
    assert result.documents_without_consensus == 0


def test_only_the_first_n_replicates_vote(tmp_path, flat_extraction, fake_scoring):
    # Six replicates ran but the recipe asked for cons@3, so seeds 3-5 must
    # not reach the vote -- otherwise reducing a tier that overran its own
    # seed list would silently report a wider consensus than it claims to.
    for seed in range(3):
        write_records(tmp_path, "aime24|0", seed, [ok_record("0", "204")])
    for seed in range(3, 6):
        write_records(tmp_path, "aime24|0", seed, [ok_record("0", "13")])

    metric = FakeMetric("pass@k:k=1", verdicts={"204": 1.0, "13": 0.0})
    result = consensus.consensus_for_task(
        task="aime24|0",
        request={"n": 3, "metric": "pass@k:k=1"},
        config=FakeConfig(metric),
        seeds=[0, 1, 2, 3, 4, 5],
        output_dir=tmp_path,
    )

    assert result.value == pytest.approx(1.0)


def test_a_document_without_consensus_scores_zero_and_is_counted(
    tmp_path, flat_extraction, fake_scoring
):
    for seed in range(3):
        write_records(
            tmp_path,
            "aime24|0",
            seed,
            [ok_record("0", "204"), ok_record("1", "")],
        )

    metric = FakeMetric("pass@k:k=1", verdicts={"204": 1.0})
    result = consensus.consensus_for_task(
        task="aime24|0",
        request={"n": 3, "metric": "pass@k:k=1"},
        config=FakeConfig(metric),
        seeds=[0, 1, 2],
        output_dir=tmp_path,
    )

    assert result.value == pytest.approx(0.5)
    # The zero is reported as a generation failure, not left to read as a
    # reasoning failure.
    assert result.documents_without_consensus == 1


def test_a_metric_the_task_does_not_declare_is_rejected(tmp_path):
    write_records(tmp_path, "aime24|0", 0, [ok_record("0", "204")])

    with pytest.raises(ValueError, match="does not declare"):
        consensus.consensus_for_task(
            task="aime24|0",
            request={"n": 2, "metric": "maj@n"},
            config=FakeConfig(FakeMetric("pass@k:k=1")),
            seeds=[0, 1],
            output_dir=tmp_path,
        )


def test_asking_for_more_replicates_than_ran_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="needs 64 replicates but only 3 ran"):
        consensus.consensus_for_task(
            task="aime24|0",
            request={"n": 64, "metric": "pass@k:k=1"},
            config=FakeConfig(FakeMetric("pass@k:k=1")),
            seeds=[0, 1, 2],
            output_dir=tmp_path,
        )


def test_records_written_before_gold_was_carried_name_the_fix(
    tmp_path, flat_extraction, fake_scoring
):
    for seed in (0, 1):
        stale = ok_record("0", "204")
        del stale["gold"]
        write_records(tmp_path, "aime24|0", seed, [stale])

    with pytest.raises(ValueError, match="re-run the tier"):
        consensus.consensus_for_task(
            task="aime24|0",
            request={"n": 2, "metric": "pass@k:k=1"},
            config=FakeConfig(FakeMetric("pass@k:k=1")),
            seeds=[0, 1],
            output_dir=tmp_path,
        )


def test_no_consensus_requested_produces_no_consensus_block(tmp_path):
    assert (
        consensus.consensus_metrics({"seeds": [0], "consensus": {}}, {}, tmp_path) == {}
    )
    assert consensus.consensus_metrics({"seeds": [0]}, {}, tmp_path) == {}


def test_consensus_metrics_reports_the_summary_shape(
    tmp_path, flat_extraction, fake_scoring
):
    for seed in range(3):
        write_records(tmp_path, "aime24|0", seed, [ok_record("0", "204")])

    results = consensus.consensus_metrics(
        {
            "seeds": [0, 1, 2],
            "consensus": {"aime24|0": {"n": 3, "metric": "pass@k:k=1"}},
        },
        {"aime24|0": FakeConfig(FakeMetric("pass@k:k=1", verdicts={"204": 1.0}))},
        tmp_path,
    )

    assert results == {
        "aime24|0": {
            "name": "cons@3",
            "value": 1.0,
            "n": 3,
            "metric": "pass@k:k=1",
            "documents": 1,
            "documents_without_consensus": 0,
        }
    }


# --- the LightEval boundary ------------------------------------------------


@pytest.mark.integration
def test_extraction_reads_the_final_answer_not_an_abandoned_one():
    # The model boxes 13 while working, rejects it, and answers 204. Voting on
    # the raw text would count this sample for 13 -- and would do it precisely
    # on the documents where the model reconsidered.
    completion = (
        "Let me try 13. \\boxed{13} No, that double-counts.\n"
        "</think>\nTherefore, the final answer is: $\\boxed{204}$."
    )
    assert consensus.extract_answer(completion) == "204"


@pytest.mark.integration
def test_a_completion_with_no_boxed_answer_extracts_to_nothing():
    assert consensus.extract_answer("I could not work this one out.") == ""
