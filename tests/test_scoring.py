"""Tests for `open_r1_tpu.evaluation.scoring`.

`compute_scores`, `coerce_score`, and `coerce_fields` are pure dispatch logic
tested against small fake metric objects, with no LightEval installation
needed. Everything that touches a real LightEval `Doc`/`ModelResponse`/metric
-- `build_doc`, `build_model_response`, and the reasoning-tag strip
regression guard -- is `@pytest.mark.integration` (deselected by default; run
with `pytest -m integration` once the `eval` extra is installed).
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from open_r1_tpu.evaluation import scoring, taskpack
from open_r1_tpu.tracing import scores as tracing_scores


class FakeMetric:
    """Stands in for a `lighteval.metrics.utils.metric_utils.Metric`: only
    the attributes and call shape `compute_scores` actually uses.
    """

    def __init__(self, metric_name, fn, batched_compute=False):
        self.metric_name = metric_name
        self.batched_compute = batched_compute
        self._fn = fn

    def compute_sample(self, **kwargs):
        return self._fn(**kwargs)


class FakeLangfuseClient:
    def __init__(self):
        self.score_calls = []

    def create_score(self, **kwargs):
        self.score_calls.append(kwargs)

    def _create_trace_tags_via_ingestion(self, **kwargs):
        pass


class _StubDoc:
    def __init__(self, query, choices):
        self.query = query
        self.choices = choices


# --- compute_scores dispatch, no LightEval needed ---------------------------


def test_single_metric_produces_one_score():
    metric = FakeMetric("acc", lambda model_response, doc: {"acc": 1.0})
    result = scoring.compute_scores(doc="D", model_response="R", metrics=[metric])
    assert result.scores == {"acc": 1.0}
    assert result.failed_metrics == ()
    assert result.errors == {}


def test_metric_grouping_produces_several_named_scores():
    metric = FakeMetric(["a", "b"], lambda model_response, doc: {"a": 1.0, "b": 0.0})
    result = scoring.compute_scores(doc="D", model_response="R", metrics=[metric])
    assert result.scores == {"a": 1.0, "b": 0.0}


def test_batched_metric_receives_a_single_item_batch():
    def fn(responses, docs):
        assert responses == ["R"]
        assert docs == ["D"]
        return {"batched": [42.0]}

    metric = FakeMetric("batched", fn, batched_compute=True)
    result = scoring.compute_scores(doc="D", model_response="R", metrics=[metric])
    assert result.scores == {"batched": 42.0}


def test_colliding_metric_names_raise_naming_both():
    first = FakeMetric("first", lambda model_response, doc: {"dup": 1.0})
    second = FakeMetric("second", lambda model_response, doc: {"dup": 0.0})
    with pytest.raises(ValueError, match="dup") as excinfo:
        scoring.compute_scores(doc="D", model_response="R", metrics=[first, second])
    assert "first" in str(excinfo.value)
    assert "second" in str(excinfo.value)


def test_a_raising_metric_is_counted_failed_and_does_not_abort_the_batch():
    def boom(**kwargs):
        raise RuntimeError("kaboom")

    bad = FakeMetric("bad", boom)
    good = FakeMetric("good", lambda model_response, doc: {"good": 1.0})
    result = scoring.compute_scores(doc="D", model_response="R", metrics=[bad, good])

    assert result.scores == {"good": 1.0}
    assert result.failed_metrics == ("bad",)
    assert "kaboom" in result.errors["bad"]


def test_a_metric_returning_none_is_not_zeroed():
    metric = FakeMetric("maybe", lambda model_response, doc: {"maybe": None})
    result = scoring.compute_scores(doc="D", model_response="R", metrics=[metric])
    # compute_scores carries the value through unchanged -- coercion (and its
    # "never coerce None to zero" rule) happens at the Langfuse-posting layer.
    assert result.scores == {"maybe": None}


# --- build_doc's own validation, no LightEval needed -------------------------


def test_build_doc_rejects_an_empty_query_or_choices():
    def bad_prompt_function(row, task_name):
        return _StubDoc(query="", choices=["x"])

    with pytest.raises(ValueError, match="empty"):
        scoring.build_doc(bad_prompt_function, {"a": 1}, "some_task")


# --- type coercion (Task 3d's table) -----------------------------------------


def test_type_coercion_table():
    assert scoring.coerce_score(True) == (1.0, "NUMERIC")
    assert scoring.coerce_score(False) == (0.0, "NUMERIC")
    assert scoring.coerce_score(3) == (3.0, "NUMERIC")
    assert scoring.coerce_score(3.5) == (3.5, "NUMERIC")
    assert scoring.coerce_score("extracted") == ("extracted", "CATEGORICAL")
    assert scoring.coerce_score(None) is None


def test_coerce_score_skips_a_list_valued_metric_rather_than_averaging_it():
    # e.g. ifeval's inst_level_*_acc: one bool per instruction in the
    # document, whose corpus meaning needs the metric's own corpus_level_fn,
    # not a per-document post.
    assert scoring.coerce_score([True, False, True]) is None


def test_coerce_score_rejects_an_unrecognised_shape():
    with pytest.raises(TypeError):
        scoring.coerce_score({"nested": 1})


def test_coerce_fields_drops_none_and_list_values():
    coerced = scoring.coerce_fields(
        {"a": 1.0, "b": None, "c": [True, False], "d": "label"}
    )
    assert coerced == {"a": (1.0, "NUMERIC"), "d": ("label", "CATEGORICAL")}


def test_run_level_fields():
    assert scoring.run_level_fields(completion_tokens=120, finish_reason="length") == {
        "completion_tokens": 120,
        "truncated": True,
    }
    assert scoring.run_level_fields(completion_tokens=None, finish_reason="stop") == {
        "completion_tokens": None,
        "truncated": False,
    }


def test_coerced_fields_post_through_tracing_scores_with_the_right_data_type():
    fields = scoring.coerce_fields(
        {
            "extractive_match": 1.0,
            "closed": True,
            "label": "ok",
            "skip_me": None,
            "grouped_list": [True, False],
        }
    )
    client = FakeLangfuseClient()
    tracing_scores.post_scores(
        client, "trace-1", fields, tier="t", seed=0, task="gsm8k|0", source="runner"
    )
    by_name = {c["name"]: (c["value"], c["data_type"]) for c in client.score_calls}
    assert by_name["extractive_match"] == (1.0, "NUMERIC")
    assert by_name["closed"] == (1.0, "NUMERIC")
    assert by_name["label"] == ("ok", "CATEGORICAL")
    assert "skip_me" not in by_name
    assert "grouped_list" not in by_name


# --- integration: needs the real LightEval Doc/ModelResponse/metrics -------


@pytest.mark.integration
def test_build_doc_uses_the_task_pack_prompt_function():
    resolved = taskpack.resolve_task_configs(["math_500|0"])
    config = resolved["math_500|0"]
    from datasets import load_dataset

    row = load_dataset(
        config.hf_repo, config.hf_subset, split=f"{config.evaluation_splits[0]}[:1]"
    )[0]

    doc = scoring.build_doc(config.prompt_function, row, "math_500")
    assert doc.query
    assert doc.choices
    assert doc.gold_index == 0


@pytest.mark.integration
def test_build_doc_carries_specific_through_when_the_task_defines_it():
    resolved = taskpack.resolve_task_configs(["ifeval|0"])
    config = resolved["ifeval|0"]
    from datasets import load_dataset

    row = load_dataset(
        config.hf_repo, config.hf_subset, split=f"{config.evaluation_splits[0]}[:1]"
    )[0]

    doc = scoring.build_doc(config.prompt_function, row, "ifeval")
    assert doc.specific is not None
    assert "instructions_id_list" in doc.specific


@pytest.mark.integration
def test_build_model_response_strips_reasoning_tags_but_keeps_the_raw_text():
    raw = "<think>scratch work</think>\nfinal answer"
    response = scoring.build_model_response(raw)
    assert response.text == [raw]
    assert response.text_post_processed == ["\nfinal answer"]


@pytest.mark.integration
def test_reasoning_tag_strip_prevents_an_abandoned_boxed_answer_from_winning():
    """The regression guard for the subtlest failure mode in the whole
    Langfuse-native eval plan: an abandoned candidate answer inside
    `<think>...</think>` must not be extractable once the tags are stripped,
    even though it is extractable -- and wins, because `\\boxed{}` has
    extraction priority over a plain "ANSWER:" marker -- when it is not.
    """
    from lighteval.tasks.requests import Doc

    resolved = taskpack.resolve_task_configs(["gsm8k|0"])
    metric = resolved["gsm8k|0"].metrics[0]
    # gold is 18; matches gsm8k_prompt's own choices convention (a leading
    # space before the answer text).
    doc = Doc(query="Question: whatever\nAnswer:", choices=[" 18"], gold_index=0)

    raw = (
        "<think>My first guess is $\\boxed{42}$. Wait, let me redo the "
        "arithmetic.</think>\nANSWER: 18"
    )
    stripped_response = scoring.build_model_response(raw)
    assert "boxed" not in stripped_response.text_post_processed[0]
    # Simulate scoring the same completion *without* stripping, by forcing
    # text_post_processed back to the raw text.
    unstripped_response = replace(stripped_response, text_post_processed=[raw])

    stripped_result = scoring.compute_scores(doc, stripped_response, [metric])
    unstripped_result = scoring.compute_scores(doc, unstripped_response, [metric])

    assert stripped_result.scores["extractive_match"] == 1.0
    assert unstripped_result.scores["extractive_match"] == 0.0
    assert stripped_result.scores != unstripped_result.scores
