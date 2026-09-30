"""Tests for `open_r1_tpu.evaluation.taskpack`.

The pure helpers run everywhere. Tests that read the installed LightEval's
registry skip without the eval extra, and the one that renders real examples
from the Hub is marked `network`.
"""

from __future__ import annotations

import pytest

from open_r1_tpu.evaluation import taskpack

# --- pure helpers, no LightEval needed --------------------------------------


def test_bare_name_strips_fewshot_suffix():
    assert taskpack._bare_name("gsm8k|0") == "gsm8k"
    assert taskpack._bare_name("gpqa:diamond|0") == "gpqa:diamond"


def test_bare_name_rejects_missing_name():
    with pytest.raises(ValueError, match=r"name\|num_fewshot"):
        taskpack._bare_name("|0")


def test_diff_strict_reports_every_moved_field():
    committed = {
        "hf_repo": "a",
        "hf_subset": "b",
        "generation_size": 256,
        "stop_sequence": ["Question:"],
    }
    derived = {
        "hf_repo": "a",
        "hf_subset": "c",
        "generation_size": 512,
        "stop_sequence": ["Question:"],
    }
    errors = taskpack._diff_strict("gsm8k|0", committed, derived)
    assert any("hf_subset" in e for e in errors)
    assert any("generation_size" in e for e in errors)
    assert not any("stop_sequence" in e for e in errors)
    assert not any("hf_repo" in e for e in errors)


# --- dataset_fingerprint/dataset_name: pure TaskSpec helpers, no LightEval -


def _spec(**overrides) -> taskpack.TaskSpec:
    fields = {
        "name": "gsm8k|0",
        "hf_repo": "openai/gsm8k",
        "hf_subset": "main",
        "hf_revision": None,
        "hf_avail_splits": ["train", "test"],
        "evaluation_splits": ["test"],
        "few_shots_split": None,
        "few_shots_select": None,
        "num_fewshots": 0,
        "generation_size": 256,
        "stop_sequence": ["Question:"],
        "version": 0,
        "prompt_function_ref": "lighteval.tasks.tasks.gsm8k:gsm8k_prompt",
        "metrics": [
            taskpack.MetricSpec(
                metric_name="extractive_match",
                category="GENERATIVE",
                batched_compute=False,
                higher_is_better=True,
                sample_level_fn_class="lighteval.metrics.utils.MultilingualExtractiveMatchMetric",
                corpus_level_fn={"extractive_match": "mean"},
            )
        ],
        "example": {"query": "irrelevant"},
    }
    fields.update(overrides)
    return taskpack.TaskSpec(**fields)


def test_dataset_fingerprint_is_stable_for_the_same_spec():
    assert taskpack.dataset_fingerprint(_spec()) == taskpack.dataset_fingerprint(
        _spec()
    )


def test_dataset_fingerprint_is_eight_hex_characters():
    fingerprint = taskpack.dataset_fingerprint(_spec())
    assert len(fingerprint) == 8
    assert all(c in "0123456789abcdef" for c in fingerprint)


def test_dataset_fingerprint_changes_when_a_scoring_field_changes():
    baseline = taskpack.dataset_fingerprint(_spec())
    assert taskpack.dataset_fingerprint(_spec(hf_revision="abc123")) != baseline
    assert (
        taskpack.dataset_fingerprint(_spec(prompt_function_ref="other:fn")) != baseline
    )
    changed_metric = taskpack.MetricSpec(
        metric_name="other_metric",
        category="GENERATIVE",
        batched_compute=False,
        higher_is_better=True,
        sample_level_fn_class="lighteval.metrics.utils.MultilingualExtractiveMatchMetric",
        corpus_level_fn={"other_metric": "mean"},
    )
    assert taskpack.dataset_fingerprint(_spec(metrics=[changed_metric])) != baseline


def test_dataset_fingerprint_ignores_non_scoring_fields():
    # generation_size/stop_sequence are recorded for visibility but the
    # recipe always wins over them (module docstring); example is
    # best-effort. None of these should move the fingerprint.
    baseline = taskpack.dataset_fingerprint(_spec())
    assert taskpack.dataset_fingerprint(_spec(generation_size=1)) == baseline
    assert taskpack.dataset_fingerprint(_spec(stop_sequence=[])) == baseline
    assert taskpack.dataset_fingerprint(_spec(example={})) == baseline
    assert taskpack.dataset_fingerprint(_spec(version=99)) == baseline


def test_dataset_name_format():
    spec = _spec()
    name = taskpack.dataset_name("gsm8k|0", spec)
    assert name == f"gsm8k|0@{taskpack.dataset_fingerprint(spec)}"


def test_a_capped_task_gets_its_own_dataset_name():
    spec = _spec()
    capped = taskpack.dataset_name("gsm8k|0", spec, max_samples=200)
    assert capped == f"gsm8k|0@{taskpack.dataset_fingerprint(spec)}[:200]"
    assert capped != taskpack.dataset_name("gsm8k|0", spec)
    assert taskpack.dataset_name("gsm8k|0", spec, max_samples=None) == (
        taskpack.dataset_name("gsm8k|0", spec)
    )


def test_a_large_example_specific_is_summarised_not_committed():
    small = {"instruction_id_list": ["punctuation:no_comma"]}
    assert taskpack._example_specific(small) == small
    assert taskpack._example_specific(None) is None
    large = {"inputs": ["x" * 5_000], "outputs": ["y"]}
    summary = taskpack._example_specific(large)
    assert summary["keys"] == ["inputs", "outputs"]
    assert summary["omitted_chars"] > 5_000


def test_verify_missing_pack_file_is_a_named_error(tmp_path):
    errors, warnings = taskpack.verify_task_specs(tmp_path / "nope.yaml", ["gsm8k|0"])
    assert errors and "could not read task pack" in errors[0]
    assert not warnings


def test_verify_reports_tasks_the_pack_does_not_cover(tmp_path):
    pack_path = tmp_path / "taskpack.yaml"
    taskpack.write_taskpack(
        pack_path, {"lighteval_version": "0.0.0", "tasks": {"gsm8k|0": {}}}
    )
    errors, _ = taskpack.verify_task_specs(pack_path, ["gsm8k|0", "math_500|0"])
    assert any("math_500|0" in e for e in errors)


# --- with the installed LightEval registry ----------------------------------


@pytest.fixture
def lighteval_registry():
    pytest.importorskip("lighteval")


@pytest.fixture
def unfetched_examples(monkeypatch, lighteval_registry):
    """Derive without downloading anything: every example is unavailable."""
    monkeypatch.setattr(
        taskpack, "_render_example", lambda config: {"unavailable": "not fetched"}
    )


@pytest.mark.network
def test_derive_and_verify_round_trip(tmp_path, lighteval_registry):
    pack = taskpack.derive_taskpack(["gsm8k|0", "math_500|0"])
    assert set(pack["tasks"]) == {"gsm8k|0", "math_500|0"}

    math500 = pack["tasks"]["math_500|0"]
    # The known upstream/recipe divergence this pack exists to make visible.
    assert math500["generation_size"] == 32768
    assert math500["metrics"][0]["metric_name"] == "pass@k:k=1&n=1"

    gsm8k = pack["tasks"]["gsm8k|0"]
    assert gsm8k["metrics"][0]["metric_name"] == "extractive_match"
    assert gsm8k["stop_sequence"] == ["Question:"]

    pack_path = tmp_path / "taskpack.yaml"
    taskpack.write_taskpack(pack_path, pack)
    errors, warnings = taskpack.verify_task_specs(pack_path, ["gsm8k|0", "math_500|0"])
    assert errors == []
    # Neither dataset is gated, so both examples render on both sides.
    assert warnings == []


def test_verify_names_the_exact_key_that_moved(tmp_path, unfetched_examples):
    pack = taskpack.derive_taskpack(["math_500|0"])
    pack["tasks"]["math_500|0"]["generation_size"] = 1
    pack_path = tmp_path / "taskpack.yaml"
    taskpack.write_taskpack(pack_path, pack)

    errors, _ = taskpack.verify_task_specs(pack_path, ["math_500|0"])
    assert len(errors) == 1
    assert "generation_size" in errors[0]
    assert "committed=1" in errors[0]
    assert "derived=32768" in errors[0]


def test_resolve_task_configs_raises_naming_the_bad_task(lighteval_registry):
    with pytest.raises(ValueError, match="not-a-real-task"):
        taskpack.resolve_task_configs(["not-a-real-task|0"])


def test_ifeval_metric_grouping_is_recorded_as_a_list(unfetched_examples):
    pack = taskpack.derive_taskpack(["ifeval|0"])
    metric_names = pack["tasks"]["ifeval|0"]["metrics"][0]["metric_name"]
    assert set(metric_names) == {
        "prompt_level_strict_acc",
        "inst_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_loose_acc",
    }


def test_an_unreachable_dataset_degrades_the_example_rather_than_failing(
    tmp_path, monkeypatch, lighteval_registry
):
    def gated(*args, **kwargs):
        raise PermissionError("gpqa is gated on the Hub")

    monkeypatch.setattr("datasets.load_dataset", gated)
    pack = taskpack.derive_taskpack(["gpqa:diamond|0"])
    assert "gated" in pack["tasks"]["gpqa:diamond|0"]["example"]["unavailable"]

    pack_path = tmp_path / "taskpack.yaml"
    taskpack.write_taskpack(pack_path, pack)
    errors, warnings = taskpack.verify_task_specs(pack_path, ["gpqa:diamond|0"])
    # Neither side could render the example, which is not a mismatch.
    assert errors == []
    assert warnings == []
