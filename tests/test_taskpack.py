"""Tests for `open_r1_tpu.evaluation.taskpack`.

Most of this module's real work -- reading `LightevalTaskConfig` off the
installed LightEval's own registry -- needs `lighteval` installed, so those
tests are `@pytest.mark.integration` per the project convention (deselected
by default; run with `pytest -m integration` once the `eval` extra is
installed). The pure helpers (`_bare_name`, `import_prompt_function`, the diff
logic) are tested unconditionally.
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


def test_import_prompt_function_resolves_a_stdlib_callable():
    fn = taskpack.import_prompt_function("json:loads")
    import json

    assert fn is json.loads


def test_import_prompt_function_rejects_a_bare_module():
    with pytest.raises(ValueError, match="module:qualname"):
        taskpack.import_prompt_function("json")


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


# --- integration: needs the real LightEval registry -------------------------


@pytest.mark.integration
def test_derive_and_verify_round_trip(tmp_path):
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
    # No warnings expected here: neither dataset is gated, so both examples
    # should render on both sides.
    assert warnings == []


@pytest.mark.integration
def test_verify_names_the_exact_key_that_moved(tmp_path):
    pack = taskpack.derive_taskpack(["math_500|0"])
    pack["tasks"]["math_500|0"]["generation_size"] = 1
    pack_path = tmp_path / "taskpack.yaml"
    taskpack.write_taskpack(pack_path, pack)

    errors, _ = taskpack.verify_task_specs(pack_path, ["math_500|0"])
    assert len(errors) == 1
    assert "generation_size" in errors[0]
    assert "committed=1" in errors[0]
    assert "derived=32768" in errors[0]


@pytest.mark.integration
def test_resolve_task_configs_raises_naming_the_bad_task():
    with pytest.raises(ValueError, match="not-a-real-task"):
        taskpack.resolve_task_configs(["not-a-real-task|0"])


@pytest.mark.integration
def test_ifeval_metric_grouping_is_recorded_as_a_list():
    pack = taskpack.derive_taskpack(["ifeval|0"])
    metric_names = pack["tasks"]["ifeval|0"]["metrics"][0]["metric_name"]
    assert set(metric_names) == {
        "prompt_level_strict_acc",
        "inst_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_loose_acc",
    }


@pytest.mark.integration
def test_gated_dataset_degrades_example_to_a_warning_not_a_failure(tmp_path):
    pack = taskpack.derive_taskpack(["gpqa:diamond|0"])
    assert "unavailable" in pack["tasks"]["gpqa:diamond|0"]["example"]

    pack_path = tmp_path / "taskpack.yaml"
    taskpack.write_taskpack(pack_path, pack)
    errors, warnings = taskpack.verify_task_specs(pack_path, ["gpqa:diamond|0"])
    # Both sides are equally unable to reach the gated dataset -- that is not
    # a mismatch, so no warning either.
    assert errors == []
    assert warnings == []
