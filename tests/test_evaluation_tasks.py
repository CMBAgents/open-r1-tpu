"""Tests for `open_r1_tpu.evaluation.tasks`.

The pure helpers run everywhere; tests that read the installed LightEval's
registry skip without the eval extra.
"""

from __future__ import annotations

import pytest

from open_r1_tpu.evaluation import tasks

# --- pure helpers, no LightEval needed --------------------------------------


def test_bare_name_strips_fewshot_suffix():
    assert tasks._bare_name("gsm8k|0") == "gsm8k"
    assert tasks._bare_name("gpqa:diamond|0") == "gpqa:diamond"


def test_bare_name_rejects_missing_name():
    with pytest.raises(ValueError, match=r"name\|num_fewshot"):
        tasks._bare_name("|0")


def _fields(**overrides) -> dict:
    fields = {
        "hf_repo": "openai/gsm8k",
        "hf_subset": "main",
        "hf_revision": None,
        "prompt_function_ref": "lighteval.tasks.tasks.gsm8k:gsm8k_prompt",
        "metrics": [{"metric_name": "extractive_match"}],
    }
    fields.update(overrides)
    return fields


def test_dataset_fingerprint_is_eight_hex_characters_and_stable():
    fingerprint = tasks.dataset_fingerprint(_fields())
    assert fingerprint == tasks.dataset_fingerprint(_fields())
    assert len(fingerprint) == 8
    assert all(c in "0123456789abcdef" for c in fingerprint)


def test_dataset_fingerprint_changes_when_a_scoring_field_changes():
    baseline = tasks.dataset_fingerprint(_fields())
    assert tasks.dataset_fingerprint(_fields(hf_revision="abc123")) != baseline
    assert tasks.dataset_fingerprint(_fields(prompt_function_ref="x:y")) != baseline
    changed = _fields(metrics=[{"metric_name": "other_metric"}])
    assert tasks.dataset_fingerprint(changed) != baseline


# --- with the installed LightEval registry ----------------------------------


@pytest.fixture
def lighteval_registry():
    pytest.importorskip("lighteval")


def test_resolve_task_configs_raises_naming_the_bad_task(lighteval_registry):
    with pytest.raises(ValueError, match="not-a-real-task"):
        tasks.resolve_task_configs(["not-a-real-task|0"])


def test_dataset_names_are_unchanged_for_the_pinned_lighteval(lighteval_registry):
    # Existing Langfuse datasets are found by these names; a change here
    # starts every task's dataset afresh.
    resolved = tasks.resolve_task_configs(["gsm8k|0", "math_500|0"])
    assert tasks.dataset_name("gsm8k|0", resolved["gsm8k|0"]) == "gsm8k|0@bfd761c2"
    assert (
        tasks.dataset_name("math_500|0", resolved["math_500|0"])
        == "math_500|0@a48cc6f4"
    )


def test_a_capped_task_gets_its_own_dataset_name(lighteval_registry):
    config = tasks.resolve_task_configs(["gsm8k|0"])["gsm8k|0"]
    full = tasks.dataset_name("gsm8k|0", config)
    assert tasks.dataset_name("gsm8k|0", config, max_samples=200) == f"{full}[:200]"
    assert tasks.dataset_name("gsm8k|0", config, max_samples=None) == full


def test_scoring_fields_record_what_the_protocol_depends_on(lighteval_registry):
    resolved = tasks.resolve_task_configs(["math_500|0", "ifeval|0"])
    math500 = tasks.scoring_fields(resolved["math_500|0"])
    # MATH-500 is scored as pass@1 over one sample, not plain extractive match.
    assert math500["metrics"][0]["metric_name"] == "pass@k:k=1&n=1"
    ifeval = tasks.scoring_fields(resolved["ifeval|0"])
    assert set(ifeval["metrics"][0]["metric_name"]) == {
        "prompt_level_strict_acc",
        "inst_level_strict_acc",
        "prompt_level_loose_acc",
        "inst_level_loose_acc",
    }
