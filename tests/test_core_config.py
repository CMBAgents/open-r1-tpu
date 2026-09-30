"""`open_r1_tpu.core.config`, the schema-agnostic recipe loader: dotted
overrides, `extends` merging, the prompt-file reader and the unknown-key
checks every stage's validator builds on. Recipes here pass a no-op validator
rather than any stage's schema.
"""

from pathlib import Path

import pytest
import yaml

from open_r1_tpu.core.config import (
    check_sections,
    load_config,
    parse_override,
    read_prompt_file,
    reject_unknown_keys,
)


def _noop_validator(config):
    pass


def _write_yaml(path: Path, content: dict) -> None:
    path.write_text(yaml.safe_dump(content), encoding="utf-8")


# --- dotted overrides ----------------------------------------------------


def test_parse_override_uses_yaml_types():
    assert parse_override("training.max_steps=12") == ("training.max_steps", 12)
    assert parse_override("export.enabled=false") == ("export.enabled", False)
    assert parse_override("model.mesh.shape=[1, 8]") == (
        "model.mesh.shape",
        [1, 8],
    )
    assert parse_override("a.b=x=y") == ("a.b", "x=y")


def test_parse_override_needs_an_equals_sign():
    with pytest.raises(ValueError, match=r"expected section\.key=value"):
        parse_override("training.max_steps")


def test_overrides_replace_nested_keys_and_create_missing_mappings(tmp_path):
    _write_yaml(tmp_path / "recipe.yaml", {"a": {"x": 1}, "b": None})

    config = load_config(
        tmp_path / "recipe.yaml", ["a.x=2", "b.c.d=3"], validator=_noop_validator
    )

    assert config == {"a": {"x": 2}, "b": {"c": {"d": 3}}}


@pytest.mark.parametrize(
    ("override", "error"),
    [("a..x=1", "Invalid override key"), ("a.x.y=1", "not a configuration mapping")],
)
def test_malformed_overrides_are_rejected(tmp_path, override, error):
    _write_yaml(tmp_path / "recipe.yaml", {"a": {"x": 1}})
    with pytest.raises(ValueError, match=error):
        load_config(tmp_path / "recipe.yaml", [override], validator=_noop_validator)


def test_a_recipe_must_be_a_mapping(tmp_path):
    (tmp_path / "recipe.yaml").write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a mapping"):
        load_config(tmp_path / "recipe.yaml", validator=_noop_validator)


# --- extends -----------------------------------------------------------


def test_extends_merges_mappings_and_child_wins_on_scalar_conflict(tmp_path):
    _write_yaml(tmp_path / "base.yaml", {"a": {"x": 1, "y": 2}, "b": "base"})
    _write_yaml(
        tmp_path / "child.yaml",
        {"extends": "base.yaml", "a": {"y": 20, "z": 3}, "b": "child"},
    )

    config = load_config(tmp_path / "child.yaml", validator=_noop_validator)

    assert config == {"a": {"x": 1, "y": 20, "z": 3}, "b": "child"}


def test_extends_replaces_lists_and_scalars_wholesale(tmp_path):
    _write_yaml(tmp_path / "base.yaml", {"tasks": ["a", "b"], "seeds": [0, 1]})
    _write_yaml(tmp_path / "child.yaml", {"extends": "base.yaml", "tasks": ["c"]})

    config = load_config(tmp_path / "child.yaml", validator=_noop_validator)

    assert config == {"tasks": ["c"], "seeds": [0, 1]}


def test_extends_path_is_relative_to_the_declaring_file(tmp_path):
    (tmp_path / "nested").mkdir()
    _write_yaml(tmp_path / "base.yaml", {"a": 1})
    _write_yaml(tmp_path / "nested" / "child.yaml", {"extends": "../base.yaml", "b": 2})

    config = load_config(tmp_path / "nested" / "child.yaml", validator=_noop_validator)

    assert config == {"a": 1, "b": 2}


def test_extends_key_never_reaches_the_validator(tmp_path):
    _write_yaml(tmp_path / "base.yaml", {"a": 1})
    _write_yaml(tmp_path / "child.yaml", {"extends": "base.yaml", "b": 2})
    seen = {}

    load_config(tmp_path / "child.yaml", validator=seen.update)

    assert "extends" not in seen


def test_a_base_recipe_with_its_own_extends_raises(tmp_path):
    _write_yaml(tmp_path / "grandparent.yaml", {"a": 1})
    _write_yaml(tmp_path / "base.yaml", {"extends": "grandparent.yaml", "a": 2})
    _write_yaml(tmp_path / "child.yaml", {"extends": "base.yaml", "b": 3})

    with pytest.raises(ValueError, match="extends"):
        load_config(tmp_path / "child.yaml", validator=_noop_validator)


def test_extends_merges_before_dotted_overrides_apply(tmp_path):
    _write_yaml(tmp_path / "base.yaml", {"a": {"x": 1}})
    _write_yaml(tmp_path / "child.yaml", {"extends": "base.yaml", "a": {"y": 2}})

    config = load_config(tmp_path / "child.yaml", ["a.x=9"], validator=_noop_validator)

    assert config == {"a": {"x": 9, "y": 2}}


def test_a_recipe_without_extends_loads_as_before(tmp_path):
    _write_yaml(tmp_path / "solo.yaml", {"a": 1})

    assert load_config(tmp_path / "solo.yaml", validator=_noop_validator) == {"a": 1}


# --- read_prompt_file ----------------------------------------------------


def test_read_prompt_file_strips_a_trailing_newline(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_text("Reason carefully.\n", encoding="utf-8")

    assert read_prompt_file(path) == "Reason carefully."


def test_read_prompt_file_with_no_trailing_newline_is_unchanged(tmp_path):
    path = tmp_path / "prompt.txt"
    path.write_bytes(b"Reason carefully.")

    assert read_prompt_file(path) == "Reason carefully."


def test_read_prompt_file_passes_a_null_path_through():
    assert read_prompt_file(None) is None


def test_read_prompt_file_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="system prompt file not found"):
        read_prompt_file(tmp_path / "absent.txt")


def test_training_and_eval_resolve_the_same_file_to_identical_text(tmp_path):
    # Both sides funnel through read_prompt_file, but this exercises each
    # caller's own resolution path end to end rather than trusting the
    # shared helper alone -- a recipe that names the same file for both
    # stages must not drift even if one caller's plumbing changes.
    from open_r1_tpu.evaluation.config import resolve_settings as eval_resolve_settings
    from open_r1_tpu.sft.preflight import _preflight_example

    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Shared prompt text.\n", encoding="utf-8")

    _, encode_kwargs = _preflight_example({"system_prompt_file": str(prompt_path)})
    eval_settings = eval_resolve_settings(
        {
            "eval": {"tasks": ["gsm8k|0"], "seeds": [0]},
            "server": {
                "model_path": "artifacts/model",
                "turn_end_token": "<|im_end|>",
                "max_concurrency": 8,
                "fail_fast_after": 10,
            },
            "sampling": {
                "temperature": 0.6,
                "top_p": 0.95,
                "max_new_tokens": 128,
                "system_prompt_file": str(prompt_path),
            },
            "reporting": {
                "reasoning_start": "<think>",
                "reasoning_end": "</think>",
                "answer_marker": "\\boxed{",
            },
        }
    )

    assert encode_kwargs["system_prompt"] == "Shared prompt text."
    assert eval_settings["system_prompt"] == encode_kwargs["system_prompt"]


# --- unknown keys ----------------------------------------------------------


def test_unknown_key_names_its_section_and_the_nearest_match():
    with pytest.raises(
        ValueError, match=r"Unknown key eval\.seed; did you mean 'seeds'"
    ):
        reject_unknown_keys("eval", {"tasks": [], "seed": 0}, {"tasks", "seeds"})


def test_unknown_key_without_a_near_match_gets_no_suggestion():
    with pytest.raises(ValueError) as excinfo:
        reject_unknown_keys("eval", {"zzz": 1}, {"tasks", "seeds"})
    assert str(excinfo.value) == "Unknown key eval.zzz"


def test_known_keys_pass():
    reject_unknown_keys("eval", {"tasks": []}, {"tasks", "seeds"})


def test_sections_are_required_optional_or_unknown():
    check_sections({"model": {}, "export": {}}, ["model"], ["export"])
    check_sections({"model": {}}, ["model"], ["export"])
    with pytest.raises(ValueError, match="Missing configuration section: model"):
        check_sections({"model": None}, ["model"])
    with pytest.raises(ValueError, match="export must be a configuration mapping"):
        check_sections({"model": {}, "export": True}, ["model"], ["export"])
    with pytest.raises(
        ValueError, match="Unknown configuration section modle; did you mean 'model'"
    ):
        check_sections({"modle": {}}, ["model"])
