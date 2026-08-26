"""Tests for `open_r1_tpu.core.config`'s loader mechanics: `extends` merging
and the shared prompt-file reader. Both are schema-agnostic, so these use a
no-op validator rather than either the training or evaluation schema.
"""

from pathlib import Path

import pytest
import yaml

from open_r1_tpu.core.config import load_config, read_prompt_file


def _noop_validator(config):
    pass


def _write_yaml(path: Path, content: dict) -> None:
    path.write_text(yaml.safe_dump(content), encoding="utf-8")


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


def test_read_prompt_file_missing_file_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match="system prompt file not found"):
        read_prompt_file(tmp_path / "absent.txt")


# A test proving training and eval resolve a shared prompt file to identical
# text lives in test_config.py, once both sides read system_prompt_file.
