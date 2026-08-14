from pathlib import Path

import pytest

from open_r1_tpu.config import load_config, parse_override


RECIPE = (
    Path(__file__).parents[1]
    / "recipes/OpenR1-Distill-Qwen3-1.7B/sft/config_distill.yaml"
)


def test_parse_override_uses_yaml_types():
    assert parse_override("training.max_steps=12") == ("training.max_steps", 12)
    assert parse_override("export.enabled=false") == ("export.enabled", False)
    assert parse_override("model.mesh.shape=[1, 8]") == (
        "model.mesh.shape",
        [1, 8],
    )


def test_load_config_applies_nested_overrides():
    config = load_config(
        RECIPE,
        ["dataset.max_examples=128", "model.mesh.shape=[1, 8]"],
    )
    assert config["dataset"]["max_examples"] == 128
    assert config["model"]["mesh"]["shape"] == [1, 8]


def test_invalid_mesh_is_rejected():
    with pytest.raises(ValueError, match="axis_names"):
        load_config(RECIPE, ["model.mesh.axis_names=[fsdp]"])

