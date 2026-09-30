"""The parts of `open_r1_tpu.model.loading` that need no TPU, JAX or Tunix."""

from pathlib import Path

import pytest

from open_r1_tpu.core.config import load_config
from open_r1_tpu.model.loading import absolute_checkpoint_dir, validate_mesh_config
from open_r1_tpu.sft.config import validate_sft_config

RECIPE = (
    Path(__file__).parents[1] / "recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml"
)


@pytest.mark.parametrize(
    ("mesh", "error"),
    [
        (None, "shape"),
        ({"shape": [], "axis_names": []}, "shape"),
        ({"shape": [2, 0], "axis_names": ["fsdp", "tp"]}, "shape"),
        ({"shape": [2, 2], "axis_names": ["fsdp"]}, "axis_names"),
    ],
)
def test_invalid_meshes_are_rejected(mesh, error):
    with pytest.raises(ValueError, match=error):
        validate_mesh_config({"mesh": mesh})


def test_relative_checkpoint_dir_is_made_absolute():
    # Orbax raises "Checkpoint path should be absolute" for a relative
    # directory, which the recipe uses by default.
    resolved = absolute_checkpoint_dir("artifacts/run/checkpoints")
    assert Path(resolved).is_absolute()
    assert resolved.endswith("artifacts/run/checkpoints")


def test_absolute_and_remote_checkpoint_dirs_are_preserved():
    assert absolute_checkpoint_dir("/data/run/checkpoints") == ("/data/run/checkpoints")
    # Resolving a URI against the working directory would corrupt the scheme.
    assert absolute_checkpoint_dir("gs://bucket/run/checkpoints") == (
        "gs://bucket/run/checkpoints"
    )


def test_recipe_checkpoint_dir_resolves_to_an_absolute_path():
    config = load_config(RECIPE, validator=validate_sft_config)
    assert Path(
        absolute_checkpoint_dir(config["training"]["checkpoint_dir"])
    ).is_absolute()
