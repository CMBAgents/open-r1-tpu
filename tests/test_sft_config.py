"""SFT recipes and `open_r1_tpu.sft.config.validate_sft_config`."""

import re
from pathlib import Path

import pytest

from open_r1_tpu.core.config import load_config
from open_r1_tpu.sft.config import validate_sft_config

REPO = Path(__file__).parents[1]
RECIPE = REPO / "recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml"
SFT_RECIPES = sorted(REPO.glob("recipes/*/sft/*.yaml"))


def load(overrides=()):
    return load_config(RECIPE, list(overrides), validator=validate_sft_config)


@pytest.mark.parametrize("recipe", SFT_RECIPES, ids=lambda path: str(path.parent))
def test_every_committed_sft_recipe_validates(recipe):
    load_config(recipe, [], validator=validate_sft_config)


def test_overrides_reach_nested_recipe_keys():
    config = load(["dataset.max_examples=128", "model.mesh.shape=[1, 8]"])
    assert config["dataset"]["max_examples"] == 128
    assert config["model"]["mesh"]["shape"] == [1, 8]


def test_distill_recipe_matches_its_tested_setup():
    config = load()

    # Four chips as [fsdp, tp] = [2, 2]: tp halves the float32 logits of a
    # 26,624-token window, which do not fit one chip whole.
    assert config["model"]["mesh"]["shape"] == [2, 2]
    # LoRA freezes the tied embedding matrix, whose rows carry <|im_end|>.
    assert "lora_config" not in config["model"]
    # The config-edited RoPE-300k base, staged locally so its config.json is
    # what the export copies.
    assert config["model"]["model_source"] == "local"
    assert config["model"]["model_path"].endswith("Qwen2.5-Math-1.5B-RoPE-300k")
    # The splash kernel needs its block size to divide the sequence length,
    # and fails at trace time rather than at load time.
    block_size = config["model"]["flash_attention_block_size"]
    assert config["dataset"]["max_length"] % block_size == 0
    # Deployment-specific values belong in the environment, not the recipe.
    assert config["training"]["wandb"]["entity"] is None
    assert config["training"]["project_name"] == "open-r1-tpu"


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("dataset.overlength_policy=clip", "overlength_policy"),
        ("dataset.message_schema=3", "message_schema"),
        ("dataset.message_schema={from: human}", "unknown"),
        ("dataset.message_schema={role_map: {human: 1}}", "role_map"),
    ],
)
def test_invalid_dataset_policy_config_is_rejected(override, error):
    with pytest.raises(ValueError, match=error):
        load([override])


def test_invalid_mesh_is_rejected():
    with pytest.raises(ValueError, match="axis_names"):
        load(["model.mesh.axis_names=[fsdp]"])


def test_missing_section_is_rejected():
    with pytest.raises(ValueError, match="Missing configuration section: optimizer"):
        load(["optimizer=null"])


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("training.wandb.mode=invalid", "mode"),
        ("training.wandb.tags=[tpu, 1]", "tags"),
    ],
)
def test_invalid_wandb_config_is_rejected(override, error):
    with pytest.raises(ValueError, match=error):
        load([override])


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("dataset.eval_fraction=1.0", "eval_fraction"),
        ("dataset.eval_fraction=-0.1", "eval_fraction"),
        ("dataset.eval_max_examples=0", "eval_max_examples"),
        ("training.transcripts.enabled=maybe", "enabled"),
        ("training.transcripts.every_n_steps=0", "every_n_steps"),
        ("training.transcripts.max_new_tokens=-1", "max_new_tokens"),
        ("training.transcripts.prompts=[]", "prompts"),
        ("training.transcripts.prompts=[1, 2]", "prompts"),
    ],
)
def test_invalid_inspection_config_is_rejected(override, error):
    with pytest.raises(ValueError, match=error):
        load([override])


@pytest.mark.parametrize(
    ("override", "error"),
    [
        (
            "exports={enabled: false}",
            "configuration section exports; did you mean 'export'",
        ),
        ("dataset.max_lenght=512", "key dataset.max_lenght; did you mean 'max_length'"),
        ("optimizer.learning_rat=1e-5", "did you mean 'learning_rate'"),
        ("training.max_step=4", "did you mean 'max_steps'"),
        ("training.wandb.entitiy=team", "did you mean 'entity'"),
        ("training.transcripts.every_n_step=5", "did you mean 'every_n_steps'"),
        ("export.output_directory=/tmp/x", "did you mean 'output_dir'"),
        ("dataset.question_column=q", "Unknown key dataset.question_column"),
    ],
)
def test_unknown_keys_are_rejected_with_the_nearest_match(override, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        load([override])


def test_keys_that_pass_through_to_tunix_are_not_checked():
    config = load(
        [
            "model.some_tunix_field=1",
            "tokenizer.some_tunix_field=1",
            "training.checkpointing_options.some_tunix_field=1",
        ]
    )
    assert config["model"]["some_tunix_field"] == 1


def test_export_must_be_a_mapping():
    with pytest.raises(ValueError, match="export must be a configuration mapping"):
        load(["export=true"])
