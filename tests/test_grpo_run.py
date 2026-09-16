from pathlib import Path

import pytest

from open_r1_tpu.core.config import load_config
from open_r1_tpu.grpo.run import validate_grpo_config

RECIPE = (
    Path(__file__).parents[1]
    / "recipes/OpenR1-Distill-Qwen2.5-Math-1.5B/grpo/config_grpo.yaml"
)


def test_recipe_loads_and_targets_one_device():
    config = load_config(RECIPE, [], validator=validate_grpo_config)
    assert config["model"]["mesh"]["shape"] == [1, 1]
    assert config["model"]["lora_config"]["rank"] == 64


def test_recipe_names_the_merged_sft_export_not_the_pre_sft_base():
    config = load_config(RECIPE, [], validator=validate_grpo_config)
    assert (
        config["model"]["model_path"]
        == "artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/merged"
    )


def test_recipe_rope_theta_matches_the_sft_recipe():
    sft_recipe = (
        Path(__file__).parents[1]
        / "recipes/OpenR1-Distill-Qwen2.5-Math-1.5B/sft/config_distill.yaml"
    )
    grpo_config = load_config(RECIPE, [], validator=validate_grpo_config)
    sft_config = load_config(sft_recipe)
    assert grpo_config["model"]["rope_theta"] == sft_config["model"]["rope_theta"]


def test_recipe_kv_cache_covers_prompt_plus_generation():
    config = load_config(RECIPE, [], validator=validate_grpo_config)
    rollout = config["rollout"]
    assert rollout["kv_cache_size"] >= (
        rollout["max_prompt_length"] + rollout["max_tokens_to_generate"]
    )


def test_recipe_export_is_disabled_until_verified():
    config = load_config(RECIPE, [], validator=validate_grpo_config)
    assert config["export"]["enabled"] is False


def test_missing_lora_config_is_rejected():
    with pytest.raises(ValueError, match="lora_config"):
        load_config(RECIPE, ["model.lora_config=null"], validator=validate_grpo_config)


def test_invalid_mesh_is_rejected():
    with pytest.raises(ValueError, match="axis_names"):
        load_config(
            RECIPE, ["model.mesh.axis_names=[fsdp]"], validator=validate_grpo_config
        )


def test_undersized_kv_cache_is_rejected():
    with pytest.raises(ValueError, match="kv_cache_size"):
        load_config(
            RECIPE, ["rollout.kv_cache_size=10"], validator=validate_grpo_config
        )


def test_zero_num_generations_is_rejected():
    with pytest.raises(ValueError, match="num_generations"):
        load_config(RECIPE, ["grpo.num_generations=0"], validator=validate_grpo_config)


def test_export_enabled_without_verification_is_rejected():
    with pytest.raises(ValueError, match="i_have_verified_qwen2_lora_export"):
        load_config(RECIPE, ["export.enabled=true"], validator=validate_grpo_config)


def test_export_enabled_with_verification_is_accepted():
    config = load_config(
        RECIPE,
        ["export.enabled=true", "export.i_have_verified_qwen2_lora_export=true"],
        validator=validate_grpo_config,
    )
    assert config["export"]["enabled"] is True


def test_missing_section_is_rejected():
    with pytest.raises(ValueError, match="Missing configuration section: grpo"):
        load_config(RECIPE, ["grpo=null"], validator=validate_grpo_config)
