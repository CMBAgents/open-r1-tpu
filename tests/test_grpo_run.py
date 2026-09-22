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


# ---------------------------------------------------------------------------
# rollout.eos_token_ids / rollout.completion_stop_strings
# ---------------------------------------------------------------------------


def test_eos_token_ids_must_be_a_list_of_non_negative_integers():
    with pytest.raises(ValueError, match="eos_token_ids"):
        load_config(
            RECIPE, ["rollout.eos_token_ids=[]"], validator=validate_grpo_config
        )
    with pytest.raises(ValueError, match="eos_token_ids"):
        load_config(
            RECIPE, ["rollout.eos_token_ids=[-1]"], validator=validate_grpo_config
        )
    with pytest.raises(ValueError, match="eos_token_ids"):
        load_config(RECIPE, ["rollout.eos_token_ids=7"], validator=validate_grpo_config)
    config = load_config(
        RECIPE, ["rollout.eos_token_ids=[0]"], validator=validate_grpo_config
    )
    assert config["rollout"]["eos_token_ids"] == [0]


def test_completion_stop_strings_must_be_a_list_of_strings():
    with pytest.raises(ValueError, match="completion_stop_strings"):
        load_config(
            RECIPE,
            ["rollout.completion_stop_strings=[]"],
            validator=validate_grpo_config,
        )
    with pytest.raises(ValueError, match="completion_stop_strings"):
        load_config(
            RECIPE,
            ["rollout.completion_stop_strings=x"],
            validator=validate_grpo_config,
        )


# ---------------------------------------------------------------------------
# The GSM8K recipes for the rowanai and general-purpose bases
# ---------------------------------------------------------------------------

GSM8K_RECIPES = {
    "rowanai": Path(__file__).parents[1]
    / "recipes/rowanai/grpo/config_rowanai_gsm8k.yaml",
    "qwen": Path(__file__).parents[1]
    / "recipes/rowanai/grpo/config_qwen_base_gsm8k.yaml",
}


@pytest.mark.parametrize("arm", sorted(GSM8K_RECIPES))
def test_gsm8k_recipe_loads_with_k_8_on_one_device(arm):
    config = load_config(GSM8K_RECIPES[arm], [], validator=validate_grpo_config)
    assert config["grpo"]["num_generations"] == 8
    assert config["model"]["mesh"]["shape"] == [1, 1]
    assert config["model"]["lora_config"]["rank"] == 64
    assert config["dataset"]["data_files"] == "data/gsm8k-pre1905/train.jsonl"
    assert (
        config["dataset"]["question_column"],
        config["dataset"]["answer_column"],
    ) == (
        "prompt",
        "solution",
    )
    assert config["export"]["enabled"] is False
    rollout = config["rollout"]
    assert rollout["kv_cache_size"] >= (
        rollout["max_prompt_length"] + rollout["max_tokens_to_generate"]
    )


def test_gsm8k_arms_differ_only_in_model_tokenizer_and_output_paths():
    rowanai = load_config(GSM8K_RECIPES["rowanai"], [], validator=validate_grpo_config)
    qwen = load_config(GSM8K_RECIPES["qwen"], [], validator=validate_grpo_config)
    for section in ("dataset", "optimizer", "grpo"):
        assert rowanai[section] == qwen[section], section
    for key in (
        "max_prompt_length",
        "max_tokens_to_generate",
        "kv_cache_size",
        "temperature",
    ):
        assert rowanai["rollout"][key] == qwen["rollout"][key], key


def test_rowanai_gsm8k_recipe_stops_on_document_eos_not_the_turn_marker():
    # rowanai's <|im_end|> is ordinary tokens starting with "<" (ID 29); the
    # default stop token would end generation at the "<" of <think>.
    config = load_config(GSM8K_RECIPES["rowanai"], [], validator=validate_grpo_config)
    assert config["rollout"]["eos_token_ids"] == [0]
    assert config["rollout"]["completion_stop_strings"] == ["<|im_end|>"]
    assert config["model"]["architecture"] == "llama"
    assert config["model"]["rope_theta"] == 50000


def test_qwen_gsm8k_recipe_uses_the_general_purpose_base_with_its_own_template():
    config = load_config(GSM8K_RECIPES["qwen"], [], validator=validate_grpo_config)
    assert config["model"]["model_id"] == "Qwen/Qwen2.5-1.5B"
    assert config["model"]["rope_theta"] == 1000000
    assert config["tokenizer"]["chat_template"] is None
    assert "eos_token_ids" not in config["rollout"]


@pytest.mark.parametrize("arm", sorted(GSM8K_RECIPES))
def test_gsm8k_recipes_start_from_the_worked_solution_sft_export(arm):
    config = load_config(GSM8K_RECIPES[arm], [], validator=validate_grpo_config)
    expected = {"rowanai": "v4-rowanai", "qwen": "v4-qwen-base"}[arm]
    assert (
        config["model"]["model_path"]
        == f"artifacts/rowanai-clean-worked/{expected}/merged"
    )


# ---------------------------------------------------------------------------
# training.eval_rollouts_path and the rollout recorder
# ---------------------------------------------------------------------------


def test_eval_rollouts_path_requires_an_eval_split():
    with pytest.raises(ValueError, match="eval_fraction"):
        load_config(
            RECIPE,
            ["training.eval_rollouts_path=/tmp/x.jsonl"],
            validator=validate_grpo_config,
        )
    with pytest.raises(ValueError, match="eval_rollouts_path"):
        load_config(
            RECIPE,
            ["training.eval_rollouts_path=''", "dataset.eval_fraction=0.1"],
            validator=validate_grpo_config,
        )


def test_rollout_recorder_writes_one_line_per_completion(tmp_path):
    import json

    from open_r1_tpu.grpo.rewards import DEFAULT_REWARD_FNS
    from open_r1_tpu.grpo.run import build_rollout_recorder

    path = tmp_path / "nested" / "eval_rollouts.jsonl"
    record = build_rollout_recorder(str(path), DEFAULT_REWARD_FNS)
    prompts = ["p1", "p2"]
    completions = ["<think>2+2</think>\\boxed{4}", "no idea"]
    n = record(
        prompts,
        completions,
        [6.0, -2.5],
        100,
        "eval",
        question=["q1", "q2"],
        answer=["4", "7"],
        not_a_column=3,
    )
    assert n == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["step"] for r in rows] == [100, 100]
    assert rows[0]["mode"] == "eval"
    assert rows[0]["question"] == "q1" and rows[0]["answer"] == "4"
    assert rows[0]["completion"] == completions[0]
    assert rows[0]["reward"] == 6.0
    assert rows[0]["rewards"] == {"format_reward": 3.0, "correctness_reward": 3.0}
    assert rows[1]["rewards"]["correctness_reward"] == 0.0
    assert "not_a_column" not in rows[0]
    # Appends on the next call rather than overwriting.
    record(prompts[:1], completions[:1], [6.0], 200, "eval", answer=["4"])
    assert len(path.read_text().splitlines()) == 3


def test_rollout_recorder_accepts_numpy_columns(tmp_path):
    """Tunix passes dataset columns as numpy arrays, not lists."""
    import json

    import numpy as np

    from open_r1_tpu.grpo.rewards import DEFAULT_REWARD_FNS
    from open_r1_tpu.grpo.run import build_rollout_recorder

    path = tmp_path / "eval_rollouts.jsonl"
    record = build_rollout_recorder(str(path), DEFAULT_REWARD_FNS)
    n = record(
        np.array(["p1", "p2"]),
        np.array(["<think>2+2</think>\\boxed{4}", "no idea"]),
        [6.0, -2.5],
        0,
        "eval",
        question=np.array(["q1", "q2"]),
        answer=np.array(["4", "7"]),
        a_scalar=np.int64(3),
    )
    assert n == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["answer"] == "4" and rows[0]["question"] == "q1"
    assert rows[0]["rewards"]["correctness_reward"] == 3.0
    assert rows[1]["rewards"]["correctness_reward"] == 0.0
    assert "a_scalar" not in rows[0]


@pytest.mark.parametrize("arm", sorted(GSM8K_RECIPES))
def test_gsm8k_recipes_flash_block_divides_prompt_length(arm):
    """Splash attention requires the block size to divide the query length."""
    config = load_config(GSM8K_RECIPES[arm], [], validator=validate_grpo_config)
    block = config["model"]["flash_attention_block_size"]
    prompt_len = config["rollout"]["max_prompt_length"]
    total_len = prompt_len + config["rollout"]["max_tokens_to_generate"]
    assert prompt_len % block == 0
    assert total_len % block == 0


@pytest.mark.parametrize("arm", sorted(GSM8K_RECIPES))
def test_gsm8k_recipes_record_a_64_prompt_eval_split(arm):
    config = load_config(GSM8K_RECIPES[arm], [], validator=validate_grpo_config)
    assert config["dataset"]["eval_fraction"] > 0
    assert config["dataset"]["eval_max_examples"] == 64
    assert config["training"]["eval_every_n_steps"] == 100
    assert config["training"]["eval_rollouts_path"].endswith("eval_rollouts.jsonl")
