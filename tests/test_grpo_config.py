"""GRPO recipes and `open_r1_tpu.grpo.config.validate_grpo_config`."""

import re
from pathlib import Path

import pytest

from open_r1_tpu.core.config import load_config
from open_r1_tpu.grpo.config import validate_grpo_config

REPO = Path(__file__).parents[1]
RECIPE = REPO / "recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml"
GRPO_RECIPES = sorted(REPO.glob("recipes/*/grpo/*.yaml"))


@pytest.mark.parametrize("recipe", GRPO_RECIPES, ids=lambda path: path.stem)
def test_every_committed_grpo_recipe_validates(recipe):
    load_config(recipe, [], validator=validate_grpo_config)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("rollouts={}", "configuration section rollouts; did you mean 'rollout'"),
        ("grpo.num_generation=4", "key grpo.num_generation; did you mean"),
        ("rollout.temprature=1.0", "did you mean 'temperature'"),
        ("dataset.answer_col=gold", "did you mean 'answer_column'"),
        ("training.eval_rollout_path=x", "did you mean 'eval_rollouts_path'"),
        ("training.gradient_accumulation_steps=2", "Unknown key training.gradient"),
        ("training.wandb.tag=[a]", "did you mean 'tags'"),
        ("optimizer.max_grad=1.0", "did you mean 'max_grad_norm'"),
        ("export.enable=true", "did you mean 'enabled'"),
    ],
)
def test_unknown_keys_are_rejected_with_the_nearest_match(override, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        load_config(RECIPE, [override], validator=validate_grpo_config)


@pytest.mark.parametrize("recipe", GRPO_RECIPES, ids=lambda path: path.stem)
def test_every_recipe_kv_cache_covers_prompt_plus_generation(recipe):
    config = load_config(recipe, [], validator=validate_grpo_config)
    rollout = config["rollout"]
    assert rollout["kv_cache_size"] >= (
        rollout["max_prompt_length"] + rollout["max_tokens_to_generate"]
    )


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


def test_export_can_be_disabled():
    config = load_config(
        RECIPE, ["export.enabled=false"], validator=validate_grpo_config
    )
    assert config["export"]["enabled"] is False


def test_missing_section_is_rejected():
    with pytest.raises(ValueError, match="Missing configuration section: grpo"):
        load_config(RECIPE, ["grpo=null"], validator=validate_grpo_config)


# ---------------------------------------------------------------------------
# rollout.eos_token_ids
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


@pytest.mark.parametrize("names", [["nope"], [], "answer_correctness_reward"])
def test_grpo_reward_functions_are_validated(names):
    with pytest.raises(ValueError):
        load_config(
            RECIPE,
            [f"grpo.reward_functions={names!r}"],
            validator=validate_grpo_config,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        ["training.train_micro_batch_size=0"],
        ["training.train_micro_batch_size=3"],
        ["training.rollout_micro_batch_size=-1"],
        ["training.compute_logps_micro_batch_size=true"],
        ["dataset.eval_batch_size=0"],
    ],
)
def test_grpo_micro_batch_sizes_are_validated(overrides):
    with pytest.raises(ValueError, match="batch_size"):
        load_config(RECIPE, overrides, validator=validate_grpo_config)


def test_grpo_micro_batch_sizes_accept_a_divisor():
    config = load_config(
        RECIPE,
        [
            "training.train_micro_batch_size=1",
            "training.compute_logps_micro_batch_size=1",
        ],
        validator=validate_grpo_config,
    )
    assert config["training"]["train_micro_batch_size"] == 1


# ---------------------------------------------------------------------------
# training.eval_rollouts_path
# ---------------------------------------------------------------------------


def test_eval_rollouts_path_requires_an_eval_split():
    with pytest.raises(ValueError, match="eval_fraction"):
        load_config(
            RECIPE, ["dataset.eval_fraction=0.0"], validator=validate_grpo_config
        )
    with pytest.raises(ValueError, match="eval_rollouts_path"):
        load_config(
            RECIPE, ["training.eval_rollouts_path=''"], validator=validate_grpo_config
        )


# ---------------------------------------------------------------------------
# grpo.loss_agg_mode / grpo.kl_loss_mode
# ---------------------------------------------------------------------------


def test_loss_options_are_optional_and_checked():
    config = load_config(RECIPE, [], validator=validate_grpo_config)
    del config["grpo"]["loss_agg_mode"], config["grpo"]["kl_loss_mode"]
    validate_grpo_config(config)
    with pytest.raises(ValueError, match="loss_agg_mode"):
        load_config(
            RECIPE, ["grpo.loss_agg_mode=token-sum"], validator=validate_grpo_config
        )
    with pytest.raises(ValueError, match="kl_loss_mode"):
        load_config(RECIPE, ["grpo.kl_loss_mode=k3"], validator=validate_grpo_config)


# ---------------------------------------------------------------------------
# The SimpleRL-Zoo positive control
# ---------------------------------------------------------------------------

SIMPLERL_RECIPE = REPO / "recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml"
QWEN25_BASE = REPO / "models/Qwen2.5-1.5B"
ABEL = "Question:\n{}\nAnswer:\nLet's think step by step.\n"


def _render_like_transformers(template, messages):
    # Transformers renders chat templates in this environment.
    jinja2 = pytest.importorskip("jinja2")
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    del jinja2
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    return env.from_string(template).render(
        messages=messages, add_generation_prompt=True
    )


def test_simplerl_recipe_matches_the_published_setup():
    config = load_config(SIMPLERL_RECIPE, [], validator=validate_grpo_config)
    assert config["model"]["model_path"] == "models/Qwen2.5-1.5B"
    assert config["dataset"]["data_files"].endswith(
        "simplelr_abel_level3to5/train.parquet"
    )
    assert config["dataset"]["system_prompt_file"] is None
    assert config["grpo"]["reward_functions"] == ["math_answer_reward"]
    assert config["grpo"]["num_generations"] == 8
    assert config["grpo"]["beta"] == 1e-4
    assert config["grpo"]["loss_agg_mode"] == "token-mean"
    assert config["grpo"]["kl_loss_mode"] == "low_var_kl"
    assert config["rollout"]["temperature"] == 1.0
    assert config["rollout"]["top_p"] == 1.0 and config["rollout"]["top_k"] is None
    # <|endoftext|>, <|im_end|>, "Question", "Answer", "Problem".
    assert config["rollout"]["eos_token_ids"] == [151643, 151645, 14582, 16141, 31198]
    assert config["optimizer"]["min_lr_ratio"] == 1.0
    # The tested configuration: bfloat16 or splash attention corrupts rollouts.
    assert config["model"]["dtype"] == config["model"]["load_dtype"] == "float32"
    assert config["model"]["use_flash_attention"] is False
    assert config["model"]["mesh"]["shape"] == [4, 1]
    assert config["export"]["enabled"] is True
    block = config["model"]["flash_attention_block_size"]
    rollout = config["rollout"]
    assert rollout["max_prompt_length"] % block == 0
    assert (
        rollout["max_prompt_length"] + rollout["max_tokens_to_generate"]
    ) % block == 0
    assert rollout["kv_cache_size"] >= (
        rollout["max_prompt_length"] + rollout["max_tokens_to_generate"]
    )


def test_simplerl_chat_template_renders_their_prompt():
    template = load_config(SIMPLERL_RECIPE, [], validator=validate_grpo_config)[
        "tokenizer"
    ]["chat_template"]
    problem = "Compute $\\sin 315^\\circ$.\n\nGive an exact value."
    user = [{"role": "user", "content": problem}]
    assert _render_like_transformers(template, user) == ABEL.format(problem)
    # LightEval's GSM8K query and MATH-500 instruction are unwrapped.
    gsm8k = [{"role": "user", "content": "Question: How many?\nAnswer:"}]
    assert _render_like_transformers(template, gsm8k) == ABEL.format("How many?")
    math_500 = [
        {
            "role": "user",
            "content": "Solve the following problem. The final line of your "
            'response MUST be of the following format: "ANSWER: $ANSWER" '
            "(without quotes) where $ANSWER is the final answer. Think step by "
            "step before answering.\n\n" + problem,
        }
    ]
    assert _render_like_transformers(template, math_500) == ABEL.format(problem)
    # A system turn is dropped; an assistant turn ends in <|endoftext|>.
    turn = [{"role": "system", "content": "s"}, *user]
    turn.append({"role": "assistant", "content": "1"})
    assert _render_like_transformers(template, turn) == (
        ABEL.format(problem) + "1<|endoftext|>"
    )


def test_simplerl_turn_end_is_endoftext_with_the_real_tokenizer():
    transformers = pytest.importorskip("transformers")
    if not (QWEN25_BASE / "tokenizer_config.json").is_file():
        pytest.skip(f"needs the staged Qwen2.5-1.5B tokenizer at {QWEN25_BASE}")
    from open_r1_tpu.model.tokenizing import assistant_turn_end_id

    config = load_config(SIMPLERL_RECIPE, [], validator=validate_grpo_config)
    tokenizer = transformers.AutoTokenizer.from_pretrained(QWEN25_BASE)
    tokenizer.chat_template = config["tokenizer"]["chat_template"]
    assert assistant_turn_end_id(tokenizer) == 151643
    assert [tokenizer.decode([i]) for i in config["rollout"]["eos_token_ids"]] == [
        "<|endoftext|>",
        "<|im_end|>",
        "Question",
        "Answer",
        "Problem",
    ]
