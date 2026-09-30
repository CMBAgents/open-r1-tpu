"""Tests for `open_r1_tpu.evaluation.config`: the committed eval recipes,
recipe validation, and the resolved settings.
"""

import json
from pathlib import Path

import pytest

from open_r1_tpu.evaluation import config as eval_config
from open_r1_tpu.evaluation import server as eval_server
from open_r1_tpu.evaluation.stack import vllm_tpu_image_tag

RECIPE_DIR = Path(__file__).parents[1] / "recipes/Qwen2.5-Math-1.5B/eval"
TIER0 = RECIPE_DIR / "tier0_smoke.yaml"
TIER1 = RECIPE_DIR / "tier1_core.yaml"
TIER2 = RECIPE_DIR / "tier2_headline.yaml"
TIER3 = RECIPE_DIR / "tier3_regression.yaml"
DISTILL_DIR = Path(__file__).parents[1] / "recipes/DeepSeek-R1-Distill-Qwen-1.5B/eval"
SIMPLERL_DIR = Path(__file__).parents[1] / "recipes/Qwen2.5-1.5B/eval"
ALL_TIERS = [
    TIER0,
    TIER1,
    TIER2,
    TIER3,
    *sorted(DISTILL_DIR.glob("tier*.yaml")),
    *sorted(SIMPLERL_DIR.glob("tier*.yaml")),
]


def minimal_config(**overrides):
    config = {
        "eval": {"tier": "t", "tasks": ["suite|task|0"], "seeds": [0]},
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
            "system_prompt_file": None,
        },
        "reporting": {
            "reasoning_start": "<think>",
            "reasoning_end": "</think>",
            "answer_marker": "\\boxed{",
        },
    }
    for section, values in overrides.items():
        config[section] = {**config[section], **values}
    return config


# --- recipes ---------------------------------------------------------------


@pytest.mark.parametrize("recipe", ALL_TIERS, ids=lambda p: p.stem)
def test_every_tier_recipe_loads(recipe):
    settings = eval_config.resolve_settings(eval_config.load_eval_config(recipe))

    assert settings["tasks"]
    assert settings["seeds"]
    assert settings["max_new_tokens"] > 0
    assert settings["serve_command"] == ["scripts/run_vllm_tpu_container.sh"]
    assert settings["server_image"] == vllm_tpu_image_tag()


def test_tier_recipes_keep_deployment_values_neutral():
    for recipe in ALL_TIERS:
        config = eval_config.load_eval_config(recipe)
        assert config["reporting"]["wandb"]["entity"] is None
        assert config["reporting"]["wandb"]["project_name"] == "open-r1-tpu"
        assert not config["server"]["model_path"].startswith("gs://")


# The DeepSeek-R1-Distill-Qwen-1.5B recipes exist to replicate the published
# model card, so the rows and the recipe set are checked against each other.
# CodeForces (954) is absent on purpose: LightEval 0.13.0 ships no CodeForces
# task and no Elo harness, and the published rating is a percentile placement
# against human contestants rather than a benchmark accuracy.
DISTILL_TIERS = sorted(DISTILL_DIR.glob("tier*.yaml"))
MODEL_CARD_TASKS = {
    "aime24|0",  # 28.9 pass@1, 52.7 cons@64
    "math_500|0",  # 83.9 pass@1
    "gpqa:diamond|0",  # 33.8 pass@1
    "lcb:codegeneration|0",  # 16.9 pass@1
}


def test_the_reference_recipes_cover_every_measurable_model_card_row():
    covered = set()
    for recipe in DISTILL_TIERS:
        covered.update(eval_config.load_eval_config(recipe)["eval"]["tasks"])

    assert covered >= MODEL_CARD_TASKS


def test_every_model_card_tier_uses_the_published_token_budget():
    # A trace cut off by the cap scores as wrong, so evaluating below the
    # 32768 tokens the published numbers were measured at would undershoot
    # them for a reason that has nothing to do with the model.
    for recipe in DISTILL_TIERS:
        settings = eval_config.resolve_settings(eval_config.load_eval_config(recipe))
        if not MODEL_CARD_TASKS & set(settings["tasks"]):
            continue
        if recipe.name == "tier1_core.yaml":
            # math_500 runs there at the project's comparison protocol, not
            # the card's; the tier-1 parity test below pins what it must
            # match instead.
            continue
        assert settings["max_new_tokens"] == 32768, recipe.name
        assert settings["max_model_len"] > 32768, recipe.name


def test_the_reference_recipes_never_send_a_system_prompt():
    # DeepSeek's usage guidance for the distills: no system prompt at all.
    # The chat template already opens the reasoning block, and the published
    # numbers were measured this way.
    for recipe in DISTILL_TIERS:
        settings = eval_config.resolve_settings(eval_config.load_eval_config(recipe))
        assert settings["system_prompt"] is None, recipe.name


def test_the_simplerl_tier_runs_their_generation_settings():
    # SimpleRL-Zoo scored its trained models at temperature 1.0, top-p 0.95
    # and 16,000 new tokens, on the plain-text prompt the served chat
    # template renders, which has no system turn.
    settings = eval_config.resolve_settings(
        eval_config.load_eval_config(SIMPLERL_DIR / "tier1_core.yaml")
    )
    assert settings["tasks"] == ["gsm8k|0", "math_500|0"]
    assert settings["temperature"] == 1.0
    assert settings["top_p"] == 0.95
    assert settings["max_new_tokens"] == 16000
    assert settings["system_prompt"] is None
    assert settings["turn_end_token"] == "<|endoftext|>"
    assert len(settings["seeds"]) == 3


def test_the_reference_tier1_runs_the_comparison_protocol():
    # Tier 1 measures the published distill and the project's own 1.5B export
    # on MATH-500 under identical generation parameters; only the prompt side
    # may differ (DeepSeek's distills run without a system prompt, the
    # project export keeps the prompt it was trained with).
    reference = eval_config.resolve_settings(
        eval_config.load_eval_config(DISTILL_DIR / "tier1_core.yaml")
    )
    project = eval_config.resolve_settings(eval_config.load_eval_config(TIER1))

    assert reference["tasks"] == ["math_500|0"]
    for key in (
        "tasks",
        "seeds",
        "max_samples",
        "consensus",
        "max_model_len",
        "temperature",
        "top_p",
        "max_new_tokens",
    ):
        assert reference[key] == project[key], key
    assert reference["system_prompt"] is None
    assert project["system_prompt"] is not None


def test_the_aime_tier_asks_for_the_consensus_number_the_card_reports():
    settings = eval_config.resolve_settings(
        eval_config.load_eval_config(DISTILL_DIR / "tier2_headline.yaml")
    )

    assert settings["consensus"] == {"aime24|0": {"n": 64, "metric": "pass@k:k=1"}}
    # cons@64 votes over the replicates, so the tier has to generate 64 of
    # them -- and pass@1 on a 30-problem benchmark needs them anyway.
    assert len(settings["seeds"]) == 64


def test_the_qwen25_math_tiers_declare_enough_context_for_serving():
    # Stock Qwen2.5-Math-1.5B declares max_position_embeddings 4096 and an
    # export copies its base's config, so vLLM could refuse every tier's
    # window at startup. The recipes declare a larger context through
    # --hf-overrides and leave rope_theta alone: an export must be served at
    # the theta it was trained at.
    for recipe in sorted(RECIPE_DIR.glob("tier*.yaml")):
        settings = eval_config.resolve_settings(eval_config.load_eval_config(recipe))
        command = eval_server.vllm_serve_command(settings)

        overrides = json.loads(command[command.index("--hf-overrides") + 1])
        assert overrides["max_position_embeddings"] > settings["max_model_len"], (
            recipe.name
        )
        assert "rope_theta" not in overrides, recipe.name


def test_smoke_tier_is_greedy_and_capped():
    settings = eval_config.resolve_settings(eval_config.load_eval_config(TIER0))

    assert settings["temperature"] == 0.0
    assert settings["max_samples"] == 200
    # Greedy decoding has no sampling variance, so extra seeds buy nothing.
    assert settings["seeds"] == [0]


def test_headline_tier_runs_enough_seeds_for_a_30_problem_benchmark():
    settings = eval_config.resolve_settings(eval_config.load_eval_config(TIER2))

    assert len(settings["seeds"]) >= 10


def test_regression_tier_drops_the_reasoning_system_prompt():
    # Asking for a reasoning trace is itself an instruction-following failure
    # on IFEval, so this tier measures the model as a plain assistant.
    settings = eval_config.resolve_settings(eval_config.load_eval_config(TIER3))

    assert settings["system_prompt"] is None


def test_system_prompt_file_resolves_to_the_files_text(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Reason first.\n", encoding="utf-8")

    settings = eval_config.resolve_settings(
        minimal_config(sampling={"system_prompt_file": str(prompt_path)})
    )

    # The trailing newline is stripped so an editor-added one cannot make an
    # otherwise-identical file differ.
    assert settings["system_prompt"] == "Reason first."


def test_an_explicit_null_system_prompt_file_yields_no_prompt():
    settings = eval_config.resolve_settings(
        minimal_config(sampling={"system_prompt_file": None})
    )

    assert settings["system_prompt"] is None


def test_a_missing_system_prompt_file_is_a_clear_error():
    with pytest.raises(ValueError, match="system prompt file not found"):
        eval_config.resolve_settings(
            minimal_config(
                sampling={"system_prompt_file": "recipes/does/not/exist.txt"}
            )
        )


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "section", ["eval", "server", "sampling", "reporting"], ids=str
)
def test_every_section_is_required(section):
    config = minimal_config()
    del config[section]

    with pytest.raises(ValueError, match=section):
        eval_config.validate_eval_config(config)


@pytest.mark.parametrize(
    "key", ["temperature", "top_p", "max_new_tokens", "system_prompt_file"]
)
def test_a_missing_required_sampling_key_names_itself(key):
    config = minimal_config()
    del config["sampling"][key]

    with pytest.raises(ValueError, match=rf"sampling\.{key}"):
        eval_config.validate_eval_config(config)


def test_an_empty_sampling_section_is_rejected():
    # Every measurement-affecting key is required, so an empty section is
    # missing all of them rather than falling back to a default. (Passing
    # sampling={} through minimal_config would merge onto its defaults rather
    # than emptying the section, so the section is replaced directly here.)
    config = minimal_config()
    config["sampling"] = {}

    with pytest.raises(ValueError, match="sampling"):
        eval_config.validate_eval_config(config)


@pytest.mark.parametrize("key", ["reasoning_start", "reasoning_end", "answer_marker"])
def test_a_missing_required_reporting_key_names_itself(key):
    config = minimal_config()
    del config["reporting"][key]

    with pytest.raises(ValueError, match=rf"reporting\.{key}"):
        eval_config.validate_eval_config(config)


@pytest.mark.parametrize(
    ("section", "key", "suggestion"),
    [
        ("eval", "tsaks", "tasks"),
        ("server", "modle_path", "model_path"),
        ("sampling", "temperatur", "temperature"),
        ("reporting", "reasoning_strat", "reasoning_start"),
    ],
)
def test_an_unknown_key_is_rejected_with_a_close_match_suggestion(
    section, key, suggestion
):
    config = minimal_config(**{section: {key: "x"}})

    with pytest.raises(
        ValueError, match=rf"Unknown key {section}\.{key}.*{suggestion}"
    ):
        eval_config.validate_eval_config(config)


def test_an_unknown_wandb_key_is_rejected():
    with pytest.raises(ValueError, match=r"Unknown key reporting\.wandb\.entty"):
        eval_config.validate_eval_config(
            minimal_config(reporting={"wandb": {"entty": None}})
        )


def test_a_typo_d_dotted_override_is_rejected_the_same_way():
    # Overrides apply before validation runs, so a typo'd override becomes an
    # unknown key here rather than silently doing nothing.
    with pytest.raises(ValueError, match=r"Unknown key sampling\.max_new_token"):
        eval_config.load_eval_config(TIER0, ["sampling.max_new_token=4096"])


def test_a_typo_d_section_in_an_override_is_rejected():
    with pytest.raises(
        ValueError,
        match="Unknown configuration section samplng; did you mean 'sampling'",
    ):
        eval_config.load_eval_config(TIER0, ["samplng.temperature=0.9"])


def test_wandb_requires_project_name_and_mode_when_enabled():
    with pytest.raises(ValueError, match=r"wandb\.project_name"):
        eval_config.validate_eval_config(
            minimal_config(reporting={"wandb": {"enabled": True}})
        )


def test_wandb_can_omit_project_name_and_mode_when_disabled():
    eval_config.validate_eval_config(
        minimal_config(reporting={"wandb": {"enabled": False}})
    )


def test_a_missing_turn_end_token_names_itself():
    config = minimal_config()
    del config["server"]["turn_end_token"]

    with pytest.raises(ValueError, match=r"server\.turn_end_token"):
        eval_config.validate_eval_config(config)


def test_reasoning_start_may_be_null_but_not_empty():
    # Null means the serving chat template opens the reasoning block inside
    # the prompt itself, so completions carry only the closing tag.
    eval_config.validate_eval_config(
        minimal_config(reporting={"reasoning_start": None})
    )

    with pytest.raises(ValueError, match=r"reporting\.reasoning_start"):
        eval_config.validate_eval_config(
            minimal_config(reporting={"reasoning_start": ""})
        )


@pytest.mark.parametrize("system_prompt_file", ["", 3, []])
def test_system_prompt_file_must_be_a_non_empty_string_or_null(system_prompt_file):
    with pytest.raises(ValueError, match=r"system_prompt_file"):
        eval_config.validate_eval_config(
            minimal_config(sampling={"system_prompt_file": system_prompt_file})
        )


@pytest.mark.parametrize("tasks", [[], "gsm8k", [""], [1]], ids=str)
def test_tasks_must_be_a_non_empty_list_of_strings(tasks):
    with pytest.raises(ValueError, match=r"eval\.tasks"):
        eval_config.validate_eval_config(minimal_config(eval={"tasks": tasks}))


def test_a_consensus_request_must_name_a_task_the_tier_runs():
    with pytest.raises(ValueError, match=r"eval\.tasks does not run"):
        eval_config.validate_eval_config(
            minimal_config(
                eval={
                    "tasks": ["aime24|0"],
                    "seeds": [0, 1],
                    "consensus": {"math_500|0": {"n": 2, "metric": "pass@k:k=1"}},
                }
            )
        )


def test_a_consensus_cannot_vote_over_more_replicates_than_the_tier_runs():
    # Caught at load time rather than after the generations have been paid
    # for: the replicates are the samples, so cons@64 over ten seeds is not a
    # number that exists.
    with pytest.raises(ValueError, match="only 10 replicate"):
        eval_config.validate_eval_config(
            minimal_config(
                eval={
                    "tasks": ["aime24|0"],
                    "seeds": list(range(10)),
                    "consensus": {"aime24|0": {"n": 64, "metric": "pass@k:k=1"}},
                }
            )
        )


@pytest.mark.parametrize("n", [1, 0, -1, "64", True], ids=str)
def test_a_consensus_over_fewer_than_two_samples_is_rejected(n):
    with pytest.raises(ValueError, match="at least 2"):
        eval_config.validate_eval_config(
            minimal_config(
                eval={
                    "tasks": ["aime24|0"],
                    "seeds": list(range(64)),
                    "consensus": {"aime24|0": {"n": n, "metric": "pass@k:k=1"}},
                }
            )
        )


def test_a_consensus_must_name_the_metric_that_judges_it():
    # A task declares several metrics (aime24 declares pass@k:k=1 and
    # avg@n:n=1); picking one by position would make a headline number depend
    # on LightEval's declaration order.
    with pytest.raises(ValueError, match=r"eval\.consensus\['aime24\|0'\]\.metric"):
        eval_config.validate_eval_config(
            minimal_config(
                eval={
                    "tasks": ["aime24|0"],
                    "seeds": [0, 1],
                    "consensus": {"aime24|0": {"n": 2}},
                }
            )
        )


def test_an_unknown_consensus_key_is_rejected():
    with pytest.raises(ValueError, match=r"Unknown key eval\.consensus"):
        eval_config.validate_eval_config(
            minimal_config(
                eval={
                    "tasks": ["aime24|0"],
                    "seeds": [0, 1],
                    "consensus": {"aime24|0": {"n": 2, "metric": "pass@k:k=1", "k": 1}},
                }
            )
        )


def test_repeated_seeds_are_rejected():
    # Two identical seeds produce two identical runs and a standard deviation
    # of zero, which reads as a precise result rather than a duplicated one.
    with pytest.raises(ValueError, match="repeat"):
        eval_config.validate_eval_config(minimal_config(eval={"seeds": [0, 0]}))


def test_context_window_must_leave_room_for_the_prompt():
    with pytest.raises(ValueError, match="max_model_len"):
        eval_config.validate_eval_config(
            minimal_config(
                server={"max_model_len": 4096}, sampling={"max_new_tokens": 4096}
            )
        )


@pytest.mark.parametrize(
    "sampling",
    [{"temperature": -0.1}, {"top_p": 0.0}, {"top_p": 1.5}, {"max_new_tokens": 0}],
    ids=str,
)
def test_invalid_sampling_parameters_are_rejected(sampling):
    with pytest.raises(ValueError, match="sampling"):
        eval_config.validate_eval_config(minimal_config(sampling=sampling))


def test_invalid_wandb_mode_is_rejected():
    with pytest.raises(ValueError, match=r"wandb\.mode"):
        eval_config.validate_eval_config(
            minimal_config(reporting={"wandb": {"mode": "sometimes"}})
        )


@pytest.mark.parametrize(
    "image",
    ["vllm/vllm-tpu:latest", "vllm/vllm-tpu:v0.27.0", "image@sha256:short"],
)
def test_server_image_must_be_the_derived_tag_or_an_immutable_digest(image):
    with pytest.raises(ValueError, match="derived local"):
        eval_config.validate_eval_config(minimal_config(server={"image": image}))


def test_an_external_server_can_explicitly_disable_the_image():
    eval_config.validate_eval_config(minimal_config(server={"image": None}))


@pytest.mark.parametrize("serve_command", [[], "vllm serve", [""], [1]], ids=str)
def test_an_invalid_serve_command_is_rejected(serve_command):
    with pytest.raises(ValueError, match="serve_command"):
        eval_config.validate_eval_config(
            minimal_config(server={"serve_command": serve_command})
        )


# --- resolved settings -------------------------------------------------------


def test_served_model_name_defaults_to_the_export_directory():
    settings = eval_config.resolve_settings(minimal_config())

    assert settings["served_model_name"] == "model"
    assert settings["base_url"] == "http://127.0.0.1:8000/v1"


def test_the_default_server_is_the_container_wrapper_with_the_derived_image():
    settings = eval_config.resolve_settings(minimal_config())

    assert settings["serve_command"] == ["scripts/run_vllm_tpu_container.sh"]
    assert settings["server_image"] == vllm_tpu_image_tag()


def test_a_custom_serve_command_has_no_image_unless_the_recipe_names_one():
    settings = eval_config.resolve_settings(
        minimal_config(server={"serve_command": ["vllm", "serve"]})
    )

    assert settings["server_image"] is None
