import json
import os
from pathlib import Path

import pytest

from open_r1_tpu.evaluation import run as evaluate
from open_r1_tpu.evaluation.stack import VLLM_TPU_BASE_IMAGE, vllm_tpu_image_tag

RECIPE_DIR = Path(__file__).parents[1] / "recipes/Qwen3-1.7B-Math/eval"
TIER0 = RECIPE_DIR / "tier0_smoke.yaml"
TIER1 = RECIPE_DIR / "tier1_core.yaml"
TIER2 = RECIPE_DIR / "tier2_headline.yaml"
TIER3 = RECIPE_DIR / "tier3_regression.yaml"
DISTILL_DIR = Path(__file__).parents[1] / "recipes/DeepSeek-R1-Distill-Qwen-1.5B/eval"
ALL_TIERS = [TIER0, TIER1, TIER2, TIER3, *sorted(DISTILL_DIR.glob("tier*.yaml"))]


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
    settings = evaluate.resolve_settings(evaluate.load_eval_config(recipe))

    assert settings["tasks"]
    assert settings["seeds"]
    assert settings["max_new_tokens"] > 0
    assert settings["serve_command"] == ["scripts/run_vllm_tpu_container.sh"]
    assert settings["server_image"] == vllm_tpu_image_tag()


def test_tier_recipes_keep_deployment_values_neutral():
    for recipe in ALL_TIERS:
        config = evaluate.load_eval_config(recipe)
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
        covered.update(evaluate.load_eval_config(recipe)["eval"]["tasks"])

    assert covered >= MODEL_CARD_TASKS


def test_every_model_card_tier_uses_the_published_token_budget():
    # A trace cut off by the cap scores as wrong, so evaluating below the
    # 32768 tokens the published numbers were measured at would undershoot
    # them for a reason that has nothing to do with the model.
    for recipe in DISTILL_TIERS:
        settings = evaluate.resolve_settings(evaluate.load_eval_config(recipe))
        if not MODEL_CARD_TASKS & set(settings["tasks"]):
            continue
        assert settings["max_new_tokens"] == 32768, recipe.name
        assert settings["max_model_len"] > 32768, recipe.name


def test_the_reference_recipes_never_send_a_system_prompt():
    # DeepSeek's usage guidance for the distills: no system prompt at all.
    # The chat template already opens the reasoning block, and the published
    # numbers were measured this way.
    for recipe in DISTILL_TIERS:
        settings = evaluate.resolve_settings(evaluate.load_eval_config(recipe))
        assert settings["system_prompt"] is None, recipe.name


def test_the_aime_tier_asks_for_the_consensus_number_the_card_reports():
    settings = evaluate.resolve_settings(
        evaluate.load_eval_config(DISTILL_DIR / "tier2_headline.yaml")
    )

    assert settings["consensus"] == {"aime24|0": {"n": 64, "metric": "pass@k:k=1"}}
    # cons@64 votes over the replicates, so the tier has to generate 64 of
    # them -- and pass@1 on a 30-problem benchmark needs them anyway.
    assert len(settings["seeds"]) == 64


def test_smoke_tier_is_greedy_and_capped():
    settings = evaluate.resolve_settings(evaluate.load_eval_config(TIER0))

    assert settings["temperature"] == 0.0
    assert settings["max_samples"] == 200
    # Greedy decoding has no sampling variance, so extra seeds buy nothing.
    assert settings["seeds"] == [0]


def test_headline_tier_runs_enough_seeds_for_a_30_problem_benchmark():
    settings = evaluate.resolve_settings(evaluate.load_eval_config(TIER2))

    assert len(settings["seeds"]) >= 10


def test_regression_tier_drops_the_reasoning_system_prompt():
    # Asking for a reasoning trace is itself an instruction-following failure
    # on IFEval, so this tier measures the model as a plain assistant.
    settings = evaluate.resolve_settings(evaluate.load_eval_config(TIER3))

    assert settings["system_prompt"] is None


def test_system_prompt_file_resolves_to_the_files_text(tmp_path):
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Reason first.\n", encoding="utf-8")

    settings = evaluate.resolve_settings(
        minimal_config(sampling={"system_prompt_file": str(prompt_path)})
    )

    # The trailing newline is stripped so an editor-added one cannot make an
    # otherwise-identical file differ.
    assert settings["system_prompt"] == "Reason first."


def test_an_explicit_null_system_prompt_file_yields_no_prompt():
    settings = evaluate.resolve_settings(
        minimal_config(sampling={"system_prompt_file": None})
    )

    assert settings["system_prompt"] is None


def test_a_missing_system_prompt_file_is_a_clear_error():
    with pytest.raises(ValueError, match="system prompt file not found"):
        evaluate.resolve_settings(
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
        evaluate.validate_eval_config(config)


@pytest.mark.parametrize(
    "key", ["temperature", "top_p", "max_new_tokens", "system_prompt_file"]
)
def test_a_missing_required_sampling_key_names_itself(key):
    config = minimal_config()
    del config["sampling"][key]

    with pytest.raises(ValueError, match=rf"sampling\.{key}"):
        evaluate.validate_eval_config(config)


def test_an_empty_sampling_section_is_rejected():
    # Every measurement-affecting key is required, so an empty section is
    # missing all of them rather than falling back to a default. (Passing
    # sampling={} through minimal_config would merge onto its defaults rather
    # than emptying the section, so the section is replaced directly here.)
    config = minimal_config()
    config["sampling"] = {}

    with pytest.raises(ValueError, match="sampling"):
        evaluate.validate_eval_config(config)


@pytest.mark.parametrize("key", ["reasoning_start", "reasoning_end", "answer_marker"])
def test_a_missing_required_reporting_key_names_itself(key):
    config = minimal_config()
    del config["reporting"][key]

    with pytest.raises(ValueError, match=rf"reporting\.{key}"):
        evaluate.validate_eval_config(config)


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
        evaluate.validate_eval_config(config)


def test_an_unknown_wandb_key_is_rejected():
    with pytest.raises(ValueError, match=r"Unknown key reporting\.wandb\.entty"):
        evaluate.validate_eval_config(
            minimal_config(reporting={"wandb": {"entty": None}})
        )


def test_a_typo_d_dotted_override_is_rejected_the_same_way():
    # Overrides apply before validation runs, so a typo'd override becomes an
    # unknown key here rather than silently doing nothing.
    with pytest.raises(ValueError, match=r"Unknown key sampling\.max_new_token"):
        evaluate.load_eval_config(TIER0, ["sampling.max_new_token=4096"])


def test_wandb_requires_project_name_and_mode_when_enabled():
    with pytest.raises(ValueError, match=r"wandb\.project_name"):
        evaluate.validate_eval_config(
            minimal_config(reporting={"wandb": {"enabled": True}})
        )


def test_wandb_can_omit_project_name_and_mode_when_disabled():
    evaluate.validate_eval_config(
        minimal_config(reporting={"wandb": {"enabled": False}})
    )


def test_a_missing_turn_end_token_names_itself():
    config = minimal_config()
    del config["server"]["turn_end_token"]

    with pytest.raises(ValueError, match=r"server\.turn_end_token"):
        evaluate.validate_eval_config(config)


def test_reasoning_start_may_be_null_but_not_empty():
    # Null means the serving chat template opens the reasoning block inside
    # the prompt itself, so completions carry only the closing tag.
    evaluate.validate_eval_config(minimal_config(reporting={"reasoning_start": None}))

    with pytest.raises(ValueError, match=r"reporting\.reasoning_start"):
        evaluate.validate_eval_config(minimal_config(reporting={"reasoning_start": ""}))


@pytest.mark.parametrize("system_prompt_file", ["", 3, []])
def test_system_prompt_file_must_be_a_non_empty_string_or_null(system_prompt_file):
    with pytest.raises(ValueError, match=r"system_prompt_file"):
        evaluate.validate_eval_config(
            minimal_config(sampling={"system_prompt_file": system_prompt_file})
        )


@pytest.mark.parametrize("tasks", [[], "gsm8k", [""], [1]], ids=str)
def test_tasks_must_be_a_non_empty_list_of_strings(tasks):
    with pytest.raises(ValueError, match=r"eval\.tasks"):
        evaluate.validate_eval_config(minimal_config(eval={"tasks": tasks}))


def test_a_consensus_request_must_name_a_task_the_tier_runs():
    with pytest.raises(ValueError, match=r"eval\.tasks does not run"):
        evaluate.validate_eval_config(
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
        evaluate.validate_eval_config(
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
        evaluate.validate_eval_config(
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
        evaluate.validate_eval_config(
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
        evaluate.validate_eval_config(
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
        evaluate.validate_eval_config(minimal_config(eval={"seeds": [0, 0]}))


def test_context_window_must_leave_room_for_the_prompt():
    with pytest.raises(ValueError, match="max_model_len"):
        evaluate.validate_eval_config(
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
        evaluate.validate_eval_config(minimal_config(sampling=sampling))


def test_invalid_wandb_mode_is_rejected():
    with pytest.raises(ValueError, match=r"wandb\.mode"):
        evaluate.validate_eval_config(
            minimal_config(reporting={"wandb": {"mode": "sometimes"}})
        )


@pytest.mark.parametrize(
    "image",
    ["vllm/vllm-tpu:latest", "vllm/vllm-tpu:v0.27.0", "image@sha256:short"],
)
def test_server_image_must_be_the_derived_tag_or_an_immutable_digest(image):
    with pytest.raises(ValueError, match="derived local"):
        evaluate.validate_eval_config(minimal_config(server={"image": image}))


def test_an_external_server_can_explicitly_disable_the_image():
    evaluate.validate_eval_config(minimal_config(server={"image": None}))


# --- resolved settings and command construction ----------------------------


def test_served_model_name_defaults_to_the_export_directory():
    settings = evaluate.resolve_settings(minimal_config())

    assert settings["served_model_name"] == "model"
    assert settings["base_url"] == "http://127.0.0.1:8000/v1"


def test_the_server_binary_can_live_outside_this_environment():
    # tpu-inference does not support this project's Python, so vLLM is reached
    # wherever it is installed rather than imported from here.
    settings = evaluate.resolve_settings(
        minimal_config(server={"serve_command": ["/opt/vllm-venv/bin/vllm", "serve"]})
    )

    command = evaluate.vllm_serve_command(settings)

    assert command[:2] == ["/opt/vllm-venv/bin/vllm", "serve"]
    assert command[2] == "artifacts/model"


def test_the_default_server_is_the_derived_local_tpu_container():
    settings = evaluate.resolve_settings(minimal_config())

    command = evaluate.vllm_serve_command(settings)

    assert command[:4] == [
        "scripts/run_vllm_tpu_container.sh",
        "--image",
        vllm_tpu_image_tag(),
        "--",
    ]
    assert command[4] == "artifacts/model"


def test_a_containerised_server_command_is_accepted():
    settings = evaluate.resolve_settings(
        minimal_config(
            server={"serve_command": ["docker", "run", "--rm", "img", "serve"]}
        )
    )

    assert evaluate.vllm_serve_command(settings)[:4] == [
        "docker",
        "run",
        "--rm",
        "img",
    ]


@pytest.mark.parametrize("serve_command", [[], "vllm serve", [""], [1]], ids=str)
def test_an_invalid_serve_command_is_rejected(serve_command):
    with pytest.raises(ValueError, match="serve_command"):
        evaluate.validate_eval_config(
            minimal_config(server={"serve_command": serve_command})
        )


def test_the_server_disables_prefix_caching():
    # A prefix-cache hit changes the prefill's kernel shape and therefore the
    # bf16 logits, so greedy completions would depend on server cache state.
    settings = evaluate.resolve_settings(minimal_config())

    assert "--no-enable-prefix-caching" in evaluate.vllm_serve_command(settings)


def test_serve_command_carries_the_recipe_port_and_window():
    settings = evaluate.resolve_settings(
        minimal_config(server={"port": 9001, "max_model_len": 20480})
    )

    command = evaluate.vllm_serve_command(settings)

    assert command[command.index("--port") + 1] == "9001"
    assert command[command.index("--max-model-len") + 1] == "20480"
    assert command[command.index("--served-model-name") + 1] == "model"


# --- reduction -------------------------------------------------------------


def test_aggregate_reports_mean_and_spread_across_seeds():
    aggregated = evaluate.aggregate_across_seeds(
        {
            0: {"t": {"acc": 0.40}},
            1: {"t": {"acc": 0.50}},
            2: {"t": {"acc": 0.60}},
        }
    )

    assert aggregated["t"]["acc"]["mean"] == pytest.approx(0.50)
    assert aggregated["t"]["acc"]["std"] == pytest.approx(0.10)
    assert aggregated["t"]["acc"]["n"] == 3


def test_a_single_seed_reports_no_spread_rather_than_zero_spread():
    # Zero spread from one sample is the exact overclaim this pipeline exists
    # to prevent.
    aggregated = evaluate.aggregate_across_seeds({0: {"t": {"acc": 0.4}}})

    assert aggregated["t"]["acc"]["std"] is None
    assert aggregated["t"]["acc"]["n"] == 1


def test_build_summary_records_the_stack_and_the_sampling_parameters():
    settings = evaluate.resolve_settings(minimal_config())

    summary = evaluate.build_summary(
        settings,
        {0: {"t": {"acc": 0.4}}},
        {0: {"format_rate": 1.0, "truncation_rate": None}},
        {
            "image_id": "sha256:local-image",
            "service_versions": {
                "vllm-tpu": "0.27.0",
                "tpu-inference": "0.27.0",
            },
        },
    )

    assert summary["sampling"]["temperature"] == 0.6
    # Replicates are unseeded on this backend, and an archived summary listing
    # `seeds: [0, 1, 2]` must not be read as reproducible sample-by-sample.
    assert summary["seeded_replicates"] is False
    assert set(summary["stack"]) >= {
        "python",
        "lighteval",
        "openai",
        "latex2sympy2-extended",
    }
    # vLLM runs outside this environment, so the derived service-image contract
    # and complete command are recorded rather than an importable package version.
    assert summary["serve_command"] == ["scripts/run_vllm_tpu_container.sh"]
    assert summary["server_image"] == vllm_tpu_image_tag()
    assert summary["server_command"][:3] == [
        "scripts/run_vllm_tpu_container.sh",
        "--image",
        vllm_tpu_image_tag(),
    ]
    assert summary["server_image_provenance"] == {
        "spec_tag": vllm_tpu_image_tag(),
        "image_id": "sha256:local-image",
        "base_image": VLLM_TPU_BASE_IMAGE,
        "service_versions": {
            "vllm-tpu": "0.27.0",
            "tpu-inference": "0.27.0",
        },
    }
    assert summary["tasks_metrics"]["t"]["acc"]["mean"] == pytest.approx(0.4)
    assert summary["generation"]["format_rate"]["mean"] == pytest.approx(1.0)
    # Absent in every seed, so it stays absent rather than becoming 0.0.
    assert summary["generation"]["truncation_rate"]["mean"] is None


def test_summary_rows_flatten_one_row_per_metric():
    summary = {
        "tier": "t1",
        "tasks_metrics": {"task": {"acc": {"mean": 0.5, "std": 0.1, "n": 3}}},
    }

    assert evaluate.summary_rows(summary) == [["t1", "task", "acc", 0.5, 0.1, 3]]


# --- filesystem ------------------------------------------------------------


def test_summary_round_trips_through_disk(tmp_path):
    path = tmp_path / "nested" / "summary.json"

    evaluate.write_summary(str(path), {"tier": "t", "n": 1})

    assert evaluate.read_json(path) == {"tier": "t", "n": 1}


# --- integration -----------------------------------------------------------
#
# These need a vLLM server already serving the recipe's model. Start one with
# `SKIP_SERVER=0` via scripts/run_eval_tpu.sh, or point OPEN_R1_TPU_EVAL_URL at
# a server that is already up, then run `pytest -m integration`.


@pytest.fixture
def live_settings(tmp_path):
    """Settings pointed at a running server, with the work kept tiny."""
    settings = evaluate.resolve_settings(evaluate.load_eval_config(TIER0))
    base_url = os.environ.get("OPEN_R1_TPU_EVAL_URL")
    if base_url:
        settings["base_url"] = base_url
    settings["max_samples"] = 1
    settings["output_dir"] = str(tmp_path)
    settings["summary_path"] = str(tmp_path / "summary.json")
    settings["wandb"] = {"enabled": False}
    return settings


@pytest.mark.integration
def test_the_served_model_answers_before_a_benchmark_is_committed_to_it(
    live_settings,
):
    import urllib.request

    evaluate.wait_for_server(live_settings["base_url"], timeout_secs=120)

    with urllib.request.urlopen(
        live_settings["base_url"].rstrip("/") + "/models", timeout=30
    ) as response:
        served = json.loads(response.read())

    # The name litellm asks for must be the name vLLM answers to, or every
    # request comes back as model-not-found rather than a completion.
    assert live_settings["served_model_name"] in {
        entry["id"] for entry in served["data"]
    }


@pytest.mark.integration
def test_the_configured_model_export_passes_preflight(live_settings):
    from open_r1_tpu.evaluation.preflight import check_export_dir

    errors, _ = check_export_dir(
        live_settings["model_path"], live_settings["turn_end_token"]
    )

    assert errors == []
