import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from open_r1_tpu.evaluation import run as evaluate
from open_r1_tpu.evaluation.stack import VLLM_TPU_BASE_IMAGE, vllm_tpu_image_tag

RECIPE_DIR = Path(__file__).parents[1] / "recipes/Qwen3-1.7B-Math/eval"
TIER0 = RECIPE_DIR / "tier0_smoke.yaml"
TIER1 = RECIPE_DIR / "tier1_core.yaml"
TIER2 = RECIPE_DIR / "tier2_headline.yaml"
TIER3 = RECIPE_DIR / "tier3_regression.yaml"
ALL_TIERS = [TIER0, TIER1, TIER2, TIER3]


def minimal_config(**overrides):
    config = {
        "eval": {"tier": "t", "tasks": ["suite|task|0"], "seeds": [0]},
        "server": {"model_path": "artifacts/model"},
        "sampling": {},
        "reporting": {},
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
    config = evaluate.litellm_model_config(settings, seed=0)
    assert "system_prompt" not in config["model_parameters"]


def test_the_system_prompt_travels_in_the_model_config_not_the_command_line():
    # `endpoint litellm` has no --system-prompt flag; system_prompt is a field
    # on LightEval's ModelConfig, read from the model_parameters mapping.
    settings = evaluate.resolve_settings(evaluate.load_eval_config(TIER0))

    config = evaluate.litellm_model_config(settings, seed=0)
    command = evaluate.lighteval_command(settings, "m.yaml", "out")

    assert config["model_parameters"]["system_prompt"] == settings["system_prompt"]
    assert not any(argument.startswith("--system") for argument in command)


# --- validation ------------------------------------------------------------


@pytest.mark.parametrize(
    "section", ["eval", "server", "sampling", "reporting"], ids=str
)
def test_every_section_is_required(section):
    config = minimal_config()
    del config[section]

    with pytest.raises(ValueError, match=section):
        evaluate.validate_eval_config(config)


@pytest.mark.parametrize("tasks", [[], "gsm8k", [""], [1]], ids=str)
def test_tasks_must_be_a_non_empty_list_of_strings(tasks):
    with pytest.raises(ValueError, match=r"eval\.tasks"):
        evaluate.validate_eval_config(minimal_config(eval={"tasks": tasks}))


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


def test_stop_tokens_default_to_the_chat_turn_boundary():
    # Qwen3-Base's EOS is <|endoftext|>, but the chat template ends a turn with
    # <|im_end|>; without this the model writes the user's next turn too.
    settings = evaluate.resolve_settings(minimal_config())

    assert settings["stop"] == ["<|im_end|>"]


def test_stop_tokens_can_be_disabled_explicitly():
    settings = evaluate.resolve_settings(minimal_config(sampling={"stop": []}))

    assert settings["stop"] == []
    assert (
        "stop_tokens"
        not in evaluate.litellm_model_config(settings, 0)["model_parameters"][
            "generation_parameters"
        ]
    )


def test_litellm_config_threads_the_seed_and_targets_the_local_server():
    settings = evaluate.resolve_settings(minimal_config())

    parameters = evaluate.litellm_model_config(settings, 7)["model_parameters"]

    assert parameters["model_name"] == "hosted_vllm/model"
    assert parameters["base_url"] == "http://127.0.0.1:8000/v1"
    assert parameters["generation_parameters"]["seed"] == 7


def test_lighteval_command_joins_tasks_and_appends_extra_args():
    settings = evaluate.resolve_settings(
        minimal_config(
            eval={
                "tasks": ["suite|a|0", "suite|b|0"],
                "max_samples": 8,
                "extra_args": ["--custom-tasks", "tasks.py"],
            }
        )
    )

    command = evaluate.lighteval_command(settings, "model.yaml", "out")

    assert "suite|a|0,suite|b|0" in command
    assert command[-2:] == ["--custom-tasks", "tasks.py"]
    assert "--save-details" in command
    assert command[command.index("--max-samples") + 1] == "8"


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


def test_serve_command_carries_the_recipe_port_and_window():
    settings = evaluate.resolve_settings(
        minimal_config(server={"port": 9001, "max_model_len": 20480})
    )

    command = evaluate.vllm_serve_command(settings)

    assert command[command.index("--port") + 1] == "9001"
    assert command[command.index("--max-model-len") + 1] == "20480"
    assert command[command.index("--served-model-name") + 1] == "model"


# --- reading LightEval details ---------------------------------------------


def write_details_shard(
    output_dir, records, *, task="gsm8k", timestamp="2026-01-01", legacy=False
):
    """Write a real LightEval detail shard.

    Mirrors the layout the harness actually produces -- Parquet, one row per
    document, generations nested in a struct under `model_response` -- so the
    reader is exercised against the real format rather than a dict that happens
    to match today's field names. `legacy` writes the dunder-wrapped column
    names LightEval used up to 0.12.
    """
    directory = Path(output_dir) / "details" / "model" / timestamp
    directory.mkdir(parents=True, exist_ok=True)
    response, doc = (
        ("__model_response__", "__doc__") if legacy else ("model_response", "doc")
    )
    table = pa.table(
        {
            response: records,
            doc: [{"gold": "4"}] * len(records),
        }
    )
    path = directory / f"details_{task}_{timestamp}.parquet"
    pq.write_table(table, path)
    return path


def read_stats(tmp_path, records, **kwargs):
    """Round-trip records through a real Parquet shard and summarize them."""
    write_details_shard(tmp_path, records)
    responses = evaluate.read_detail_responses(evaluate.find_details_files(tmp_path))
    return evaluate.completion_stats(responses, **kwargs)


def test_completions_survive_a_parquet_round_trip(tmp_path):
    write_details_shard(tmp_path, [{"text": ["a", "b"], "output_tokens": [1, 2]}])

    responses = evaluate.read_detail_responses(evaluate.find_details_files(tmp_path))

    assert evaluate.extract_completions(responses[0]) == ["a", "b"]
    assert evaluate.extract_token_counts(responses[0]) == [1, 2]


def test_completions_are_read_from_a_json_string_column(tmp_path):
    # Some releases serialize the struct rather than nesting it.
    write_details_shard(tmp_path, [json.dumps({"text": ["a"], "output_tokens": [3]})])

    responses = evaluate.read_detail_responses(evaluate.find_details_files(tmp_path))

    assert evaluate.extract_completions(responses[0]) == ["a"]


def test_unrecognised_response_shape_names_the_keys_it_found():
    # LightEval renames these fields between releases. Guessing would report a
    # format rate of zero rather than an error.
    with pytest.raises(ValueError, match="surprise"):
        evaluate.extract_completions({"surprise": "x"})


def test_a_shard_without_the_generation_column_is_rejected(tmp_path):
    directory = tmp_path / "details" / "model" / "2026-01-01"
    directory.mkdir(parents=True)
    pq.write_table(pa.table({"doc": ["x"]}), directory / "details_t_2026-01-01.parquet")

    with pytest.raises(ValueError, match="model_response"):
        evaluate.read_detail_responses(evaluate.find_details_files(tmp_path))


def test_token_id_lists_are_counted_rather_than_summed(tmp_path):
    # Some releases store the generated token ids in place of a count.
    stats = read_stats(
        tmp_path,
        [{"text": ["<think>a</think>\\boxed{1}"], "output_tokens": [[7, 8, 9]]}],
        max_new_tokens=100,
    )

    assert stats["mean_completion_tokens"] == pytest.approx(3.0)


def test_completion_stats_separate_format_from_correctness(tmp_path):
    stats = read_stats(
        tmp_path,
        [
            # Well formed: closed trace and a boxed answer.
            {"text": ["<think>reason</think> so \\boxed{4}"], "output_tokens": [10]},
            # Reasoned but never gave a boxed answer.
            {"text": ["<think>reason</think> four"], "output_tokens": [10]},
            # Ran out of budget mid-trace.
            {"text": ["<think>reason and reason"], "output_tokens": [100]},
        ],
        max_new_tokens=100,
    )

    assert stats["completions"] == 3
    assert stats["format_rate"] == pytest.approx(1 / 3)
    assert stats["reasoning_closed_rate"] == pytest.approx(2 / 3)
    assert stats["answer_marker_rate"] == pytest.approx(1 / 3)
    assert stats["truncation_rate"] == pytest.approx(1 / 3)
    assert stats["mean_completion_tokens"] == pytest.approx(40.0)


def test_a_closing_tag_before_an_opening_one_is_not_a_closed_trace(tmp_path):
    stats = read_stats(
        tmp_path,
        [{"text": ["</think> answer \\boxed{4} <think>"], "output_tokens": [5]}],
        max_new_tokens=100,
    )

    assert stats["reasoning_closed_rate"] == 0.0
    assert stats["format_rate"] == 0.0


def test_missing_token_counts_report_null_rather_than_a_guess(tmp_path):
    stats = read_stats(
        tmp_path,
        [{"text": ["<think>a</think>\\boxed{1}"], "output_tokens": None}],
        max_new_tokens=100,
    )

    assert stats["truncation_rate"] is None
    assert stats["mean_completion_tokens"] is None
    # Character length always survives, so length is never wholly lost.
    assert stats["mean_completion_chars"] > 0
    assert stats["format_rate"] == 1.0


# --- reduction -------------------------------------------------------------


def test_normalise_results_drops_stderr_and_non_numeric_metrics():
    results = {
        "results": {
            "task|0": {
                "extractive_match": 0.5,
                "extractive_match_stderr": 0.1,
                "pass@4": 0.75,
                "note": "text",
                "flag": True,
            },
        }
    }

    assert evaluate.normalise_results(results) == {
        "task|0": {"extractive_match": 0.5, "pass@4": 0.75}
    }


def test_normalise_results_rejects_a_file_with_no_results():
    with pytest.raises(ValueError, match="results"):
        evaluate.normalise_results({"config_general": {}})


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

    assert summary["sampling"]["temperature"] == evaluate.DEFAULT_TEMPERATURE
    assert set(summary["stack"]) >= {
        "python",
        "lighteval",
        "litellm",
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


def test_results_file_is_found_under_the_output_directory(tmp_path):
    results = tmp_path / "results" / "model"
    results.mkdir(parents=True)
    (results / "results_2026-01-01T00-00-00.json").write_text("{}")
    (results / "results_2026-02-01T00-00-00.json").write_text("{}")

    found = evaluate.find_results_file(tmp_path)

    assert found.name == "results_2026-02-01T00-00-00.json"


def test_a_missing_results_file_explains_that_the_harness_failed(tmp_path):
    with pytest.raises(FileNotFoundError, match="harness"):
        evaluate.find_results_file(tmp_path)


def test_only_the_newest_run_of_details_is_read(tmp_path):
    # Re-running into the same output directory must not average the previous
    # run's generations into the new one.
    base = tmp_path / "details" / "model"
    for timestamp in ("2026-01-01", "2026-02-01"):
        (base / timestamp).mkdir(parents=True)
        (base / timestamp / f"details_gsm8k_{timestamp}.parquet").write_bytes(b"")

    found = evaluate.find_details_files(tmp_path)

    assert [path.parent.name for path in found] == ["2026-02-01"]


def test_no_details_directory_is_not_an_error(tmp_path):
    assert evaluate.find_details_files(tmp_path) == []


# --- reduction over real harness output ------------------------------------


def write_results_file(seed_dir, scores):
    """Write a results file in the layout LightEval produces."""
    results = Path(seed_dir) / "results" / "model"
    results.mkdir(parents=True, exist_ok=True)
    path = results / "results_2026-01-01T00-00-00.json"
    path.write_text(json.dumps({"results": scores, "config_general": {}}))
    return path


def test_reduction_reads_a_real_results_file_off_disk(tmp_path):
    # Everything run_seed does after the subprocess returns. The subprocess
    # itself is covered by the integration tests below, which run the harness
    # for real rather than standing in for it.
    write_results_file(tmp_path, {"t|0": {"acc": 0.5, "acc_stderr": 0.2}})

    metrics = evaluate.normalise_results(
        evaluate.read_json(evaluate.find_results_file(tmp_path))
    )

    assert metrics == {"t|0": {"acc": 0.5}}


def test_a_summary_survives_the_full_reduction_from_real_files(tmp_path):
    for seed in (1, 2):
        seed_dir = tmp_path / f"seed-{seed}"
        write_results_file(seed_dir, {"t|0": {"acc": 0.4 + 0.2 * seed}})
        write_details_shard(
            seed_dir, [{"text": ["<think>a</think>\\boxed{1}"], "output_tokens": [9]}]
        )

    settings = evaluate.resolve_settings(minimal_config())
    per_seed_metrics = {}
    per_seed_stats = {}
    for seed in (1, 2):
        seed_dir = tmp_path / f"seed-{seed}"
        per_seed_metrics[seed] = evaluate.normalise_results(
            evaluate.read_json(evaluate.find_results_file(seed_dir))
        )
        per_seed_stats[seed] = evaluate.completion_stats(
            evaluate.read_detail_responses(evaluate.find_details_files(seed_dir)),
            max_new_tokens=settings["max_new_tokens"],
        )

    summary = evaluate.build_summary(settings, per_seed_metrics, per_seed_stats)
    evaluate.write_summary(str(tmp_path / "summary.json"), summary)

    assert summary["tasks_metrics"]["t|0"]["acc"]["mean"] == pytest.approx(0.7)
    assert summary["tasks_metrics"]["t|0"]["acc"]["std"] is not None
    assert summary["generation"]["format_rate"]["mean"] == pytest.approx(1.0)
    assert evaluate.read_json(tmp_path / "summary.json") == json.loads(
        json.dumps(summary)
    )


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
def test_run_seed_drives_lighteval_against_the_live_server(live_settings, tmp_path):
    metrics, stats = evaluate.run_seed(live_settings, 0, tmp_path)

    # One task, one sample: the numbers are meaningless, the plumbing is not.
    assert metrics
    assert all(
        isinstance(value, float) for m in metrics.values() for value in m.values()
    )
    assert stats["completions"] >= 1


@pytest.mark.integration
def test_the_configured_model_export_passes_preflight(live_settings):
    from open_r1_tpu.evaluation.preflight import check_export_dir

    errors, _ = check_export_dir(live_settings["model_path"])

    assert errors == []


def test_a_shard_written_by_an_older_lighteval_is_still_readable(tmp_path):
    # LightEval wrapped its detail columns in dunders up to 0.12. Results
    # already on disk stay readable after an upgrade.
    write_details_shard(
        tmp_path,
        [json.dumps({"text": ["<think>x</think> \\boxed{4}"]})],
        legacy=True,
    )

    responses = evaluate.read_detail_responses(evaluate.find_details_files(tmp_path))

    assert len(responses) == 1
