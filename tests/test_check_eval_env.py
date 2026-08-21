import json
from pathlib import Path

import pytest

from open_r1_tpu import check_eval_env
from open_r1_tpu.check_eval_env import check_export_dir, check_task_names
from open_r1_tpu.evaluate import load_eval_config, resolve_settings


def write_export(tmp_path, *, tokenizer_config=None, weights="model.safetensors"):
    directory = tmp_path / "merged"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    if weights:
        (directory / weights).write_bytes(b"")
    if tokenizer_config is not None:
        (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    return directory


def test_a_complete_export_passes(tmp_path):
    directory = write_export(
        tmp_path,
        tokenizer_config={"chat_template": "{{ x }}", "eos_token": "<|im_end|>"},
    )

    errors, warnings = check_export_dir(str(directory))

    assert errors == []
    assert warnings == []


def test_a_missing_directory_is_reported_rather_than_crashing(tmp_path):
    errors, _ = check_export_dir(str(tmp_path / "absent"))

    assert any("not a directory" in error for error in errors)


def test_missing_weights_are_an_error(tmp_path):
    directory = write_export(
        tmp_path, tokenizer_config={"chat_template": "x"}, weights=""
    )

    errors, _ = check_export_dir(str(directory))

    assert any("safetensors" in error for error in errors)


def test_a_sharded_export_is_accepted(tmp_path):
    directory = write_export(
        tmp_path,
        tokenizer_config={"chat_template": "x"},
        weights="model.safetensors.index.json",
    )

    errors, _ = check_export_dir(str(directory))

    assert errors == []


def test_a_missing_chat_template_is_an_error(tmp_path):
    # Without it the server falls back to raw completion and every prompt
    # reaches the model in a format it was never trained on.
    directory = write_export(tmp_path, tokenizer_config={"eos_token": "<|im_end|>"})

    errors, _ = check_export_dir(str(directory))

    assert any("chat template" in error for error in errors)


def test_a_sidecar_template_file_counts_as_a_template(tmp_path):
    directory = write_export(tmp_path, tokenizer_config={"eos_token": "<|im_end|>"})
    (directory / "chat_template.jinja").write_text("{{ x }}")

    errors, _ = check_export_dir(str(directory))

    assert errors == []


def test_a_base_model_eos_warns_about_running_into_the_next_turn(tmp_path):
    # Qwen3-Base names <|endoftext|> as EOS while the chat template closes a
    # turn with <|im_end|>.
    directory = write_export(
        tmp_path,
        tokenizer_config={"chat_template": "x", "eos_token": "<|endoftext|>"},
    )

    errors, warnings = check_export_dir(str(directory))

    assert errors == []
    assert any("<|im_end|>" in warning for warning in warnings)


def test_a_structured_eos_token_is_understood(tmp_path):
    directory = write_export(
        tmp_path,
        tokenizer_config={
            "chat_template": "x",
            "eos_token": {"content": "<|im_end|>"},
        },
    )

    _, warnings = check_export_dir(str(directory))

    assert warnings == []


def test_unparseable_tokenizer_config_is_reported(tmp_path):
    directory = write_export(tmp_path, tokenizer_config={"chat_template": "x"})
    (directory / "tokenizer_config.json").write_text("{not json")

    errors, _ = check_export_dir(str(directory))

    assert any("valid JSON" in error for error in errors)


# A small stand-in for LightEval's registry. The real one is built by a
# constructor that reaches the network, so the unit tests supply the set of
# known names directly; the integration test below is the one that checks the
# recipes against what is actually installed.
KNOWN = {
    "lighteval|gsm8k",
    "lighteval|math_500",
    "lighteval|gpqa:diamond",
    "lighteval|gpqa:mc",
    "extended|ifeval",
    "extended|olympiad_bench:OE_TO_maths_en_COMP",
}


def test_task_names_in_the_registry_pass():
    errors, warnings = check_task_names(
        ["lighteval|gsm8k|0|0", "extended|ifeval|0|0"], known=KNOWN
    )

    assert errors == []
    assert warnings == []


def test_a_task_missing_from_every_suite_is_an_error():
    errors, _ = check_task_names(["lighteval|amc23|0|0"], known=KNOWN)

    assert any("not in LightEval's registry" in error for error in errors)


def test_the_right_name_in_the_wrong_suite_names_the_right_one():
    errors, _ = check_task_names(["lighteval|ifeval|0|0"], known=KNOWN)

    assert len(errors) == 1
    assert "extended|ifeval" in errors[0]


def test_a_near_miss_is_suggested():
    errors, _ = check_task_names(["lighteval|gpqa:diamnod|0|0"], known=KNOWN)

    assert "lighteval|gpqa:diamond" in errors[0]


def test_a_malformed_task_string_is_an_error():
    errors, _ = check_task_names(["gsm8k"], known=KNOWN)

    assert any("suite|name" in error for error in errors)


def test_an_unloaded_suite_warns_rather_than_failing():
    # Community and multilingual tasks are not loaded, so their names cannot be
    # checked. Failing on them would reject a valid recipe.
    errors, warnings = check_task_names(["community|whatever|0|0"], known=KNOWN)

    assert errors == []
    assert any("unchecked" in warning for warning in warnings)


def test_an_unreadable_registry_warns_rather_than_failing_everything(monkeypatch):
    monkeypatch.setattr(check_eval_env, "registry_task_names", lambda: None)

    errors, warnings = check_eval_env.check_task_names(["lighteval|gsm8k|0|0"])

    assert errors == []
    assert any("unchecked" in warning for warning in warnings)


@pytest.mark.integration
def test_every_recipe_task_resolves_against_the_installed_lighteval():
    known = check_eval_env.registry_task_names()
    assert known is not None, "LightEval's registry could not be read"

    for recipe in sorted(Path("recipes").glob("*/eval/*.yaml")):
        settings = resolve_settings(load_eval_config(str(recipe)))
        errors, _ = check_task_names(settings["tasks"], known=known)
        assert errors == [], f"{recipe}: {errors}"
