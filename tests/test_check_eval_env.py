import json
from pathlib import Path

import pytest

from open_r1_tpu.evaluation import preflight as check_eval_env
from open_r1_tpu.evaluation.preflight import (
    check_dependency_versions,
    check_export_dir,
    check_server_runtime,
    check_task_names,
)
from open_r1_tpu.evaluation.run import load_eval_config, resolve_settings
from open_r1_tpu.evaluation.stack import (
    EVALUATION_PACKAGE_VERSIONS,
    EVALUATION_PYTHON_VERSION,
    VLLM_TPU_SERVICE_VERSIONS,
    vllm_tpu_image_tag,
)


def write_export(tmp_path, *, tokenizer_config=None, weights="model.safetensors"):
    directory = tmp_path / "merged"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    if weights:
        (directory / weights).write_bytes(b"")
    if tokenizer_config is not None:
        (directory / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    return directory


def test_the_validated_dependency_stack_passes():
    assert (
        check_dependency_versions(
            EVALUATION_PACKAGE_VERSIONS,
            python_version=EVALUATION_PYTHON_VERSION,
        )
        == []
    )


def test_dependency_drift_is_an_error():
    installed = dict(EVALUATION_PACKAGE_VERSIONS)
    installed["lighteval"] = "99.0.0"

    errors = check_dependency_versions(
        installed,
        python_version=EVALUATION_PYTHON_VERSION,
    )

    assert any("lighteval is 99.0.0, expected 0.13.0" in error for error in errors)


def test_python_drift_is_an_error():
    errors = check_dependency_versions(
        EVALUATION_PACKAGE_VERSIONS,
        python_version="3.13.13",
    )

    assert any("Python is 3.13.13" in error for error in errors)


def test_external_server_runtime_is_reported_as_unchecked():
    settings = resolve_settings(
        {
            "eval": {"tasks": ["gsm8k|0"], "seeds": [0]},
            "server": {
                "model_path": "models/example",
                "serve_command": ["vllm", "serve"],
                "image": None,
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
    )

    errors, warnings = check_server_runtime(settings)

    assert errors == []
    assert any("not reproducibility-checked" in warning for warning in warnings)


def test_container_runtime_check_uses_the_derived_image_and_versions(tmp_path):
    checker = tmp_path / "run_vllm_tpu_container.sh"
    checker.write_text(
        "#!/usr/bin/env bash\n"
        f"[[ $1 == --image && $2 == '{vllm_tpu_image_tag()}' ]] || exit 1\n"
        "if [[ $3 == --check ]]; then exit 0; fi\n"
        "if [[ $3 == --provenance ]]; then\n"
        "  printf '%s' "
        '\'{"image_id":"sha256:local","service_versions":\'\n'
        "  printf '%s\\n' "
        '\'{"vllm-tpu":"0.27.0","tpu-inference":"0.27.0"}}\'\n'
        "  exit 0\n"
        "fi\n"
        "exit 1\n"
    )
    checker.chmod(0o755)
    settings = {
        "serve_command": [str(checker)],
        "server_image": vllm_tpu_image_tag(),
    }

    assert check_server_runtime(settings) == ([], [])


def test_container_runtime_check_names_the_build_command_when_image_is_absent(tmp_path):
    checker = tmp_path / "run_vllm_tpu_container.sh"
    checker.write_text(
        "#!/usr/bin/env bash\n"
        "echo 'Run scripts/run_vllm_tpu_container.sh --build first.' >&2\n"
        "exit 1\n"
    )
    checker.chmod(0o755)

    errors, warnings = check_server_runtime(
        {"serve_command": [str(checker)], "server_image": vllm_tpu_image_tag()}
    )

    assert warnings == []
    assert "scripts/run_vllm_tpu_container.sh --build" in errors[0]


def test_container_runtime_check_rejects_wrong_service_versions(tmp_path):
    checker = tmp_path / "run_vllm_tpu_container.sh"
    checker.write_text(
        "#!/usr/bin/env bash\n"
        "if [[ $3 == --check ]]; then exit 0; fi\n"
        "printf '%s' "
        '\'{"image_id":"sha256:local","service_versions":\'\n'
        "printf '%s\\n' "
        '\'{"vllm-tpu":"0.26.0","tpu-inference":"0.27.0"}}\'\n'
    )
    checker.chmod(0o755)

    errors, _ = check_server_runtime(
        {"serve_command": [str(checker)], "server_image": vllm_tpu_image_tag()}
    )

    assert any(
        f"vllm-tpu 0.26.0, expected {VLLM_TPU_SERVICE_VERSIONS['vllm-tpu']}" in error
        for error in errors
    )


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
    "gsm8k",
    "math_500",
    "gpqa:diamond",
    "gpqa:mc",
    "ifeval",
    "olympiad_bench:OE_TO_maths_en_COMP",
}


def test_task_names_in_the_registry_pass():
    errors, warnings = check_task_names(["gsm8k|0", "ifeval|0"], known=KNOWN)

    assert errors == []
    assert warnings == []


def test_a_task_missing_from_the_registry_is_an_error():
    errors, _ = check_task_names(["amc23|0"], known=KNOWN)

    assert any("not in LightEval's registry" in error for error in errors)


def test_a_suite_prefix_warns_and_names_the_form_that_replaced_it():
    # 0.13 keys its registry by bare name and discards a leading suite without
    # saying anything useful, so the recipe drifts from what actually ran.
    errors, warnings = check_task_names(["lighteval|gsm8k|0"], known=KNOWN)

    assert errors == []
    assert len(warnings) == 1
    assert "'gsm8k|0'" in warnings[0]


def test_a_near_miss_is_suggested():
    errors, _ = check_task_names(["gpqa:diamnod|0"], known=KNOWN)

    assert "gpqa:diamond" in errors[0]


def test_a_task_string_with_no_few_shot_field_is_an_error():
    errors, _ = check_task_names(["gsm8k"], known=KNOWN)

    assert any("name|num_fewshot" in error for error in errors)


def test_the_removed_fourth_task_field_is_rejected():
    # LightEval 0.13 fails to resolve a four-field name rather than warning.
    errors, _ = check_task_names(["lighteval|gsm8k|0|0"], known=KNOWN)

    assert len(errors) == 1
    assert "'gsm8k|0'" in errors[0]


def test_an_unreadable_registry_warns_rather_than_failing_everything(monkeypatch):
    monkeypatch.setattr(check_eval_env, "registry_task_names", lambda: None)

    errors, warnings = check_eval_env.check_task_names(["gsm8k|0"])

    assert errors == []
    assert any("unchecked" in warning for warning in warnings)


@pytest.mark.integration
def test_every_recipe_task_resolves_against_the_installed_lighteval():
    known = check_eval_env.registry_task_names()
    assert known is not None, "LightEval's registry could not be read"

    # base.yaml is not a standalone recipe -- it has no eval:/sampling: of its
    # own and is only ever reached through another recipe's `extends`.
    for recipe in sorted(Path("recipes").glob("*/eval/*.yaml")):
        if recipe.name == "base.yaml":
            continue
        settings = resolve_settings(load_eval_config(str(recipe)))
        errors, warnings = check_task_names(settings["tasks"], known=known)
        assert errors == [], f"{recipe}: {errors}"
        assert warnings == [], f"{recipe}: {warnings}"
