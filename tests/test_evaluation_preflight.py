"""Tests for `open_r1_tpu.evaluation.preflight`."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from open_r1_tpu.evaluation import preflight
from open_r1_tpu.evaluation.config import load_eval_config, resolve_settings
from open_r1_tpu.evaluation.preflight import (
    check_dependency_versions,
    check_export_dir,
    check_server_runtime,
)
from open_r1_tpu.evaluation.stack import (
    EVALUATION_PACKAGE_VERSIONS,
    EVALUATION_PYTHON_VERSION,
    VLLM_TPU_SERVICE_VERSIONS,
    vllm_tpu_image_tag,
)


def test_config_is_required(monkeypatch):
    # No default recipe for an expensive run: a missing --config must fail
    # argument parsing rather than silently picking one.
    monkeypatch.setattr(sys, "argv", ["preflight"])

    with pytest.raises(SystemExit):
        preflight.main()


# Qwen3's own id for <|im_end|>; the value only has to be internally
# consistent within a fixture, since `check_export_dir` reads it from the
# files rather than hardcoding it.
TURN_END_TOKEN_ID = 151645


def write_export(
    tmp_path,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    weights: str = "model.safetensors",
    generation_config: Any = "default",
):
    """Write a merged-export fixture. Defaults to a complete, passing export.

    `tokenizer_config`, when given, is merged onto a base that names
    <|im_end|>'s token id via `added_tokens_decoder`, the shape a real Qwen3
    export carries; pass `{"added_tokens_decoder": {}}` in it to simulate an
    export whose tokenizer files do not name the id. `generation_config`
    defaults to one whose `eos_token_id` includes that id; pass an explicit
    mapping, or None, to test a mismatched or missing one.
    """
    directory = tmp_path / "merged"
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    if weights:
        (directory / weights).write_bytes(b"")
    if tokenizer_config is not None:
        merged_tokenizer_config = {
            "added_tokens_decoder": {
                str(TURN_END_TOKEN_ID): {"content": "<|im_end|>", "special": True}
            },
            **tokenizer_config,
        }
        (directory / "tokenizer_config.json").write_text(
            json.dumps(merged_tokenizer_config)
        )
    if generation_config == "default":
        generation_config = {"eos_token_id": [151643, TURN_END_TOKEN_ID]}
    if generation_config is not None:
        (directory / "generation_config.json").write_text(json.dumps(generation_config))
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
                "turn_end_token": "<|im_end|>",
                "serve_command": ["vllm", "serve"],
                "image": None,
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
    directory = write_export(tmp_path, tokenizer_config={"chat_template": "{{ x }}"})

    assert check_export_dir(str(directory), "<|im_end|>") == []


def test_a_missing_directory_is_reported_rather_than_crashing(tmp_path):
    errors = check_export_dir(str(tmp_path / "absent"), "<|im_end|>")

    assert any("not a directory" in error for error in errors)


def test_missing_weights_are_an_error(tmp_path):
    directory = write_export(
        tmp_path, tokenizer_config={"chat_template": "x"}, weights=""
    )

    errors = check_export_dir(str(directory), "<|im_end|>")

    assert any("safetensors" in error for error in errors)


def test_a_sharded_export_is_accepted(tmp_path):
    directory = write_export(
        tmp_path,
        tokenizer_config={"chat_template": "x"},
        weights="model.safetensors.index.json",
    )

    errors = check_export_dir(str(directory), "<|im_end|>")

    assert errors == []


def test_a_missing_chat_template_is_an_error(tmp_path):
    # Without it the server falls back to raw completion and every prompt
    # reaches the model in a format it was never trained on.
    directory = write_export(tmp_path, tokenizer_config={})

    errors = check_export_dir(str(directory), "<|im_end|>")

    assert any("chat template" in error for error in errors)


def test_a_sidecar_template_file_counts_as_a_template(tmp_path):
    directory = write_export(tmp_path, tokenizer_config={})
    (directory / "chat_template.jinja").write_text("{{ x }}")

    errors = check_export_dir(str(directory), "<|im_end|>")

    assert errors == []


def test_unparseable_tokenizer_config_is_reported(tmp_path):
    directory = write_export(tmp_path, tokenizer_config={"chat_template": "x"})
    (directory / "tokenizer_config.json").write_text("{not json")

    errors = check_export_dir(str(directory), "<|im_end|>")

    assert any("valid JSON" in error for error in errors)


# --- turn-end token id / generation_config.json ------------------------------
#
# A stop string cannot end a turn (vLLM matches it against text with special
# tokens stripped), so the export's generation_config.json must name the
# turn-end token as an EOS, and anything else is an error.


def test_missing_generation_config_is_a_hard_error(tmp_path):
    directory = write_export(
        tmp_path, tokenizer_config={"chat_template": "x"}, generation_config=None
    )

    errors = check_export_dir(str(directory), "<|im_end|>")

    assert any("generation_config.json" in error for error in errors)


def test_eos_token_id_missing_the_turn_end_id_is_a_hard_error(tmp_path):
    directory = write_export(
        tmp_path,
        tokenizer_config={"chat_template": "x"},
        # <|endoftext|>'s id only, the way Qwen3-Base's own generation config
        # reads before it is corrected to include the chat turn-end token.
        generation_config={"eos_token_id": [151643]},
    )

    errors = check_export_dir(str(directory), "<|im_end|>")

    assert any("eos_token_id" in error and "<|im_end|>" in error for error in errors)


def test_a_scalar_eos_token_id_is_accepted(tmp_path):
    directory = write_export(
        tmp_path,
        tokenizer_config={"chat_template": "x"},
        generation_config={"eos_token_id": TURN_END_TOKEN_ID},
    )

    errors = check_export_dir(str(directory), "<|im_end|>")

    assert errors == []


def test_unresolvable_turn_end_token_id_is_a_hard_error(tmp_path):
    # Neither tokenizer_config.json nor tokenizer.json names the id, so there
    # is nothing to check the generation config against.
    directory = write_export(
        tmp_path,
        tokenizer_config={"chat_template": "x", "added_tokens_decoder": {}},
    )

    errors = check_export_dir(str(directory), "<|im_end|>")

    assert any("cannot verify" in error for error in errors)


def test_the_turn_end_token_id_falls_back_to_tokenizer_json(tmp_path):
    directory = write_export(
        tmp_path,
        tokenizer_config={"chat_template": "x", "added_tokens_decoder": {}},
    )
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {"added_tokens": [{"id": TURN_END_TOKEN_ID, "content": "<|im_end|>"}]}
        )
    )

    errors = check_export_dir(str(directory), "<|im_end|>")

    assert errors == []


def test_a_model_with_its_own_turn_end_token_passes(tmp_path):
    # DeepSeek's distills carry no <|im_end|> at all: the turn ends on the
    # tokenizer's end-of-sentence token, which only tokenizer.json names, and
    # generation_config.json holds its id as a bare scalar.
    directory = write_export(
        tmp_path,
        tokenizer_config={"chat_template": "x", "added_tokens_decoder": {}},
        generation_config={"eos_token_id": 151643},
    )
    (directory / "tokenizer.json").write_text(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 151643, "content": "<｜end▁of▁sentence｜>"}  # noqa: RUF001
                ]
            }
        )
    )

    errors = check_export_dir(str(directory), "<｜end▁of▁sentence｜>")  # noqa: RUF001

    assert errors == []


def test_every_recipe_task_resolves_against_the_installed_lighteval():
    pytest.importorskip("lighteval")
    from open_r1_tpu.evaluation.taskpack import resolve_task_configs

    # base.yaml is not a standalone recipe -- it has no eval:/sampling: of its
    # own and is only ever reached through another recipe's `extends`.
    recipes = Path(__file__).parents[1] / "recipes"
    for recipe in sorted(recipes.glob("*/eval/*.yaml")):
        if recipe.name == "base.yaml":
            continue
        settings = resolve_settings(load_eval_config(str(recipe)))
        resolved = resolve_task_configs(settings["tasks"])
        assert sorted(resolved) == sorted(settings["tasks"]), recipe
