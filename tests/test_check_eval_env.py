import json

from open_r1_tpu.check_eval_env import check_export_dir


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
