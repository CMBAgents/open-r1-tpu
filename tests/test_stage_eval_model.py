import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).parents[1]
RECIPE = REPO / "recipes/Qwen2.5-1.5B/grpo/simplerl-zoo.yaml"

_spec = importlib.util.spec_from_file_location(
    "stage_eval_model", REPO / "scripts/stage_eval_model.py"
)
assert _spec is not None and _spec.loader is not None
stage_eval_model = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(stage_eval_model)


def _source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    (source / "model.safetensors").write_bytes(b"weights")
    (source / "tokenizer_config.json").write_text(
        json.dumps({"chat_template": "chatml", "eos_token": "<|endoftext|>"})
    )
    (source / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 151643, "top_k": 20, "bos_token_id": 151643})
    )
    return source


def test_stage_writes_the_recipe_template_and_stop_tokens(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "staged"
    stage_eval_model.main(
        ["--recipe", str(RECIPE), "--source", str(source), "--output", str(output)]
    )

    from open_r1_tpu.core.config import load_config
    from open_r1_tpu.grpo.config import validate_grpo_config

    recipe = load_config(RECIPE, [], validator=validate_grpo_config)
    tokenizer_config = json.loads((output / "tokenizer_config.json").read_text())
    generation_config = json.loads((output / "generation_config.json").read_text())
    assert tokenizer_config["chat_template"] == recipe["tokenizer"]["chat_template"]
    assert tokenizer_config["eos_token"] == "<|endoftext|>"
    assert generation_config["eos_token_id"] == recipe["rollout"]["eos_token_ids"]
    # Sampling defaults go, so every staged model samples only as asked.
    assert "top_k" not in generation_config
    assert generation_config["bos_token_id"] == 151643
    assert (output / "model.safetensors").read_bytes() == b"weights"
    # The source is untouched.
    source_config = json.loads((source / "tokenizer_config.json").read_text())
    assert source_config["chat_template"] == "chatml"


def test_stage_rewrites_a_separate_template_file(tmp_path):
    source = _source(tmp_path)
    (source / "chat_template.jinja").write_text("chatml")
    output = tmp_path / "staged"
    stage_eval_model.stage(source, output, "abel", [151643], overwrite=False)
    assert (output / "chat_template.jinja").read_text() == "abel"


def test_stage_refuses_to_overwrite_unless_asked(tmp_path):
    source = _source(tmp_path)
    output = tmp_path / "staged"
    stage_eval_model.stage(source, output, "abel", [151643], overwrite=False)
    with pytest.raises(FileExistsError):
        stage_eval_model.stage(source, output, "abel", [151643], overwrite=False)
    stage_eval_model.stage(source, output, "abel", [151643], overwrite=True)


def test_stage_refuses_the_source_itself(tmp_path):
    source = _source(tmp_path)
    with pytest.raises(ValueError, match="refusing"):
        stage_eval_model.stage(source, source, "abel", [151643], overwrite=True)


def test_a_recipe_without_serving_settings_is_rejected():
    with pytest.raises(ValueError, match="chat_template"):
        stage_eval_model.serving_settings({"tokenizer": {}, "rollout": {}})
    with pytest.raises(ValueError, match="eos_token_ids"):
        stage_eval_model.serving_settings(
            {"tokenizer": {"chat_template": "x"}, "rollout": {}}
        )
