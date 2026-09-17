"""Checks against the actual staged tokenizers; private assets stay untracked."""

import json
from pathlib import Path

import pytest

from open_r1_tpu.core.config import load_config
from open_r1_tpu.model.export import write_turn_end_generation_config
from open_r1_tpu.model.tokenizing import as_token_ids
from open_r1_tpu.sft.data import encode_reasoning_example


@pytest.fixture(params=["rowanai", "qwen"])
def chat_arm(request):
    transformers = pytest.importorskip("transformers")
    config = load_config(f"recipes/rowanai/sft/config_{request.param}.yaml")
    path = Path(config["tokenizer"]["tokenizer_path"])
    if not (path / "tokenizer.json").exists():
        pytest.skip("Stage the comparison tokenizer to run this private-asset check")
    return (
        request.param,
        config,
        transformers.AutoTokenizer.from_pretrained(path, local_files_only=True),
    )


def test_chatml_supervises_reply_and_complete_turn_marker(chat_arm):
    arm, config, tokenizer = chat_arm
    messages = [
        {"role": "user", "content": "Explain why 2 < 3."},
        {"role": "assistant", "content": "Two is smaller than three."},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False)
    assert rendered.endswith("Two is smaller than three.<|im_end|>\n")
    prefix = as_token_ids(
        tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True
        )
    )
    encoded = encode_reasoning_example(
        {"messages": messages}, tokenizer, max_length=3072, require_reasoning_tags=False
    )
    assert encoded is not None
    assert encoded.prompt_length == len(prefix)
    assert not encoded.input_mask[: len(prefix)].any()
    assert encoded.input_mask[len(prefix) : encoded.unpadded_length].all()
    if arm == "rowanai":
        assert len(tokenizer) == 32000
        assert tokenizer.chat_template == config["tokenizer"]["chat_template"]
        assert len(tokenizer.encode("<|im_end|>", add_special_tokens=False)) == 7
    else:
        assert config["tokenizer"]["chat_template"] is None
        assert tokenizer.encode("<|im_end|>", add_special_tokens=False) == [151645]
        assert "Please reason step by step" in rendered


def test_export_stops_on_the_correct_boundary(chat_arm, tmp_path):
    arm, config, tokenizer = chat_arm
    path = tmp_path / "generation_config.json"
    base = json.loads(
        (Path(config["model"]["model_path"]) / "generation_config.json").read_text()
    )
    path.write_text(json.dumps(base))
    write_turn_end_generation_config(
        output_dir=str(tmp_path),
        tokenizer=tokenizer,
        stop_strings=config["export"].get("stop_strings"),
    )
    written = json.loads(path.read_text())
    if arm == "rowanai":
        assert written["eos_token_id"] == 0
        assert written["stop_strings"] == ["<|im_end|>"]
        assert (
            written["eos_token_id"]
            != tokenizer.encode("<", add_special_tokens=False)[0]
        )
    else:
        assert written["eos_token_id"][0] == 151645
        assert "stop_strings" not in written


def test_rowanai_stop_string_does_not_stop_on_an_angle_bracket(chat_arm):
    arm, _, tokenizer = chat_arm
    if arm != "rowanai":
        pytest.skip("Qwen has a dedicated turn-end token")
    torch = pytest.importorskip("torch")
    from transformers import StopStringCriteria

    stop = StopStringCriteria(tokenizer, ["<|im_end|>"])
    for text, expected in [
        ("2 < 3", False),
        ("answer<|im_", False),
        ("answer<|im_end|>", True),
    ]:
        ids = torch.tensor([tokenizer.encode(text, add_special_tokens=False)])
        assert stop(ids, None).item() == expected


@pytest.mark.parametrize("value", ["", [], [""], [1]])
def test_invalid_stop_strings_fail_before_training(value):
    with pytest.raises(ValueError, match=r"export\.stop_strings"):
        load_config(
            "recipes/rowanai/sft/config_rowanai.yaml",
            ["export.stop_strings=" + json.dumps(value)],
        )
