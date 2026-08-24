import json
from pathlib import Path

import pytest

from open_r1_tpu.core.config import load_config
from open_r1_tpu.training import transcripts

RECIPE = (
    Path(__file__).parents[1]
    / "recipes/OpenR1-Distill-Qwen3-1.7B/sft/config_distill.yaml"
)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        rendered = "".join(
            f"<|{message['role']}|>{message['content']}" for message in messages
        )
        return rendered + "<|assistant|>"


class FakeSamplerOutput:
    def __init__(self, text, tokens):
        self.text = text
        self.tokens = tokens


def test_settings_default_to_disabled_with_derived_cache():
    config = load_config(RECIPE)
    settings = load_settings(config)

    assert settings["enabled"] is False
    assert settings["every_n_steps"] == 500
    # An explicit null cache_size means "derive from the sequence budget".
    assert settings["cache_size"] == (
        config["dataset"]["max_length"] + settings["max_new_tokens"]
    )
    assert settings["prompts"] == list(transcripts.DEFAULT_PROMPTS)
    assert settings["reasoning_end"] == "</think>"


def load_settings(config):
    return transcripts.resolve_settings(config)


def test_explicit_prompts_and_cache_size_win():
    config = load_config(
        RECIPE,
        [
            "training.transcripts.enabled=true",
            "training.transcripts.prompts=[first, second]",
            "training.transcripts.cache_size=99",
        ],
    )
    settings = load_settings(config)

    assert settings["enabled"] is True
    assert settings["prompts"] == ["first", "second"]
    assert settings["cache_size"] == 99


@pytest.mark.parametrize(
    ("step", "every", "expected"),
    [
        (0, 500, False),
        (500, 500, True),
        (501, 500, False),
        (1000, 500, True),
        (500, 0, False),
    ],
)
def test_should_sample_fires_only_on_the_interval(step, every, expected):
    assert transcripts.should_sample(step, every) is expected


def test_rendered_prompt_matches_the_training_prefix():
    rendered = transcripts.render_prompt(
        FakeTokenizer(), "What is 2+2?", "Think first."
    )
    assert rendered == ("<|system|>Think first.<|user|>What is 2+2?<|assistant|>")


def test_rendered_prompt_omits_an_absent_system_turn():
    rendered = transcripts.render_prompt(FakeTokenizer(), "What is 2+2?", None)
    assert rendered == "<|user|>What is 2+2?<|assistant|>"


def test_record_flags_a_closed_reasoning_trace():
    record = transcripts.build_record(
        500,
        "prompt",
        "<think>weighing it up</think>The answer is 4.",
        reasoning_start="<think>",
        reasoning_end="</think>",
        max_new_tokens=256,
        generated_tokens=32,
    )

    assert record["step"] == 500
    assert record["has_reasoning_start"] is True
    assert record["has_reasoning_end"] is True
    assert record["reasoning_balanced"] is True
    assert record["hit_token_cap"] is False


def test_record_flags_an_unclosed_reasoning_trace():
    # The failure teacher-forced loss cannot show: the model never closes.
    record = transcripts.build_record(
        500,
        "prompt",
        "<think>and on and on and on",
        reasoning_start="<think>",
        reasoning_end="</think>",
        max_new_tokens=256,
        generated_tokens=256,
    )

    assert record["has_reasoning_end"] is False
    assert record["reasoning_balanced"] is False
    # Truncation explains the missing tag, so the flag says "inconclusive".
    assert record["hit_token_cap"] is True


def test_record_rejects_a_reversed_trace():
    record = transcripts.build_record(
        1,
        "prompt",
        "</think>backwards<think>",
        reasoning_start="<think>",
        reasoning_end="</think>",
        max_new_tokens=256,
        generated_tokens=8,
    )
    assert record["reasoning_balanced"] is False


def test_records_are_appended_as_jsonl(tmp_path):
    output = tmp_path / "nested" / "transcripts.jsonl"
    transcripts.write_records(str(output), [{"step": 1}])
    transcripts.write_records(str(output), [{"step": 2}, {"step": 3}])

    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["step"] for line in lines] == [1, 2, 3]


def test_sample_transcripts_summarizes_every_prompt():
    settings = {
        "prompts": ["first", "second"],
        "system_prompt": "Think first.",
        "max_new_tokens": 256,
        "temperature": 0.0,
        "seed": 42,
        "reasoning_start": "<think>",
        "reasoning_end": "</think>",
    }
    captured = {}

    def fake_sampler(**kwargs):
        captured.update(kwargs)
        return FakeSamplerOutput(
            text=["<think>a</think>A", "<think>b"],
            tokens=[[1, 2, 3], [4, 5]],
        )

    records = transcripts.sample_transcripts(
        fake_sampler, FakeTokenizer(), settings, step=500
    )

    assert captured["max_generation_steps"] == 256
    assert captured["temperature"] == 0.0
    assert len(captured["input_strings"]) == 2
    assert captured["input_strings"][0].endswith("<|assistant|>")
    assert [record["step"] for record in records] == [500, 500]
    assert [record["reasoning_balanced"] for record in records] == [True, False]
    assert [record["generated_tokens"] for record in records] == [3, 2]


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        ({"use_flash_attention": True, "flash_attention_block_size": 1024}, 1024),
        ({"use_flash_attention": True, "flash_attention_block_size": 256}, 256),
        # Rounded up to a power of two so the block still divides the prefill.
        ({"use_flash_attention": True, "flash_attention_block_size": 768}, 1024),
        ({"use_flash_attention": True}, 1024),
        # Without the kernel there is no divisibility rule to satisfy.
        ({"use_flash_attention": False}, None),
        ({}, None),
    ],
)
def test_prompt_length_satisfies_the_splash_kernel(model_config, expected):
    assert transcripts.flash_attention_prompt_length(model_config) == expected


def test_default_settings_pad_prompts_to_the_attention_block():
    config = load_config(RECIPE)
    settings = transcripts.resolve_settings(config)

    # Short prompts would otherwise pad to 128 and raise
    # "q_block_size=1024 should divide q_seq_len=128".
    assert (
        settings["max_prompt_length"] == (config["model"]["flash_attention_block_size"])
    )
    # The cache has to cover the padded prompt plus the completion.
    assert settings["cache_size"] == (
        settings["max_prompt_length"] + settings["max_new_tokens"]
    )


def test_disabling_flash_attention_leaves_prompt_padding_to_the_sampler():
    config = load_config(RECIPE, ["model.use_flash_attention=false"])
    settings = transcripts.resolve_settings(config)

    assert settings["max_prompt_length"] is None
    assert settings["cache_size"] == (
        config["dataset"]["max_length"] + settings["max_new_tokens"]
    )


def test_sampler_receives_the_padded_prompt_length():
    settings = {
        "prompts": ["first"],
        "system_prompt": None,
        "max_new_tokens": 64,
        "temperature": 0.0,
        "seed": 42,
        "max_prompt_length": 1024,
        "reasoning_start": "<think>",
        "reasoning_end": "</think>",
    }
    captured = {}

    def fake_sampler(**kwargs):
        captured.update(kwargs)
        return FakeSamplerOutput(text=["<think>a</think>A"], tokens=[[1, 2]])

    transcripts.sample_transcripts(fake_sampler, FakeTokenizer(), settings, step=2)
    assert captured["max_prompt_length"] == 1024


def test_sampler_omits_prompt_length_when_unset():
    settings = {
        "prompts": ["first"],
        "system_prompt": None,
        "max_new_tokens": 64,
        "temperature": 0.0,
        "seed": 42,
        "max_prompt_length": None,
        "reasoning_start": "<think>",
        "reasoning_end": "</think>",
    }
    captured = {}

    def fake_sampler(**kwargs):
        captured.update(kwargs)
        return FakeSamplerOutput(text=["ok"], tokens=[[1]])

    transcripts.sample_transcripts(fake_sampler, FakeTokenizer(), settings, step=2)
    assert "max_prompt_length" not in captured
