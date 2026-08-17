import json
from pathlib import Path

import pytest

from open_r1_tpu import transcripts
from open_r1_tpu.config import load_config


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
    assert rendered == (
        "<|system|>Think first.<|user|>What is 2+2?<|assistant|>"
    )


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
