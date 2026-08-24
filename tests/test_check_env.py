from pathlib import Path

from open_r1_tpu.core.config import load_config
from open_r1_tpu.training.data import encode_reasoning_example
from open_r1_tpu.training.preflight import _preflight_example

OT3_RECIPE = (
    Path(__file__).parents[1] / "recipes/Qwen3-1.7B-OT3/sft/config_distill.yaml"
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(character) for character in rendered]


def test_probe_defaults_to_role_and_content_messages():
    record, encode_kwargs = _preflight_example({})

    assert list(record) == ["messages"]
    assert [message["role"] for message in record["messages"]] == ["user", "assistant"]
    assert encode_kwargs["messages_column"] == "messages"
    assert encode_kwargs["system_prompt"] is None


def test_probe_is_written_in_the_recipe_corpus_vocabulary():
    config = load_config(OT3_RECIPE)

    record, encode_kwargs = _preflight_example(config["dataset"])

    # A hardcoded role/content probe would be filtered here and reported as a
    # tokenizer fault that is not there.
    assert record["conversations"] == [
        {"from": "human", "value": "What is 2 + 2?"},
        {"from": "gpt", "value": "<think>Adding gives four.</think>4"},
    ]
    # A wider window than the preflight's own 256: this tokenizer spends one
    # token per character, so the recipe's system prompt alone fills that.
    encoded = encode_reasoning_example(
        record, FakeTokenizer(), max_length=1024, **encode_kwargs
    )
    assert encoded is not None
    assert encoded.input_mask[encoded.prompt_length : encoded.unpadded_length].all()
