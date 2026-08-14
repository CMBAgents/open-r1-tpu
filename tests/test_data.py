from collections import UserDict

import numpy as np

from open_r1_tpu.data import encode_reasoning_example, normalize_messages


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return [ord(character) for character in rendered]


class MappingTokenizer(FakeTokenizer):
    def apply_chat_template(self, *args, **kwargs):
        return UserDict({"input_ids": super().apply_chat_template(*args, **kwargs)})


def _record(completion="<think>two plus two is four</think>4"):
    return {
        "messages": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": completion},
        ]
    }


def test_reasoning_trace_uses_assistant_only_loss_mask():
    tokenizer = FakeTokenizer()
    encoded = encode_reasoning_example(
        _record(), tokenizer, max_length=256, system_prompt="Reason first."
    )
    assert encoded is not None
    assert not encoded.input_mask[: encoded.prompt_length].any()
    assert encoded.input_mask[encoded.prompt_length : encoded.unpadded_length].all()
    assert not encoded.input_mask[encoded.unpadded_length :].any()
    assert encoded.input_tokens.dtype == np.int32


def test_missing_reasoning_tags_is_filtered():
    assert (
        encode_reasoning_example(
            _record("The answer is 4."), FakeTokenizer(), max_length=256
        )
        is None
    )


def test_overlength_trace_is_filtered_not_truncated():
    assert (
        encode_reasoning_example(_record(), FakeTokenizer(), max_length=20)
        is None
    )


def test_full_sequence_loss_can_be_enabled():
    encoded = encode_reasoning_example(
        _record(),
        FakeTokenizer(),
        max_length=256,
        assistant_only_loss=False,
    )
    assert encoded is not None
    assert encoded.input_mask[: encoded.unpadded_length].all()


def test_messages_may_be_json_encoded():
    messages = normalize_messages(
        '[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]'
    )
    assert messages[-1] == {"role": "assistant", "content": "a"}


def test_mapping_tokenizer_output_is_supported():
    encoded = encode_reasoning_example(
        _record(), MappingTokenizer(), max_length=256
    )
    assert encoded is not None
    assert encoded.input_mask.sum() > 0
