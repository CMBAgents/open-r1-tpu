import sys
from collections import UserDict
from types import SimpleNamespace
from typing import Any

import numpy as np

from open_r1_tpu import data as data_module
from open_r1_tpu.data import encode_reasoning_example, normalize_messages


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt) -> Any:
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


def test_multi_turn_supervises_every_assistant_turn():
    tokenizer = FakeTokenizer()
    record = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "<think>greet</think>Hello"},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<think>easy</think>4"},
        ]
    }
    encoded = encode_reasoning_example(record, tokenizer, max_length=256)
    assert encoded is not None

    rendered = "".join(
        f"<{m['role']}>{m['content']}</{m['role']}>" for m in record["messages"]
    )
    first_start = len("<user>Hi</user><assistant>")
    first_end = len("<user>Hi</user><assistant><think>greet</think>Hello</assistant>")
    # The first assistant turn is supervised through its closing tag.
    assert encoded.input_mask[first_start:first_end].all()
    # The second user turn and the second assistant header are not.
    assert not encoded.input_mask[first_end : encoded.prompt_length].any()
    # The final turn is supervised to the end of the sequence, as before.
    final_target = "<think>easy</think>4</assistant>"
    assert encoded.prompt_length == len(rendered) - len(final_target)
    assert encoded.input_mask[encoded.prompt_length : encoded.unpadded_length].all()
    assert not encoded.input_mask[:first_start].any()
    assert not encoded.input_mask[encoded.unpadded_length :].any()


def test_assistant_first_conversation_is_filtered():
    record = {
        "messages": [
            {"role": "assistant", "content": "<think>t</think>hello"},
        ]
    }
    assert encode_reasoning_example(record, FakeTokenizer(), max_length=256) is None


class ScaffoldPromptTokenizer(FakeTokenizer):
    """Appends a scaffold to the generation prompt that the full render omits,
    like Qwen3 with enable_thinking=False, so prefixes are not stable."""

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant><pad>"
        return [ord(character) for character in rendered]


def test_prefix_unstable_template_drops_example():
    record = {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "<think>a</think>hello"},
            {"role": "user", "content": "More"},
            {"role": "assistant", "content": "<think>b</think>done"},
        ]
    }
    assert (
        encode_reasoning_example(record, ScaffoldPromptTokenizer(), max_length=256)
        is None
    )


def test_missing_reasoning_tags_is_filtered():
    assert (
        encode_reasoning_example(
            _record("The answer is 4."), FakeTokenizer(), max_length=256
        )
        is None
    )


def test_overlength_trace_is_filtered_not_truncated():
    assert encode_reasoning_example(_record(), FakeTokenizer(), max_length=20) is None


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
    encoded = encode_reasoning_example(_record(), MappingTokenizer(), max_length=256)
    assert encoded is not None
    assert encoded.input_mask.sum() > 0


def test_local_parquet_data_files_are_forwarded(monkeypatch):
    captured = {}

    class FakeDataset:
        def __len__(self):
            return 1

    def fake_load_dataset(name, config, **kwargs):
        captured.update(name=name, config=config, **kwargs)
        return FakeDataset()

    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset)
    )
    monkeypatch.setattr(
        data_module,
        "build_grain_dataset",
        lambda data_source, *args, **kwargs: data_source,
    )
    result, eval_result = data_module.load_reasoning_datasets(
        {
            "name": "parquet",
            "config": None,
            "data_files": "data/Mixture-of-Thoughts/all/*.parquet",
            "train_split": "train",
            "batch_size": 1,
            "max_length": 128,
        },
        FakeTokenizer(),
    )
    assert isinstance(result, FakeDataset)
    assert eval_result is None
    assert captured == {
        "name": "parquet",
        "config": None,
        "split": "train",
        "data_files": "data/Mixture-of-Thoughts/all/*.parquet",
    }


def test_eval_split_is_capped_to_a_bounded_number_of_steps(monkeypatch):
    class FakeSplit:
        def __init__(self, size):
            self.size = size
            self.selected = None

        def __len__(self):
            return self.size

        def select(self, indices):
            self.selected = list(indices)
            return self

    class FakeRaw:
        def __init__(self):
            self.train = FakeSplit(900)
            self.eval = FakeSplit(100)
            self.split_kwargs = None

        def __len__(self):
            return 1000

        def train_test_split(self, **kwargs):
            self.split_kwargs = kwargs
            return {"train": self.train, "test": self.eval}

    raw = FakeRaw()
    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: raw)
    )
    monkeypatch.setattr(
        data_module,
        "build_grain_dataset",
        lambda data_source, *args, **kwargs: data_source,
    )

    train_ds, eval_ds = data_module.load_reasoning_datasets(
        {
            "name": "parquet",
            "config": None,
            "batch_size": 1,
            "max_length": 128,
            "eval_fraction": 0.1,
            "eval_max_examples": 16,
            "seed": 7,
        },
        FakeTokenizer(),
    )

    assert train_ds is raw.train
    assert eval_ds is raw.eval
    assert raw.split_kwargs == {"test_size": 0.1, "seed": 7}
    # A tenth of a large corpus would otherwise run for thousands of steps.
    assert raw.eval.selected == list(range(16))


def test_eval_split_cap_never_exceeds_the_split(monkeypatch):
    class FakeSplit:
        def __init__(self, size):
            self.size = size
            self.selected = None

        def __len__(self):
            return self.size

        def select(self, indices):
            self.selected = list(indices)
            return self

    class FakeRaw:
        def __init__(self):
            self.train = FakeSplit(9)
            self.eval = FakeSplit(4)

        def __len__(self):
            return 13

        def train_test_split(self, **kwargs):
            return {"train": self.train, "test": self.eval}

    raw = FakeRaw()
    monkeypatch.setitem(
        sys.modules, "datasets", SimpleNamespace(load_dataset=lambda *a, **k: raw)
    )
    monkeypatch.setattr(
        data_module,
        "build_grain_dataset",
        lambda data_source, *args, **kwargs: data_source,
    )

    data_module.load_reasoning_datasets(
        {
            "name": "parquet",
            "config": None,
            "batch_size": 1,
            "max_length": 128,
            "eval_fraction": 0.3,
            "eval_max_examples": 64,
        },
        FakeTokenizer(),
    )
    assert raw.eval.selected == list(range(4))
