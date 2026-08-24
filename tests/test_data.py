import sys
from collections import UserDict
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from open_r1_tpu.training import data as data_module
from open_r1_tpu.training.data import encode_reasoning_example, normalize_messages


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


def _rendered(messages):
    """The exact string FakeTokenizer renders, one token per character."""
    return "".join(
        f"<{message['role']}>{message['content']}</{message['role']}>"
        for message in messages
    )


def test_overlength_trace_is_filtered_not_truncated():
    assert encode_reasoning_example(_record(), FakeTokenizer(), max_length=20) is None
    # The default is what a live run depends on; state it explicitly too.
    assert (
        encode_reasoning_example(
            _record(), FakeTokenizer(), max_length=20, overlength_policy="drop"
        )
        is None
    )


def test_the_overlength_policy_only_affects_overlength_examples():
    dropped = encode_reasoning_example(_record(), FakeTokenizer(), max_length=256)
    truncated = encode_reasoning_example(
        _record(), FakeTokenizer(), max_length=256, overlength_policy="truncate"
    )
    assert dropped is not None and truncated is not None
    assert dropped.unpadded_length == truncated.unpadded_length
    assert dropped.prompt_length == truncated.prompt_length
    assert dropped.input_tokens.tolist() == truncated.input_tokens.tolist()
    assert dropped.input_mask.tolist() == truncated.input_mask.tolist()


def test_unknown_overlength_policy_is_rejected():
    with pytest.raises(ValueError, match="overlength_policy"):
        encode_reasoning_example(
            _record(), FakeTokenizer(), max_length=256, overlength_policy="clip"
        )


def test_truncation_cuts_the_tail_and_appends_no_terminator():
    tokenizer = FakeTokenizer()
    record = _record()
    rendered = _rendered(record["messages"])
    # Cut mid-reasoning, before the trace closes its <think> block.
    max_length = rendered.index("</think>")

    encoded = encode_reasoning_example(
        record, tokenizer, max_length=max_length, overlength_policy="truncate"
    )

    assert encoded is not None
    # The packer slices by unpadded_length, so it must be the cut length.
    assert encoded.unpadded_length == max_length
    assert encoded.input_tokens.tolist() == [ord(c) for c in rendered[:max_length]]
    kept = "".join(chr(token) for token in encoded.input_tokens.tolist())
    # The template's closing sequence went with the tail, and nothing replaced
    # it: a terminator here would teach the model to stop mid-reasoning.
    assert not kept.endswith("</assistant>")
    assert "</think>" not in kept
    assert tokenizer.eos_token_id not in encoded.input_tokens.tolist()
    assert not encoded.input_mask[: encoded.prompt_length].any()
    assert encoded.input_mask[encoded.prompt_length :].all()


def test_truncated_example_packs_at_its_truncated_length():
    record = _record()
    max_length = _rendered(record["messages"]).index("</think>")
    encoded = encode_reasoning_example(
        record, FakeTokenizer(), max_length=max_length, overlength_policy="truncate"
    )
    assert encoded is not None

    (window,) = data_module.pack_encoded_examples(
        [encoded], max_length=max_length, pad_id=0
    )

    # A pre-truncation unpadded_length would overrun the window instead.
    assert window["input_tokens"].tolist() == encoded.input_tokens.tolist()
    assert window["segment_ids"].tolist() == [1] * max_length


def _multi_turn_record():
    return {
        "messages": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "<think>greet</think>Hello"},
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<think>easy</think>4"},
        ]
    }


def test_truncation_clamps_a_span_that_crosses_the_boundary():
    record = _multi_turn_record()
    first_start = len("<user>Hi</user><assistant>")
    max_length = first_start + 5

    encoded = encode_reasoning_example(
        record, FakeTokenizer(), max_length=max_length, overlength_policy="truncate"
    )

    assert encoded is not None
    assert encoded.unpadded_length == max_length
    # Spans come from the untruncated render, so the mask still starts exactly
    # at the assistant header and simply stops at the boundary.
    assert not encoded.input_mask[:first_start].any()
    assert encoded.input_mask[first_start:max_length].all()
    # Nothing of the final turn survives, so the whole window is context.
    assert encoded.prompt_length == max_length


def test_truncation_discards_a_turn_that_starts_past_the_boundary():
    record = _multi_turn_record()
    first_start = len("<user>Hi</user><assistant>")
    first_end = len("<user>Hi</user><assistant><think>greet</think>Hello</assistant>")
    max_length = first_end + 3

    encoded = encode_reasoning_example(
        record, FakeTokenizer(), max_length=max_length, overlength_policy="truncate"
    )

    assert encoded is not None
    assert encoded.input_mask[first_start:first_end].all()
    # The second assistant turn begins after the cut and carries no loss.
    assert not encoded.input_mask[first_end:].any()
    assert encoded.unpadded_length == max_length


def test_a_prompt_longer_than_the_window_is_still_dropped():
    record = _record()
    prompt = "<user>What is 2+2?</user><assistant>"

    for max_length in (len(prompt) - 4, len(prompt)):
        # No assistant token fits, so there is nothing trainable in the window.
        assert (
            encode_reasoning_example(
                record,
                FakeTokenizer(),
                max_length=max_length,
                overlength_policy="truncate",
            )
            is None
        )


def test_full_sequence_loss_drops_a_window_holding_only_the_prompt():
    record = _record()
    prompt = "<user>What is 2+2?</user><assistant>"
    assert (
        encode_reasoning_example(
            record,
            FakeTokenizer(),
            max_length=len(prompt),
            assistant_only_loss=False,
            overlength_policy="truncate",
        )
        is None
    )


SHAREGPT_SCHEMA = {
    "role_key": "from",
    "content_key": "value",
    "role_map": {"human": "user", "gpt": "assistant"},
}


def _sharegpt_record():
    return {
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "<think>two plus two is four</think>4"},
        ]
    }


def test_sharegpt_rows_are_mapped_onto_role_and_content():
    encoded = encode_reasoning_example(
        _sharegpt_record(),
        FakeTokenizer(),
        max_length=256,
        messages_column="conversations",
        message_schema=data_module.message_schema_from_config(SHAREGPT_SCHEMA),
    )

    assert encoded is not None
    assert encoded.prompt_length == len("<user>What is 2+2?</user><assistant>")
    assert not encoded.input_mask[: encoded.prompt_length].any()
    assert encoded.input_mask[encoded.prompt_length : encoded.unpadded_length].all()


def test_sharegpt_rows_are_dropped_without_a_schema():
    # The failure that motivates the schema: no error, just an empty dataset.
    assert (
        encode_reasoning_example(
            _sharegpt_record(),
            FakeTokenizer(),
            max_length=256,
            messages_column="conversations",
        )
        is None
    )


def test_the_default_schema_leaves_role_content_rows_untouched():
    messages = normalize_messages(
        [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    )
    assert messages == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]


def test_an_unmapped_role_keeps_its_own_name():
    schema = data_module.message_schema_from_config({"role_map": {"human": "user"}})
    messages = normalize_messages(
        [{"role": "system", "content": "s"}, {"role": "human", "content": "q"}],
        schema=schema,
    )
    assert [message["role"] for message in messages] == ["system", "user"]


@pytest.mark.parametrize(
    ("block", "error"),
    [
        ("not a mapping", "configuration mapping"),
        ({"from": "human"}, "unknown"),
        ({"role_key": ""}, "role_key"),
        ({"content_key": 3}, "content_key"),
        ({"role_map": {"human": 1}}, "role_map"),
    ],
)
def test_an_invalid_message_schema_is_rejected(block, error):
    with pytest.raises(ValueError, match=error):
        data_module.message_schema_from_config(block)


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


def _encoded(tokens, supervised_from, *, max_length=16, pad_id=0):
    """Build an EncodedExample directly, bypassing the tokenizer."""
    padded = np.full((max_length,), pad_id, dtype=np.int32)
    padded[: len(tokens)] = tokens
    mask = np.zeros((max_length,), dtype=np.bool_)
    mask[supervised_from : len(tokens)] = True
    return data_module.EncodedExample(
        input_tokens=padded,
        input_mask=mask,
        prompt_length=supervised_from,
        unpadded_length=len(tokens),
    )


def test_packing_concatenates_examples_with_segment_geometry():
    examples = [
        _encoded([11, 12, 13], 1),
        _encoded([21, 22, 23, 24], 2),
    ]
    windows = list(data_module.pack_encoded_examples(examples, max_length=10, pad_id=0))
    assert len(windows) == 1
    window = windows[0]
    assert window["input_tokens"].tolist() == [11, 12, 13, 21, 22, 23, 24, 0, 0, 0]
    assert window["segment_ids"].tolist() == [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
    # RoPE positions restart at every example boundary.
    assert window["positions"].tolist() == [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
    assert window["input_mask"].tolist() == [
        False,
        True,
        True,
        False,
        False,
        True,
        True,
        False,
        False,
        False,
    ]
    assert window["input_tokens"].dtype == np.int32
    assert window["segment_ids"].dtype == np.int32
    assert window["positions"].dtype == np.int32


def test_packing_never_supervises_a_segment_start():
    # A fully supervised example: after the causal shift its first token would
    # be predicted from the preceding segment's final position.
    examples = [_encoded([11, 12], 0), _encoded([21, 22], 0)]
    (window,) = data_module.pack_encoded_examples(examples, max_length=8, pad_id=0)
    assert window["input_mask"].tolist() == [
        False,
        True,
        False,
        True,
        False,
        False,
        False,
        False,
    ]


def test_packing_uses_first_fit_across_open_windows():
    examples = [
        _encoded([1] * 6, 0),
        _encoded([2] * 5, 0),
        _encoded([3] * 2, 0),
    ]
    windows = list(data_module.pack_encoded_examples(examples, max_length=8, pad_id=0))
    assert len(windows) == 2
    # The third example returns to the first window's remaining space.
    assert windows[0]["input_tokens"].tolist() == [1, 1, 1, 1, 1, 1, 3, 3]
    assert windows[0]["segment_ids"].tolist() == [1, 1, 1, 1, 1, 1, 2, 2]
    assert windows[1]["segment_ids"].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]


def test_packing_emits_the_fullest_window_when_the_pool_is_full():
    examples = [
        _encoded([1] * 3, 0),
        _encoded([2] * 4, 0),
        _encoded([3] * 4, 0),
    ]
    windows = list(
        data_module.pack_encoded_examples(
            examples, max_length=6, pad_id=0, open_windows=2
        )
    )
    assert len(windows) == 3
    # The 4-token window is fuller than the 3-token one, so it is emitted
    # first to make room; the others flush at the end of the stream.
    assert windows[0]["input_tokens"].tolist()[:4] == [2, 2, 2, 2]
    assert windows[1]["input_tokens"].tolist()[:3] == [1, 1, 1]
    assert windows[2]["input_tokens"].tolist()[:4] == [3, 3, 3, 3]


def test_packing_rejects_examples_longer_than_the_window():
    with pytest.raises(ValueError, match="longer than the packing window"):
        list(
            data_module.pack_encoded_examples(
                [_encoded([1] * 9, 0)], max_length=8, pad_id=0
            )
        )


def test_packed_batches_stack_windows_and_drop_the_remainder():
    examples = [_encoded([value] * 8, 0, max_length=8) for value in (1, 2, 3)]
    dataset = data_module.PackedBatchDataset(
        examples, max_length=8, pad_id=0, batch_size=2
    )
    batches = list(dataset)
    assert len(batches) == 1
    batch = batches[0]
    assert set(batch) == set(data_module.PACKED_FIELDS)
    assert batch["input_tokens"].shape == (2, 8)
    assert batch["input_tokens"][0].tolist() == [1] * 8
    assert batch["input_tokens"][1].tolist() == [2] * 8
    # The trainer restarts iteration on resume and eval; the wrapper must be
    # re-iterable and deterministic.
    assert [b["input_tokens"].tolist() for b in dataset] == [
        batch["input_tokens"].tolist()
    ]


def test_load_reasoning_datasets_forwards_the_packing_flag(monkeypatch):
    captured = {}

    class FakeDataset:
        def __len__(self):
            return 1

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *a, **k: FakeDataset()),
    )

    def fake_build(data_source, *args, **kwargs):
        captured.update(kwargs)
        return data_source

    monkeypatch.setattr(data_module, "build_grain_dataset", fake_build)
    data_module.load_reasoning_datasets(
        {
            "name": "parquet",
            "config": None,
            "batch_size": 2,
            "max_length": 128,
            "packing": True,
        },
        FakeTokenizer(),
    )
    assert captured["packing"] is True


def test_load_reasoning_datasets_builds_the_schema_and_policy(monkeypatch):
    captured = {}

    class FakeDataset:
        def __len__(self):
            return 1

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *a, **k: FakeDataset()),
    )
    monkeypatch.setattr(
        data_module,
        "build_grain_dataset",
        lambda data_source, *args, **kwargs: captured.update(kwargs) or data_source,
    )

    data_module.load_reasoning_datasets(
        {
            "name": "parquet",
            "config": None,
            "batch_size": 1,
            "max_length": 128,
            "messages_column": "conversations",
            "message_schema": SHAREGPT_SCHEMA,
            "overlength_policy": "truncate",
        },
        FakeTokenizer(),
    )

    encode_kwargs = captured["encode_kwargs"]
    assert encode_kwargs["overlength_policy"] == "truncate"
    assert encode_kwargs["messages_column"] == "conversations"
    schema = encode_kwargs["message_schema"]
    assert (schema.role_key, schema.content_key) == ("from", "value")
    assert dict(schema.role_map) == {"human": "user", "gpt": "assistant"}


def test_load_reasoning_datasets_defaults_to_the_drop_policy(monkeypatch):
    captured = {}

    class FakeDataset:
        def __len__(self):
            return 1

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        SimpleNamespace(load_dataset=lambda *a, **k: FakeDataset()),
    )
    monkeypatch.setattr(
        data_module,
        "build_grain_dataset",
        lambda data_source, *args, **kwargs: captured.update(kwargs) or data_source,
    )

    data_module.load_reasoning_datasets(
        {"name": "parquet", "config": None, "batch_size": 1, "max_length": 128},
        FakeTokenizer(),
    )

    encode_kwargs = captured["encode_kwargs"]
    assert encode_kwargs["overlength_policy"] == "drop"
    assert encode_kwargs["message_schema"] is data_module.DEFAULT_MESSAGE_SCHEMA
