"""Reasoning-trace normalisation, tokenisation, packing and loading for SFT."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from open_r1_tpu.core.config import read_prompt_file
from open_r1_tpu.model.tokenizing import (
    assistant_closing_ids,
    find_subsequence,
    render_ids,
)


@dataclass(frozen=True)
class EncodedExample:
    """One fixed-length causal-LM example and its supervised-token mask."""

    input_tokens: np.ndarray
    input_mask: np.ndarray
    prompt_length: int
    unpadded_length: int


@dataclass(frozen=True)
class MessageSchema:
    """How one corpus spells the fields and the roles of a chat message.

    ShareGPT-style corpora store a turn as ``{"from": "human", "value": ...}``
    instead of ``{"role": "user", "content": ...}``. ``role_key`` and
    ``content_key`` name the fields to read; ``role_map`` renames role values.
    Roles absent from the map keep their own name, so partial maps are allowed.
    """

    role_key: str = "role"
    content_key: str = "content"
    role_map: Mapping[str, str] = MappingProxyType({})


DEFAULT_MESSAGE_SCHEMA = MessageSchema()

# What to do with a rendered conversation longer than the training window.
DROP_OVERLENGTH = "drop"
TRUNCATE_OVERLENGTH = "truncate"
OVERLENGTH_POLICIES = (DROP_OVERLENGTH, TRUNCATE_OVERLENGTH)


def message_schema_from_config(value: Any) -> MessageSchema:
    """Build a :class:`MessageSchema` from a ``dataset.message_schema`` block.

    Validated strictly, because a mismatched schema filters every record and
    leaves an empty dataset rather than an error.
    """
    if value is None:
        return DEFAULT_MESSAGE_SCHEMA
    if not isinstance(value, Mapping):
        raise ValueError("dataset.message_schema must be a configuration mapping")
    unknown = sorted(set(value) - {"role_key", "content_key", "role_map"})
    if unknown:
        raise ValueError(f"unknown dataset.message_schema keys: {unknown}")
    role_key = value.get("role_key", DEFAULT_MESSAGE_SCHEMA.role_key)
    content_key = value.get("content_key", DEFAULT_MESSAGE_SCHEMA.content_key)
    for name, key in (("role_key", role_key), ("content_key", content_key)):
        if not isinstance(key, str) or not key:
            raise ValueError(
                f"dataset.message_schema.{name} must be a non-empty string"
            )
    role_map = value.get("role_map") or {}
    if not isinstance(role_map, Mapping) or not all(
        isinstance(source, str) and isinstance(target, str)
        for source, target in role_map.items()
    ):
        raise ValueError("dataset.message_schema.role_map must map strings to strings")
    return MessageSchema(
        role_key=role_key,
        content_key=content_key,
        role_map=MappingProxyType(dict(role_map)),
    )


def normalize_messages(
    value: Any, *, schema: MessageSchema = DEFAULT_MESSAGE_SCHEMA
) -> list[dict[str, str]]:
    """Normalise a conversation to role/content dicts, reading it per ``schema``."""
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("messages must be a sequence")

    messages: list[dict[str, str]] = []
    for message in value:
        if not isinstance(message, dict):
            raise ValueError("each message must be a mapping")
        role = message.get(schema.role_key)
        content = message.get(schema.content_key)
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError(
                "each message needs string "
                f"{schema.role_key!r} and {schema.content_key!r} fields"
            )
        messages.append({"role": schema.role_map.get(role, role), "content": content})
    if not messages:
        raise ValueError("messages cannot be empty")
    return messages


def prepare_messages(
    record: dict[str, Any],
    *,
    messages_column: str,
    system_prompt: str | None,
    message_schema: MessageSchema = DEFAULT_MESSAGE_SCHEMA,
) -> list[dict[str, str]]:
    """Read a conversation, inject the reasoning system prompt, and validate it."""
    if messages_column not in record:
        raise ValueError(f"record has no {messages_column!r} column")
    messages = normalize_messages(record[messages_column], schema=message_schema)
    if messages[-1]["role"] != "assistant":
        raise ValueError("the final message must be an assistant reasoning trace")
    if system_prompt and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def _assistant_turn_spans(
    tokenizer: Any, messages: list[dict[str, str]], full_ids: list[int]
) -> list[tuple[int, int]] | None:
    """Locate every assistant turn of the rendered conversation in ``full_ids``.

    A turn starts where the render of the preceding messages with the
    generation prompt ends, which must be an exact prefix. Its end cannot come
    from a sub-render, because some templates (Qwen3's) render the last
    assistant message differently, so it is the next occurrence of the closing
    sequence every assistant turn ends with. Any mismatch returns None, so the
    example is dropped rather than trained with a wrong mask.
    """
    closing = assistant_closing_ids(tokenizer)
    spans: list[tuple[int, int]] = []
    previous_end = 0
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        if index == 0:
            return None
        prefix = render_ids(tokenizer, messages[:index], add_generation_prompt=True)
        start = len(prefix)
        if start < previous_end or start >= len(full_ids):
            return None
        if full_ids[:start] != prefix:
            return None
        if index == len(messages) - 1:
            end = len(full_ids)
        else:
            found = find_subsequence(full_ids, closing, start)
            if found is None:
                return None
            end = found + len(closing)
        spans.append((start, end))
        previous_end = end
    return spans or None


def _pad_id(tokenizer: Any) -> int:
    pad_id_method: Any = getattr(tokenizer, "pad_id", None)
    if callable(pad_id_method):
        resolved: Any = pad_id_method()  # callable() narrows Any to object
        return int(resolved)
    pad_id: Any = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        eos_id: Any = getattr(tokenizer, "eos_token_id", None)
        if eos_id is None:
            raise ValueError("tokenizer defines neither a pad token nor an EOS token")
        return int(eos_id)
    return int(pad_id)


def encode_reasoning_example(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    messages_column: str = "messages",
    system_prompt: str | None = None,
    assistant_only_loss: bool = True,
    require_reasoning_tags: bool = True,
    reasoning_start: str = "<think>",
    reasoning_end: str = "</think>",
    message_schema: MessageSchema = DEFAULT_MESSAGE_SCHEMA,
    overlength_policy: str = DROP_OVERLENGTH,
) -> EncodedExample | None:
    """Tokenise one trace, or return None to filter it.

    With ``assistant_only_loss`` every assistant turn is supervised, located
    by exact chat-template prefixes; user and system tokens never carry loss.
    ``prompt_length`` is the context length of the final turn.

    A render longer than ``max_length`` is dropped under the default ``drop``
    policy, since cutting a reasoning trace teaches incomplete chains and
    usually removes the answer. ``truncate`` keeps the prompt and as much of
    the trace as fits, for corpora whose traces were themselves generated
    under a context cap. The truncated sequence is left unterminated:
    appending a terminator would teach the model to stop mid-reasoning. An
    example whose prompt alone fills the window is dropped either way.
    """
    if overlength_policy not in OVERLENGTH_POLICIES:
        raise ValueError(
            "overlength_policy must be one of "
            f"{', '.join(OVERLENGTH_POLICIES)}; got {overlength_policy!r}"
        )
    try:
        messages = prepare_messages(
            record,
            messages_column=messages_column,
            system_prompt=system_prompt,
            message_schema=message_schema,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    completion = messages[-1]["content"]
    if require_reasoning_tags and (
        reasoning_start not in completion or reasoning_end not in completion
    ):
        return None

    try:
        full_ids = render_ids(tokenizer, messages, add_generation_prompt=False)
    except (TypeError, ValueError):
        return None

    if len(full_ids) < 2:
        return None
    if len(full_ids) > max_length and overlength_policy == DROP_OVERLENGTH:
        return None
    # Right-truncation: the prompt survives and nothing replaces the cut tail.
    length = min(len(full_ids), max_length)

    mask = np.zeros((max_length,), dtype=np.bool_)
    try:
        if assistant_only_loss:
            # Located in the untruncated render, which the whole-message
            # prefix renders are matched against.
            spans = _assistant_turn_spans(tokenizer, messages, full_ids)
            if spans is None:
                return None
            for start, end in spans:
                if start >= length:
                    continue
                mask[start : min(end, length)] = True
            if not mask.any():
                return None
            prompt_length = min(spans[-1][0], length)
        else:
            prompt_ids = render_ids(
                tokenizer, messages[:-1], add_generation_prompt=True
            )
            if (
                len(prompt_ids) >= len(full_ids)
                or full_ids[: len(prompt_ids)] != prompt_ids
            ):
                # prompt_length still needs an exact prefix, never a guess.
                return None
            if len(prompt_ids) >= length:
                return None
            prompt_length = len(prompt_ids)
            mask[:length] = True
    except (TypeError, ValueError):
        return None

    tokens = np.full((max_length,), _pad_id(tokenizer), dtype=np.int32)
    tokens[:length] = np.asarray(full_ids[:length], dtype=np.int32)

    return EncodedExample(
        input_tokens=tokens,
        input_mask=mask,
        prompt_length=prompt_length,
        # Post-truncation: the packer slices by this.
        unpadded_length=length,
    )


# Fields of one packed window and of the batches the trainer receives when
# packing is enabled. segment_ids are 1..K per packed example and 0 on padding;
# positions restart at 0 for each segment so RoPE sees per-example offsets.
PACKED_FIELDS = ("input_tokens", "input_mask", "positions", "segment_ids")


class _OpenWindow:
    """A partially filled fixed-length window accepting further examples."""

    def __init__(self, max_length: int, pad_id: int):
        self.input_tokens = np.full((max_length,), pad_id, dtype=np.int32)
        self.input_mask = np.zeros((max_length,), dtype=np.bool_)
        self.positions = np.zeros((max_length,), dtype=np.int32)
        self.segment_ids = np.zeros((max_length,), dtype=np.int32)
        self.used = 0
        self.segments = 0

    def append(self, example: EncodedExample) -> None:
        length = int(example.unpadded_length)
        start, end = self.used, self.used + length
        self.input_tokens[start:end] = example.input_tokens[:length]
        self.input_mask[start:end] = example.input_mask[:length]
        # After the causal shift a segment's first token would be predicted
        # from the previous segment, so it never carries loss.
        self.input_mask[start] = False
        self.positions[start:end] = np.arange(length, dtype=np.int32)
        self.segments += 1
        self.segment_ids[start:end] = self.segments
        self.used = end

    def as_dict(self) -> dict[str, np.ndarray]:
        return {field: getattr(self, field) for field in PACKED_FIELDS}


def pack_encoded_examples(
    examples: Iterable[EncodedExample],
    *,
    max_length: int,
    pad_id: int,
    open_windows: int = 8,
) -> Iterable[dict[str, np.ndarray]]:
    """Pack variable-length examples into fixed windows, first fit.

    Up to ``open_windows`` windows accept examples at once; an example lands in
    the first window with room. When none has room and the pool is full, the
    fullest window is emitted, which keeps the emitted fill fraction high
    without unbounded buffering. All windows flush at the end of the stream.
    """
    if open_windows < 1:
        raise ValueError("open_windows must be at least 1")
    pool: list[_OpenWindow] = []
    for example in examples:
        length = int(example.unpadded_length)
        if length > max_length:
            # The encoder never emits one; a hand-built example could.
            raise ValueError("example is longer than the packing window")
        window = next((w for w in pool if w.used + length <= max_length), None)
        if window is None:
            if len(pool) == open_windows:
                fullest = max(pool, key=lambda w: w.used)
                pool.remove(fullest)
                yield fullest.as_dict()
            window = _OpenWindow(max_length, pad_id)
            pool.append(window)
        window.append(example)
    for window in pool:
        yield window.as_dict()


class PackedBatchDataset:
    """Re-iterable packing + batching stage over encoded examples.

    Yields ``{field: [batch, max_length]}`` dict batches carrying the packed
    geometry (positions, segment_ids) that ``TrainingInput`` cannot represent;
    the trainer treats batches as pytrees, so dicts pass through unchanged.
    A short final batch is dropped, matching ``batch(drop_remainder=True)``.
    """

    def __init__(
        self,
        source: Iterable[EncodedExample],
        *,
        max_length: int,
        pad_id: int,
        batch_size: int,
        open_windows: int = 8,
    ):
        self._source = source
        self._max_length = max_length
        self._pad_id = pad_id
        self._batch_size = batch_size
        self._open_windows = open_windows

    def __iter__(self) -> Any:
        buffer: list[dict[str, np.ndarray]] = []
        windows = pack_encoded_examples(
            iter(self._source),
            max_length=self._max_length,
            pad_id=self._pad_id,
            open_windows=self._open_windows,
        )
        for window in windows:
            buffer.append(window)
            if len(buffer) == self._batch_size:
                yield {
                    field: np.stack([window[field] for window in buffer])
                    for field in PACKED_FIELDS
                }
                buffer = []


def build_grain_dataset(
    data_source: Any,
    tokenizer: Any,
    *,
    batch_size: int,
    num_epochs: int,
    shuffle: bool,
    seed: int,
    encode_kwargs: dict[str, Any],
    packing: bool = False,
) -> Iterable[Any]:
    """Build a lazy, fixed-shape Grain dataset for Tunix ``PeftTrainer``."""
    from grain import python as grain
    from tunix.sft.peft_trainer import TrainingInput

    dataset = grain.MapDataset.source(data_source)
    if shuffle:
        dataset = dataset.shuffle(seed=seed)
    if num_epochs > 1:
        dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(
        lambda record: encode_reasoning_example(record, tokenizer, **encode_kwargs)
    )
    # The filter drops the Nones, which the type checker cannot see.
    dataset = cast(
        "grain.MapDataset[EncodedExample]",
        dataset.filter(lambda example: example is not None),
    )
    if packing:
        # Packing is stateful across examples, so it runs after Grain as a
        # re-iterable wrapper that also batches.
        return PackedBatchDataset(
            dataset.to_iter_dataset(),
            max_length=int(encode_kwargs["max_length"]),
            pad_id=_pad_id(tokenizer),
            batch_size=batch_size,
        )
    # Tunix declares TrainingInput with flax.struct.dataclass(frozen=True),
    # which hides its fields from type checkers.
    dataset = dataset.map(
        lambda example: TrainingInput(
            input_tokens=example.input_tokens,  # pyright: ignore[reportCallIssue]
            input_mask=example.input_mask,  # pyright: ignore[reportCallIssue]
        )
    )
    # Filtered MapDatasets no longer have a one-to-one index mapping, so Grain
    # requires conversion to an IterDataset before batching.
    dataset = dataset.to_iter_dataset()
    dataset = dataset.batch(batch_size=batch_size, drop_remainder=True)
    return dataset


def load_reasoning_datasets(config: dict[str, Any], tokenizer: Any) -> tuple[Any, Any]:
    """Load a Hugging Face reasoning dataset and return Tunix-ready iterables."""
    from datasets import load_dataset

    load_kwargs = {"split": config.get("train_split", "train")}
    if config.get("data_files") is not None:
        load_kwargs["data_files"] = config["data_files"]
    raw = load_dataset(config["name"], config.get("config"), **load_kwargs)
    max_examples = config.get("max_examples")
    if max_examples is not None:
        raw = raw.select(range(min(int(max_examples), len(raw))))

    eval_fraction = float(config.get("eval_fraction", 0.0))
    if eval_fraction:
        split = raw.train_test_split(
            test_size=eval_fraction, seed=int(config.get("seed", 42))
        )
        train_source, eval_source = split["train"], split["test"]
        # The trainer walks the whole eval split at every eval, so cap it.
        eval_max_examples = config.get("eval_max_examples")
        if eval_max_examples is not None:
            eval_source = eval_source.select(
                range(min(int(eval_max_examples), len(eval_source)))
            )
    else:
        train_source, eval_source = raw, None

    encode_kwargs = {
        "max_length": int(config["max_length"]),
        "messages_column": config.get("messages_column", "messages"),
        "system_prompt": read_prompt_file(config.get("system_prompt_file")),
        "assistant_only_loss": bool(config.get("assistant_only_loss", True)),
        "require_reasoning_tags": bool(config.get("require_reasoning_tags", True)),
        "reasoning_start": config.get("reasoning_start", "<think>"),
        "reasoning_end": config.get("reasoning_end", "</think>"),
        # Renamed per record rather than by rewriting the corpus, which for a
        # million-row dataset would cost far more.
        "message_schema": message_schema_from_config(config.get("message_schema")),
        "overlength_policy": str(config.get("overlength_policy", DROP_OVERLENGTH)),
    }
    common = {
        "tokenizer": tokenizer,
        "batch_size": int(config["batch_size"]),
        "seed": int(config.get("seed", 42)),
        "encode_kwargs": encode_kwargs,
        "packing": bool(config.get("packing", False)),
    }
    train_ds = build_grain_dataset(
        train_source,
        num_epochs=int(config.get("num_train_epochs", 1)),
        shuffle=True,
        **common,
    )
    eval_ds = None
    if eval_source is not None:
        eval_ds = build_grain_dataset(
            eval_source,
            num_epochs=1,
            shuffle=False,
            **common,
        )
    return train_ds, eval_ds
