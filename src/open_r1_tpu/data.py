"""Reasoning-trace preprocessing and Grain input pipelines."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from typing import Any, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class EncodedExample:
    """One fixed-length causal-LM example and its supervised-token mask."""

    input_tokens: np.ndarray
    input_mask: np.ndarray
    prompt_length: int
    unpadded_length: int


def normalize_messages(value: Any) -> list[dict[str, str]]:
    """Normalize a Hugging Face ``messages`` value to role/content dictionaries."""
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
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("each message needs string role and content fields")
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("messages cannot be empty")
    return messages


def prepare_messages(
    record: dict[str, Any],
    *,
    messages_column: str,
    system_prompt: str | None,
) -> list[dict[str, str]]:
    """Read a conversation, inject the reasoning system prompt, and validate it."""
    if messages_column not in record:
        raise ValueError(f"record has no {messages_column!r} column")
    messages = normalize_messages(record[messages_column])
    if messages[-1]["role"] != "assistant":
        raise ValueError("the final message must be an assistant reasoning trace")
    if system_prompt and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def _as_token_ids(value: Any) -> list[int]:
    # Recent Transformers versions return BatchEncoding, which implements the
    # mapping protocol but is not necessarily a plain dict.
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("chat template unexpectedly returned a token batch")
        value = value[0]
    if not isinstance(value, list) or not all(isinstance(token, int) for token in value):
        raise ValueError("chat template did not return a list of token IDs")
    return value


def _render_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    return _as_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )


def _pad_id(tokenizer: Any) -> int:
    pad_id_method = getattr(tokenizer, "pad_id", None)
    if callable(pad_id_method):
        return int(pad_id_method())
    pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
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
) -> EncodedExample | None:
    """Tokenize one trace, preserving only complete examples.

    Overlength examples are deliberately filtered instead of truncated. Cutting a
    reasoning trace teaches the model incomplete chains and often removes the final
    answer, so it is a poor default for reasoning distillation.
    """
    try:
        messages = prepare_messages(
            record,
            messages_column=messages_column,
            system_prompt=system_prompt,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None

    completion = messages[-1]["content"]
    if require_reasoning_tags and (
        reasoning_start not in completion or reasoning_end not in completion
    ):
        return None

    try:
        full_ids = _render_ids(
            tokenizer, messages, add_generation_prompt=False
        )
        prompt_ids = _render_ids(
            tokenizer, messages[:-1], add_generation_prompt=True
        )
    except (TypeError, ValueError):
        return None

    if len(full_ids) > max_length or len(full_ids) < 2:
        return None
    if len(prompt_ids) >= len(full_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
        # Assistant-only loss requires an exact prompt prefix. Silently using a
        # guessed boundary would train on the wrong tokens for some templates.
        return None

    prompt_length = len(prompt_ids)
    tokens = np.full((max_length,), _pad_id(tokenizer), dtype=np.int32)
    tokens[: len(full_ids)] = np.asarray(full_ids, dtype=np.int32)

    mask = np.zeros((max_length,), dtype=np.bool_)
    if assistant_only_loss:
        mask[prompt_length : len(full_ids)] = True
    else:
        mask[: len(full_ids)] = True

    return EncodedExample(
        input_tokens=tokens,
        input_mask=mask,
        prompt_length=prompt_length,
        unpadded_length=len(full_ids),
    )


def build_grain_dataset(
    data_source: Any,
    tokenizer: Any,
    *,
    batch_size: int,
    num_epochs: int,
    shuffle: bool,
    seed: int,
    encode_kwargs: dict[str, Any],
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
        lambda record: encode_reasoning_example(
            record, tokenizer, **encode_kwargs
        )
    )
    dataset = dataset.filter(lambda example: example is not None)
    dataset = dataset.map(
        lambda example: TrainingInput(
            input_tokens=example.input_tokens,
            input_mask=example.input_mask,
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
        # A fraction of a large corpus is a large evaluation set, and the
        # trainer walks all of it at every eval. Cap it so evaluation costs a
        # bounded number of steps rather than scaling with the corpus.
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
        "system_prompt": config.get("system_prompt"),
        "assistant_only_loss": bool(config.get("assistant_only_loss", True)),
        "require_reasoning_tags": bool(config.get("require_reasoning_tags", True)),
        "reasoning_start": config.get("reasoning_start", "<think>"),
        "reasoning_end": config.get("reasoning_end", "</think>"),
    }
    common = {
        "tokenizer": tokenizer,
        "batch_size": int(config["batch_size"]),
        "seed": int(config.get("seed", 42)),
        "encode_kwargs": encode_kwargs,
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
