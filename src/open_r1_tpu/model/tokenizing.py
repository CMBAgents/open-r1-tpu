"""Chat-template helpers shared by the SFT and GRPO training stages."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def as_token_ids(value: Any) -> list[int]:
    """Coerce an ``apply_chat_template(tokenize=True)`` result to a token list.

    Recent Transformers return a ``BatchEncoding`` mapping, not a list.
    """
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError("chat template unexpectedly returned a token batch")
        value = value[0]
    if not isinstance(value, list) or not all(
        isinstance(token, int) for token in value
    ):
        raise ValueError("chat template did not return a list of token IDs")
    return value


def user_turn(prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
    """One user message, after a system message when ``system_prompt`` is set."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def render_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    return as_token_ids(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )


# Keyed by id(tokenizer); the tokenizer itself is stored alongside the ids so
# the key cannot be recycled while the cache entry is alive.
_closing_ids_cache: dict[int, tuple[Any, list[int]]] = {}


def assistant_closing_ids(tokenizer: Any) -> list[int]:
    """Derive the token sequence that terminates every rendered assistant turn.

    It is the longest common token suffix of two probe conversations whose
    assistant replies differ in one character (``<|im_end|>\\n`` for Qwen).
    """
    cached = _closing_ids_cache.get(id(tokenizer))
    if cached is not None and cached[0] is tokenizer:
        return cached[1]
    probes = []
    for filler in ("0", "1"):
        probes.append(
            render_ids(
                tokenizer,
                [
                    {"role": "user", "content": "probe"},
                    {"role": "assistant", "content": filler},
                ],
                add_generation_prompt=False,
            )
        )
    first, second = probes
    length = 0
    limit = min(len(first), len(second))
    while length < limit and first[-1 - length] == second[-1 - length]:
        length += 1
    if length == 0:
        raise ValueError("chat template has no fixed assistant closing sequence")
    closing = first[len(first) - length :]
    _closing_ids_cache[id(tokenizer)] = (tokenizer, closing)
    return closing


def assistant_turn_end_id(tokenizer: Any) -> int:
    """The first token of the sequence that closes every assistant turn.

    A chat model must emit it for generation to stop (Qwen's ``<|im_end|>``),
    and a base model's generation config usually names a different EOS.
    """
    return assistant_closing_ids(tokenizer)[0]


def find_subsequence(haystack: list[int], needle: list[int], start: int) -> int | None:
    for position in range(start, len(haystack) - len(needle) + 1):
        if haystack[position : position + len(needle)] == needle:
            return position
    return None
