"""Chat-template token helpers shared by the SFT and GRPO training stages.

Public names here (rather than the leading-underscore style used for
same-module helpers elsewhere in this project) because both
:mod:`open_r1_tpu.sft.data` and :mod:`open_r1_tpu.grpo.run` import them across
a package boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def as_token_ids(value: Any) -> list[int]:
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
    if not isinstance(value, list) or not all(
        isinstance(token, int) for token in value
    ):
        raise ValueError("chat template did not return a list of token IDs")
    return value


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

    Two single-turn probe conversations whose assistant replies differ only in
    their final character are rendered; the longest common token suffix is the
    template's closing sequence (``<|im_end|>\\n`` for Qwen-style templates).
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

    This is the token a chat model must emit for generation to stop (Qwen's
    ``<|im_end|>``). Base-model generation configs name a different EOS, and a
    stop *string* for it never matches under serving defaults that strip
    special tokens from decoded text, so servers need it as a token-level EOS.
    """
    return assistant_closing_ids(tokenizer)[0]


def find_subsequence(haystack: list[int], needle: list[int], start: int) -> int | None:
    for position in range(start, len(haystack) - len(needle) + 1):
        if haystack[position : position + len(needle)] == needle:
            return position
    return None
