#!/usr/bin/env python3
"""Chat interactively with a locally staged Qwen2 or Qwen3 model on the TPUs.

Run from the repository root on the TPU VM, with the project environment
active::

    python scripts/chat_tpu.py --model-path models/Qwen2.5-Math-1.5B \
      --model-name qwen2.5-math-1.5b

The architecture is detected from the directory's ``config.json``. To talk to a
training run's own weights, before it has finished or been exported, pass the
SFT recipe it was trained with; its latest checkpoint is restored on top of the
local base the run started from (``model.model_path``, or
``model.model_download_path`` for a Hub model) unless ``--model-path`` says
otherwise::

    python scripts/chat_tpu.py \
      --recipe recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml

Whether the checkpoint holds every parameter or LoRA adapters alone, and the
adapters' geometry, are read from the recipe's ``model.lora_config``: adapters
restored under the wrong geometry produce confident nonsense, not an error.

The model is loaded from ``model.safetensors`` without contacting the Hugging
Face Hub. Type ``/help`` at the prompt for the interactive commands.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any

from open_r1_tpu.model.checkpoint import (
    load_sampler,
    pad_input_strings_for_fsdp,
    resolve_model_dir,
    tunix_mesh_context,
)
from open_r1_tpu.model.tokenizing import as_token_ids

DEFAULT_MODEL_PATH = "models/Qwen2.5-Math-1.5B"
DEFAULT_MAX_PROMPT_LENGTH = 1024
# Empty, so no system message is sent and the chat template supplies its own
# default, if it has one. Pass --system-prompt to use a recipe's prompt.
DEFAULT_SYSTEM_PROMPT = ""

# The chat template ends every turn with <|im_end|>, but Qwen base models name
# <|endoftext|> as their EOS, and the sampler stops only on the tokens it is
# given. Without <|im_end|> the model writes the user's next turn as well.
TURN_END_TOKENS = ("<|im_end|>", "<|endoftext|>")

# Qwen3's template opens an assistant turn with an empty reasoning block when
# the message has no <think> trace, so a model trained on such data learns to
# emit one. It is scaffolding, not content.
EMPTY_REASONING = re.compile(r"\A<think>\s*</think>\s*")

# Reasoning traces are shown in grey so the answer stands out. Display only:
# history keeps the reply as generated, and no codes are written when stdout is
# not a terminal or NO_COLOR is set.
REASONING_COLOUR = "\033[90m"  # bright black, rendered grey by terminals
COLOUR_RESET = "\033[0m"
REASONING_START = "<think>"
REASONING_END = "</think>"


def parse_args() -> argparse.Namespace:
    """Parse command-line options without importing the TPU runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=None,
        help=(
            "Local Qwen2 or Qwen3 directory containing model.safetensors and "
            "config.json (default: the recipe's local base model with --recipe, "
            f"else {DEFAULT_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help=(
            "SFT recipe whose latest checkpoint to restore on top of "
            "--model-path. Omit to talk to the base weights."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Tunix model name, such as qwen2.5-math-1.5b (default: the recipe's "
            "model.model_name with --recipe, else read from config.json's "
            "_name_or_path, which Hub downloads and exports lack)."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Checkpoint root to restore from, overriding the recipe's "
            "training.checkpoint_dir."
        ),
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to restore (default: the latest written).",
    )
    parser.add_argument(
        "--system-prompt",
        default=DEFAULT_SYSTEM_PROMPT,
        help=(
            "System prompt prepended to every conversation; pass an empty "
            "string to omit it."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=8192,
        help=(
            "Maximum tokens to generate for each assistant reply (default: "
            "8192). Reasoning models write their whole thinking trace before "
            "the answer, so a smaller budget truncates them mid-thought."
        ),
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=DEFAULT_MAX_PROMPT_LENGTH,
        help=(
            "Fixed prompt-token budget, including chat-template tokens "
            f"(default: {DEFAULT_MAX_PROMPT_LENGTH})."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature; use 0 for greedy decoding (default: 0.6).",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help=(
            "Nucleus sampling threshold (default: 0.95). Ignored when "
            "--temperature is 0."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed; it advances once per assistant reply (default: 42).",
    )
    return parser.parse_args()


def recipe_model(recipe: str) -> tuple[str | None, str]:
    """An SFT recipe's local base, as its merged export finds it (None if it
    has none), and its Tunix model name."""
    from open_r1_tpu.core.config import load_config
    from open_r1_tpu.model.export import local_base_model_path
    from open_r1_tpu.sft.config import validate_sft_config

    config = load_config(recipe, validator=validate_sft_config)
    try:
        base = local_base_model_path(config)
    except ValueError:
        base = None
    return base, str(config["model"]["model_name"])


def validate_options(args: argparse.Namespace) -> None:
    """Fail before the expensive model load for invalid options."""
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_prompt_length <= 0:
        raise ValueError("--max-prompt-length must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if args.checkpoint_dir and not args.recipe:
        raise ValueError(
            "--checkpoint-dir needs --recipe, which says whether the "
            "checkpoint holds full-model parameters or LoRA adapters, and "
            "for adapters supplies the geometry they were written under"
        )
    if args.recipe:
        recipe_path = Path(args.recipe).expanduser().resolve()
        if not recipe_path.is_file():
            raise FileNotFoundError(f"Recipe does not exist: {recipe_path}")
        args.recipe = str(recipe_path)
        base, model_name = recipe_model(args.recipe)
        if args.model_name is None:
            args.model_name = model_name
        if args.model_path is None:
            if base is None:
                raise ValueError(
                    f"{args.recipe} names no local base model; pass --model-path"
                )
            args.model_path = base

    if args.model_path is None:
        args.model_path = DEFAULT_MODEL_PATH
    args.model_path = resolve_model_dir(args.model_path)


def messages_with_prompt(
    history: list[dict[str, str]], system_prompt: str
) -> list[dict[str, str]]:
    """Build a chat-template message list from retained conversation turns."""
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history)
    return messages


def render_prompt(
    tokenizer: Any, history: list[dict[str, str]], system_prompt: str
) -> str:
    """Render the exact generation prefix expected by the local tokenizer."""
    return tokenizer.apply_chat_template(
        messages_with_prompt(history, system_prompt),
        tokenize=False,
        add_generation_prompt=True,
    )


def token_id(tokenizer: Any, token: str) -> int | None:
    """Look up one special token, through the adapter or its HF tokenizer."""
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        inner = getattr(tokenizer, "tokenizer", None)
        convert = getattr(inner, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    try:
        resolved = convert(token)
    except (KeyError, TypeError, ValueError):
        return None
    return resolved if isinstance(resolved, int) else None


def stop_token_ids(tokenizer: Any) -> list[int]:
    """Resolve the tokens that end an assistant turn."""
    stop_ids: list[int] = []
    for token in TURN_END_TOKENS:
        resolved = token_id(tokenizer, token)
        if resolved is not None and resolved not in stop_ids:
            stop_ids.append(resolved)
    if not stop_ids:
        raise ValueError("tokenizer defines none of " + ", ".join(TURN_END_TOKENS))
    return stop_ids


def clean_reply(completion: str) -> str:
    """Drop any turn-ending marker the sampler echoed back."""
    text = completion
    for token in TURN_END_TOKENS:
        if text.endswith(token):
            text = text[: -len(token)]
            break
    return text.strip()


def visible_reply(completion: str) -> str:
    """Hide the template's empty reasoning scaffold from the transcript."""
    return EMPTY_REASONING.sub("", clean_reply(completion), count=1).strip()


def colour_reasoning(reply: str) -> str:
    """Wrap the reasoning trace, when one is present, in grey.

    ``<think>`` and ``</think>`` are ordinary token sequences to these
    tokenizers and ChatML does not pre-open the block, so a reply may carry a
    full pair, a bare closing marker, or an opening marker whose trace ran out
    of tokens. Everything through the closing marker is trace; with only an
    opening marker, the whole reply is.
    """
    end = reply.find(REASONING_END)
    if end >= 0:
        split = end + len(REASONING_END)
        return f"{REASONING_COLOUR}{reply[:split]}{COLOUR_RESET}{reply[split:]}"
    if reply.startswith(REASONING_START):
        return f"{REASONING_COLOUR}{reply}{COLOUR_RESET}"
    return reply


def prompt_token_count(
    tokenizer: Any, history: list[dict[str, str]], system_prompt: str
) -> int:
    """Count rendered prompt tokens without assuming a particular tokenizer class."""
    token_ids = tokenizer.apply_chat_template(
        messages_with_prompt(history, system_prompt),
        tokenize=True,
        add_generation_prompt=True,
    )
    return len(as_token_ids(token_ids))


def fit_history(
    tokenizer: Any,
    history: list[dict[str, str]],
    system_prompt: str,
    max_prompt_length: int,
) -> tuple[list[dict[str, str]] | None, int, int]:
    """Remove oldest complete turns until the next prompt fits the KV cache.

    ``history`` must end in the just-entered user message. A single overlong
    user message cannot be truncated safely, so it is rejected instead.
    """
    retained = list(history)
    removed_turns = 0
    count = prompt_token_count(tokenizer, retained, system_prompt)
    while count > max_prompt_length and len(retained) > 1:
        # After a completed reply, history alternates user and assistant, so
        # discarding two items keeps complete conversation turns intact.
        del retained[:2]
        removed_turns += 1
        count = prompt_token_count(tokenizer, retained, system_prompt)
    if count > max_prompt_length:
        return None, count, removed_turns
    return retained, count, removed_turns


def sampler_top_p(temperature: float, top_p: float) -> float | None:
    """Pick the top_p that puts the Tunix sampler in the intended mode.

    The pinned sampler greedy-decodes whenever top_p is None, silently ignoring
    temperature and seed. So temperature 0 maps to None, and any positive
    temperature must carry a top_p for either setting to take effect.
    """
    return None if temperature == 0 else top_p


def chat_loop(args: argparse.Namespace, tokenizer: Any, sampler: Any) -> None:
    """Read user turns, sample completions, and retain a bounded history."""
    history: list[dict[str, str]] = []
    reply_number = 0
    stop_ids = stop_token_ids(tokenizer)
    colour = sys.stdout.isatty() and "NO_COLOR" not in os.environ
    print("Ready. Type /help for commands, /reset to clear history, or /exit to quit.")

    while True:
        try:
            user_text = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            print("Goodbye.")
            return
        if user_text == "/reset":
            history.clear()
            print("Conversation cleared.")
            continue
        if user_text == "/help":
            print("Commands: /help, /reset, /exit (or /quit).")
            continue

        candidate = [*history, {"role": "user", "content": user_text}]
        fitted, token_count, removed_turns = fit_history(
            tokenizer,
            candidate,
            args.system_prompt,
            args.max_prompt_length,
        )
        if fitted is None:
            print(
                "That message is too long for the prompt budget "
                f"({token_count} > {args.max_prompt_length} tokens). "
                "Send a shorter message or restart with a larger "
                "--max-prompt-length."
            )
            continue
        history = fitted
        if removed_turns:
            print(
                f"[Dropped {removed_turns} oldest turn(s) to fit the "
                f"{args.max_prompt_length}-token prompt budget.]"
            )

        prompt = render_prompt(tokenizer, history, args.system_prompt)
        output = sampler(
            input_strings=pad_input_strings_for_fsdp(
                [prompt], getattr(args, "sampler_fsdp_size", 1)
            ),
            max_generation_steps=args.max_new_tokens,
            temperature=args.temperature,
            top_p=sampler_top_p(args.temperature, args.top_p),
            seed=args.seed + reply_number,
            max_prompt_length=args.max_prompt_length,
            eos_tokens=stop_ids,
        )
        completion = clean_reply(str(output.text[0]))
        # The template drops a prior turn's reasoning block when it renders
        # the next prompt, so the history keeps the reply as generated.
        history.append({"role": "assistant", "content": completion})
        reply_number += 1
        reply = visible_reply(completion)
        if colour:
            reply = colour_reasoning(reply)
        print(f"\nassistant> {reply}")


def main() -> None:
    args = parse_args()
    try:
        validate_options(args)
        print(f"Loading local model from {args.model_path} ...")
        runtime = load_sampler(
            args.model_path,
            seed=args.seed,
            cache_size=args.max_prompt_length + args.max_new_tokens,
            model_name=args.model_name,
            recipe=args.recipe,
            checkpoint_dir=args.checkpoint_dir,
            step=args.step,
        )
        args.sampler_fsdp_size = runtime.fsdp_size
        print("Model loaded. The first reply will compile the TPU decode path.")
        with tunix_mesh_context(runtime.mesh):
            chat_loop(args, runtime.tokenizer, runtime.sampler)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
