#!/usr/bin/env python3
"""Continue raw text with locally staged Qwen3-1.7B-Base weights on one TPU.

No system prompt, role markers, chat template, or history is added. For a
single completion::

    python scripts/complete_qwen_tpu.py \
      --model-path models/Qwen3-1.7B-Base \
      "The capital of France is"

Omit the positional prompt to enter multiple independent prompts interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from chat_qwen_tpu import (
    DEFAULT_MODEL_PATH,
    load_runtime,
    tunix_mesh_context,
)

DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_MAX_PROMPT_LENGTH = 2048


def parse_args() -> argparse.Namespace:
    """Parse CLI options without importing the TPU runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Raw text to continue; omit it for an interactive prompt loop.",
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=(
            "Local Qwen3-1.7B-Base directory containing model.safetensors "
            f"(default: {DEFAULT_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "Maximum continuation length in tokens "
            f"(default: {DEFAULT_MAX_NEW_TOKENS})."
        ),
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=DEFAULT_MAX_PROMPT_LENGTH,
        help=(f"Fixed raw-prompt token budget (default: {DEFAULT_MAX_PROMPT_LENGTH})."),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Generation seed (default: 42).",
    )
    return parser.parse_args()


def validate_options(args: argparse.Namespace) -> None:
    """Validate fixed shapes and the local model before loading it."""
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_prompt_length <= 0:
        raise ValueError("--max-prompt-length must be positive")
    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"No model.safetensors found in local model directory: {model_path}"
        )
    args.model_path = str(model_path)


def load_completion_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, Any]:
    """Load Qwen with the padding-aware, non-splash attention path.

    The pinned Tunix sampler left-pads fixed-length prompts but does not pass
    segment IDs into Qwen splash attention, so splash would expose real prompt
    tokens to pad/EOS embeddings. Ordinary attention consumes the padding mask.
    """
    return load_runtime(args, use_flash_attention=False)


def generate_completion(sampler: Any, prompt: str, args: argparse.Namespace) -> str:
    """Generate from exactly ``prompt`` without chat formatting or history."""
    if not prompt:
        raise ValueError("prompt cannot be empty")
    prompt_tokens = sampler.tokenize(prompt)
    if len(prompt_tokens) > args.max_prompt_length:
        raise ValueError(
            "prompt is too long for the configured prompt budget "
            f"({len(prompt_tokens)} > {args.max_prompt_length} tokens)"
        )

    output = sampler(
        input_strings=[prompt],
        max_generation_steps=args.max_new_tokens,
        # With no top_p or beam_size, Tunix performs deterministic greedy
        # next-token selection; temperature is unused in that mode.
        temperature=1.0,
        seed=args.seed,
        max_prompt_length=args.max_prompt_length,
    )
    return str(output.text[0])


def print_completion(sampler: Any, prompt: str, args: argparse.Namespace) -> None:
    """Generate and display the original text plus its exact continuation."""
    continuation = generate_completion(sampler, prompt, args)
    print(f"\nresult> {prompt}{continuation}")


def interactive_loop(sampler: Any, args: argparse.Namespace) -> None:
    """Complete independent prompts until the user exits."""
    print("Ready. Each prompt is independent. Type /exit (or /quit) to stop.")
    while True:
        try:
            prompt = input("\nprompt> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            return
        if prompt.strip() in {"/exit", "/quit"}:
            print("Goodbye.")
            return
        if not prompt:
            continue
        try:
            print_completion(sampler, prompt, args)
        except ValueError as exc:
            print(f"error: {exc}")


def main() -> None:
    args = parse_args()
    try:
        validate_options(args)
        print(f"Loading local model from {args.model_path} ...")
        mesh, _tokenizer, sampler, _model = load_completion_runtime(args)
        print("Model loaded. The first completion will compile the TPU decode path.")
        with tunix_mesh_context(mesh):
            if args.prompt is None:
                interactive_loop(sampler, args)
            else:
                print_completion(sampler, args.prompt, args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
