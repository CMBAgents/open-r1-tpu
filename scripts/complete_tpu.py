#!/usr/bin/env python3
"""Continue raw text with a locally staged Qwen2 or Qwen3 model on the TPUs.

No system prompt, role markers, chat template, or history is added, so this
shows a base model's native next-token behaviour. For a single completion::

    python scripts/complete_tpu.py \
      --model-path models/Qwen2.5-Math-1.5B --model-name qwen2.5-math-1.5b \
      "The capital of France is"

Omit the prompt to enter independent prompts interactively. Decoding is
greedy. The architecture is detected from the directory's ``config.json``.
"""

from __future__ import annotations

import argparse
from typing import Any

from open_r1_tpu.model.checkpoint import (
    load_sampler,
    pad_input_strings_for_fsdp,
    resolve_model_dir,
    tunix_mesh_context,
)

DEFAULT_MODEL_PATH = "models/Qwen2.5-Math-1.5B"
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_MAX_PROMPT_LENGTH = 2048


def parse_args() -> argparse.Namespace:
    """Parse command-line options without importing the TPU runtime."""
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
            "Local Qwen2 or Qwen3 directory containing model.safetensors and "
            f"config.json (default: {DEFAULT_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Tunix model name, such as qwen2.5-math-1.5b (default: read from "
            "config.json's _name_or_path, which Hub downloads and exports lack)."
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
        help=f"Fixed raw-prompt token budget (default: {DEFAULT_MAX_PROMPT_LENGTH}).",
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
    args.model_path = resolve_model_dir(args.model_path)


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
        input_strings=pad_input_strings_for_fsdp(
            [prompt], getattr(args, "sampler_fsdp_size", 1)
        ),
        max_generation_steps=args.max_new_tokens,
        # With no top_p or beam_size, Tunix decodes greedily and ignores
        # temperature.
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
        runtime = load_sampler(
            args.model_path,
            seed=args.seed,
            cache_size=args.max_prompt_length + args.max_new_tokens,
            model_name=args.model_name,
        )
        args.sampler_fsdp_size = runtime.fsdp_size
        print("Model loaded. The first completion will compile the TPU decode path.")
        with tunix_mesh_context(runtime.mesh):
            if args.prompt is None:
                interactive_loop(runtime.sampler, args)
            else:
                print_completion(runtime.sampler, args.prompt, args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
