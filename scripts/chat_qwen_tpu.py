#!/usr/bin/env python3
"""Chat interactively with locally staged Qwen3 weights on one TPU.

Run from the repository root on the TPU VM after activating the project
environment::

    python scripts/chat_qwen_tpu.py --model-path models/Qwen3-1.7B-Base

To talk to a training run's own weights, pass the recipe it was trained with.
The LoRA adapters from its latest checkpoint are then restored on top of the
base model, which is how a run can be inspected before it has finished and
exported merged weights::

    python scripts/chat_qwen_tpu.py \
      --recipe recipes/Qwen3-1.7B-Instruct/sft/config_instruct.yaml

The rank, alpha and target modules come from that recipe rather than from
flags of their own: adapters restored under a different LoRA geometry than
they were trained with produce confident nonsense rather than an error.

The model is loaded with Tunix/JAX directly from ``model.safetensors``. It
never contacts the Hugging Face Hub. Type ``/help`` at the prompt for the
interactive commands.
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any


DEFAULT_MODEL_PATH = "models/Qwen3-1.7B-Base"
DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."
FLASH_ATTENTION_BLOCK_SIZE = 1024


@contextmanager
def tunix_mesh_context(mesh: Any):
    """Activate the legacy physical mesh still read by the pinned Tunix."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`with mesh:` context manager has been deprecated.*",
            category=DeprecationWarning,
        )
        with mesh:
            yield


def parse_args() -> argparse.Namespace:
    """Parse command-line options without importing the TPU runtime."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help=(
            "Local Qwen3-1.7B-Base directory containing model.safetensors "
            f"(default: {DEFAULT_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help=(
            "Training recipe whose latest LoRA checkpoint to restore on top "
            "of --model-path. Omit to talk to the base weights."
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
        help="System prompt prepended to every conversation; pass an empty string to omit it.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate for each assistant reply (default: 512).",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=FLASH_ATTENTION_BLOCK_SIZE,
        help=(
            "Fixed prompt-token budget, including chat-template tokens "
            f"(default: {FLASH_ATTENTION_BLOCK_SIZE})."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature; use 0 for greedy decoding (default: 0.6).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Sampling seed; it advances once per assistant reply (default: 42).",
    )
    return parser.parse_args()


def validate_options(args: argparse.Namespace) -> None:
    """Fail before the expensive model load for invalid sampler settings."""
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_prompt_length <= 0:
        raise ValueError("--max-prompt-length must be positive")
    if args.max_prompt_length % FLASH_ATTENTION_BLOCK_SIZE != 0:
        raise ValueError(
            "--max-prompt-length must be divisible by "
            f"{FLASH_ATTENTION_BLOCK_SIZE} when flash attention is enabled"
        )
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {model_path}")
    if not (model_path / "model.safetensors").is_file():
        raise FileNotFoundError(
            f"No model.safetensors found in local model directory: {model_path}"
        )
    args.model_path = str(model_path)

    if args.checkpoint_dir and not args.recipe:
        raise ValueError(
            "--checkpoint-dir needs --recipe, which supplies the LoRA rank, "
            "alpha and target modules the checkpoint was written under"
        )
    if args.recipe:
        recipe_path = Path(args.recipe).expanduser().resolve()
        if not recipe_path.is_file():
            raise FileNotFoundError(f"Recipe does not exist: {recipe_path}")
        args.recipe = str(recipe_path)


def model_config(
    model_path: str,
    seed: int,
    *,
    use_flash_attention: bool = True,
    lora_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the inference-safe subset of the project's Qwen TPU settings."""
    config: dict[str, Any] = {
        "model_name": "qwen3-1.7b-base",
        "model_source": "local",
        "model_path": model_path,
        "rng_seed": seed,
        "dtype": "bfloat16",
        "load_dtype": "bfloat16",
        "remat_config": "DECODER",
        "use_flash_attention": use_flash_attention,
        "flash_attention_block_size": FLASH_ATTENTION_BLOCK_SIZE,
        "mesh": {"shape": [1, 1], "axis_names": ["fsdp", "tp"]},
    }
    if lora_config:
        # Adapters must be restored into the same geometry they were trained
        # under, so this is read from the recipe rather than given its own
        # defaults here.
        config["lora_config"] = lora_config
    return config


def recipe_lora_settings(recipe_path: str) -> tuple[dict[str, Any], str]:
    """Read the LoRA geometry and checkpoint root a recipe trains with."""
    from open_r1_tpu.config import load_config

    config = load_config(recipe_path)
    lora_config = config["model"].get("lora_config")
    if not lora_config:
        raise ValueError(
            f"Recipe trains no LoRA adapters, so it has none to restore: "
            f"{recipe_path}"
        )
    return lora_config, config["training"]["checkpoint_dir"]


def available_steps(checkpoint_root: str) -> list[int]:
    """List the steps written under a checkpoint root, newest last."""
    root_path = Path(checkpoint_root)
    if not root_path.is_dir():
        return []
    return sorted(
        int(entry.name)
        for entry in root_path.iterdir()
        if entry.is_dir() and entry.name.isdigit()
    )


def resolve_step(checkpoint_root: str, step: int | None) -> int | None:
    """Reject a step that was never written, naming the ones that were.

    A run stopped between saves has no checkpoint at the step its log last
    reported, which is the step someone naturally asks for.
    """
    if step is None:
        return None
    steps = available_steps(checkpoint_root)
    if steps and step not in steps:
        written = ", ".join(str(value) for value in steps)
        raise FileNotFoundError(
            f"No checkpoint at step {step} under {checkpoint_root}. Steps "
            f"written and still kept: {written}. Training saves every "
            "training.checkpointing_options.save_interval_steps steps and "
            "keeps max_to_keep of them, so a run stopped between saves has "
            "no checkpoint at the step it stopped on."
        )
    return step


def restore_lora_checkpoint(
    model: Any, checkpoint_dir: str, step: int | None = None
) -> int:
    """Restore LoRA parameters into `model`, returning the step restored."""
    from open_r1_tpu.sft import _absolute_checkpoint_dir
    from tunix.sft import checkpoint_manager as checkpoint_manager_lib

    root = _absolute_checkpoint_dir(checkpoint_dir)
    step = resolve_step(root, step)
    # Deliberately built with Tunix's default options rather than the recipe's:
    # this client only reads, and the recipe's max_to_keep carries a
    # preservation policy that has no business running from a chat session.
    # The step naming is not configurable through a recipe, so the default
    # reads what training wrote.
    manager = checkpoint_manager_lib.CheckpointManager(root_directory=root)
    if manager.latest_step() is None:
        raise FileNotFoundError(
            f"No checkpoint has been written under {root}. Training writes "
            "one every training.checkpointing_options.save_interval_steps "
            "optimizer steps."
        )
    # Training saves adapters alone, since model.lora_config is set and Tunix
    # passes save_only_lora_params through to Orbax.
    restored_step, _metadata = manager.maybe_restore(
        model, optimizer=None, step=step, restore_only_lora_params=True
    )
    manager.close()
    return int(restored_step)


def tokenizer_config(model_path: str) -> dict[str, Any]:
    """Use the tokenizer staged beside the model, without a Hub lookup."""
    return {
        "tokenizer_path": model_path,
        "tokenizer_type": "huggingface",
        "add_bos": False,
        "add_eos": False,
        "chat_template": None,
    }


def messages_with_prompt(
    history: list[dict[str, str]], system_prompt: str
) -> list[dict[str, str]]:
    """Build a Qwen chat-template message list from retained conversation turns."""
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


def as_token_ids(value: Any) -> list[int]:
    """Normalize the tokenizer's list, array, or batch-of-one output."""
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


def prompt_token_count(
    tokenizer: Any, history: list[dict[str, str]], system_prompt: str
) -> int:
    """Count rendered prompt tokens without assuming a particular tokenizer class."""
    token_ids = tokenizer.apply_chat_template(
        messages_with_prompt(history, system_prompt),
        tokenize=True,
        add_generation_prompt=True,
    )
    # Recent Transformers versions return a BatchEncoding mapping by default;
    # it is mapping-like but is not necessarily a literal dict.
    if isinstance(token_ids, Mapping):
        token_ids = token_ids["input_ids"]
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


def load_runtime(
    args: argparse.Namespace, *, use_flash_attention: bool = True
) -> tuple[Any, Any, Any, Any]:
    """Create one TPU mesh, the local model, its tokenizer, and a sampler."""
    import jax
    from tunix.cli.utils import model as model_utils
    from tunix.generate import sampler as sampler_lib
    from tunix.utils import mesh as mesh_utils

    devices = jax.devices()
    if len(devices) != 1 or any(device.platform != "tpu" for device in devices):
        visible = ", ".join(f"{device.platform}:{device.id}" for device in devices)
        raise RuntimeError(
            "This client expects exactly one visible TPU device; JAX sees "
            f"[{visible}]. Run it on the configured TPU VM."
        )

    mesh = mesh_utils.create_mesh((1, 1), ("fsdp", "tp"))
    # Reuse the application's local safetensors loader so the inference clients
    # and SFT use the same Qwen architecture and dtype.
    from open_r1_tpu.sft import _create_model

    lora_config = None
    checkpoint_dir = args.checkpoint_dir
    if args.recipe:
        recipe_lora, recipe_checkpoint_dir = recipe_lora_settings(args.recipe)
        lora_config = recipe_lora
        checkpoint_dir = checkpoint_dir or recipe_checkpoint_dir

    config = {
        "model": model_config(
            args.model_path,
            args.seed,
            use_flash_attention=use_flash_attention,
            lora_config=lora_config,
        ),
        "tokenizer": tokenizer_config(args.model_path),
    }
    model, tokenizer_path = _create_model(config, mesh)
    if lora_config:
        restored = restore_lora_checkpoint(model, checkpoint_dir, args.step)
        # Named explicitly because it is rarely the step the run stopped on.
        print(f"Restored LoRA adapters from step {restored}.")
    tokenizer = model_utils.create_tokenizer(config["tokenizer"], tokenizer_path)
    model_runtime_config = getattr(model, "config", None)
    cache_config = sampler_lib.CacheConfig(
        cache_size=args.max_prompt_length + args.max_new_tokens,
        num_layers=int(getattr(model_runtime_config, "num_layers")),
        num_kv_heads=int(getattr(model_runtime_config, "num_kv_heads")),
        head_dim=int(getattr(model_runtime_config, "head_dim")),
    )
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=cache_config,
    )
    return mesh, tokenizer, sampler, model


def chat_loop(args: argparse.Namespace, tokenizer: Any, sampler: Any) -> None:
    """Read user turns, sample completions, and retain a bounded history."""
    history: list[dict[str, str]] = []
    reply_number = 0
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
            input_strings=[prompt],
            max_generation_steps=args.max_new_tokens,
            temperature=args.temperature,
            seed=args.seed + reply_number,
            max_prompt_length=args.max_prompt_length,
        )
        completion = str(output.text[0])
        history.append({"role": "assistant", "content": completion})
        reply_number += 1
        print(f"\nassistant> {completion.strip()}")


def main() -> None:
    args = parse_args()
    try:
        validate_options(args)
        print(f"Loading local model from {args.model_path} ...")
        mesh, tokenizer, sampler, _model = load_runtime(args)
        print("Model loaded. The first reply will compile the TPU decode path.")
        # The pinned Tunix sampler still reads JAX's legacy thread-local
        # physical mesh, just like PeftTrainer. jax.set_mesh(mesh) alone leaves
        # that legacy mesh empty and fails when the sampler resolves P("tp").
        with tunix_mesh_context(mesh):
            chat_loop(args, tokenizer, sampler)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
