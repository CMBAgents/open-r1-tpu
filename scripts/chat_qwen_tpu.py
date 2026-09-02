#!/usr/bin/env python3
"""Chat interactively with locally staged Qwen weights on the VM's TPUs.

Run from the repository root on the TPU VM after activating the project
environment::

    python scripts/chat_qwen_tpu.py --model-path models/Qwen3-1.7B-Base

Qwen2.5-1.5B weights are detected from their local ``config.json`` and use the
matching Tunix architecture automatically::

    python scripts/chat_qwen_tpu.py --model-path models/Qwen2.5-Math-1.5B

To talk to a training run's own weights, pass the recipe it was trained with.
Its latest checkpoint is then restored on top of the base model, which is how
a run can be inspected before it has finished and exported merged weights::

    python scripts/chat_qwen_tpu.py \
      --recipe recipes/Qwen3-1.7B-Instruct/sft/config_instruct.yaml

A full fine-tune's checkpoint replaces every parameter; a LoRA recipe's
checkpoint restores adapters alone. Which applies is read from the recipe's
model.lora_config rather than from flags of its own: adapters restored under
a different LoRA geometry than they were trained with produce confident
nonsense rather than an error.

The model is loaded with Tunix/JAX directly from ``model.safetensors``. It
never contacts the Hugging Face Hub. Type ``/help`` at the prompt for the
interactive commands.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

DEFAULT_MODEL_PATH = "models/Qwen3-1.7B-Base"
# Empty by default because SFT trains on conversations as they come, and the
# recipes leave dataset.system_prompt_file null. Injecting a system prompt
# here would put the model in front of a message type it rarely saw in
# training.
DEFAULT_SYSTEM_PROMPT = ""
FLASH_ATTENTION_BLOCK_SIZE = 1024

# Qwen3's chat template ends every turn with <|im_end|>, but the base model's
# tokenizer_config.json names <|endoftext|> as its EOS. The sampler stops on
# the tokenizer's EOS unless told otherwise, so a chat model left to itself
# runs past the end of its reply and writes the user's next turn as well.
TURN_END_TOKENS = ("<|im_end|>", "<|endoftext|>")

# That same template opens every assistant turn with an empty reasoning block
# when the message carries no <think> trace, so a model trained on a corpus
# without traces learns to emit one. It is scaffolding, not content.
EMPTY_REASONING = re.compile(r"\A<think>\s*</think>\s*")

# Reasoning traces are dimmed to grey so the answer stands out in the
# transcript. Display-only: history keeps the reply exactly as generated,
# and the codes are withheld when stdout is not a terminal (or NO_COLOR is
# set), so piped transcripts stay plain text.
REASONING_COLOUR = "\033[90m"  # bright black, rendered grey by terminals
COLOUR_RESET = "\033[0m"
REASONING_START = "<think>"
REASONING_END = "</think>"


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
            "Local supported Qwen directory containing model.safetensors and "
            "config.json "
            f"(default: {DEFAULT_MODEL_PATH})"
        ),
    )
    parser.add_argument(
        "--recipe",
        default=None,
        help=(
            "Training recipe whose latest checkpoint to restore on top of "
            "--model-path. Omit to talk to the base weights."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Canonical Tunix model name, only needed when the local config.json "
            "has no _name_or_path."
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
            "8192). Reasoning-distilled models write their whole thinking "
            "trace before the answer, so a budget that looks generous for "
            "chat truncates them mid-thought."
        ),
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


def validate_options(args: argparse.Namespace) -> None:
    """Fail before the expensive model load for invalid sampler settings."""
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive")
    if args.max_prompt_length <= 0:
        raise ValueError("--max-prompt-length must be positive")
    if args.temperature < 0:
        raise ValueError("--temperature cannot be negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")

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
            "--checkpoint-dir needs --recipe, which says whether the "
            "checkpoint holds full-model parameters or LoRA adapters, and "
            "for adapters supplies the geometry they were written under"
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
    use_flash_attention: bool,
    model_name: str,
    mesh_shape: tuple[int, int],
    lora_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the inference-safe subset of the detected Qwen TPU settings."""
    config: dict[str, Any] = {
        "model_name": model_name,
        "model_source": "local",
        "model_path": model_path,
        "rng_seed": seed,
        # float32, not training's bfloat16: under the pinned Tunix on JAX
        # 0.10.2 / v6e, the jitted Qwen2 forward returns all-NaN logits for
        # any KV cache of 1536 slots or more, and the sampler then emits
        # token id 0 ("!") forever. The same computation is clean when run
        # eagerly, at caches up to 1408 slots, and in float32 (verified to
        # 27,648 slots), so this is a bfloat16 compilation defect rather
        # than a weights or masking problem. Reasoning replies need caches
        # far beyond 1408, and an interactive client can afford float32.
        "dtype": "float32",
        "load_dtype": "float32",
        "remat_config": "DECODER",
        "use_flash_attention": use_flash_attention,
        "flash_attention_block_size": FLASH_ATTENTION_BLOCK_SIZE,
        "mesh": {"shape": list(mesh_shape), "axis_names": ["fsdp", "tp"]},
    }
    if lora_config:
        # Adapters must be restored into the same geometry they were trained
        # under, so this is read from the recipe rather than given its own
        # defaults here.
        config["lora_config"] = lora_config
    return config


def model_settings_for_path(
    model_path: str, model_name_override: str | None = None
) -> tuple[str, int]:
    """Read the Tunix model name and tensor-parallel width from ``config.json``.

    ``_name_or_path`` is the source model id that Hugging Face preserves in a
    local config. Tunix uses its lowercase final path component as its model
    name. An explicit ``--model-name`` handles exported configs that omit the
    source id without introducing architecture-specific defaults here.
    """
    config_path = Path(model_path) / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"No config.json found in local model directory: {config_path.parent}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in local model config: {config_path}") from exc
    if not isinstance(config, dict):
        raise ValueError(f"Local model config must be a JSON object: {config_path}")

    source_name = model_name_override or config.get("_name_or_path")
    if not isinstance(source_name, str) or not source_name.strip():
        raise ValueError(
            f"Local model config {config_path} has no _name_or_path; pass "
            "--model-name with the canonical Tunix model name."
        )
    num_kv_heads = config.get("num_key_value_heads")
    if not isinstance(num_kv_heads, int) or num_kv_heads <= 0:
        raise ValueError(
            f"Local model config {config_path} has invalid num_key_value_heads: "
            f"{num_kv_heads!r}"
        )
    return source_name.rsplit("/", maxsplit=1)[-1].lower(), num_kv_heads


def recipe_restore_settings(recipe_path: str) -> tuple[dict[str, Any] | None, str]:
    """Read the LoRA geometry (None for a full fine-tune) and checkpoint root."""
    from open_r1_tpu.core.config import load_config

    config = load_config(recipe_path)
    return config["model"].get("lora_config"), config["training"]["checkpoint_dir"]


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


def restore_checkpoint(
    model: Any, checkpoint_dir: str, step: int | None = None, *, lora_only: bool
) -> int:
    """Restore checkpoint parameters into `model`, returning the step restored."""
    from tunix.sft import checkpoint_manager as checkpoint_manager_lib

    from open_r1_tpu.training.run import _absolute_checkpoint_dir

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
    # A LoRA run saves adapters alone (model.lora_config makes Tunix pass
    # save_only_lora_params through to Orbax); a full fine-tune saves every
    # parameter. The restore must match what training wrote.
    restored_step, _metadata = manager.maybe_restore(
        model, optimizer=None, step=step, restore_only_lora_params=lora_only
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

    <think> and </think> are ordinary token sequences to this corpus's
    tokenisers and ChatML does not pre-open the block, so a reply may carry
    a full pair, a bare closing marker, or an unclosed opening one when the
    trace ran out the token budget. Everything through the closing marker
    is trace; with only an opening marker the entire reply is.
    """
    end = reply.find(REASONING_END)
    if end >= 0:
        split = end + len(REASONING_END)
        return f"{REASONING_COLOUR}{reply[:split]}{COLOUR_RESET}{reply[split:]}"
    if reply.startswith(REASONING_START):
        return f"{REASONING_COLOUR}{reply}{COLOUR_RESET}"
    return reply


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


def mesh_shape_for_devices(
    devices: Sequence[Any], num_kv_heads: int
) -> tuple[int, int]:
    """Use every visible TPU through FSDP and tensor parallelism.

    Tensor parallelism cannot exceed the model's KV-head count. Remaining
    chips form the FSDP axis; callers pad the interactive batch to that width
    before sampling.
    """
    if num_kv_heads <= 0:
        raise ValueError("num_kv_heads must be positive")
    if not devices:
        raise RuntimeError(
            "This client needs at least one visible TPU device; JAX sees no "
            "devices. Run it on a TPU VM with the TPU runtime configured."
        )
    if any(device.platform != "tpu" for device in devices):
        visible = ", ".join(f"{device.platform}:{device.id}" for device in devices)
        raise RuntimeError(
            "This client requires every visible JAX device to be a TPU; JAX sees "
            f"[{visible}]. Run it on a TPU VM with the TPU runtime configured."
        )
    tp_size = min(len(devices), num_kv_heads)
    if num_kv_heads % tp_size != 0 or len(devices) % tp_size != 0:
        raise RuntimeError(
            "This client needs a visible TPU-chip count that can form a tensor-"
            f"parallel width dividing its {num_kv_heads} KV heads; JAX sees "
            f"{len(devices)} TPU devices."
        )
    return len(devices) // tp_size, tp_size


def sampler_top_p(temperature: float, top_p: float) -> float | None:
    """Pick the top_p that puts the Tunix sampler in the intended mode.

    The pinned sampler selects its sampling mode from top_p alone: with
    top_p=None it greedy-decodes and silently ignores temperature and seed.
    So temperature 0 maps to top_p=None (greedy), and any positive
    temperature must carry a top_p for either knob to take effect at all.
    """
    return None if temperature == 0 else top_p


def pad_input_strings_for_fsdp(input_strings: list[str], fsdp_size: int) -> list[str]:
    """Repeat the final prompt so the sampler batch divides the FSDP width."""
    if fsdp_size <= 0:
        raise ValueError("FSDP mesh size must be positive")
    if not input_strings:
        raise ValueError("at least one input string is required")
    remainder = len(input_strings) % fsdp_size
    if remainder == 0:
        return input_strings
    return [
        *input_strings,
        *([input_strings[-1]] * (fsdp_size - remainder)),
    ]


def load_runtime(
    args: argparse.Namespace, *, use_flash_attention: bool = False
) -> tuple[Any, Any, Any, Any]:
    """Create an all-visible-TPU mesh, the local model, tokenizer, and sampler."""
    import jax
    from tunix.cli.utils import model as model_utils
    from tunix.generate import sampler as sampler_lib
    from tunix.utils import mesh as mesh_utils

    model_name, num_kv_heads = model_settings_for_path(
        args.model_path, getattr(args, "model_name", None)
    )
    mesh_shape = mesh_shape_for_devices(tuple(jax.devices()), num_kv_heads)
    mesh = mesh_utils.create_mesh(mesh_shape, ("fsdp", "tp"))
    args.sampler_fsdp_size = mesh_shape[0]
    # Reuse the application's local safetensors loader so the inference clients
    # and SFT use the same Qwen architecture. The dtype deliberately differs:
    # see model_config for why inference runs in float32.
    from open_r1_tpu.training.run import _create_model

    lora_config = None
    checkpoint_dir = args.checkpoint_dir
    if args.recipe:
        lora_config, recipe_checkpoint_dir = recipe_restore_settings(args.recipe)
        checkpoint_dir = checkpoint_dir or recipe_checkpoint_dir

    config = {
        "model": model_config(
            args.model_path,
            args.seed,
            use_flash_attention=use_flash_attention,
            model_name=model_name,
            mesh_shape=mesh_shape,
            lora_config=lora_config,
        ),
        "tokenizer": tokenizer_config(args.model_path),
    }
    model, tokenizer_path = _create_model(config, mesh)
    if args.recipe:
        restored = restore_checkpoint(
            model, checkpoint_dir, args.step, lora_only=bool(lora_config)
        )
        # Named explicitly because it is rarely the step the run stopped on.
        what = "LoRA adapters" if lora_config else "full model parameters"
        print(f"Restored {what} from step {restored}.")
    tokenizer = model_utils.create_tokenizer(config["tokenizer"], tokenizer_path)
    model_runtime_config = getattr(model, "config", None)
    if model_runtime_config is None:
        raise SystemExit("model exposes no config; cannot size the KV cache")
    cache_config = sampler_lib.CacheConfig(
        cache_size=args.max_prompt_length + args.max_new_tokens,
        num_layers=int(model_runtime_config.num_layers),
        num_kv_heads=int(model_runtime_config.num_kv_heads),
        head_dim=int(model_runtime_config.head_dim),
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
