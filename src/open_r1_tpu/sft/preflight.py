"""Preflight the installed training stack against an SFT recipe before
consuming TPU time.

Run with::

  python -m open_r1_tpu.sft.preflight --config \
    recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml
"""

from __future__ import annotations

import math
import os
from typing import Any

from open_r1_tpu.core.cli import parse_recipe_args, recipe_parser
from open_r1_tpu.core.config import load_config, read_prompt_file
from open_r1_tpu.core.packages import installed_version
from open_r1_tpu.model.export import merged_lora_saver, safetensors_entry_fn
from open_r1_tpu.model.loading import create_tokenizer
from open_r1_tpu.sft.config import validate_sft_config
from open_r1_tpu.sft.data import (
    encode_reasoning_example,
    message_schema_from_config,
)


def _preflight_example(
    dataset: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a probe conversation in the shape the recipe's own corpus uses.

    A hardcoded role/content record would fail the boundary check on, say, a
    ShareGPT-style corpus and report a tokenizer fault that is not there, so
    the probe is written through the inverted role map, which also exercises
    the mapping.
    """
    schema = message_schema_from_config(dataset.get("message_schema"))
    sources = {target: source for source, target in schema.role_map.items()}
    column = dataset.get("messages_column", "messages")
    record = {
        column: [
            {schema.role_key: sources.get(role, role), schema.content_key: content}
            for role, content in (
                ("user", "What is 2 + 2?"),
                ("assistant", "<think>Adding gives four.</think>4"),
            )
        ]
    }
    encode_kwargs = {
        "messages_column": column,
        "system_prompt": read_prompt_file(dataset.get("system_prompt_file")),
        "message_schema": schema,
    }
    return record, encode_kwargs


def main() -> None:
    args = parse_recipe_args(recipe_parser(__doc__))
    config = load_config(args.config, args.overrides, validator=validate_sft_config)

    import jax
    import optax
    from tunix.sft import peft_trainer
    from tunix.sft import utils as sft_utils

    errors: list[str] = []
    devices = jax.devices()
    mesh_size = math.prod(config["model"]["mesh"]["shape"])
    if len(devices) != mesh_size:
        errors.append(f"recipe needs {mesh_size} devices but JAX sees {len(devices)}")
    non_tpu = [str(device) for device in devices if device.platform != "tpu"]
    if non_tpu:
        errors.append(f"non-TPU JAX devices detected: {non_tpu}")
    if config["model"]["model_source"] == "huggingface" and not os.environ.get(
        "HF_TOKEN"
    ):
        errors.append("HF_TOKEN is unset; Tunix's downloader requires it")

    required_api = {
        "PeftTrainer.with_loss_fn": hasattr(peft_trainer.PeftTrainer, "with_loss_fn"),
        "PeftTrainer.with_gen_model_input_fn": hasattr(
            peft_trainer.PeftTrainer, "with_gen_model_input_fn"
        ),
        "LossOutput": hasattr(sft_utils, "LossOutput"),
        "WeightedMetric": hasattr(sft_utils, "WeightedMetric"),
        "integer-label cross entropy": hasattr(
            optax, "softmax_cross_entropy_with_integer_labels"
        ),
    }
    errors.extend(
        f"installed stack lacks {name}"
        for name, available in required_api.items()
        if not available
    )
    if str(config["training"]["checkpoint_dir"]).startswith("gs://"):
        try:
            import gcsfs  # noqa: F401
        except ImportError:
            errors.append("GCS checkpointing requires the gcsfs package")

    tokenizer = create_tokenizer(config, config["tokenizer"]["tokenizer_path"])
    sample, encode_kwargs = _preflight_example(config["dataset"])
    encoded = encode_reasoning_example(
        sample,
        tokenizer,
        max_length=256,
        **encode_kwargs,
    )
    if encoded is None:
        errors.append(
            "the installed tokenizer did not produce a valid assistant boundary"
        )

    # Export runs after the last training step, so an unsupported model found
    # there costs the whole run. Check the branch export_model will take.
    if config.get("export", {}).get("enabled", False):
        model_name = str(config["model"]["model_name"])
        try:
            if config["model"].get("lora_config"):
                merged_lora_saver(model_name)
            else:
                safetensors_entry_fn(model_name)
        except NotImplementedError as exc:
            errors.append(str(exc))

    print(f"JAX {jax.__version__}; Tunix {installed_version('google-tunix')}")
    print(f"Devices ({len(devices)}): {devices}")
    if encoded is not None:
        print(
            "Chat template: "
            f"{encoded.prompt_length} prompt tokens, "
            f"{int(encoded.input_mask.sum())} supervised tokens"
        )
    if errors:
        raise SystemExit("TPU preflight failed:\n- " + "\n- ".join(errors))
    print("TPU preflight passed.")


if __name__ == "__main__":
    main()
