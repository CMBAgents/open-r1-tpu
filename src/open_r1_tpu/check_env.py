"""Preflight the installed Tunix/JAX stack before consuming TPU time."""

from __future__ import annotations

import argparse
import math
import os
from importlib import metadata

from open_r1_tpu.config import load_config
from open_r1_tpu.data import encode_reasoning_example

DEFAULT_CONFIG = "recipes/OpenR1-Distill-Qwen3-1.7B/sft/config_distill.yaml"


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    config = load_config(args.config, args.overrides)

    import jax
    import optax
    from tunix.cli.utils import model as model_utils
    from tunix.models import automodel
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

    tokenizer = model_utils.create_tokenizer(
        config["tokenizer"], config["tokenizer"]["tokenizer_path"]
    )
    if config["tokenizer"].get("chat_template"):
        tokenizer.tokenizer.chat_template = config["tokenizer"]["chat_template"]
    sample = {
        "messages": [
            {"role": "user", "content": "What is 2 + 2?"},
            {
                "role": "assistant",
                "content": "<think>Adding gives four.</think>4",
            },
        ]
    }
    encoded = encode_reasoning_example(
        sample,
        tokenizer,
        max_length=256,
        system_prompt=config["dataset"].get("system_prompt"),
    )
    if encoded is None:
        errors.append(
            "the installed tokenizer did not produce a valid assistant boundary"
        )

    if config.get("export", {}).get("enabled", False):
        params_module = automodel.get_model_module(
            config["model"]["model_name"], automodel.ModelModule.PARAMS
        )
        if not callable(
            getattr(params_module, "save_lora_merged_model_as_safetensors", None)
        ):
            errors.append("the installed Tunix model lacks merged-LoRA export")

    print(f"JAX {jax.__version__}; Tunix {_version('google-tunix')}")
    print(f"Devices ({len(devices)}): {devices}")
    if encoded is not None:
        print(
            "Qwen chat template: "
            f"{encoded.prompt_length} prompt tokens, "
            f"{int(encoded.input_mask.sum())} supervised tokens"
        )
    if errors:
        raise SystemExit("TPU preflight failed:\n- " + "\n- ".join(errors))
    print("TPU preflight passed.")


if __name__ == "__main__":
    main()
