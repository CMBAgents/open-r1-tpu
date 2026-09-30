#!/usr/bin/env python3
"""Export merged Hugging Face weights from a saved SFT checkpoint.

Training exports only its final state. This restores any saved step of a run
on top of the recipe's base model and writes it through the same
``export_model``, so intermediate checkpoints can be evaluated like the final
export. A full fine-tune's checkpoint replaces every parameter; a LoRA recipe's
(one that sets ``model.lora_config``) restores the adapters, which the export
merges into the base weights. The adapters' geometry comes from the recipe, so
pass the recipe the run was trained with.

Run from the repository root on the TPU VM, with the project environment
active::

    python scripts/export_checkpoint.py \
        --recipe recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml \
        --step 200 \
        --output artifacts/OpenR1-Distill-Qwen2.5-Math-1.5B/checkpoint-200/merged

The model is loaded on the recipe's ``model.mesh``. An existing ``--output``
directory is replaced.
"""

from __future__ import annotations

import argparse
from typing import Any

from open_r1_tpu.core.config import load_config
from open_r1_tpu.model.checkpoint import restore_checkpoint
from open_r1_tpu.model.export import export_model
from open_r1_tpu.model.loading import create_mesh, create_model
from open_r1_tpu.sft.config import validate_sft_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe", required=True, help="SFT recipe the run was trained with"
    )
    parser.add_argument(
        "--step", type=int, required=True, help="checkpoint step to restore"
    )
    parser.add_argument("--output", required=True, help="merged export directory")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help=(
            "Checkpoint root to restore from, overriding the recipe's "
            "training.checkpoint_dir."
        ),
    )
    return parser.parse_args(argv)


def read_recipe(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any] | None, str]:
    """Return the recipe with export enabled into ``--output``, its LoRA
    geometry (None for a full fine-tune), and the checkpoint root."""
    config = load_config(args.recipe, validator=validate_sft_config)
    config["export"] = {
        **config.get("export", {}),
        "output_dir": args.output,
        "overwrite": True,
        "enabled": True,
    }
    checkpoint_dir = args.checkpoint_dir or config["training"]["checkpoint_dir"]
    return config, config["model"].get("lora_config"), checkpoint_dir


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config, lora_config, checkpoint_dir = read_recipe(args)

    from tunix.cli.utils import model as model_utils

    mesh = create_mesh(config)
    print(f"Loading base model from {config['model']['model_path']} ...", flush=True)
    model, local_model_path = create_model(config, mesh)

    print(f"Restoring step {args.step} from {checkpoint_dir} ...", flush=True)
    restored = restore_checkpoint(
        model, checkpoint_dir, args.step, lora_only=bool(lora_config)
    )
    what = "LoRA adapters" if lora_config else "full model parameters"
    print(f"Restored {what} from step {restored}.", flush=True)

    tokenizer_config = {
        **config["tokenizer"],
        "tokenizer_path": local_model_path,
        "chat_template": None,
    }
    tokenizer = model_utils.create_tokenizer(tokenizer_config, local_model_path)
    export_model(
        config=config,
        model=model,
        tokenizer=tokenizer,
        local_model_path=local_model_path,
    )
    print(f"Exported step {restored} to {args.output}")


if __name__ == "__main__":
    main()
