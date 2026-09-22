#!/usr/bin/env python3
"""Export a merged HF-format model from a saved mid-training SFT checkpoint.

`run.py` only exports the final in-memory model state at the end of a run.
This restores an arbitrary saved step from the recipe's own checkpoint
directory (the same restore Tunix's `CheckpointManager` performs, as used
interactively in `chat_qwen_tpu.py`) and runs it through the same
`export_model` used at the end of every training run, so beginning/middle
checkpoints can be evaluated the same way as the final export.

Usage (from the repo root, with the pinned environment active):
    python3 scripts/export_checkpoint.py \
        --recipe recipes/rowanai/sft/config_rowanai_clean_v4_worked.yaml \
        --step 200 \
        --output artifacts/rowanai-clean-worked/v4-rowanai/checkpoint-200/merged
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="the arm's SFT recipe")
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
    return parser.parse_args()


def restore_checkpoint(model: Any, checkpoint_dir: str, step: int) -> int:
    """Restore full-model parameters at `step`, returning the step restored.

    Mirrors `chat_qwen_tpu.restore_checkpoint`: these recipes train no LoRA
    adapter, so the checkpoint always holds every parameter.
    """
    from tunix.sft import checkpoint_manager as checkpoint_manager_lib

    from open_r1_tpu.model.loading import absolute_checkpoint_dir

    root = absolute_checkpoint_dir(checkpoint_dir)
    manager = checkpoint_manager_lib.CheckpointManager(root_directory=root)
    if manager.latest_step() is None:
        raise FileNotFoundError(f"No checkpoint has been written under {root}")
    restored_step, _metadata = manager.maybe_restore(
        model, optimizer=None, step=step, restore_only_lora_params=False
    )
    manager.close()
    return int(restored_step)


def main() -> None:
    args = parse_args()

    from tunix.cli.utils import model as model_utils
    from tunix.utils import mesh as mesh_utils

    from open_r1_tpu.core.config import load_config
    from open_r1_tpu.model.export import export_model
    from open_r1_tpu.model.loading import create_model

    config = load_config(args.recipe)
    if config["model"].get("lora_config"):
        raise SystemExit(
            "export_checkpoint.py only supports full fine-tunes; "
            f"{args.recipe} sets model.lora_config."
        )

    mesh_config = config["model"]["mesh"]
    mesh = mesh_utils.create_mesh(
        tuple(mesh_config["shape"]), tuple(mesh_config["axis_names"])
    )
    print(f"Loading base model from {config['model']['model_path']} ...", flush=True)
    model, local_model_path = create_model(config, mesh)

    checkpoint_dir = args.checkpoint_dir or config["training"]["checkpoint_dir"]
    print(f"Restoring step {args.step} from {checkpoint_dir} ...", flush=True)
    restored = restore_checkpoint(model, checkpoint_dir, args.step)
    print(f"Restored full model parameters from step {restored}.", flush=True)

    tokenizer_config = dict(config["tokenizer"])
    tokenizer_config["tokenizer_path"] = local_model_path
    tokenizer_config["chat_template"] = None
    tokenizer = model_utils.create_tokenizer(tokenizer_config, local_model_path)

    export_config = dict(config)
    export_config["export"] = {
        **config.get("export", {}),
        "output_dir": args.output,
        "overwrite": True,
        "enabled": True,
    }
    export_model(
        config=export_config,
        model=model,
        tokenizer=tokenizer,
        local_model_path=local_model_path,
    )
    print(f"Exported step {restored} to {args.output}")


if __name__ == "__main__":
    main()
