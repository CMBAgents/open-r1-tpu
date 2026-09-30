#!/usr/bin/env python3
"""Set a staged model's RoPE theta and declared context in its config.json.

Makes a config-only long-context variant of a base model, as open-r1 did for
open-r1/Qwen2.5-Math-7B-RoPE-300k: the weights are untouched, and training
then adapts them to the new theta. The export copies this config.json, so
the served model declares the theta it was trained at.

    python scripts/set_rope.py models/Qwen2.5-Math-1.5B-RoPE-300k \\
        --rope-theta 300000 --max-position-embeddings 32768
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def set_rope(
    model_dir: str | Path, *, rope_theta: float, max_position_embeddings: int
) -> dict:
    path = Path(model_dir) / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["rope_theta"] = rope_theta
    config["max_position_embeddings"] = max_position_embeddings
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", help="Directory holding config.json")
    parser.add_argument("--rope-theta", type=float, required=True)
    parser.add_argument("--max-position-embeddings", type=int, required=True)
    args = parser.parse_args()
    config = set_rope(
        args.model_dir,
        rope_theta=args.rope_theta,
        max_position_embeddings=args.max_position_embeddings,
    )
    print(
        f"{args.model_dir}: rope_theta {config['rope_theta']}, "
        f"max_position_embeddings {config['max_position_embeddings']}"
    )


if __name__ == "__main__":
    main()
