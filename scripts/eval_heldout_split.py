#!/usr/bin/env python3
"""Generate answers for a held-out split and score the answer-key rows.

Runs the direct Tunix sampler against a merged export, one fresh single turn
per held-out question -- which is the distribution these models were trained
on, so no history is carried between rows.

Scoring covers only the rows whose gold is an answer-key entry rather than a
worked solution; the rest are generated and recorded, with format statistics,
but not marked right or wrong. `open_r1_tpu.sft.heldout_match` decides which is
which and does the comparison.

Usage (from the repo root, with the pinned environment active):
    python3 scripts/eval_heldout_split.py \
        --recipe recipes/rowanai/sft/config_rowanai_split.yaml \
        --model-path artifacts/rowanai-split/rowanai/merged \
        --test-file data/rowanai/test_split.jsonl \
        --output artifacts/rowanai-split/rowanai/heldout.jsonl
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import sys
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

TURN_END = "<|im_end|>"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", help="the arm's SFT recipe")
    parser.add_argument("--model-path", help="merged export dir")
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help=(
            "re-score the completions already in --output instead of "
            "generating, so a scoring change can be applied to a finished run "
            "without taking the chip again"
        ),
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help=(
            "compute and load dtype for generation. float32 by default because "
            "Qwen2.5-Math carries attention biases up to 21.5 (and weights to "
            "402), which overflow the plain attention path in bfloat16: the "
            "logits go non-finite and argmax returns token 0, so the "
            "completion decodes as a run of '!'"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="first N rows only")
    args = parser.parse_args()
    if not args.rescore and not (args.recipe and args.model_path):
        parser.error("--recipe and --model-path are required unless --rescore")
    return args


@contextmanager
def tunix_mesh_context(mesh: Any):
    """Activate the legacy physical mesh the pinned Tunix sampler still reads."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"`with mesh:` context manager has been deprecated.*",
            category=DeprecationWarning,
        )
        with mesh:
            yield


def load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any]:
    """Build the mesh, model, tokenizer and sampler for one merged export."""
    from tunix.cli.utils import model as model_utils
    from tunix.generate import sampler as sampler_lib
    from tunix.utils import mesh as mesh_utils

    from open_r1_tpu.core.config import load_config
    from open_r1_tpu.model.loading import create_model

    config = copy.deepcopy(load_config(args.recipe, []))
    model_config = config["model"]
    model_config["model_source"] = "local"
    model_config["model_path"] = args.model_path
    # Rematerialisation only pays off in training, and flash attention is not
    # needed at these lengths. Generation runs in float32: see --dtype.
    model_config["remat_config"] = "NONE"
    model_config["use_flash_attention"] = False
    model_config["dtype"] = args.dtype
    model_config["load_dtype"] = args.dtype
    model_config.pop("lora_config", None)

    mesh_config = model_config["mesh"]
    mesh = mesh_utils.create_mesh(
        tuple(mesh_config["shape"]), tuple(mesh_config["axis_names"])
    )
    model, tokenizer_path = create_model(config, mesh)

    # Read the template and vocabulary back from the export, so what is scored
    # is exactly what the export would serve.
    tokenizer_config = dict(config["tokenizer"])
    tokenizer_config["tokenizer_path"] = args.model_path
    tokenizer_config["chat_template"] = None
    tokenizer = model_utils.create_tokenizer(tokenizer_config, tokenizer_path)

    runtime_config = getattr(model, "config", None)
    if runtime_config is None:
        raise SystemExit("model exposes no config; cannot size the KV cache")
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=args.max_prompt_length + args.max_new_tokens,
            num_layers=int(runtime_config.num_layers),
            num_kv_heads=int(runtime_config.num_kv_heads),
            head_dim=int(runtime_config.head_dim),
        ),
    )
    return mesh, tokenizer, sampler


def turn_end_ids(tokenizer: Any) -> list[int] | None:
    """The single token that ends an assistant turn, when there is one.

    Qwen has a real `<|im_end|>` token. This corpus's other base renders the
    same marker as ordinary text, so no token id can stop on it and the
    completion has to be cut by string instead.
    """
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        convert = getattr(getattr(tokenizer, "tokenizer", None), "ids", None)
    if convert is None:
        return None
    try:
        resolved = convert(TURN_END)
    except (KeyError, TypeError, ValueError):
        return None
    return [resolved] if isinstance(resolved, int) else None


def render(tokenizer: Any, question: str) -> str:
    """One fresh single turn, matching the training distribution exactly."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


def cut_at_turn_end(text: str) -> tuple[str, bool]:
    """Trim a completion at the turn-end marker; report whether it stopped."""
    index = text.find(TURN_END)
    if index == -1:
        return text.strip(), False
    return text[:index].strip(), True


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.test_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} held-out rows from {args.test_file}", flush=True)

    started = time.monotonic()
    if args.rescore:
        previous = [
            json.loads(line)
            for line in Path(args.output).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [record["id"] for record in previous] != [row["id"] for row in rows]:
            raise SystemExit(f"{args.output} does not line up with {args.test_file}")
        # Already cut at the turn-end marker when it was written.
        completions = [record["completion"] for record in previous]
        flags = [bool(record["stopped_at_turn_end"]) for record in previous]
        print(f"re-scoring {len(completions)} saved completions", flush=True)
        return report(args, rows, completions, started, stopped_flags=flags)

    print(f"Loading {args.model_path} in {args.dtype} ...", flush=True)
    mesh, tokenizer, sampler = load_runtime(args)
    eos_tokens = turn_end_ids(tokenizer)
    print(f"Turn-end token ids: {eos_tokens or 'none (cutting by string)'}", flush=True)

    prompts = [render(tokenizer, row["messages"][0]["content"]) for row in rows]
    completions: list[str] = []
    # When the turn ends on a token id the sampler consumes it, so the marker
    # never reaches the text and stopping has to be read off the token count.
    generated: list[int] = []
    with tunix_mesh_context(mesh):
        for start in range(0, len(prompts), args.batch_size):
            batch = prompts[start : start + args.batch_size]
            wanted = len(batch)
            # Keep one compiled shape: pad the final batch and drop the extras.
            while len(batch) < args.batch_size:
                batch.append(batch[-1])
            output = sampler(
                input_strings=batch,
                max_generation_steps=args.max_new_tokens,
                max_prompt_length=args.max_prompt_length,
                echo=False,
                # top_p unset is what puts the pinned sampler in greedy mode.
                temperature=0.0,
                eos_tokens=eos_tokens,
                pad_output=False,
            )
            completions.extend(output.text[:wanted])
            generated.extend(len(list(ids)) for ids in output.tokens[:wanted])
            done = min(start + args.batch_size, len(prompts))
            print(
                f"  {done}/{len(prompts)} in {time.monotonic() - started:.0f}s",
                flush=True,
            )

    flags = None
    if eos_tokens is not None:
        flags = [count < args.max_new_tokens for count in generated]
    return report(args, rows, completions, started, stopped_flags=flags)


def report(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    completions: list[str],
    started: float,
    *,
    stopped_flags: list[bool] | None = None,
) -> None:
    """Score the completions, write the records, and print the summary.

    ``stopped_flags`` overrides the verdict read from the text: a model that
    stops on a token id leaves no marker to find, and a re-scored completion
    was cut when it was written.
    """
    from open_r1_tpu.sft.heldout_match import (
        is_gradable,
        matches_lenient,
        matches_strict,
    )

    records = []
    for index, (row, completion) in enumerate(zip(rows, completions, strict=True)):
        text, stopped = cut_at_turn_end(completion)
        if stopped_flags is not None:
            stopped = stopped_flags[index]
        gold = row["messages"][1]["content"]
        gradable = is_gradable(row)
        records.append(
            {
                "id": row["id"],
                "domain": row["metadata"]["domain"],
                "difficulty": row["metadata"]["difficulty"],
                "extraction_type": row["source"]["extraction_type"],
                "source_file": row["source"]["file"],
                "question": row["messages"][0]["content"],
                "gold": gold,
                "completion": text,
                "stopped_at_turn_end": stopped,
                "completion_chars": len(text),
                "gradable": gradable,
                "correct_strict": matches_strict(text, gold) if gradable else None,
                "correct_lenient": matches_lenient(text, gold) if gradable else None,
            }
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    graded = [r for r in records if r["gradable"]]
    strict = sum(1 for r in graded if r["correct_strict"])
    lenient = sum(1 for r in graded if r["correct_lenient"])
    stopped = sum(1 for r in records if r["stopped_at_turn_end"])
    chars = sorted(r["completion_chars"] for r in records)
    by_domain: dict[str, list[int]] = collections.defaultdict(list)
    for record in graded:
        by_domain[record["domain"]].append(int(bool(record["correct_strict"])))

    print(f"\nwrote {out_path}")
    print(f"scored {len(records)} rows in {time.monotonic() - started:.0f}s")
    print(f"gradable: {len(graded)}/{len(records)}")
    if graded:
        print(f"  strict  {strict}/{len(graded)} = {100 * strict / len(graded):.1f}%")
        print(f"  lenient {lenient}/{len(graded)} = {100 * lenient / len(graded):.1f}%")
        for domain, hits in sorted(by_domain.items()):
            print(f"    {domain}: {sum(hits)}/{len(hits)}")
    print(f"stopped at turn end: {stopped}/{len(records)}")
    print(f"completion chars: median {chars[len(chars) // 2]}, max {chars[-1]}")
    # A completion of one repeated character means the logits went non-finite
    # and argmax fell back to token 0. Silent when it happens, so say it loudly.
    degenerate = sum(1 for r in records if len(set(r["completion"])) == 1)
    if degenerate:
        print(
            f"WARNING: {degenerate}/{len(records)} completions are a single "
            "repeated character; the generation is numerically broken"
        )


if __name__ == "__main__":
    main()
