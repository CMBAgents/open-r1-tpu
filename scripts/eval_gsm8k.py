#!/usr/bin/env python3
"""Score a merged export on GSM8K-style questions whose gold is one number.

Greedy generation, one fresh turn per question with the recipe's system prompt,
through the same Tunix sampler and float32 load as eval_heldout_split.py. Each
completion is cut at the turn-end marker and scored two ways:

- correct: its final number (a ``\\boxed{}`` answer if any, else the last number
  written) equals gold. This is the reading the GSM8K GRPO reward uses.
- answer_line_correct: it ends in an ``Answer: N`` line and N equals gold, the
  shape the GSM8K SFT recipes teach.

Greedy decoding is deterministic, so there is no seed spread to report; the
summary gives a 95% Wilson interval for the sampling error of the question set.

The test file is JSONL with ``id``, ``prompt`` (the bare question) and
``solution`` (the gold number), the columns the GSM8K GRPO recipes read.

Usage (from the repo root, with the pinned environment active):
    python3 scripts/eval_gsm8k.py \
        --recipe recipes/rowanai/sft/config_rowanai_gsm8k_sft.yaml \
        --model-path artifacts/rowanai-gsm8k-sft/v1/merged \
        --test-file data/gsm8k-pre1905/test.jsonl \
        --output artifacts/rowanai-gsm8k-sft/v1/eval/test.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
for path in (REPO_ROOT / "src", SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_heldout_split import (  # noqa: E402
    cut_at_turn_end,
    load_runtime,
    tunix_mesh_context,
    turn_end_ids,
)
from open_r1_tpu.grpo.rewards import (  # noqa: E402
    answer_correctness_reward,
    extract_final_number,
)

ANSWER_LINE = re.compile(r"^\s*Answer:\s*(\S+?)\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="SFT recipe of the export")
    parser.add_argument("--model-path", help="merged export dir")
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--rescore",
        action="store_true",
        help="re-score the completions already in --output without generating",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        help="compute and load dtype for generation; see eval_heldout_split.py",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="first N rows only")
    args = parser.parse_args()
    if not args.rescore and not args.model_path:
        parser.error("--model-path is required unless --rescore")
    return args


def system_prompt_of(recipe: str) -> str | None:
    """The system prompt the recipe trained with, so eval prompts match it."""
    from open_r1_tpu.core.config import load_config, read_prompt_file

    path = load_config(recipe, [])["dataset"].get("system_prompt_file")
    return read_prompt_file(path) if path is not None else None


def messages_for(question: str, system_prompt: str | None) -> list[dict[str, str]]:
    messages = [{"role": "user", "content": question}]
    if system_prompt:
        messages.insert(0, {"role": "system", "content": system_prompt})
    return messages


def answer_line(text: str) -> str | None:
    """The N of a final ``Answer: N`` line, or None when the last line is not one."""
    lines = text.rstrip().splitlines()
    match = ANSWER_LINE.match(lines[-1]) if lines else None
    return match.group(1) if match else None


def score(completion: str, gold: str) -> dict[str, Any]:
    """Both correctness readings of one completion, already cut at turn end."""
    line = answer_line(completion)
    return {
        "predicted": extract_final_number(completion),
        "answer_line": line,
        "correct": answer_correctness_reward([""], [completion], [gold])[0] == 1.0,
        "answer_line_correct": line is not None
        and answer_correctness_reward([""], [line], [gold])[0] == 1.0,
    }


def wilson_interval(hits: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion; stays inside [0, 1] at 0 or n."""
    if total == 0:
        return (0.0, 1.0)
    p = hits / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    half /= denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def generate(
    args: argparse.Namespace, rows: list[dict[str, Any]], system_prompt: str | None
) -> tuple[list[str], list[int]]:
    """Greedy completions and their generated-token counts, in row order."""
    print(f"Loading {args.model_path} in {args.dtype} ...", flush=True)
    mesh, tokenizer, sampler = load_runtime(args)
    # Stop on the document EOS as well as a single-token turn end: a model
    # whose turn-end marker is ordinary text is trained to emit EOS after it.
    eos_tokens = sorted(set((turn_end_ids(tokenizer) or []) + [tokenizer.eos_id()]))
    print(f"Stop token ids: {eos_tokens}", flush=True)

    prompts = [
        tokenizer.apply_chat_template(
            messages_for(row["prompt"], system_prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        for row in rows
    ]
    longest = max(len(tokenizer.encode(prompt)) for prompt in prompts)
    if longest > args.max_prompt_length:
        raise SystemExit(
            f"longest prompt is {longest} tokens, over --max-prompt-length "
            f"{args.max_prompt_length}; raise it rather than let the sampler "
            "grow the prompt past the KV cache"
        )
    print(f"First prompt:\n{prompts[0]}\n(longest prompt {longest} tokens)", flush=True)

    started = time.monotonic()
    completions: list[str] = []
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
    return completions, generated


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.test_file).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit:
        rows = rows[: args.limit]
    print(f"{len(rows)} rows from {args.test_file}", flush=True)
    started = time.monotonic()
    system_prompt = system_prompt_of(args.recipe)

    out_path = Path(args.output)
    if args.rescore:
        previous = [
            json.loads(line)
            for line in out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if [record["id"] for record in previous] != [row["id"] for row in rows]:
            raise SystemExit(f"{args.output} does not line up with {args.test_file}")
        # Already cut at the turn-end marker when written, so the marker is
        # gone from the text and whether the turn closed is read back instead.
        texts = [record["completion"] for record in previous]
        closed_flags = [bool(record["closed_turn"]) for record in previous]
        generated = [record["generated_tokens"] for record in previous]
        model_path = previous[0].get("model_path") if previous else None
    else:
        completions, generated = generate(args, rows, system_prompt)
        cut = [cut_at_turn_end(completion) for completion in completions]
        texts = [text for text, _ in cut]
        closed_flags = [closed for _, closed in cut]
        model_path = args.model_path

    records = []
    for row, text, closed, count in zip(
        rows, texts, closed_flags, generated, strict=True
    ):
        records.append(
            {
                "id": row["id"],
                "model_path": model_path,
                "question": row["prompt"],
                "gold": str(row["solution"]),
                "completion": text,
                "generated_tokens": count,
                "hit_token_cap": count >= args.max_new_tokens,
                "closed_turn": closed,
                **score(text, str(row["solution"])),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(records)
    correct = sum(record["correct"] for record in records)
    line_correct = sum(record["answer_line_correct"] for record in records)
    low, high = wilson_interval(correct, total)
    tokens = sorted(record["generated_tokens"] for record in records)
    summary = {
        "model_path": model_path,
        "test_file": args.test_file,
        "recipe": args.recipe,
        "system_prompt": system_prompt,
        "decoding": "greedy",
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "rows": total,
        "correct": correct,
        "accuracy": correct / total if total else None,
        "accuracy_wilson_95": [low, high],
        "answer_line_correct": line_correct,
        "answer_line_accuracy": line_correct / total if total else None,
        "has_answer_line": sum(r["answer_line"] is not None for r in records),
        "closed_turn": sum(r["closed_turn"] for r in records),
        "hit_token_cap": sum(r["hit_token_cap"] for r in records),
        "median_generated_tokens": tokens[total // 2] if total else None,
        "seconds": round(time.monotonic() - started),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {out_path} and {summary_path}")
    print(
        f"correct (final number) {correct}/{total} = {100 * correct / total:.1f}% "
        f"[95% CI {100 * low:.1f}-{100 * high:.1f}]"
    )
    print(f"correct (Answer: line) {line_correct}/{total}")
    print(f"has Answer: line {summary['has_answer_line']}/{total}")
    print(f"closed turn {summary['closed_turn']}/{total}")
    print(f"hit {args.max_new_tokens}-token cap {summary['hit_token_cap']}/{total}")
    print(f"median generated tokens {summary['median_generated_tokens']}")
    degenerate = sum(1 for r in records if len(set(r["completion"])) == 1)
    if degenerate:
        print(
            f"WARNING: {degenerate}/{total} completions are a single repeated "
            "character; the generation is numerically broken"
        )


if __name__ == "__main__":
    main()
