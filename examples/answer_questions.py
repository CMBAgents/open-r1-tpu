#!/usr/bin/env python3
"""Answer maths questions with a local model on the TPU and score the replies.

The tutorial notebooks in this directory use it to compare a model before and
after training. Each question is asked once with greedy decoding, so the score
is a quick check rather than a benchmark; docs/evaluation.md covers those. Run
from the repository root on the TPU VM:

    python examples/answer_questions.py models/Qwen2.5-0.5B \
      --model-name qwen2.5-0.5b \
      --questions data/gsm8k-think/test.parquet \
      --system-prompt-file examples/recipes/system_prompt.txt \
      --output artifacts/gsm8k-answers/base.jsonl

The questions file is Parquet or JSON Lines with ``question`` and ``answer``
columns. Each output line holds the question, the gold answer, the reply, each
reward function's score, and two verdicts: ``correct``, the reply's final
number (its ``\\boxed{}`` answer, or else the last number it wrote) equals the
gold answer; and ``formatted``, the reply is a closed ``<think>`` block
followed by a ``\\boxed{}`` answer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from open_r1_tpu.core.config import read_prompt_file
from open_r1_tpu.grpo.rewards import (
    answer_correctness_reward,
    correctness_reward,
    format_reward,
)
from open_r1_tpu.model.tokenizing import as_token_ids, assistant_turn_end_id, user_turn

REWARD_FNS = (format_reward, correctness_reward, answer_correctness_reward)
# Markers the sampler can leave on the end of a reply.
TURN_END_MARKERS = ("<|im_end|>", "<|endoftext|>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="Local model directory")
    parser.add_argument(
        "--model-name",
        required=True,
        help="Tunix model name, such as qwen2.5-0.5b",
    )
    parser.add_argument(
        "--questions", required=True, help="Parquet or JSON Lines questions file"
    )
    parser.add_argument("--output", required=True, help="JSON Lines file to write")
    parser.add_argument(
        "--system-prompt-file",
        default=None,
        help="System prompt sent with every question (default: none)",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Answer only the first N questions"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def read_questions(path: str, limit: int | None) -> list[dict[str, str]]:
    """Read ``question`` and ``answer`` columns, the answer as a string."""
    from datasets import load_dataset

    kind = "parquet" if path.endswith(".parquet") else "json"
    rows = load_dataset(kind, data_files=path, split="train")
    if limit is not None:
        rows = rows.select(range(min(limit, len(rows))))
    return [{"question": str(r["question"]), "answer": str(r["answer"])} for r in rows]


def strip_turn_end(reply: str) -> str:
    for marker in TURN_END_MARKERS:
        reply = reply.removesuffix(marker)
    return reply.strip()


def score(reply: str, gold: str) -> dict[str, Any]:
    """Score one reply with the GRPO reward functions and give the verdicts."""
    rewards = {
        fn.__name__: fn(prompts=[""], completions=[reply], answer=[gold])[0]
        for fn in REWARD_FNS
    }
    return {
        "correct": rewards["answer_correctness_reward"] == 1.0,
        "formatted": rewards["format_reward"] == 3.0,
        "rewards": rewards,
    }


def batches(items: list[str], size: int) -> list[tuple[list[str], int]]:
    """Split into batches of exactly ``size``, padding the last with copies of
    its final item so every batch compiles to the same shape. Each batch comes
    with the number of real items in it."""
    out = []
    for start in range(0, len(items), size):
        batch = items[start : start + size]
        real = len(batch)
        out.append((batch + [batch[-1]] * (size - real), real))
    return out


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens <= 0:
        raise SystemExit("--batch-size and --max-new-tokens must be positive")
    rows = read_questions(args.questions, args.limit)
    if not rows:
        raise SystemExit(f"No questions in {args.questions}")
    system_prompt = read_prompt_file(args.system_prompt_file)

    from open_r1_tpu.model.checkpoint import (
        load_sampler,
        pad_input_strings_for_fsdp,
        resolve_model_dir,
        tunix_mesh_context,
    )

    print(f"Loading {args.model_path} ...", flush=True)
    runtime = load_sampler(
        resolve_model_dir(args.model_path),
        seed=0,
        cache_size=args.max_prompt_length + args.max_new_tokens,
        model_name=args.model_name,
    )
    tokenizer = runtime.tokenizer
    prompts = []
    for row in rows:
        messages = user_turn(row["question"], system_prompt)
        length = len(
            as_token_ids(
                tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            )
        )
        if length > args.max_prompt_length:
            raise SystemExit(
                f"A prompt is {length} tokens, over --max-prompt-length "
                f"{args.max_prompt_length}: {row['question'][:80]!r}"
            )
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
    stop_ids = sorted({assistant_turn_end_id(tokenizer), int(tokenizer.eos_id())})

    replies: list[str] = []
    with tunix_mesh_context(runtime.mesh):
        for batch, real in batches(prompts, args.batch_size):
            output = runtime.sampler(
                input_strings=pad_input_strings_for_fsdp(batch, runtime.fsdp_size),
                max_generation_steps=args.max_new_tokens,
                max_prompt_length=args.max_prompt_length,
                eos_tokens=stop_ids,
                temperature=0.0,
                top_p=None,  # greedy
            )
            replies.extend(strip_turn_end(str(text)) for text in output.text[:real])
            print(f"Answered {len(replies)}/{len(prompts)}", flush=True)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    correct = formatted = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for row, reply in zip(rows, replies, strict=True):
            record = {
                "question": row["question"],
                "gold": row["answer"],
                "reply": reply,
                **score(reply, row["answer"]),
            }
            correct += record["correct"]
            formatted += record["formatted"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    total = len(rows)
    print(f"Correct:   {correct}/{total} ({correct / total:.1%})")
    print(f"Formatted: {formatted}/{total} ({formatted / total:.1%})")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
