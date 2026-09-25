#!/usr/bin/env python3
"""Score a base model on a plain-text probe suite: multiple choice and text.

Each probe row is a plain-text context and one or more continuations (each
starting with a space). No chat template or system prompt is applied: this
measures what pretraining alone gives the base model. The context may hold
worked examples (a few-shot suite) or none (a zero-shot, next-token suite).

Multiple-choice rows (``kind`` "choice", the default) have several options and
the index of the correct one. Three readings:

- By likelihood: the summed log-probability of each option's tokens after the
  context. ``acc`` picks the highest sum; ``acc_norm`` divides by the option's
  length in bytes first, so a longer option is not penalised for having more
  tokens (lm-evaluation-harness's two readings).
- By rank: where the correct option's first token ranks among every token the
  model could produce next (``top1``, ``top5``). This needs no wrong options.
- Greedy continuation, for rows marked ``generate``: the first line of up to
  --max-new-tokens greedy tokens must start with the gold answer, or with any
  wording in the row's ``accept`` list (a number must match exactly, so 150
  does not count for 15).

Text rows (``kind`` "text") have one option, the text to score. They report
its bits per byte, which compares models with different tokenisers.

Option tokens are found by tokenising the context and context+option and
taking the common prefix, so the scored span is exactly what the model would
see. If the tokeniser merges across the boundary, the merged token is scored
with the option, and the summary counts such options as
``boundary_merged_options``.

The probe file is JSONL with ``id``, ``task``, ``family``, ``context``,
``options``, ``answer``, ``gold`` and ``generate``, and optionally ``kind``
and ``accept``. The recipe only supplies the architecture, RoPE and tokeniser
settings for --model-path.

Usage (from the repo root, with the pinned environment active):
    python3 scripts/eval_probes.py \
        --recipe recipes/rowanai/sft/config_qwen_base_gsm8k_sft.yaml \
        --model-path models/Qwen2.5-1.5B \
        --probes data/fewshot-probes/probes.jsonl \
        --output artifacts/fewshot-probes/qwen2.5-1.5b.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
for path in (REPO_ROOT / "src", SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from eval_gsm8k import wilson_interval  # noqa: E402
from eval_heldout_split import load_runtime, tunix_mesh_context  # noqa: E402

NUMBER = re.compile(r"-?\d[\d,]*")
# Padded widths, so the scorer compiles a handful of shapes rather than one per
# batch, and short batches are not padded to the longest text in the suite.
WIDTHS = (64, 128, 256, 512, 1024, 2048)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True, help="recipe of the base model")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--probes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gen-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=12)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    return parser.parse_args()


def is_text(row: dict[str, Any]) -> bool:
    return row.get("kind") == "text"


def option_span(
    encode: Callable[[str], list[int]], context: str, option: str
) -> tuple[list[int], int, bool]:
    """Token ids of context+option, where the option's tokens start, and
    whether the tokeniser merged a token across the boundary."""
    context_ids = list(encode(context))
    full_ids = list(encode(context + option))
    start = 0
    for a, b in zip(context_ids, full_ids, strict=False):
        if a != b:
            break
        start += 1
    if start < 1 or start >= len(full_ids):
        raise ValueError(f"no scorable option tokens for {option!r}")
    return full_ids, start, start < len(context_ids)


def padded_width(length: int) -> int:
    for width in WIDTHS:
        if length <= width:
            return width
    raise ValueError(f"a {length}-token sequence is longer than {WIDTHS[-1]}")


def first_line(text: str) -> str:
    return text.split("\n", 1)[0]


def normalise(text: str) -> str:
    return " ".join(text.lower().split()).strip(" .,;:!?")


def generation_correct(text: str, gold: str) -> bool:
    """Whether the first line of a continuation starts with the gold answer."""
    line = first_line(text)
    if NUMBER.fullmatch(gold):
        match = NUMBER.search(line)
        return (
            match is not None
            and line[: match.start()].strip() == ""
            and match.group().replace(",", "") == gold
        )
    said, wanted = normalise(line), normalise(gold)
    if not said.startswith(wanted):
        return False
    rest = said[len(wanted) :]
    return rest == "" or not rest[0].isalnum()


def pick(scores: Sequence[float]) -> int:
    return max(range(len(scores)), key=lambda i: scores[i])


def score_row(
    row: dict[str, Any],
    logprobs: list[float],
    generation: str | None,
    first_ranks: list[int] | None = None,
) -> dict[str, Any]:
    """One output record from the option log-probabilities and continuation.

    ``first_ranks[i]`` is how many tokens the model rated above option i's
    first token (0 means it was the model's top choice).
    """
    options = row["options"]
    lengths = [len(option.encode("utf-8")) for option in options]
    record: dict[str, Any] = {
        "id": row["id"],
        "task": row["task"],
        "family": row["family"],
        "kind": row.get("kind", "choice"),
    }
    if is_text(row):
        (logprob,) = logprobs
        return {
            **record,
            "logprob": logprob,
            "bytes": lengths[0],
            "bits_per_byte": -logprob / math.log(2) / lengths[0],
        }
    normed = [lp / n for lp, n in zip(logprobs, lengths, strict=True)]
    pred, pred_norm = pick(logprobs), pick(normed)
    accept = row.get("accept") or [row["gold"]]
    record |= {
        "options": options,
        "answer": row["answer"],
        "gold": row["gold"],
        "logprobs": [round(lp, 4) for lp in logprobs],
        "pred": pred,
        "pred_norm": pred_norm,
        "correct": pred == row["answer"],
        "correct_norm": pred_norm == row["answer"],
        "gold_rank": None if first_ranks is None else first_ranks[row["answer"]],
        "generation": generation,
        "gen_correct": None
        if generation is None
        else any(generation_correct(generation, wording) for wording in accept),
    }
    if "middle" in row:
        record["picked_middle"] = options[pred] == row["middle"]
    return record


def accuracy(hits: int, total: int) -> dict[str, Any]:
    low, high = wilson_interval(hits, total)
    return {
        "correct": hits,
        "n": total,
        "accuracy": hits / total if total else None,
        "wilson_95": [low, high],
    }


def summarise(records: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict:
    """Accuracy per task and family beside the chance rate, and bits per byte
    for text rows. The overall block covers multiple-choice rows only."""
    n_options = {row["id"]: len(row["options"]) for row in rows}

    def block(group: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"n": len(group)}
        choice = [r for r in group if r["kind"] != "text"]
        text = [r for r in group if r["kind"] == "text"]
        if choice:
            generated = [r for r in choice if r["gen_correct"] is not None]
            ranked = [r for r in choice if r["gold_rank"] is not None]
            out |= {
                "chance": sum(1 / n_options[r["id"]] for r in choice) / len(choice),
                "acc": accuracy(sum(r["correct"] for r in choice), len(choice)),
                "acc_norm": accuracy(
                    sum(r["correct_norm"] for r in choice), len(choice)
                ),
                "gen": accuracy(
                    sum(r["gen_correct"] for r in generated), len(generated)
                )
                if generated
                else None,
                "top1": accuracy(sum(r["gold_rank"] == 0 for r in ranked), len(ranked))
                if ranked
                else None,
                "top5": accuracy(sum(r["gold_rank"] < 5 for r in ranked), len(ranked))
                if ranked
                else None,
            }
        if text:
            total_bytes = sum(r["bytes"] for r in text)
            total_bits = sum(-r["logprob"] / math.log(2) for r in text)
            out |= {"bytes": total_bytes, "bits_per_byte": total_bits / total_bytes}
        return out

    tasks: dict[str, Any] = {}
    for task in dict.fromkeys(r["task"] for r in records):
        group = [r for r in records if r["task"] == task]
        tasks[task] = block(group)
        tasks[task]["families"] = {
            family: block([r for r in group if r["family"] == family])
            for family in dict.fromkeys(r["family"] for r in group)
        }
    choice = [r for r in records if r["kind"] != "text"]
    return {"overall": block(choice) if choice else None, "tasks": tasks}


def option_scores(
    args: argparse.Namespace, model: Any, tokenizer: Any, mesh: Any, rows: list
) -> tuple[list[list[float]], list[list[int]], int]:
    """Summed log-probability and first-token rank of every option, by row."""
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx

    spans = []
    merged = 0
    for row_index, row in enumerate(rows):
        for option in row["options"]:
            ids, start, was_merged = option_span(
                tokenizer.encode, row["context"], option
            )
            merged += was_merged
            spans.append((row_index, ids, start))
    longest = max(len(ids) for _, ids, _ in spans)
    print(f"{len(spans)} option sequences, longest {longest} tokens", flush=True)

    @nnx.jit
    def token_scores(model, tokens, lengths):
        batch, width = tokens.shape
        positions = jnp.broadcast_to(jnp.arange(width), (batch, width))
        valid = jnp.arange(width)[None, :] < lengths[:, None]
        causal = jnp.tril(jnp.ones((width, width), dtype=jnp.bool_))
        mask = valid[:, None, :] & causal[None, :, :]
        logits, _ = model(tokens, positions=positions, cache=None, attention_mask=mask)
        logits = logits[:, :-1].astype(jnp.float32)
        picked = jnp.take_along_axis(logits, tokens[:, 1:, None], axis=-1)[..., 0]
        ranks = jnp.sum(logits > picked[..., None], axis=-1)
        return picked - jax.nn.logsumexp(logits, axis=-1), ranks

    # Batch neighbours of similar length, so each batch pads to a small width.
    order = sorted(range(len(spans)), key=lambda i: len(spans[i][1]))
    sums = [0.0] * len(spans)
    ranks = [0] * len(spans)
    started = time.monotonic()
    with tunix_mesh_context(mesh):
        for first in range(0, len(order), args.batch_size):
            chosen = order[first : first + args.batch_size]
            wanted = len(chosen)
            # Keep the batch size fixed: pad the final batch and drop the extras.
            chosen = chosen + [chosen[-1]] * (args.batch_size - wanted)
            width = padded_width(max(len(spans[i][1]) for i in chosen))
            tokens = np.zeros((args.batch_size, width), dtype=np.int32)
            lengths = np.zeros((args.batch_size,), dtype=np.int32)
            for slot, i in enumerate(chosen):
                ids = spans[i][1]
                tokens[slot, : len(ids)] = ids
                lengths[slot] = len(ids)
            logps, token_ranks = token_scores(
                model, jnp.asarray(tokens), jnp.asarray(lengths)
            )
            logps, token_ranks = np.asarray(logps), np.asarray(token_ranks)
            for slot, i in enumerate(chosen[:wanted]):
                _, ids, start = spans[i]
                # logps[t] is log p(token t+1 | tokens 0..t).
                sums[i] = float(logps[slot, start - 1 : len(ids) - 1].sum())
                ranks[i] = int(token_ranks[slot, start - 1])
    print(f"scored in {time.monotonic() - started:.0f}s", flush=True)

    per_row: list[list[float]] = [[] for _ in rows]
    per_row_ranks: list[list[int]] = [[] for _ in rows]
    for (row_index, _, _), total, rank in zip(spans, sums, ranks, strict=True):
        per_row[row_index].append(total)
        per_row_ranks[row_index].append(rank)
    return per_row, per_row_ranks, merged


def generations(
    args: argparse.Namespace, sampler: Any, tokenizer: Any, mesh: Any, rows: list
) -> list[str | None]:
    """Greedy continuations for the rows marked ``generate``, else None."""
    wanted_rows = [i for i, row in enumerate(rows) if row["generate"]]
    out: list[str | None] = [None] * len(rows)
    if not wanted_rows:
        return out
    prompts = [rows[i]["context"] for i in wanted_rows]
    longest = max(len(tokenizer.encode(prompt)) for prompt in prompts)
    if longest > args.max_prompt_length:
        raise SystemExit(
            f"longest prompt is {longest} tokens, over --max-prompt-length "
            f"{args.max_prompt_length}"
        )
    texts: list[str] = []
    with tunix_mesh_context(mesh):
        for first in range(0, len(prompts), args.gen_batch_size):
            batch = prompts[first : first + args.gen_batch_size]
            wanted = len(batch)
            batch = batch + [batch[-1]] * (args.gen_batch_size - wanted)
            output = sampler(
                input_strings=batch,
                max_generation_steps=args.max_new_tokens,
                max_prompt_length=args.max_prompt_length,
                echo=False,
                # top_p unset is what puts the pinned sampler in greedy mode.
                temperature=0.0,
                eos_tokens=[tokenizer.eos_id()],
                pad_output=False,
            )
            texts.extend(output.text[:wanted])
    for i, text in zip(wanted_rows, texts, strict=True):
        out[i] = text
    return out


def print_summary(summary: dict[str, Any]) -> None:
    print(
        f"{'task':<16}{'n':>4}{'chance':>8}{'acc':>7}{'norm':>7}{'top1':>7}{'gen':>7}"
    )
    rows = [(t, b) for t, b in summary["tasks"].items() if "acc" in b]
    if summary["overall"]:
        rows.append(("overall", summary["overall"]))
    for task, block in rows:
        cells = [
            block[key]["accuracy"] if block.get(key) else None
            for key in ("acc", "acc_norm", "top1", "gen")
        ]
        print(
            f"{task:<16}{block['n']:>4}{block['chance']:>8.2f}"
            + "".join(f"{'' if c is None else f'{c:.2f}':>7}" for c in cells)
        )
    for task, block in summary["tasks"].items():
        if "bits_per_byte" not in block:
            continue
        for family, part in block["families"].items():
            print(
                f"{task}/{family}: {part['bits_per_byte']:.3f} bits per byte "
                f"over {part['bytes']} bytes"
            )


def main() -> None:
    args = parse_args()
    rows = [
        json.loads(line)
        for line in Path(args.probes).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"{len(rows)} probes from {args.probes}", flush=True)
    started = time.monotonic()
    print(f"Loading {args.model_path} in {args.dtype} ...", flush=True)
    mesh, tokenizer, sampler = load_runtime(args)
    first = rows[0]["context"] + rows[0]["options"][rows[0]["answer"]]
    print(f"First probe:\n{first}", flush=True)
    print(f"Its tokens: {tokenizer.encode(first)}", flush=True)

    logprobs, ranks, merged = option_scores(
        args, sampler.transformer, tokenizer, mesh, rows
    )
    texts = generations(args, sampler, tokenizer, mesh, rows)
    records = [
        {"model_path": args.model_path, **score_row(row, lps, text, rks)}
        for row, lps, text, rks in zip(rows, logprobs, texts, ranks, strict=True)
    ]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.writelines(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        )
    summary = {
        "model_path": args.model_path,
        "recipe": args.recipe,
        "probes": args.probes,
        "dtype": args.dtype,
        "max_new_tokens": args.max_new_tokens,
        "boundary_merged_options": merged,
        **summarise(records, rows),
        "seconds": round(time.monotonic() - started),
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"\nwrote {out_path} and {summary_path}")
    print(f"options merged across the context boundary: {merged}")
    print_summary(summary)


if __name__ == "__main__":
    main()
