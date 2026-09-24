#!/usr/bin/env python3
"""Score a base model on a few-shot probe suite of one-step reasoning tasks.

Each probe row is a plain-text few-shot context ending in a query stem, a list
of candidate continuations (each starting with a space) and the index of the
correct one. No chat template or system prompt is applied: this measures what
pretraining alone gives the base model. Two readings per row:

- Multiple choice, by likelihood: the summed log-probability of each option's
  tokens after the context. ``acc`` picks the highest sum; ``acc_norm``
  divides by the option's length in bytes first, so a longer option is not
  penalised for having more tokens (lm-evaluation-harness's two readings).
- Greedy continuation, for rows marked ``generate``: the first line of up to
  --max-new-tokens greedy tokens must start with the gold answer (a number
  must match exactly, so 150 does not count for 15).

Option tokens are found by tokenising the context and context+option and
taking the common prefix, so the scored span is exactly what the model would
see. If the tokeniser merges across the boundary, the merged token is scored
with the option, and the summary counts such options as
``boundary_merged_options``.

The probe file is JSONL with ``id``, ``task``, ``family``, ``context``,
``options``, ``answer``, ``gold`` and ``generate``. The recipe only supplies
the architecture, RoPE and tokeniser settings for --model-path.

Usage (from the repo root, with the pinned environment active):
    python3 scripts/eval_fewshot_probes.py \
        --recipe recipes/rowanai/sft/config_qwen_base_gsm8k_sft.yaml \
        --model-path models/Qwen2.5-1.5B \
        --probes data/fewshot-probes/probes.jsonl \
        --output artifacts/fewshot-probes/qwen2.5-1.5b.jsonl
"""

from __future__ import annotations

import argparse
import json
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
    row: dict[str, Any], logprobs: list[float], generation: str | None
) -> dict[str, Any]:
    """One output record from the option log-probabilities and continuation."""
    options = row["options"]
    lengths = [len(option.encode("utf-8")) for option in options]
    normed = [lp / n for lp, n in zip(logprobs, lengths, strict=True)]
    pred, pred_norm = pick(logprobs), pick(normed)
    record = {
        "id": row["id"],
        "task": row["task"],
        "family": row["family"],
        "options": options,
        "answer": row["answer"],
        "gold": row["gold"],
        "logprobs": [round(lp, 4) for lp in logprobs],
        "pred": pred,
        "pred_norm": pred_norm,
        "correct": pred == row["answer"],
        "correct_norm": pred_norm == row["answer"],
        "generation": generation,
        "gen_correct": None
        if generation is None
        else generation_correct(generation, row["gold"]),
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
    """Accuracy per task and per family, with the chance rate beside it."""
    n_options = {row["id"]: len(row["options"]) for row in rows}

    def block(group: list[dict[str, Any]]) -> dict[str, Any]:
        generated = [r for r in group if r["gen_correct"] is not None]
        return {
            "n": len(group),
            "chance": sum(1 / n_options[r["id"]] for r in group) / len(group),
            "acc": accuracy(sum(r["correct"] for r in group), len(group)),
            "acc_norm": accuracy(sum(r["correct_norm"] for r in group), len(group)),
            "gen": accuracy(sum(r["gen_correct"] for r in generated), len(generated))
            if generated
            else None,
        }

    tasks: dict[str, Any] = {}
    for task in dict.fromkeys(r["task"] for r in records):
        group = [r for r in records if r["task"] == task]
        tasks[task] = block(group)
        tasks[task]["families"] = {
            family: block([r for r in group if r["family"] == family])
            for family in dict.fromkeys(r["family"] for r in group)
        }
    return {"overall": block(records), "tasks": tasks}


def option_logprobs(
    args: argparse.Namespace, model: Any, tokenizer: Any, mesh: Any, rows: list
) -> tuple[list[list[float]], int]:
    """Summed log-probability of every option of every row, in row order."""
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
    length = -(-max(len(ids) for _, ids, _ in spans) // 32) * 32
    print(f"{len(spans)} option sequences, padded to {length} tokens", flush=True)

    @nnx.jit
    def token_logps(model, tokens, lengths):
        batch, width = tokens.shape
        positions = jnp.broadcast_to(jnp.arange(width), (batch, width))
        valid = jnp.arange(width)[None, :] < lengths[:, None]
        causal = jnp.tril(jnp.ones((width, width), dtype=jnp.bool_))
        mask = valid[:, None, :] & causal[None, :, :]
        logits, _ = model(tokens, positions=positions, cache=None, attention_mask=mask)
        logits = logits[:, :-1].astype(jnp.float32)
        picked = jnp.take_along_axis(logits, tokens[:, 1:, None], axis=-1)[..., 0]
        return picked - jax.nn.logsumexp(logits, axis=-1)

    sums = [0.0] * len(spans)
    started = time.monotonic()
    with tunix_mesh_context(mesh):
        for first in range(0, len(spans), args.batch_size):
            batch = spans[first : first + args.batch_size]
            wanted = len(batch)
            # Keep one compiled shape: pad the final batch and drop the extras.
            batch = batch + [batch[-1]] * (args.batch_size - wanted)
            tokens = np.zeros((args.batch_size, length), dtype=np.int32)
            lengths = np.zeros((args.batch_size,), dtype=np.int32)
            for i, (_, ids, _) in enumerate(batch):
                tokens[i, : len(ids)] = ids
                lengths[i] = len(ids)
            logps = np.asarray(
                token_logps(model, jnp.asarray(tokens), jnp.asarray(lengths))
            )
            for i, (_, ids, start) in enumerate(batch[:wanted]):
                # logps[t] is log p(token t+1 | tokens 0..t).
                sums[first + i] = float(logps[i, start - 1 : len(ids) - 1].sum())
    print(f"scored in {time.monotonic() - started:.0f}s", flush=True)

    per_row: list[list[float]] = [[] for _ in rows]
    for (row_index, _, _), total in zip(spans, sums, strict=True):
        per_row[row_index].append(total)
    return per_row, merged


def generations(
    args: argparse.Namespace, sampler: Any, tokenizer: Any, mesh: Any, rows: list
) -> list[str | None]:
    """Greedy continuations for the rows marked ``generate``, else None."""
    wanted_rows = [i for i, row in enumerate(rows) if row["generate"]]
    prompts = [rows[i]["context"] for i in wanted_rows]
    longest = max(len(tokenizer.encode(prompt)) for prompt in prompts)
    if longest > args.max_prompt_length:
        raise SystemExit(
            f"longest prompt is {longest} tokens, over --max-prompt-length "
            f"{args.max_prompt_length}"
        )
    out: list[str | None] = [None] * len(rows)
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

    logprobs, merged = option_logprobs(args, sampler.transformer, tokenizer, mesh, rows)
    texts = generations(args, sampler, tokenizer, mesh, rows)
    records = [
        {"model_path": args.model_path, **score_row(row, lps, text)}
        for row, lps, text in zip(rows, logprobs, texts, strict=True)
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
    print(f"{'task':<14}{'n':>4}{'chance':>8}{'acc':>7}{'norm':>7}{'gen':>7}")
    for task, block in [*summary["tasks"].items(), ("overall", summary["overall"])]:
        gen = block["gen"]["accuracy"] if block["gen"] else None
        print(
            f"{task:<14}{block['n']:>4}{block['chance']:>8.2f}"
            f"{block['acc']['accuracy']:>7.2f}{block['acc_norm']['accuracy']:>7.2f}"
            f"{'' if gen is None else f'{gen:.2f}':>7}"
        )


if __name__ == "__main__":
    main()
