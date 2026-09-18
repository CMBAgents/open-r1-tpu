#!/usr/bin/env python3
"""Draw a deterministic stratified train/test split of the worked-solution corpus.

The corpus mixes two very different answer shapes: roughly half the rows have a
bare answer of a few characters ("151.") and the rest are multi-sentence worked
solutions. A uniform random split leaves that ratio, and the rarer domains, to
chance, so the split is stratified on (domain, short/long answer) instead.

Rows whose user turn is duplicated are assigned as one unit, so a question can
never appear on both sides.

Usage:
    python3 scripts/make_rowanai_split.py \
        --input data/rowanai/train.jsonl \
        --train-out data/rowanai/train_split.jsonl \
        --test-out data/rowanai/test_split.jsonl \
        --test-fraction 0.1 --seed 42
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
from pathlib import Path
from typing import Any

# A gold answer shorter than this is a bare answer rather than a worked solution.
SHORT_ANSWER_CHARS = 100


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
    return rows


def assistant_text(row: dict[str, Any]) -> str:
    for message in row["messages"]:
        if message["role"] == "assistant":
            return message["content"]
    raise ValueError(f"row {row.get('id')!r} has no assistant turn")


def user_text(row: dict[str, Any]) -> str:
    for message in row["messages"]:
        if message["role"] == "user":
            return message["content"]
    raise ValueError(f"row {row.get('id')!r} has no user turn")


def stratum_key(row: dict[str, Any]) -> tuple[str, str]:
    domain = str(row.get("metadata", {}).get("domain", "unknown"))
    length = "short" if len(assistant_text(row)) < SHORT_ANSWER_CHARS else "long"
    return (domain, length)


def group_rows(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group rows that share a user question so they stay on one side."""
    by_question: dict[str, list[dict[str, Any]]] = collections.OrderedDict()
    for row in sorted(rows, key=lambda r: str(r["id"])):
        by_question.setdefault(user_text(row), []).append(row)
    return list(by_question.values())


def split_groups(
    groups: list[list[dict[str, Any]]], test_fraction: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Largest-remainder stratified assignment of whole groups to the test side."""
    total_rows = sum(len(group) for group in groups)
    target_test = math.ceil(total_rows * test_fraction)

    by_stratum: dict[tuple[str, str], list[list[dict[str, Any]]]] = (
        collections.OrderedDict()
    )
    for group in groups:
        by_stratum.setdefault(stratum_key(group[0]), []).append(group)

    # Exact per-stratum quota, then hand the leftover rows to the strata with the
    # largest fractional part, so the totals add up to target_test exactly.
    quotas: dict[tuple[str, str], int] = {}
    remainders: list[tuple[float, tuple[str, str]]] = []
    for key in sorted(by_stratum):
        stratum_rows = sum(len(group) for group in by_stratum[key])
        exact = stratum_rows * test_fraction
        quotas[key] = math.floor(exact)
        remainders.append((exact - math.floor(exact), key))

    leftover = target_test - sum(quotas.values())
    for _, key in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if leftover <= 0:
            break
        stratum_rows = sum(len(group) for group in by_stratum[key])
        if quotas[key] < stratum_rows:
            quotas[key] += 1
            leftover -= 1

    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for key in sorted(by_stratum):
        stratum_groups = sorted(by_stratum[key], key=lambda g: str(g[0]["id"]))
        rng.shuffle(stratum_groups)
        taken = 0
        for group in stratum_groups:
            # A multi-row group is only taken whole, and only while it fits.
            if taken + len(group) <= quotas[key]:
                test.extend(group)
                taken += len(group)
            else:
                train.extend(group)

    train.sort(key=lambda r: str(r["id"]))
    test.sort(key=lambda r: str(r["id"]))
    return train, test


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def describe(name: str, rows: list[dict[str, Any]]) -> str:
    domains = collections.Counter(
        str(r.get("metadata", {}).get("domain", "unknown")) for r in rows
    )
    short = sum(1 for r in rows if len(assistant_text(r)) < SHORT_ANSWER_CHARS)
    lines = [
        f"{name}: {len(rows)} rows "
        f"({short} short-answer, {len(rows) - short} worked-solution)",
        "  domains: "
        + ", ".join(
            f"{d}={n}" for d, n in sorted(domains.items(), key=lambda x: -x[1])
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--train-out", type=Path, required=True)
    parser.add_argument("--test-out", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = load_rows(args.input)
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("input rows do not have unique ids")

    train, test = split_groups(group_rows(rows), args.test_fraction, args.seed)

    if len(train) + len(test) != len(rows):
        raise ValueError("split lost or duplicated rows")
    if {r["id"] for r in train} & {r["id"] for r in test}:
        raise ValueError("a row id appears on both sides")
    if {user_text(r) for r in train} & {user_text(r) for r in test}:
        raise ValueError("a question appears on both sides")

    write_rows(args.train_out, train)
    write_rows(args.test_out, test)

    print(f"input: {len(rows)} rows from {args.input}")
    print(describe("train", train))
    print(describe("test", test))
    print(f"wrote {args.train_out} and {args.test_out}")


if __name__ == "__main__":
    main()
