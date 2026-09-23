"""Per-step metrics about what the GRPO policy is actually writing.

Tunix already logs reward means/extremes, completion token lengths and the
actor's loss/KL/entropy. These add the behaviour those numbers hide, computed
on the completion text of every rollout in the step (train and eval alike):

``behaviour/*`` -- shape of the answers:
  answer_line_frac      a line starting "Answer:" (the format SFT's last line)
  boxed_frac            a ``\\boxed{...}`` answer
  number_found_frac     a final number the correctness reward can read
  stopped_frac          the turn-end stop string was written (only when the
                        recipe sets ``rollout.completion_stop_strings``)
  after_stop_chars_mean characters written after that stop string, which
                        still enter the policy loss
  chars_mean            scored length in characters (text before the stop)
  repeated_line_frac    some non-empty line occurs 3+ times (loops)
  distinct_4gram_mean   unique / total word 4-grams (low = repetitive)

``signal/*`` -- how much the GRPO groups (one prompt, k rollouts) can teach:
  groups_any_reward_frac   at least one rollout earned reward
  groups_mixed_frac        rewards differ within the group, so its
                           advantages are non-zero; 0 means no gradient
  groups_all_reward_frac   every rollout earned the same positive reward
  correct_per_group_mean   mean count of positive-reward rollouts per group

Every value is a per-step scalar with ``np.mean`` as its aggregation, the
shape Tunix's ``metric_fns`` hook expects: ``{name: (value, op)}``.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from open_r1_tpu.grpo.rewards import extract_boxed_answer, extract_final_number

_ANSWER_LINE = re.compile(r"^\s*Answer:", re.MULTILINE)
_WORD = re.compile(r"\w+")


def _repeated_line(text: str, times: int = 3) -> bool:
    lines = Counter(line.strip() for line in text.splitlines() if line.strip())
    return any(count >= times for count in lines.values())


def _distinct_4gram_ratio(text: str) -> float:
    words = _WORD.findall(text.lower())
    grams = [tuple(words[i : i + 4]) for i in range(len(words) - 3)]
    return len(set(grams)) / len(grams) if grams else 1.0


def _split_at_stop(text: str, stops: Sequence[str]) -> tuple[str, str | None]:
    """(text before the earliest stop string, text after it or None)."""
    best = None
    for stop in stops:
        index = text.find(stop)
        if index != -1 and (best is None or index < best[0]):
            best = (index, len(stop))
    if best is None:
        return text, None
    return text[: best[0]], text[best[0] + best[1] :]


def completion_behaviour(
    completions: Sequence[str], stop_strings: Sequence[str] | None = None
) -> dict[str, float]:
    """Text-only behaviour fractions over one step's completions."""
    stops = list(stop_strings or [])
    scored, after = [], []
    for text in completions:
        before, rest = _split_at_stop(str(text), stops)
        scored.append(before)
        after.append(rest)
    n = max(len(scored), 1)
    out = {
        "behaviour/answer_line_frac": sum(bool(_ANSWER_LINE.search(t)) for t in scored)
        / n,
        "behaviour/boxed_frac": sum(extract_boxed_answer(t) is not None for t in scored)
        / n,
        "behaviour/number_found_frac": sum(
            extract_final_number(t) is not None for t in scored
        )
        / n,
        "behaviour/chars_mean": float(np.mean([len(t) for t in scored]))
        if scored
        else 0.0,
        "behaviour/repeated_line_frac": sum(_repeated_line(t) for t in scored) / n,
        "behaviour/distinct_4gram_mean": float(
            np.mean([_distinct_4gram_ratio(t) for t in scored])
        )
        if scored
        else 1.0,
    }
    if stops:
        out["behaviour/stopped_frac"] = sum(rest is not None for rest in after) / n
        out["behaviour/after_stop_chars_mean"] = float(
            np.mean([len(rest) if rest is not None else 0 for rest in after])
        )
    return out


def group_signal(rewards: Any, num_generations: int) -> dict[str, float]:
    """How many of the step's k-rollout groups carry a learning signal."""
    values = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if num_generations <= 0 or values.size % num_generations:
        return {}
    groups = values.reshape(-1, num_generations)
    positive = groups > 0
    spread = groups.max(axis=1) - groups.min(axis=1)
    return {
        "signal/groups_any_reward_frac": float(positive.any(axis=1).mean()),
        "signal/groups_mixed_frac": float((spread > 1e-9).mean()),
        "signal/groups_all_reward_frac": float(positive.all(axis=1).mean()),
        "signal/correct_per_group_mean": float(positive.sum(axis=1).mean()),
    }


def build_behaviour_metric_fn(
    num_generations: int, stop_strings: Sequence[str] | None = None
) -> Callable[..., dict[str, tuple[float, Callable[..., Any]]]]:
    """A Tunix ``metric_fns`` entry reporting behaviour and group signal."""

    def behaviour_metrics(
        prompts: Sequence[str],
        completions: Sequence[str],
        rewards: Any,
        advantages: Any = None,
        **_: Any,
    ) -> dict[str, tuple[float, Callable[..., Any]]]:
        metrics = completion_behaviour(completions, stop_strings)
        metrics.update(group_signal(rewards, num_generations))
        return {name: (value, np.mean) for name, value in metrics.items()}

    return behaviour_metrics
