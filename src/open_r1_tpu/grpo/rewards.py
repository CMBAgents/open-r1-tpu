"""GRPO reward functions.

Each follows the calling convention of Tunix's ``SequenceRewardManager``,
``reward_fn(prompts=..., completions=..., **columns)``, where the columns are
the batch's ``question`` and ``answer`` from :mod:`open_r1_tpu.grpo.data`, and
returns one score per completion. Tunix sums the scores of every function, so
each scores on its own scale. A recipe names its rewards in
``grpo.reward_functions``; the default is ``format_reward`` plus
``correctness_reward``.

- ``format_reward``: a single closed ``<think>`` block followed by a
  ``\\boxed{...}`` answer, with signed partial credit per structural element.
- ``correctness_reward``: the boxed answer matches gold, with partial credit
  for a number within 10%.
- ``answer_correctness_reward``: 1 when the final number, boxed or else the
  last one written, equals gold; no format requirement.
- ``math_answer_reward``: SimpleRL-Zoo's 1/0 reward for MATH-style LaTeX golds.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

REASONING_START = "<think>"
REASONING_END = "</think>"
ANSWER_MARKER = r"\boxed{"

# Loose normalisation for comparing a boxed answer with a gold string. Reward
# shaping tolerates being looser than the evaluation's LaTeX-aware comparator.
_LATEX_SPACING = re.compile(r"\\[,!;:]|\\ |~")
_DFRAC_TFRAC = re.compile(r"\\d?frac")
_TRAILING_PUNCTUATION = re.compile(r"[.\s]+$")


def extract_boxed_answer(text: str) -> str | None:
    """Return the contents of the last well-formed ``\\boxed{...}``.

    Brace-matched, since answers such as ``\\frac{1}{2}`` nest braces.
    """
    marker = ANSWER_MARKER
    start = text.rfind(marker)
    if start == -1:
        return None
    depth = 0
    content_start = start + len(marker)
    for index in range(content_start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            if depth == 0:
                return text[content_start:index]
            depth -= 1
    return None  # unterminated: the cap cut the completion mid-box.


def _normalize_answer(value: str) -> str:
    value = value.strip()
    value = _LATEX_SPACING.sub("", value)
    value = _DFRAC_TFRAC.sub(r"\\frac", value)
    value = value.replace("$", "").replace("{", "").replace("}", "")
    value = value.replace(" ", "")
    value = _TRAILING_PUNCTUATION.sub("", value)
    return value


def _as_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def answers_match(predicted: str, gold: str) -> tuple[bool, bool]:
    """Compare a predicted boxed answer with gold. Returns (exact, close).

    ``exact`` is a normalised string match, enough for a bare number or a
    simple fraction. ``close`` is a match within 10% when both sides parse as
    plain numbers.
    """
    norm_pred = _normalize_answer(predicted)
    norm_gold = _normalize_answer(gold)
    if norm_pred == norm_gold:
        return True, True
    pred_float = _as_float(norm_pred)
    gold_float = _as_float(norm_gold)
    if pred_float is not None and gold_float is not None and gold_float != 0:
        ratio = pred_float / gold_float
        return False, 0.9 <= ratio <= 1.1
    return False, False


def has_closed_reasoning(text: str) -> bool:
    """Exactly one ``<think>``/``</think>`` pair, opened at or near the start."""
    return (
        text.count(REASONING_START) == 1
        and text.count(REASONING_END) == 1
        and text.find(REASONING_START) <= 0
        and text.find(REASONING_START) < text.find(REASONING_END)
    )


def format_reward(
    prompts: Sequence[str], completions: Sequence[str], **_: Any
) -> list[float]:
    """Full credit for a closed ``<think>`` block followed by a boxed answer.

    Otherwise half a point up or down per structural element present or
    missing, so a model close to the shape still gets a gradient towards it.
    """
    scores: list[float] = []
    for completion in completions:
        closed = has_closed_reasoning(completion)
        boxed = extract_boxed_answer(completion)
        if (
            closed
            and boxed is not None
            and completion.find(REASONING_END) < completion.find(ANSWER_MARKER)
        ):
            scores.append(3.0)
            continue
        score = 0.0
        score += 0.5 if completion.count(REASONING_START) == 1 else -0.5
        score += 0.5 if completion.find(REASONING_START) <= 0 else -0.5
        score += 0.5 if completion.count(REASONING_END) == 1 else -0.5
        score += 0.5 if ANSWER_MARKER in completion else -0.5
        score += 0.5 if boxed is not None else -0.5
        scores.append(score)
    return scores


def correctness_reward(
    prompts: Sequence[str], completions: Sequence[str], answer: Sequence[str], **_: Any
) -> list[float]:
    """Reward for a boxed answer matching (or close to) the gold answer."""
    if len(completions) != len(answer):
        raise ValueError(
            f"completions ({len(completions)}) and answer ({len(answer)}) "
            "must have matching length"
        )
    scores: list[float] = []
    for completion, gold in zip(completions, answer, strict=True):
        predicted = extract_boxed_answer(completion)
        if predicted is None or not gold:
            scores.append(0.0)
            continue
        exact, close = answers_match(predicted, str(gold))
        if exact:
            scores.append(3.0)
        elif close:
            scores.append(0.5)
        else:
            scores.append(-0.5)
    return scores


# A plain decimal number, thousands separators allowed. The look-behind stops
# the minus in "8-3" being read as a sign, so that span yields 8 and 3.
_NUMBER = re.compile(r"(?<!\d)-?\d[\d,]*(?:\.\d+)?")


def extract_final_number(text: str) -> str | None:
    """The completion's final answer: a boxed answer if any, else the last number.

    The last number is the conventional GSM8K fallback for free-form worked
    solutions.
    """
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    numbers = _NUMBER.findall(text)
    return numbers[-1] if numbers else None


def _plain_number(value: str) -> float | None:
    return _as_float(_normalize_answer(value).replace(",", ""))


def answer_correctness_reward(
    prompts: Sequence[str], completions: Sequence[str], answer: Sequence[str], **_: Any
) -> list[float]:
    """1.0 when the final number equals the gold number, else 0.0.

    Compared as numbers, so ``20,000``, ``$20000`` and ``20000.00`` all match
    a gold of ``20000``.
    """
    if len(completions) != len(answer):
        raise ValueError(
            f"completions ({len(completions)}) and answer ({len(answer)}) "
            "must have matching length"
        )
    scores: list[float] = []
    for completion, gold in zip(completions, answer, strict=True):
        predicted = extract_final_number(completion)
        gold_value = _plain_number(str(gold)) if gold else None
        pred_value = _plain_number(predicted) if predicted is not None else None
        correct = (
            gold_value is not None
            and pred_value is not None
            and abs(pred_value - gold_value) < 1e-6
        )
        scores.append(1.0 if correct else 0.0)
    return scores


# The MATH answer normaliser from Hendrycks et al.'s MATH release
# (math_equivalence.py): two answers are equal when their normalised strings
# are. String-level on purpose, so a pathological completion cannot hang it.


def _fix_fracs(value: str) -> str:
    """``\\frac12`` -> ``\\frac{1}{2}`` and ``\\frac1{72}`` -> ``\\frac{1}{72}``."""
    parts = value.split("\\frac")
    fixed = parts[0]
    for part in parts[1:]:
        fixed += "\\frac"
        if part.startswith("{"):
            fixed += part
            continue
        if len(part) < 2:
            return value
        numerator, rest = part[0], part[1:]
        if rest[0] == "{":
            fixed += "{" + numerator + "}" + rest
        else:
            fixed += "{" + numerator + "}{" + rest[0] + "}" + rest[1:]
    return fixed


def _fix_a_slash_b(value: str) -> str:
    """``3/4`` -> ``\\frac{3}{4}`` for two plain integers only."""
    parts = value.split("/")
    if len(parts) != 2 or not all(re.fullmatch(r"-?\d+", part) for part in parts):
        return value
    return "\\frac{" + parts[0] + "}{" + parts[1] + "}"


def _fix_sqrt(value: str) -> str:
    """``\\sqrt3`` -> ``\\sqrt{3}``."""
    parts = value.split("\\sqrt")
    fixed = parts[0]
    for part in parts[1:]:
        if part and not part.startswith("{"):
            fixed += "\\sqrt{" + part[0] + "}" + part[1:]
        else:
            fixed += "\\sqrt" + part
    return fixed


def normalize_math_answer(value: str) -> str:
    """Normalise a MATH-style answer string (Hendrycks ``_strip_string``)."""
    value = value.replace("\n", "").replace("\\!", "").replace("\\\\", "\\")
    value = value.replace("tfrac", "frac").replace("dfrac", "frac")
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("^{\\circ}", "").replace("^\\circ", "")
    value = value.replace("\\$", "").replace("$", "")
    # Units on the right: "\text{ " only ever introduces one in MATH's golds.
    if "\\text{ " in value:
        value = value.split("\\text{ ")[0]
    value = value.replace("\\%", "").replace("%", "")
    value = value.replace(" .", " 0.").replace("{.", "{0.")
    if not value:
        return value
    if value.startswith("."):
        value = "0" + value
    # "x = 5" and "k=5" -> "5", but only for a short left-hand side.
    sides = value.split("=")
    if len(sides) == 2 and len(sides[0].strip()) <= 2:
        value = sides[1]
    value = _fix_sqrt(value)
    value = value.replace(" ", "")
    value = _fix_fracs(value)
    if value == "0.5":
        value = "\\frac{1}{2}"
    return _fix_a_slash_b(value)


def math_answers_equal(predicted: str, gold: str) -> bool:
    """MATH-normalised string equality, or equal plain numbers.

    The numeric fallback catches what the normaliser leaves apart:
    ``1,000`` against ``1000`` and ``3.0`` against ``3``.
    """
    if normalize_math_answer(predicted) == normalize_math_answer(gold):
        return True
    pred_value = _plain_number(predicted)
    gold_value = _plain_number(gold)
    return (
        pred_value is not None
        and gold_value is not None
        and abs(pred_value - gold_value) < 1e-6
    )


# SimpleRL-Zoo's last-number fallback: commas dropped, then the last match.
_LAST_NUMBER = re.compile(r"-?\d*\.?\d+")


def extract_math_answer(text: str) -> str | None:
    """The completion's answer, read the way SimpleRL-Zoo reads it.

    Qwen2.5-Math's ``extract_answer`` for MATH, in its order: the last
    ``\\boxed{}``; else what follows the last "he answer is"; else what
    follows the last "final answer is"; else the last number. The trailing
    text of the two phrase cases is kept whole, as theirs is, with line
    breaks, a leading colon and a trailing full stop removed.
    """
    boxed = extract_boxed_answer(text)
    if boxed is not None:
        return boxed
    for phrase in ("he answer is", "final answer is"):
        if phrase in text:
            tail = re.sub(r"\n\s*", "", text.split(phrase)[-1].strip())
            tail = tail.removeprefix(":").removesuffix(".").removesuffix("/")
            return tail
    numbers = _LAST_NUMBER.findall(text.replace(",", ""))
    return numbers[-1] if numbers else None


def math_answer_reward(
    prompts: Sequence[str], completions: Sequence[str], answer: Sequence[str], **_: Any
) -> list[float]:
    """1.0 when the completion's answer equals the gold answer, else 0.0.

    SimpleRL-Zoo's reward, with two differences. The grader is the string
    normaliser :func:`math_answers_equal` rather than symbolic ``math_verify``,
    which misses some equivalent forms but never credits a wrong answer. And
    only the completion is searched, not the prompt with it, which matters only
    when the completion holds no number at all.
    """
    if len(completions) != len(answer):
        raise ValueError(
            f"completions ({len(completions)}) and answer ({len(answer)}) "
            "must have matching length"
        )
    scores: list[float] = []
    for completion, gold in zip(completions, answer, strict=True):
        predicted = extract_math_answer(completion)
        correct = (
            predicted is not None
            and bool(str(gold).strip())
            and math_answers_equal(predicted, str(gold))
        )
        scores.append(1.0 if correct else 0.0)
    return scores


DEFAULT_REWARD_FNS = (format_reward, correctness_reward)

REWARD_FNS_BY_NAME = {
    fn.__name__: fn
    for fn in (
        format_reward,
        correctness_reward,
        answer_correctness_reward,
        math_answer_reward,
    )
}


def reward_fns_from_names(names: Sequence[str] | None) -> tuple:
    """The reward functions a recipe's ``grpo.reward_functions`` names.

    ``None`` selects :data:`DEFAULT_REWARD_FNS`.
    """
    if names is None:
        return DEFAULT_REWARD_FNS
    unknown = [name for name in names if name not in REWARD_FNS_BY_NAME]
    if unknown:
        raise ValueError(
            f"Unknown reward function(s) {unknown}; "
            f"choose from {sorted(REWARD_FNS_BY_NAME)}"
        )
    return tuple(REWARD_FNS_BY_NAME[name] for name in names)
