"""GRPO reward functions for the reasoning-math actor.

Every function here matches the calling convention Tunix's
``tunix.rl.reward_manager.SequenceRewardManager`` uses: it invokes each
``reward_fn`` as ``reward_fn(prompts=prompts, completions=completions,
**extra_columns)``, where ``extra_columns`` is whatever the training batch
carries beyond ``prompts``/``completions`` -- here, ``question`` and
``answer`` from :mod:`open_r1_tpu.grpo.data`. Multiple reward
functions are summed per sample (``np.nansum`` over the reward-function
axis), so each function returns points on its own scale rather than a
normalized [0, 1] score, following the convention of Tunix's own
``examples/grpo_gemma.ipynb`` (3.0 for a fully correct answer, fractional
credit below that).

Two rewards are built for this core implementation, mirroring the first two
of that example's four (format-exact/format-approximate folded into one,
answer-correctness) but adapted to this project's own SFT format rather than
Gemma's ``<reasoning>``/``<answer>`` tags:

- ``format_reward``: the completion opens with ``<think>``, closes it with a
  single ``</think>``, and carries a ``\\boxed{...}`` afterward -- the shape
  SFT was teaching (see ``reporting.reasoning_start``/``reasoning_end``/
  ``answer_marker`` in every eval recipe's ``base.yaml``).
- ``correctness_reward``: the boxed answer matches the corpus's gold answer,
  with partial credit for a close numeric match.

A third, repetition-penalty reward (suppressing the verbatim-loop truncation
failure mode seen in the SFT checkpoints' evaluation completions) is
deliberately deferred rather than built here, to keep this first pipeline to
a core GRPO implementation. Revisit once format/correctness rewards alone
have been run and measured.

``answer_correctness_reward`` is a third, format-free option: 1.0 when the
final number (boxed if present, else the last number written) equals gold,
else 0.0. ``math_answer_reward`` is SimpleRL-Zoo's 1/0 reward for LaTeX
golds (MATH-style answers such as ``\\frac{3}{4}``): the answer is read as
they read it and compared with the Hendrycks MATH normaliser. A recipe picks
its rewards by name with ``grpo.reward_functions``; without it,
``DEFAULT_REWARD_FNS`` (format plus correctness) is used.

The rewards are independent reward_fns (not folded into one), so a recipe can
scale or drop either of them by omitting it from ``grpo_run``'s reward list,
and so each is unit-testable on its own without a rollout.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

REASONING_START = "<think>"
REASONING_END = "</think>"
ANSWER_MARKER = r"\boxed{"

# Loose normalization for comparing a boxed answer to the corpus's gold
# string. Not LightEval's math-verify-successor LaTeX parser (see
# pyproject.toml's lighteval pin note: "LightEval used Math-Verify for this
# until 0.13 and now parses LaTeX directly, so nothing here should name
# Math-Verify") -- reward shaping tolerates being looser than a benchmark
# scorer, and re-deriving LightEval's internal comparator here would create a
# second, driftable copy of it rather than reusing one.
_LATEX_SPACING = re.compile(r"\\[,!;:]|\\ |~")
_DFRAC_TFRAC = re.compile(r"\\d?frac")
_TRAILING_PUNCTUATION = re.compile(r"[.\s]+$")


def extract_boxed_answer(text: str) -> str | None:
    """Return the contents of the last well-formed ``\\boxed{...}``.

    Brace-matched rather than a lazy regex: ``\\boxed{\\frac{1}{2}}`` has a
    nested ``}``, which ``\\\\boxed\\{(.*?)\\}`` would truncate at the first
    close brace. The *last* occurrence is used, matching the "final answer"
    convention this project's traces and eval scoring both follow.
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
    """Compare a predicted boxed answer to gold. Returns (exact, close).

    ``exact`` is a normalized string match (order-of-magnitude looser than
    the eval harness's LaTeX-aware comparator, tight enough for the common
    case of a bare number or a simple fraction). ``close`` is a numeric
    within-10% match when both sides parse as plain floats -- credit for "in
    the right neighborhood", following ``examples/grpo_gemma.ipynb``'s
    ``check_answer``.
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

    Partial, signed credit otherwise -- one point per structural element
    present or absent (closed reasoning, a boxed answer after it, no
    duplicate tags) -- so a model that is close to the shape still gets a
    gradient toward it, matching ``match_format_approximately`` in
    ``examples/grpo_gemma.ipynb``.
    """
    scores: list[float] = []
    for completion in completions:
        closed = has_closed_reasoning(completion)
        boxed = extract_boxed_answer(completion)
        if (
            closed
            and boxed is not None
            # Both substrings are confirmed present in this branch, so a
            # plain find() ordering check is safe -- neither side is -1.
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

    For a model that writes free-form worked solutions rather than a fixed
    answer shape, the last number is the conventional GSM8K fallback (the
    "flexible extract" of common harnesses). ``\\boxed{}`` still wins when
    present, so a model that learns to box its answer is read exactly.
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

    No format requirement and no partial credit: only whether the answer is
    right. The final number is read by :func:`extract_final_number`; gold
    and prediction are compared as numbers, so ``20,000``, ``$20000`` and
    ``20000.00`` all match a gold of ``20000``.
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
# (math_equivalence.py), the de facto grader for MATH-style golds in RL
# recipes: two answers are equal when their normalised strings are. It is
# string-level on purpose -- no symbolic parsing, so it cannot hang on a
# pathological completion -- and it is looser than the eval harness's
# comparator, which reward shaping tolerates.


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

    SimpleRL-Zoo's reward: correctness only, no format reward and nothing
    below zero, with the answer read by :func:`extract_math_answer`. Two
    differences from theirs. The grader is :func:`math_answers_equal`, a
    string normaliser, where theirs is the symbolic ``math_verify``; it misses
    some equivalent forms, which costs signal but never credits a wrong
    answer. And only the completion is searched, where theirs searches prompt
    and completion together, which only matters when the completion holds no
    number at all.
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

    ``None`` keeps :data:`DEFAULT_REWARD_FNS`, so recipes written before the
    option existed are unchanged.
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
