"""Exact-answer matching for a held-out split of the worked-solution corpus.

The gradable rows are answer-key entries from 19th-century arithmetic texts, so
the golds are bare surface forms -- ``8695.``, ``\\$387.``, ``$\\frac{5}{8}$.``,
``4 hr. 44 min. 16 sec.``, ``33½.`` -- and the work is almost entirely
normalisation rather than parsing.

Two verdicts are reported per row, because neither alone is honest on its own:

``strict``
    The model's *final* line has to carry the gold and nothing that contradicts
    it. This is the headline number.
``lenient``
    The gold's values appear somewhere in the completion. A model that reaches
    the right answer and then keeps talking scores here but not above, so the
    gap between the two is a measure of answer discipline rather than of
    arithmetic.

Units are kept as comparison tokens, so ``8%`` never matches ``8``.
"""

from __future__ import annotations

import re
import unicodedata
from fractions import Fraction
from typing import Any

# A gold this long is a worked solution, not an answer-key entry.
MAX_GRADABLE_GOLD_CHARS = 60
MAX_GRADABLE_GOLD_WORDS = 4

_VULGAR = {
    "¼": "1/4",
    "½": "1/2",
    "¾": "3/4",
    "⅐": "1/7",
    "⅑": "1/9",
    "⅒": "1/10",
    "⅓": "1/3",
    "⅔": "2/3",
    "⅕": "1/5",
    "⅖": "2/5",
    "⅗": "3/5",
    "⅘": "4/5",
    "⅙": "1/6",
    "⅚": "5/6",
    "⅛": "1/8",
    "⅜": "3/8",
    "⅝": "5/8",
    "⅞": "7/8",
}

# LaTeX that carries no answer content.
_LATEX_DROP = re.compile(
    r"\\(?:left|right|displaystyle|mathrm|mathbf|text(?:rm|bf|it)?|"
    r"quad|qquad|,|;|:|!|&|#|\s)"
)
_MATH_DELIM = re.compile(r"\$\$|\\\(|\\\)|\\\[|\\\]")
# "64.75+ Ha." uses a trailing plus for "and a bit"; it is not an operator.
_APPROX_PLUS = re.compile(r"(?<=\d)\s*\+(?!\s*\d)")
_THOUSANDS = re.compile(r"(?<=\d),(?=\d{3}(?!\d))")
_FILLER = {"and", "or", "the", "is", "are", "a", "an", "of", "ans"}
# Macros that change a value rather than its presentation.
_VALUE_MACROS = ("sqrt", "pi", "infty", "log", "ln", "sin", "cos", "tan", "deg")

# Folded so that a prediction saying "feet" matches a gold saying "ft."
_UNIT_SYNONYMS = {
    "feet": "ft",
    "foot": "ft",
    "inch": "in",
    "inches": "in",
    "yard": "yd",
    "yards": "yd",
    "mile": "mi",
    "miles": "mi",
    "rod": "rd",
    "rods": "rd",
    "acres": "acre",
    "cents": "cent",
    "dollars": "dollar",
    "hour": "hr",
    "hours": "hr",
    "minute": "min",
    "minutes": "min",
    "second": "sec",
    "seconds": "sec",
    "year": "yr",
    "years": "yr",
    "month": "mo",
    "months": "mo",
    "day": "da",
    "days": "da",
    "square": "sq",
    "cubic": "cu",
    "gallon": "gal",
    "gallons": "gal",
    "pound": "lb",
    "pounds": "lb",
    "hectare": "ha",
    "hectares": "ha",
}


def _expand_frac(text: str) -> str:
    r"""Rewrite ``\frac{a}{b}`` as ``a/b``, honouring nested braces."""
    out: list[str] = []
    index = 0
    while True:
        start = text.find("\\frac", index)
        if start == -1:
            out.append(text[index:])
            return "".join(out)
        out.append(text[index:start])
        cursor = start + len("\\frac")
        parts: list[str] = []
        for _ in range(2):
            while cursor < len(text) and text[cursor] in " \t":
                cursor += 1
            if cursor < len(text) and text[cursor] == "{":
                depth = 0
                opened = cursor + 1
                cursor += 1
                while cursor < len(text):
                    if text[cursor] == "{":
                        depth += 1
                    elif text[cursor] == "}":
                        if depth == 0:
                            break
                        depth -= 1
                    cursor += 1
                parts.append(_expand_frac(text[opened:cursor]))
                cursor += 1
            elif cursor < len(text):  # the \frac12 form
                parts.append(text[cursor])
                cursor += 1
            else:
                parts.append("")
        if len(parts) == 2 and parts[0] and parts[1]:
            out.append(f" {parts[0]}/{parts[1]} ")
        index = cursor


def normalise(text: str) -> str:
    """Canonical comparison form: lowercased tokens, units folded, no LaTeX."""
    if not text:
        return ""
    result = text
    # Space-pad before NFKC: it rewrites a vulgar fraction into digits around a
    # fraction slash, which would glue onto a preceding integer and turn "33½"
    # into "331/2".
    for char, replacement in _VULGAR.items():
        result = result.replace(char, f" {replacement} ")
    result = unicodedata.normalize("NFKC", result).replace("⁄", "/")  # noqa: RUF001
    # This corpus escapes markdown metacharacters throughout.
    result = re.sub(r"\\([$%&_#])", r"\1", result)
    result = result.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    result = _expand_frac(result)
    result = _MATH_DELIM.sub(" ", result)
    result = result.replace("$", " ")
    result = _LATEX_DROP.sub(" ", result)
    # These change the value, so they must survive as comparison tokens rather
    # than being swept up with the presentational macros below.
    for macro in _VALUE_MACROS:
        result = result.replace(f"\\{macro}", f" {macro} ")
    result = re.sub(r"\\[a-zA-Z]+", " ", result)
    for char, replacement in _VULGAR.items():
        result = result.replace(char, f" {replacement} ")
    result = result.replace("−", "-").replace("–", "-").replace("—", "-")  # noqa: RUF001
    result = _THOUSANDS.sub("", result)
    result = _APPROX_PLUS.sub(" ", result)
    # A percent sign is meaning, not punctuation: keep it as its own token.
    result = result.replace("%", " % ")
    result = re.sub(r"[{}\[\]~\\]", " ", result)
    result = re.sub(r"[,;:]", " ", result)
    result = re.sub(r"\s+", " ", result).strip().lower()

    tokens: list[str] = []
    for raw in result.split(" "):
        # rstrip only: ".75" must not become "75".
        token = raw.rstrip(".").strip()
        if not token or token in _FILLER:
            continue
        tokens.append(_UNIT_SYNONYMS.get(token, token))
    return " ".join(tokens)


_NUMBER = re.compile(r"-?\d*\.?\d+(?:/-?\d*\.?\d+)?")


def _as_fraction(token: str) -> Fraction | None:
    try:
        if "/" in token:
            numerator, denominator = token.split("/", 1)
            return Fraction(Fraction(numerator), Fraction(denominator))
        return Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None


def numbers(text: str) -> list[Fraction]:
    """Exact rational values in reading order.

    An integer followed by a proper fraction is read as a mixed number, which
    is how every mixed answer in this corpus is written -- including the
    decimal-plus-fraction form ``\\$3.43¾``.
    """
    tokens = _NUMBER.findall(normalise(text))
    values: list[Fraction] = []
    index = 0
    while index < len(tokens):
        current = _as_fraction(tokens[index])
        if current is None:
            index += 1
            continue
        if index + 1 < len(tokens) and "/" not in tokens[index]:
            following = _as_fraction(tokens[index + 1])
            if (
                following is not None
                and "/" in tokens[index + 1]
                and 0 < following < 1
                and current >= 0
            ):
                # "15 5/8" is 15+5/8, but the currency form "\$3.43¾" means
                # three-quarters of the last decimal place: 3.43 + 0.0075.
                _, _, decimals = tokens[index].partition(".")
                scale = Fraction(10) ** len(decimals)
                values.append(current + following / scale)
                index += 2
                continue
        values.append(current)
        index += 1
    return values


_WORD = re.compile(r"[a-z]+|%")


def _units(text: str) -> set[str]:
    return {word for word in _WORD.findall(normalise(text)) if word not in _FILLER}


def final_line(completion: str) -> str:
    """The line a reader would take the answer from.

    The last non-empty line, unless it carries no digits while an earlier one
    does -- a trailing "Q. E. D." or "Ans." should not hide the answer above it.
    """
    lines = [line.strip() for line in (completion or "").split("\n") if line.strip()]
    if not lines:
        return ""
    for line in reversed(lines):
        if re.search(r"\d", line):
            return line
    return lines[-1]


def _values_agree(predicted: list[Fraction], gold: list[Fraction]) -> bool:
    if not gold or not predicted:
        return False
    if predicted == gold:
        return True
    # A single-value gold may sit at the end of a line that also shows working.
    return len(gold) == 1 and predicted[-1] == gold[0]


def matches_strict(completion: str, gold: str) -> bool:
    """The completion's final line carries the gold and contradicts nothing."""
    if not gold or not gold.strip():
        return False
    candidate = final_line(completion)
    if not candidate:
        return False
    if normalise(candidate) == normalise(gold):
        return True
    gold_units, candidate_units = _units(gold), _units(candidate)
    # A percentage is a different quantity from a bare number, so it has to
    # agree in both directions. Currency and unit names do not: the gold writes
    # them out and a terse model is allowed to omit them.
    if ("%" in gold_units) != ("%" in candidate_units):
        return False
    if gold_units and not gold_units <= candidate_units:
        return False
    return _values_agree(numbers(candidate), numbers(gold))


def matches_lenient(completion: str, gold: str) -> bool:
    """The gold's values appear somewhere in the completion, in order."""
    if not gold or not gold.strip():
        return False
    gold_values = numbers(gold)
    if not gold_values:
        return normalise(gold) in normalise(completion)
    found = numbers(completion or "")
    if not found:
        return False
    # Subsequence search: the gold's values in order, anywhere in the text.
    position = 0
    for value in gold_values:
        while position < len(found) and found[position] != value:
            position += 1
        if position == len(found):
            return False
        position += 1
    return True


_PROSE = re.compile(
    r"\b(?:why|because|whence|hence|therefore|proved|let|prove|"
    r"circle|triangle|parabola|equation)\b",
    re.IGNORECASE,
)


def is_gradable(row: dict[str, Any]) -> bool:
    """A row whose gold is an answer-key entry, so exact match is fair."""
    gold = row["messages"][1]["content"]
    if row["source"]["extraction_type"] not in {"question_answer", "answer_only"}:
        return False
    if len(gold) > MAX_GRADABLE_GOLD_CHARS or "\n" in gold:
        return False
    if not re.search(r"\d", gold):
        return False
    if _PROSE.search(gold):
        return False
    if len(_units(gold)) > MAX_GRADABLE_GOLD_WORDS:
        return False
    return bool(numbers(gold))
