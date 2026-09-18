"""Answer matching for a held-out split of the worked-solution corpus.

The cases below are the real surface forms the corpus's answer keys use, not
invented ones: bare integers with a trailing stop, escaped currency, vulgar and
LaTeX fractions, the currency form where the fraction belongs to the last
decimal place, and compound unit answers.
"""

from __future__ import annotations

import pytest

from open_r1_tpu.sft.heldout_match import (
    final_line,
    is_gradable,
    matches_lenient,
    matches_strict,
    normalise,
    numbers,
)


def row(gold: str, *, extraction_type: str = "question_answer") -> dict:
    return {
        "id": "row",
        "source": {"extraction_type": extraction_type, "file": "book.md"},
        "messages": [
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": gold},
        ],
        "metadata": {"domain": "arithmetic", "difficulty": "elementary"},
    }


@pytest.mark.parametrize(
    ("completion", "gold"),
    [
        ("8695.", "8695."),
        ("The sum of these is 8695.", "8695."),
        ("110.", "\\$110."),
        ("17,565.", "\\$17,565."),
        (".75", "$\\frac{3}{4}$."),
        ("15 5/8", "$15\\frac{5}{8}$."),
        ("3.4375", "\\$3.43\u00be."),
        ("$3.43\\frac{3}{4}$", "\\$3.43\u00be."),
        ("4 hours 44 minutes 16 seconds", "4 hr. 44 min. 16 sec."),
        ("$\\sqrt{2}$", "$\\sqrt{2}$."),
    ],
)
def test_strict_accepts_equivalent_surface_forms(completion: str, gold: str) -> None:
    assert matches_strict(completion, gold)


@pytest.mark.parametrize(
    ("completion", "gold"),
    [
        ("100.", "110."),
        ("25%.", "\\$25."),
        ("25.", "25%."),
        ("8 ft.", "8%."),
        ("2", "$\\sqrt{2}$."),
        ("", "8695."),
    ],
)
def test_strict_rejects_different_quantities(completion: str, gold: str) -> None:
    assert not matches_strict(completion, gold)


def test_strict_reads_the_last_line_that_carries_a_number() -> None:
    assert matches_strict("9 x 12 = 108.\nSo the cost is 110.", "\\$110.")
    assert matches_strict("110.\nQ. E. D.", "\\$110.")


def test_strict_needs_the_answer_last_but_lenient_does_not() -> None:
    wandered = "The answer is 110.\nBut on reflection it is 97."
    assert not matches_strict(wandered, "\\$110.")
    assert matches_lenient(wandered, "\\$110.")


def test_lenient_still_rejects_an_absent_value() -> None:
    assert not matches_lenient("The answer is 97.", "\\$110.")


def test_normalise_keeps_a_leading_decimal_point() -> None:
    assert numbers(".75") == numbers("$\\frac{3}{4}$")
    assert normalise(".75") == ".75"


def test_numbers_reads_a_mixed_number_as_one_value() -> None:
    assert len(numbers("$15\\frac{5}{8}$.")) == 1
    assert numbers("$15\\frac{5}{8}$.")[0] == numbers("125/8")[0]


def test_final_line_skips_a_trailing_line_without_digits() -> None:
    assert final_line("the answer is 12.\nQ. E. D.") == "the answer is 12."
    assert final_line("") == ""


@pytest.mark.parametrize(
    ("gold", "extraction_type", "expected"),
    [
        ("8695.", "question_answer", True),
        ("\\$387.", "question_answer", True),
        ("17 cents. Why?", "answer_only", False),  # prose cue
        ("Let $ABCD$ be a square, of which the diagonals are equal.", "A", False),
        ("Q. E. D.", "question_answer", False),  # no number
        ("x" * 200, "question_answer", False),  # a worked solution
    ],
)
def test_is_gradable(gold: str, extraction_type: str, expected: bool) -> None:
    assert is_gradable(row(gold, extraction_type=extraction_type)) is expected


@pytest.mark.parametrize(
    ("completion", "gold", "expected"),
    [
        ("15.", "15 cents.", True),  # a terse answer may omit the unit
        ("148.", "148 acres.", True),
        ("15 dollars.", "15 cents.", False),  # but may not contradict it
        ("2", "$\\sqrt{2}$.", False),  # a root changes the value, not the unit
        ("8.", "8%.", False),
    ],
)
def test_strict_tolerates_a_missing_unit_but_not_a_wrong_one(
    completion: str, gold: str, expected: bool
) -> None:
    assert matches_strict(completion, gold) is expected


def test_lenient_is_a_superset_of_strict() -> None:
    # Working before the answer inserts numbers between the gold's values, so
    # lenient counts occurrences rather than looking for an unbroken run.
    cases = [
        ("working 5 then 9\n15.", "15 cents."),
        ("4 hr. 44 min. 16 sec.", "4 hr. 44 min. 16 sec."),
        ("8695.", "8695."),
    ]
    for completion, gold in cases:
        assert matches_strict(completion, gold)
        assert matches_lenient(completion, gold)


def test_is_gradable_rejects_an_equation_whose_exponents_read_as_values() -> None:
    equation = "$$ = \\frac{x^2}{b^2 - x^2} + \\frac{y^2}{b^2} = 1. $$"
    assert not is_gradable(row(equation, extraction_type="answer_only"))
