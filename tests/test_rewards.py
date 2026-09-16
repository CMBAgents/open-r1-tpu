"""Reward functions run on CPU with plain strings -- no TPU, JAX, or Tunix
needed, so these run in every environment this repository's tests already
run in (see AGENTS.md: "this container has no project deps").
"""

from open_r1_tpu.grpo.rewards import (
    answers_match,
    correctness_reward,
    extract_boxed_answer,
    format_reward,
    has_closed_reasoning,
)

# ---------------------------------------------------------------------------
# extract_boxed_answer
# ---------------------------------------------------------------------------


def test_extract_boxed_answer_finds_simple_content():
    assert extract_boxed_answer(r"so the answer is \boxed{42}.") == "42"


def test_extract_boxed_answer_handles_nested_braces():
    # A lazy `\boxed\{(.*?)\}` regex would stop at the first `}` here and
    # return "\\frac{1", not "\\frac{1}{2}" -- exactly the bug brace-matching
    # avoids.
    text = r"\boxed{\frac{1}{2}}"
    assert extract_boxed_answer(text) == r"\frac{1}{2}"


def test_extract_boxed_answer_uses_the_last_occurrence():
    text = r"first try \boxed{1}, no wait, \boxed{2}"
    assert extract_boxed_answer(text) == "2"


def test_extract_boxed_answer_returns_none_when_absent():
    assert extract_boxed_answer("no boxed answer here") is None


def test_extract_boxed_answer_returns_none_when_truncated_mid_box():
    # The generation cap cut the completion before the closing brace.
    assert extract_boxed_answer(r"\boxed{1/2 and then it just keeps going") is None


# ---------------------------------------------------------------------------
# answers_match
# ---------------------------------------------------------------------------


def test_answers_match_exact_numeric():
    exact, close = answers_match("42", "42")
    assert exact and close


def test_answers_match_normalizes_latex_spacing_and_fracs():
    exact, close = answers_match(r"\dfrac{1}{2}", r"\frac{1}{2}")
    assert exact and close


def test_answers_match_close_numeric_within_ten_percent():
    exact, close = answers_match("104", "100")
    assert not exact
    assert close


def test_answers_match_rejects_far_numeric():
    exact, close = answers_match("500", "100")
    assert not exact and not close


def test_answers_match_rejects_mismatched_symbolic_answers():
    exact, close = answers_match(r"\frac{1}{3}", r"\frac{1}{2}")
    assert not exact and not close


# ---------------------------------------------------------------------------
# has_closed_reasoning
# ---------------------------------------------------------------------------


def test_has_closed_reasoning_true_for_one_clean_pair():
    assert has_closed_reasoning("<think>work</think>\\boxed{1}")


def test_has_closed_reasoning_false_when_never_opened():
    assert not has_closed_reasoning("just an answer \\boxed{1}")


def test_has_closed_reasoning_false_when_never_closed():
    assert not has_closed_reasoning("<think>still going and going")


def test_has_closed_reasoning_false_on_duplicate_tags():
    assert not has_closed_reasoning("<think>a</think><think>b</think>\\boxed{1}")


# ---------------------------------------------------------------------------
# reward functions (the Tunix reward_fn calling convention)
# ---------------------------------------------------------------------------


def test_format_reward_full_marks_for_closed_reasoning_then_boxed_answer():
    completion = "<think>step by step</think>so \\boxed{42}"
    (score,) = format_reward(["prompt"], [completion])
    assert score == 3.0


def test_format_reward_penalizes_boxed_answer_before_reasoning_closes():
    # The shape is inverted: <think> never closes before the box appears.
    completion = "<think>\\boxed{42} still thinking"
    (score,) = format_reward(["prompt"], [completion])
    assert score < 3.0


def test_format_reward_partial_credit_for_missing_pieces():
    (score,) = format_reward(["prompt"], ["no tags or box at all"])
    assert score < 0


def test_correctness_reward_matches_shape_and_scores_exact_answer():
    completions = ["<think>x</think>\\boxed{42}", "<think>x</think>\\boxed{7}"]
    answers = ["42", "42"]
    scores = correctness_reward(["p", "p"], completions, answers)
    assert scores == [3.0, -0.5]


def test_correctness_reward_gives_partial_credit_for_close_numeric_answer():
    completions = ["<think>x</think>\\boxed{104}"]
    scores = correctness_reward(["p"], completions, ["100"])
    assert scores == [0.5]


def test_correctness_reward_zero_when_no_boxed_answer():
    scores = correctness_reward(["p"], ["still reasoning, no answer yet"], ["42"])
    assert scores == [0.0]


def test_correctness_reward_rejects_mismatched_lengths():
    import pytest

    with pytest.raises(ValueError):
        correctness_reward(["p", "p"], ["a", "b"], ["only one gold"])
