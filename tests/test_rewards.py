"""Reward functions run on CPU with plain strings -- no TPU, JAX, or Tunix
needed, so these run in every environment this repository's tests already
run in (see AGENTS.md: "this container has no project deps").
"""

import pytest

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


# ---------------------------------------------------------------------------
# with_completion_stop_strings
# ---------------------------------------------------------------------------


def test_truncate_at_stop_strings_cuts_at_earliest_match():
    from open_r1_tpu.grpo.rewards import truncate_at_stop_strings

    text = "<think>x</think>\\boxed{4}<|im_end|>\n<|im_start|>user\nmore"
    assert (
        truncate_at_stop_strings(text, ["<|im_end|>"]) == "<think>x</think>\\boxed{4}"
    )
    assert truncate_at_stop_strings(text, ["<|im_start|>", "<|im_end|>"]) == (
        "<think>x</think>\\boxed{4}"
    )
    assert truncate_at_stop_strings("no marker", ["<|im_end|>"]) == "no marker"


def test_with_completion_stop_strings_scores_only_the_text_before_the_marker():
    from open_r1_tpu.grpo.rewards import with_completion_stop_strings

    good = "<think>2+2</think>\\boxed{4}"
    # After the marker: a second <think> and a wrong boxed answer, which
    # would ruin both rewards if scored.
    tail = "<|im_end|>\n<|im_start|>assistant\n<think>again</think>\\boxed{9}"
    wrapped = with_completion_stop_strings(
        [format_reward, correctness_reward], ["<|im_end|>"]
    )
    assert [fn.__name__ for fn in wrapped] == ["format_reward", "correctness_reward"]
    scores = [
        fn(prompts=["p"], completions=[good + tail], answer=["4"])[0] for fn in wrapped
    ]
    assert scores == [3.0, 3.0]
    raw = [
        fn(prompts=["p"], completions=[good + tail], answer=["4"])[0]
        for fn in (format_reward, correctness_reward)
    ]
    assert raw != [3.0, 3.0]


def test_with_completion_stop_strings_without_stops_returns_the_functions_unchanged():
    from open_r1_tpu.grpo.rewards import with_completion_stop_strings

    assert with_completion_stop_strings([format_reward], None) == [format_reward]
    assert with_completion_stop_strings([format_reward], []) == [format_reward]


# ---------------------------------------------------------------------------
# answer_correctness_reward: correctness only, no format requirement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("so the total is 18 dollars.", "18"),
        ("First 3 + 4 = 7, then 7 x 2 = 14.\nThe answer is 14", "14"),
        ("He earns $20,000 in all.", "20,000"),
        ("working 12 and 30 ... \\boxed{42} then 99", "42"),
        ("8-3 leaves 5", "5"),
        ("a debt of -7 remains", "-7"),
        ("no digits here", None),
    ],
)
def test_extract_final_number(text, expected):
    from open_r1_tpu.grpo.rewards import extract_final_number

    assert extract_final_number(text) == expected


def test_answer_correctness_reward_scores_only_the_final_answer():
    from open_r1_tpu.grpo.rewards import answer_correctness_reward

    completions = [
        "The baker earned 453 x 12 = 5436 and 126 x 7 = 882, so 6318.",
        "He had $20,000.",
        "The answer is 20000.00",
        "<think>2+2</think>\\boxed{5}",
        "The answer is 6318, I think; check: 6317",
        "no number at all",
    ]
    gold = ["6318", "20000", "20000", "4", "6318", "4"]
    assert answer_correctness_reward(
        prompts=[""] * 6, completions=completions, answer=gold
    ) == [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]


def test_answer_correctness_reward_rejects_mismatched_lengths():
    from open_r1_tpu.grpo.rewards import answer_correctness_reward

    with pytest.raises(ValueError, match="matching length"):
        answer_correctness_reward(prompts=["p"], completions=["1"], answer=[])


def test_reward_fns_from_names():
    from open_r1_tpu.grpo.rewards import (
        DEFAULT_REWARD_FNS,
        answer_correctness_reward,
        reward_fns_from_names,
    )

    assert reward_fns_from_names(None) == DEFAULT_REWARD_FNS
    assert reward_fns_from_names(["answer_correctness_reward"]) == (
        answer_correctness_reward,
    )
    with pytest.raises(ValueError, match="Unknown reward"):
        reward_fns_from_names(["nope"])
