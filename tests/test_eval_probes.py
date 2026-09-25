import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "eval_probes.py"
SPEC = importlib.util.spec_from_file_location("eval_probes", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
probes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probes)


def word_encode(text):
    """A toy tokeniser: one id per space-led word, like a byte-level BPE."""
    vocab = {}
    return [
        vocab.setdefault(word, len(vocab) + 1)
        for word in text.replace(" ", "\0 ").split("\0")
        if word
    ]


def test_option_span_starts_after_the_context():
    ids, start, merged = probes.option_span(word_encode, "2, 4, 6, 8,", " 10")
    assert start == 4
    assert len(ids) == 5
    assert merged is False


def test_option_span_flags_a_token_merged_across_the_boundary():
    # A character tokeniser that fuses "a" followed by "b" into one token.
    def encode(text):
        out, i = [], 0
        while i < len(text):
            if text[i : i + 2] == "ab":
                out.append(100)
                i += 2
            else:
                out.append(ord(text[i]))
                i += 1
        return out

    ids, start, merged = probes.option_span(encode, "xa", "b")
    assert ids == [ord("x"), 100]
    assert start == 1
    assert merged is True


@pytest.mark.parametrize(
    ("text", "gold", "correct"),
    [
        (" 15\n20, 25", "15", True),
        (" 15.", "15", True),
        (" 150", "15", False),
        (" 1,760 yards", "1760", True),
        (" about 15", "15", False),
        ("\n15", "15", False),
        (" daughter.\nHot is to", "daughter", True),
        (" daughters", "daughter", False),
        (" Daughter", "daughter", True),
        (" conducts heat.\nAll", "conducts heat", True),
        (" conducts light", "conducts heat", False),
        (" Yes.", "Yes", True),
        (" No.", "Yes", False),
        (" Tom. Who is", "Tom", True),
    ],
)
def test_generation_correct(text, gold, correct):
    assert probes.generation_correct(text, gold) is correct


def row(id_, task, family, answer, n_options=2, generate=True):
    return {
        "id": id_,
        "task": task,
        "family": family,
        "options": [f" o{i}" for i in range(n_options)],
        "answer": answer,
        "gold": f"o{answer}",
        "generate": generate,
    }


def test_score_row_picks_by_sum_and_by_length_normalised_sum():
    probe = {**row("a-01", "a", "f", 0), "options": [" x", " longer option"]}
    # Sum prefers the short option; per byte, the long one.
    record = probes.score_row(probe, [-2.0, -6.0], " x")
    assert record["pred"] == 0 and record["correct"] is True
    assert record["pred_norm"] == 1 and record["correct_norm"] is False
    assert record["gen_correct"] is False  # gold "o0", not "x"


def test_score_row_reports_when_the_middle_name_is_picked():
    probe = {**row("o-01", "ordering", "top", 0, 3), "middle": " o1"}
    record = probes.score_row(probe, [-3.0, -1.0, -2.0], None)
    assert record["picked_middle"] is True
    assert record["gen_correct"] is None


def test_summarise_counts_per_task_and_family_with_chance():
    rows = [
        row("a-01", "a", "f1", 0, 2),
        row("a-02", "a", "f2", 1, 2),
        row("b-01", "b", "g", 0, 4, generate=False),
    ]
    records = [
        probes.score_row(rows[0], [-1.0, -2.0], " o0"),
        probes.score_row(rows[1], [-1.0, -2.0], " o0"),
        probes.score_row(rows[2], [-1.0, -2.0, -3.0, -4.0], None),
    ]
    summary = probes.summarise(records, rows)
    assert summary["tasks"]["a"]["acc"]["correct"] == 1
    assert summary["tasks"]["a"]["chance"] == pytest.approx(0.5)
    assert summary["tasks"]["a"]["families"]["f2"]["acc"]["correct"] == 0
    assert summary["tasks"]["a"]["gen"]["correct"] == 1
    assert summary["tasks"]["b"]["gen"] is None
    assert summary["overall"]["n"] == 3
    assert summary["overall"]["chance"] == pytest.approx((0.5 + 0.5 + 0.25) / 3)


def test_padded_width_rounds_up_to_the_next_bucket():
    assert probes.padded_width(1) == 64
    assert probes.padded_width(64) == 64
    assert probes.padded_width(65) == 128
    with pytest.raises(ValueError):
        probes.padded_width(5000)


def test_generation_accepts_any_listed_wording():
    probe = {
        **row("g-01", "geography", "changed", 0, 4),
        "gold": "St Petersburg",
        "accept": ["St Petersburg", "St. Petersburg", "Petersburg"],
    }
    assert probes.score_row(probe, [-1.0] * 4, " St. Petersburg, on the")["gen_correct"]
    assert not probes.score_row(probe, [-1.0] * 4, " Moscow.")["gen_correct"]


def test_score_row_reads_the_gold_options_first_token_rank():
    probe = row("a-01", "a", "f", 1, 3)
    record = probes.score_row(probe, [-1.0, -2.0, -3.0], None, [0, 4, 9])
    assert record["gold_rank"] == 4


def text_row(id_, family, text):
    return {
        "id": id_,
        "task": "prose",
        "family": family,
        "kind": "text",
        "context": "A",
        "options": [text],
        "answer": 0,
        "gold": text,
        "generate": False,
    }


def test_text_row_reports_bits_per_byte():
    record = probes.score_row(text_row("p-01", "f", " abcd"), [-math.log(2) * 10], None)
    assert record["bytes"] == 5
    assert record["bits_per_byte"] == pytest.approx(2.0)
    assert "correct" not in record


def test_summarise_keeps_text_rows_out_of_accuracy_and_overall():
    rows = [
        {**row("c-01", "choice", "f", 0, 2)},
        {**row("c-02", "choice", "f", 0, 2)},
        text_row("p-01", "old", " abcd"),
        text_row("p-02", "old", " efghijklm"),
    ]
    records = [
        probes.score_row(rows[0], [-1.0, -2.0], None, [0, 3]),
        probes.score_row(rows[1], [-2.0, -1.0], None, [7, 0]),
        probes.score_row(rows[2], [-math.log(2) * 5], None),
        probes.score_row(rows[3], [-math.log(2) * 9], None),
    ]
    summary = probes.summarise(records, rows)
    choice = summary["tasks"]["choice"]
    assert choice["top1"]["correct"] == 1 and choice["top5"]["correct"] == 1
    prose = summary["tasks"]["prose"]
    assert "acc" not in prose
    # 14 bits over 5 + 10 bytes, pooled rather than averaged per row.
    assert prose["bits_per_byte"] == pytest.approx(14 / 15)
    assert summary["overall"]["n"] == 2
