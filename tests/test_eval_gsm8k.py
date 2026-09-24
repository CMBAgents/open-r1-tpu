import importlib.util
from pathlib import Path

import pytest

from open_r1_tpu.core.config import load_config

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "eval_gsm8k.py"
SPEC = importlib.util.spec_from_file_location("eval_gsm8k", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
eval_gsm8k = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(eval_gsm8k)

SFT_RECIPE = "recipes/rowanai/sft/config_rowanai_gsm8k_sft.yaml"
FORMAT_RECIPE = "recipes/rowanai/sft/config_rowanai_gsm8k_format.yaml"


@pytest.mark.parametrize(
    ("completion", "gold", "correct", "line_correct"),
    [
        ("48 / 2 = 24.\n24 + 48 = 72.\nAnswer: 72", "72", True, True),
        ("It costs $1,000 in all.\nAnswer: 1,000", "1000", True, True),
        ("Answer: $5", "5", True, True),
        # Right number last, but no answer line: only the lenient reading.
        ("So she needs 5 more dollars.", "5", True, False),
        # An answer line that is not the last line does not count as one.
        ("Answer: 5\nThat is 12 in all.", "5", False, False),
        ("Answer: 6", "5", False, False),
        ("Answer: 5 loaves", "5", True, False),
        ("", "5", False, False),
    ],
)
def test_score_reads_final_number_and_answer_line(
    completion, gold, correct, line_correct
):
    result = eval_gsm8k.score(completion, gold)
    assert result["correct"] is correct
    assert result["answer_line_correct"] is line_correct


def test_wilson_interval_stays_in_bounds():
    assert eval_gsm8k.wilson_interval(0, 0) == (0.0, 1.0)
    low, high = eval_gsm8k.wilson_interval(0, 829)
    assert low == 0.0 and 0.0 < high < 0.01
    low, high = eval_gsm8k.wilson_interval(829, 829)
    assert 0.99 < low < 1.0 and high == 1.0
    low, high = eval_gsm8k.wilson_interval(83, 829)
    assert low < 83 / 829 < high


def test_eval_prompt_uses_the_recipe_system_prompt():
    system_prompt = eval_gsm8k.system_prompt_of(SFT_RECIPE)
    assert system_prompt is not None and "last line" in system_prompt
    messages = eval_gsm8k.messages_for("How many?", system_prompt)
    assert messages == [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": "How many?"},
    ]
    assert eval_gsm8k.messages_for("How many?", None) == [
        {"role": "user", "content": "How many?"}
    ]


def test_gsm8k_sft_continues_from_the_format_sft_with_its_prompt_and_template():
    sft = load_config(SFT_RECIPE)
    fmt = load_config(FORMAT_RECIPE)
    assert sft["model"]["model_path"] == fmt["export"]["output_dir"]
    assert sft["tokenizer"] == fmt["tokenizer"]
    assert sft["dataset"]["system_prompt_file"] == fmt["dataset"]["system_prompt_file"]
    assert not sft["dataset"]["require_reasoning_tags"]
    assert sft["export"]["enabled"]
    outputs = {
        sft["export"]["output_dir"],
        sft["training"]["checkpoint_dir"],
        fmt["export"]["output_dir"],
    }
    assert len(outputs) == 3


def test_qwen_control_matches_the_rowanai_arm_except_model_and_outputs():
    rowanai = load_config(SFT_RECIPE)
    qwen = load_config("recipes/rowanai/sft/config_qwen_base_gsm8k_sft.yaml")
    assert qwen["dataset"] == rowanai["dataset"]
    assert qwen["optimizer"] == rowanai["optimizer"]
    for key in ("max_steps", "gradient_accumulation_steps", "eval_every_n_steps"):
        assert qwen["training"][key] == rowanai["training"][key]
    assert qwen["model"]["model_path"] == "models/Qwen2.5-1.5B"
    assert qwen["tokenizer"]["chat_template"] is None
    assert "stop_strings" not in qwen["export"]
    paths = [
        (arm["training"]["checkpoint_dir"], arm["export"]["output_dir"])
        for arm in (rowanai, qwen)
    ]
    assert not set(paths[0]) & set(paths[1])
