"""The GRPO eval-rollout recorder, which needs no Tunix."""

import json

import numpy as np

from open_r1_tpu.grpo.rewards import DEFAULT_REWARD_FNS
from open_r1_tpu.grpo.run import build_rollout_recorder


def test_rollout_recorder_writes_one_line_per_completion(tmp_path):
    path = tmp_path / "nested" / "eval_rollouts.jsonl"
    record = build_rollout_recorder(str(path), DEFAULT_REWARD_FNS)
    prompts = ["p1", "p2"]
    completions = ["<think>2+2</think>\\boxed{4}", "no idea"]
    n = record(
        prompts,
        completions,
        [6.0, -2.5],
        100,
        "eval",
        question=["q1", "q2"],
        answer=["4", "7"],
        not_a_column=3,
    )
    assert n == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["step"] for r in rows] == [100, 100]
    assert rows[0]["mode"] == "eval"
    assert rows[0]["question"] == "q1" and rows[0]["answer"] == "4"
    assert rows[0]["completion"] == completions[0]
    assert rows[0]["reward"] == 6.0
    assert rows[0]["rewards"] == {"format_reward": 3.0, "correctness_reward": 3.0}
    assert rows[1]["rewards"]["correctness_reward"] == 0.0
    assert "not_a_column" not in rows[0]
    # Appends on the next call rather than overwriting.
    record(prompts[:1], completions[:1], [6.0], 200, "eval", answer=["4"])
    assert len(path.read_text().splitlines()) == 3


def test_rollout_recorder_accepts_numpy_columns(tmp_path):
    """Tunix passes dataset columns as numpy arrays, not lists."""
    path = tmp_path / "eval_rollouts.jsonl"
    record = build_rollout_recorder(str(path), DEFAULT_REWARD_FNS)
    n = record(
        np.array(["p1", "p2"]),
        np.array(["<think>2+2</think>\\boxed{4}", "no idea"]),
        [6.0, -2.5],
        0,
        "eval",
        question=np.array(["q1", "q2"]),
        answer=np.array(["4", "7"]),
        a_scalar=np.int64(3),
    )
    assert n == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["answer"] == "4" and rows[0]["question"] == "q1"
    assert rows[0]["rewards"]["correctness_reward"] == 3.0
    assert rows[1]["rewards"]["correctness_reward"] == 0.0
    assert "a_scalar" not in rows[0]
