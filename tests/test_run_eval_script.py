"""`scripts/run_eval_tpu.sh` has no default recipe: an expensive run must name
its tier on purpose. The usage-and-exit path runs before anything Docker- or
TPU-related, so it is safe to exercise from a laptop with no server up.
"""

import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_eval_tpu.sh"


def test_the_script_is_syntactically_valid():
    completed = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr


def test_a_missing_recipe_prints_usage_and_exits_nonzero():
    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin"},
        cwd=SCRIPT_PATH.parents[1],
    )

    assert completed.returncode == 1
    assert "RECIPE=" in completed.stderr


def test_an_empty_recipe_is_treated_as_missing():
    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "RECIPE": ""},
        cwd=SCRIPT_PATH.parents[1],
    )

    assert completed.returncode == 1
    assert "RECIPE=" in completed.stderr
