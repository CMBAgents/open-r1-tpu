"""`scripts/run_eval_tpu.sh` has no default recipe: an expensive run must name
its tier on purpose. `RECIPE`/`TRACE_CONFIG` validation runs before anything
Docker- or TPU-related, so it is safe to exercise from a laptop with no
server up. The happy-path tests stub `python3` in a copied `scripts/`
directory, so they too need neither Docker nor a live server: with
SKIP_SERVER=1 the real script never reaches anything that does.
"""

import shutil
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


def test_a_missing_trace_config_errors_before_launching_anything():
    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "RECIPE": "recipes/fake/eval/tier0.yaml",
        },
        cwd=SCRIPT_PATH.parents[1],
    )

    assert completed.returncode == 1
    assert "TRACE_CONFIG" in completed.stderr


def test_an_empty_trace_config_is_treated_as_missing():
    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "RECIPE": "recipes/fake/eval/tier0.yaml",
            "TRACE_CONFIG": "",
        },
        cwd=SCRIPT_PATH.parents[1],
    )

    assert completed.returncode == 1
    assert "TRACE_CONFIG" in completed.stderr


def _stubbed_scripts_dir(tmp_path, capture_file):
    """A copy of scripts/run_eval_tpu.sh alongside a stand-in `python3`, so
    the real script runs unmodified but never needs Docker, a real recipe,
    or a live server.
    """
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    copy = scripts_dir / "run_eval_tpu.sh"
    shutil.copy(SCRIPT_PATH, copy)
    copy.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "{capture_file}"\n'
    )
    fake_python.chmod(0o755)
    return scripts_dir, bin_dir


def test_runs_the_experiment_entry_point_with_tracing_config(tmp_path):
    capture_file = tmp_path / "argv.txt"
    scripts_dir, bin_dir = _stubbed_scripts_dir(tmp_path, capture_file)
    tracing_config = tmp_path / "tracing.yaml"
    tracing_config.write_text("# unused; python3 is stubbed\n")

    completed = subprocess.run(
        ["bash", str(scripts_dir / "run_eval_tpu.sh")],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "RECIPE": "recipes/fake/eval/tier0.yaml",
            "SKIP_SERVER": "1",
            "TRACE_CONFIG": str(tracing_config),
        },
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    argv_lines = capture_file.read_text().splitlines()
    assert argv_lines == [
        "-m",
        "open_r1_tpu.evaluation.experiment",
        "--config",
        "recipes/fake/eval/tier0.yaml",
        "--tracing-config",
        str(tracing_config),
    ]


def test_forwards_overrides_after_tracing_config(tmp_path):
    capture_file = tmp_path / "argv.txt"
    scripts_dir, bin_dir = _stubbed_scripts_dir(tmp_path, capture_file)
    tracing_config = tmp_path / "tracing.yaml"
    tracing_config.write_text("# unused; python3 is stubbed\n")

    completed = subprocess.run(
        ["bash", str(scripts_dir / "run_eval_tpu.sh"), "reporting.wandb.enabled=false"],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "RECIPE": "recipes/fake/eval/tier0.yaml",
            "SKIP_SERVER": "1",
            "TRACE_CONFIG": str(tracing_config),
        },
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    argv_lines = capture_file.read_text().splitlines()
    assert argv_lines[-1] == "reporting.wandb.enabled=false"
