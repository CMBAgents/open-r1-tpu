"""`scripts/run_eval_tpu.sh` has no default recipe: an expensive run must name
its tier on purpose. `RECIPE` validation runs before anything Docker- or
TPU-related, so it is safe to exercise with no server up. The happy-path tests
stub `python3` in a copied `scripts/` directory, so they too need neither
Docker nor a live server: with SKIP_SERVER=1 the real script never reaches
anything that does, except the one test of the server's lifecycle.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "run_eval_tpu.sh"
SERVER_HELPER_PATH = SCRIPT_PATH.parent / "lib" / "vllm_server.sh"


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


def _stubbed_scripts_dir(tmp_path, capture_file):
    """A copy of scripts/run_eval_tpu.sh and the helper it sources alongside a
    stand-in `python3`, so the real script runs unmodified but never needs
    Docker, a real recipe, or a live server.
    """
    scripts_dir = tmp_path / "scripts"
    (scripts_dir / "lib").mkdir(parents=True)
    copy = scripts_dir / "run_eval_tpu.sh"
    shutil.copy(SCRIPT_PATH, copy)
    copy.chmod(0o755)
    shutil.copy(SERVER_HELPER_PATH, scripts_dir / "lib" / SERVER_HELPER_PATH.name)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        f'#!/usr/bin/env bash\nprintf "%s\\n" "$@" >> "{capture_file}"\n'
    )
    fake_python.chmod(0o755)
    return scripts_dir, bin_dir


def test_runs_the_evaluation_entry_point_with_tracing_config(tmp_path):
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
        "open_r1_tpu.evaluation.run",
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


def _run_stubbed(tmp_path, *args, trace_config=None):
    capture_file = tmp_path / "argv.txt"
    scripts_dir, bin_dir = _stubbed_scripts_dir(tmp_path, capture_file)
    env = {
        "PATH": f"{bin_dir}:/usr/bin:/bin",
        "RECIPE": "recipes/fake/eval/tier0.yaml",
        "SKIP_SERVER": "1",
    }
    if trace_config is not None:
        env["TRACE_CONFIG"] = trace_config
    completed = subprocess.run(
        ["bash", str(scripts_dir / "run_eval_tpu.sh"), *args],
        capture_output=True,
        text=True,
        env=env,
        cwd=tmp_path,
    )
    assert completed.returncode == 0, completed.stderr
    return capture_file.read_text().splitlines()


def test_without_a_trace_config_the_run_is_local(tmp_path):
    assert _run_stubbed(tmp_path, "reporting.wandb.enabled=false") == [
        "-m",
        "open_r1_tpu.evaluation.run",
        "--config",
        "recipes/fake/eval/tier0.yaml",
        "reporting.wandb.enabled=false",
    ]


def test_an_empty_trace_config_is_treated_as_missing(tmp_path):
    argv = _run_stubbed(tmp_path, trace_config="")
    assert "--tracing-config" not in argv


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs Linux setsid")
def test_starts_the_recipes_server_and_stops_it_afterwards(tmp_path):
    # The stubbed evaluation.server prints a stand-in server command; the real
    # script starts it in its own process group and must stop it on exit.
    capture_file = tmp_path / "argv.txt"
    scripts_dir, bin_dir = _stubbed_scripts_dir(tmp_path, capture_file)
    server = "sleep 61.5"
    (bin_dir / "python3").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" >> "{capture_file}"\n'
        f'[[ "$2" == open_r1_tpu.evaluation.server ]] && echo "{server}"\n'
        "exit 0\n"
    )

    completed = subprocess.run(
        ["bash", str(scripts_dir / "run_eval_tpu.sh"), "server.port=8123"],
        capture_output=True,
        text=True,
        env={"PATH": f"{bin_dir}:/usr/bin:/bin", "RECIPE": "r.yaml"},
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"Starting: {server}" in completed.stderr
    argv = capture_file.read_text().splitlines()
    # Both halves read the same recipe and overrides.
    assert argv == [
        "-m",
        "open_r1_tpu.evaluation.server",
        "--config",
        "r.yaml",
        "server.port=8123",
        "-m",
        "open_r1_tpu.evaluation.run",
        "--config",
        "r.yaml",
        "server.port=8123",
    ]
    leftover = subprocess.run(["pgrep", "-f", server], capture_output=True)
    assert leftover.returncode == 1, "the server outlived the script"
