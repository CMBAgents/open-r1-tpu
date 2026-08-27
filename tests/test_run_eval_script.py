"""`scripts/run_eval_tpu.sh` has no default recipe: an expensive run must name
its tier on purpose. The usage-and-exit path runs before anything Docker- or
TPU-related, so it is safe to exercise from a laptop with no server up.

The TRACE_PROXY=1 tests below stub `python3` and `scripts/run_trace_proxy.sh`
in a copied scripts/ directory, so they too need neither Docker nor a live
server: with SKIP_SERVER=1 the real script never reaches anything that does.
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


def test_trace_proxy_without_trace_config_errors_before_launching_anything():
    completed = subprocess.run(
        ["bash", str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "RECIPE": "recipes/fake/eval/tier0.yaml",
            "TRACE_PROXY": "1",
        },
        cwd=SCRIPT_PATH.parents[1],
    )

    assert completed.returncode == 1
    assert "TRACE_CONFIG" in completed.stderr


def _stubbed_scripts_dir(tmp_path, capture_file):
    """A copy of scripts/run_eval_tpu.sh alongside stand-ins for `python3`
    and `run_trace_proxy.sh`, so the real script runs unmodified but never
    needs Docker, a real recipe, or a live server.
    """
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    copy = scripts_dir / "run_eval_tpu.sh"
    shutil.copy(SCRIPT_PATH, copy)
    copy.chmod(0o755)

    fake_proxy = scripts_dir / "run_trace_proxy.sh"
    fake_proxy.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_proxy.chmod(0o755)

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_python = bin_dir / "python3"
    fake_python.write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n" "$@" >> "{capture_file}"\n'
        'if [[ "$1" == "-m" && "$2" == "open_r1_tpu.tracing.config" ]]; then\n'
        "  echo 'export TRACE_PROXY_PORT=4000'\n"
        "  echo 'export TRACE_GCS_BUCKET=fake-bucket'\n"
        "  echo 'export TRACE_GCS_PREFIX=traces/tier0/fake'\n"
        "  echo 'export TRACE_PROXY_UPSTREAM_BASE_URL=http://127.0.0.1:9/v1'\n"
        "fi\n"
    )
    fake_python.chmod(0o755)
    return scripts_dir, bin_dir


def test_trace_proxy_appends_exactly_one_base_url_override(tmp_path):
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
            "TRACE_PROXY": "1",
            "TRACE_CONFIG": str(tracing_config),
            # The upstream readiness wait would poll the stub URL forever;
            # 0 is the documented skip for callers that know the server is up.
            "TRACE_UPSTREAM_WAIT_SECS": "0",
        },
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    argv_lines = capture_file.read_text().splitlines()
    base_url_args = [line for line in argv_lines if line.startswith("server.base_url=")]
    assert base_url_args == ["server.base_url=http://127.0.0.1:4000/v1"]


def test_trace_proxy_gates_on_an_unready_upstream(tmp_path):
    """With the wait enabled and vLLM unreachable (the stub exports an
    upstream URL on a closed port), the script must fail before starting the
    proxy or the harness -- the proxy's static /v1/models would otherwise
    defeat the harness's own readiness gate.
    """
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
            "TRACE_PROXY": "1",
            "TRACE_CONFIG": str(tracing_config),
            "TRACE_UPSTREAM_WAIT_SECS": "1",
        },
        cwd=tmp_path,
    )

    assert completed.returncode == 1
    assert "not ready" in completed.stderr
    argv_lines = capture_file.read_text().splitlines()
    assert not any(line.startswith("server.base_url=") for line in argv_lines)


def test_trace_proxy_unset_forwards_no_base_url_override(tmp_path):
    capture_file = tmp_path / "argv.txt"
    scripts_dir, bin_dir = _stubbed_scripts_dir(tmp_path, capture_file)

    completed = subprocess.run(
        ["bash", str(scripts_dir / "run_eval_tpu.sh")],
        capture_output=True,
        text=True,
        env={
            "PATH": f"{bin_dir}:/usr/bin:/bin",
            "RECIPE": "recipes/fake/eval/tier0.yaml",
            "SKIP_SERVER": "1",
        },
        cwd=tmp_path,
    )

    assert completed.returncode == 0, completed.stderr
    argv_lines = capture_file.read_text().splitlines()
    assert not any(line.startswith("server.base_url=") for line in argv_lines)
