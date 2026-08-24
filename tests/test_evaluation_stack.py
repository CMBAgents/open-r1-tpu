import os
import subprocess
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

from open_r1_tpu.evaluation.stack import (
    EVALUATION_PACKAGE_VERSIONS,
    EVALUATION_PYTHON_VERSION,
    VLLM_TPU_IMAGE,
)

REPO_ROOT = Path(__file__).parents[1]


def test_python_pin_matches_the_interpreter_file():
    assert (REPO_ROOT / ".python-version").read_text().strip() == (
        EVALUATION_PYTHON_VERSION
    )


def test_eval_extra_matches_the_validated_direct_package_pins():
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    requirements = {
        Requirement(value).name: Requirement(value)
        for value in pyproject["project"]["optional-dependencies"]["eval"]
    }

    for name, expected in EVALUATION_PACKAGE_VERSIONS.items():
        requirement = requirements[name]
        assert str(requirement.specifier) == f"=={expected}"


def test_vllm_image_names_a_release_and_an_immutable_digest():
    repository, digest = VLLM_TPU_IMAGE.rsplit("@sha256:", maxsplit=1)

    assert repository.endswith(":v0.27.0")
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_container_wrapper_reports_the_same_pin():
    script = REPO_ROOT / "scripts/run_vllm_tpu_container.sh"

    assert os.access(script, os.X_OK)
    completed = subprocess.run(
        [str(script), "--print-image"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == VLLM_TPU_IMAGE


def test_container_wrapper_reserves_an_unused_cid_path_and_hides_the_token(tmp_path):
    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s\\n' "$*" >> "$FAKE_DOCKER_LOG"
case "$1" in
  info) exit 0 ;;
  container) exit 1 ;;
  run)
    shift
    cid_file=""
    while [[ $# -gt 0 ]]; do
      if [[ "$1" == --cidfile ]]; then
        cid_file="$2"
        shift 2
      else
        shift
      fi
    done
    [[ -n "$cid_file" && ! -e "$cid_file" ]]
    printf 'fake-container-id\\n' > "$cid_file"
    exit 0
    ;;
  stop) exit 0 ;;
  *) exit 0 ;;
esac
"""
    )
    fake_docker.chmod(0o755)
    model = tmp_path / "model"
    model.mkdir()
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "FAKE_DOCKER_LOG": str(docker_log),
        "HF_TOKEN": "must-not-appear-in-command",
    }

    subprocess.run(
        [
            str(REPO_ROOT / "scripts/run_vllm_tpu_container.sh"),
            "--",
            str(model),
            "--port",
            "8000",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    logged = docker_log.read_text()
    assert "run --rm" in logged
    assert "--env HF_TOKEN" in logged
    assert "must-not-appear-in-command" not in logged
    assert "stop --time 30 fake-container-id" in logged
