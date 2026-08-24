import os
import re
import subprocess
import tomllib
from pathlib import Path

from packaging.requirements import Requirement

from open_r1_tpu.evaluation.stack import (
    EVALUATION_PACKAGE_VERSIONS,
    EVALUATION_PYTHON_VERSION,
    VLLM_TPU_BASE_IMAGE,
    VLLM_TPU_IMAGE_NAME,
    vllm_tpu_image_tag,
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


def test_vllm_image_tag_is_derived_and_the_base_keeps_its_digest():
    image = vllm_tpu_image_tag()
    name, tag = image.rsplit(":", maxsplit=1)
    _, digest = VLLM_TPU_BASE_IMAGE.rsplit("@sha256:", maxsplit=1)

    assert name == VLLM_TPU_IMAGE_NAME
    assert len(tag) == 12
    assert set(tag) <= set("0123456789abcdef")
    assert len(digest) == 64
    assert set(digest) <= set("0123456789abcdef")


def test_vllm_image_tag_is_stable_and_changes_with_each_build_input(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    lockfile = tmp_path / "vllm-tpu.lock"
    dockerfile.write_text("FROM python\n")
    lockfile.write_text("package==1 --hash=sha256:abc\n")

    original = vllm_tpu_image_tag(dockerfile, lockfile)
    assert vllm_tpu_image_tag(dockerfile, lockfile) == original

    dockerfile.write_text("FROM python:3.12\n")
    assert vllm_tpu_image_tag(dockerfile, lockfile) != original
    dockerfile.write_text("FROM python\n")
    lockfile.write_text("package==2 --hash=sha256:def\n")
    assert vllm_tpu_image_tag(dockerfile, lockfile) != original


def test_docker_build_spec_uses_the_pinned_base_and_complete_hash_lock():
    docker_dir = REPO_ROOT / "docker/vllm-tpu"
    dockerfile = (docker_dir / "Dockerfile").read_text()
    requirement = (docker_dir / "vllm-tpu.in").read_text()
    lock = (docker_dir / "vllm-tpu.lock").read_text()

    assert f"FROM {VLLM_TPU_BASE_IMAGE}" in dockerfile
    assert requirement == "vllm-tpu==0.27.0\n"
    assert len(re.findall(r"^[a-z0-9][a-z0-9._-]*==[^ ]+", lock, re.MULTILINE)) == 241
    assert "vllm-tpu==0.27.0" in lock
    assert "tpu-inference==0.27.0" in lock
    assert "--hash=sha256:" in lock


def test_container_wrapper_reports_the_derived_tag():
    script = REPO_ROOT / "scripts/run_vllm_tpu_container.sh"

    assert os.access(script, os.X_OK)
    completed = subprocess.run(
        [str(script), "--print-image"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == vllm_tpu_image_tag()


def test_container_wrapper_refuses_a_stale_local_tag_before_using_docker():
    completed = subprocess.run(
        [
            str(REPO_ROOT / "scripts/run_vllm_tpu_container.sh"),
            "--image",
            f"{VLLM_TPU_IMAGE_NAME}:stale",
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "does not match this build spec" in completed.stderr
    assert vllm_tpu_image_tag() in completed.stderr


def test_container_wrapper_builds_the_derived_tag_from_the_committed_context(tmp_path):
    docker_log = tmp_path / "docker.log"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'printf \'%s\\n\' "$*" >> "$FAKE_DOCKER_LOG"\n'
        'case "$1" in\n'
        "  info|build) exit 0 ;;\n"
        "  *) exit 1 ;;\n"
        "esac\n"
    )
    fake_docker.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "FAKE_DOCKER_LOG": str(docker_log),
    }

    subprocess.run(
        [str(REPO_ROOT / "scripts/run_vllm_tpu_container.sh"), "--build"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    build_command = docker_log.read_text()
    assert f"build --tag {vllm_tpu_image_tag()}" in build_command
    assert str(REPO_ROOT / "docker/vllm-tpu") in build_command


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
