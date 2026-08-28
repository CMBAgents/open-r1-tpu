"""Versions and build inputs that define the supported evaluation environment.

Keep this module dependency-free: the container and setup shell scripts import
it through ``PYTHONPATH=src`` before the project environment necessarily
exists. Direct Python requirements are repeated in ``pyproject.toml`` and the
unit suite checks that the two declarations cannot drift apart; ``uv.lock``
then freezes the complete transitive environment.
"""

from hashlib import sha256
from pathlib import Path

EVALUATION_PYTHON_VERSION = "3.13.14"

EVALUATION_PACKAGE_VERSIONS = {
    "datasets": "5.0.1",
    "huggingface-hub": "1.28.0",
    "langfuse": "4.14.5",
    "latex2sympy2-extended": "1.0.6",
    "lighteval": "0.13.0",
    # No litellm: evaluation.runner reaches vLLM directly over openai, never
    # through litellm -- see the eval extra's own comment in pyproject.toml.
    "openai": "2.54.0",
    "pyarrow": "25.0.1",
    "xxhash": "3.8.1",
}

# The only remote image reference: the digest pins the Debian 12/Python 3.12
# base that the local service image is built from.
VLLM_TPU_BASE_IMAGE = (
    "python:3.12-slim-bookworm@"
    "sha256:a116514e19457bcb7af7efe9c3dd0b9b71e85b317694e7882a1c52aa15a78134"
)
VLLM_TPU_IMAGE_NAME = "open-r1-tpu-vllm"
VLLM_TPU_SERVICE_VERSIONS = {
    "vllm-tpu": "0.27.0",
    "tpu-inference": "0.27.0",
}


def vllm_tpu_image_tag(
    dockerfile: str | Path | None = None,
    lockfile: str | Path | None = None,
    patches: str | Path | None = None,
) -> str:
    """Derive the local image tag from the committed build inputs.

    The raw Dockerfile bytes precede the raw lockfile bytes without a separator,
    matching ``sha256(Dockerfile || vllm-tpu.lock)``. The build-time patches
    follow, each contributing its file name and then its bytes, in name order,
    so that editing, adding, removing, or renaming a patch yields a new tag and
    the wrapper refuses the image built before the change. Optional paths keep
    this function easy to exercise without Docker or the project environment.
    """
    repository_root = Path(__file__).resolve().parents[3]
    dockerfile_path = Path(dockerfile or repository_root / "docker/vllm-tpu/Dockerfile")
    lockfile_path = Path(lockfile or repository_root / "docker/vllm-tpu/vllm-tpu.lock")
    patches_path = Path(patches or repository_root / "docker/vllm-tpu/patches")
    digest = sha256(dockerfile_path.read_bytes() + lockfile_path.read_bytes())
    for patch in sorted(patches_path.glob("*.py")):
        digest.update(patch.name.encode())
        digest.update(patch.read_bytes())
    return f"{VLLM_TPU_IMAGE_NAME}:{digest.hexdigest()[:12]}"
