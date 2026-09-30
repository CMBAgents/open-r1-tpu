"""Versions and build inputs that define the supported evaluation environment.

Standard library only: shell scripts import this through `PYTHONPATH=src`
before the project environment exists. The package pins are repeated in
`pyproject.toml`'s eval extra, and a test keeps the two in step.
"""

from hashlib import sha256
from importlib import metadata
from pathlib import Path

EVALUATION_PYTHON_VERSION = "3.13.14"

EVALUATION_PACKAGE_VERSIONS = {
    "datasets": "5.0.1",
    "huggingface-hub": "1.28.0",
    "langfuse": "4.14.5",
    "latex2sympy2-extended": "1.0.6",
    "lighteval": "0.13.0",
    # No litellm: evaluation.generate reaches vLLM directly through openai.
    "openai": "2.54.0",
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


def installed_version(distribution: str) -> str:
    """The installed version of `distribution`, or "unknown" when absent."""
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def vllm_tpu_image_tag(
    dockerfile: str | Path | None = None,
    lockfile: str | Path | None = None,
    patches: str | Path | None = None,
) -> str:
    """Derive the local image tag from the committed build inputs:
    `sha256(Dockerfile || vllm-tpu.lock)`, then each build-time patch's name
    and bytes in name order, so any change to a patch yields a new tag and the
    wrapper refuses an image built before it.
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
