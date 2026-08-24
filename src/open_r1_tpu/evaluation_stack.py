"""Immutable versions that define the supported evaluation environment.

Keep this module dependency-free: the container and setup shell scripts import
it through ``PYTHONPATH=src`` before the project environment necessarily
exists. Direct Python requirements are repeated in ``pyproject.toml`` and the
unit suite checks that the two declarations cannot drift apart; ``uv.lock``
then freezes the complete transitive environment.
"""

EVALUATION_PYTHON_VERSION = "3.13.14"

EVALUATION_PACKAGE_VERSIONS = {
    "datasets": "5.0.1",
    "huggingface-hub": "1.28.0",
    "latex2sympy2-extended": "1.0.6",
    "lighteval": "0.13.0",
    "litellm": "1.97.0",
    "pyarrow": "25.0.1",
    "xxhash": "3.8.1",
}

# Tag plus registry digest: the readable tag identifies the published TPU
# image release, while the digest prevents the tag from resolving to different
# bytes later. The digest, not internal package metadata, is authoritative.
VLLM_TPU_IMAGE = (
    "docker.io/vllm/vllm-tpu:v0.27.0@"
    "sha256:d6748bc7b1b020ab6411506d4bf30f8bfabb5db2b8505328f26d1a545b479df8"
)
