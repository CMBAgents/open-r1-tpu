"""The vLLM server: its command line, its image provenance, and readiness.

The recipe is the one source of truth for the port, the served model name and
the context window: `scripts/run_eval_tpu.sh` starts the server with the
command this module prints and the harness reads the same recipe. vLLM is
never imported here -- it runs in the pinned container and is reached over
HTTP.

    python -m open_r1_tpu.evaluation.server \
      --config recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from open_r1_tpu.core.cli import parse_recipe_args, recipe_parser
from open_r1_tpu.evaluation.config import (
    DEFAULT_SERVE_COMMAND,
    load_eval_config,
    resolve_settings,
    uses_container_wrapper,
)

LOGGER = logging.getLogger(__name__)


def vllm_serve_command(settings: Mapping[str, Any]) -> list[str]:
    """The server invocation for this recipe. `server.serve_command` supplies
    everything up to the model path, since the server runs outside this
    environment (tpu-inference does not support this project's Python).
    """
    command = [*settings.get("serve_command", DEFAULT_SERVE_COMMAND)]
    if settings.get("server_image"):
        # The wrapper owns the Docker arguments; `--` passes the rest to vLLM.
        command += ["--image", str(settings["server_image"]), "--"]
    command += [
        str(settings["model_path"]),
        "--served-model-name",
        str(settings["served_model_name"]),
        "--host",
        str(settings["host"]),
        "--port",
        str(settings["port"]),
        "--tensor-parallel-size",
        str(settings.get("tensor_parallel_size", 1)),
        # Greedy output must not depend on server cache state: a prefix-cache
        # hit shortens the prefill, changing the attention kernel's shape and
        # its bf16 accumulation order, which flips argmax at near-tied logits.
        "--no-enable-prefix-caching",
    ]
    if settings.get("max_model_len"):
        command += ["--max-model-len", str(settings["max_model_len"])]
    command += list(settings.get("server_extra_args", []))
    return command


def container_image_provenance(settings: Mapping[str, Any]) -> dict[str, Any] | None:
    """The image ID and service versions, read through the container wrapper,
    or None when the server is not the wrapper's image. Runs Python in the
    built image only: it neither starts vLLM nor reserves the TPU.
    """
    image = settings.get("server_image")
    command = [
        str(part) for part in settings.get("serve_command", DEFAULT_SERVE_COMMAND)
    ]
    if image is None or not uses_container_wrapper(command):
        return None

    completed = subprocess.run(
        [*command, "--image", str(image), "--provenance"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise RuntimeError(
            "could not read vLLM container provenance"
            + (f": {detail}" if detail else "")
        )
    try:
        provenance = json.loads(completed.stdout)
    except ValueError as error:
        raise RuntimeError(
            f"vLLM container provenance was not valid JSON: {completed.stdout.strip()}"
        ) from error
    if not isinstance(provenance, dict):
        raise RuntimeError("vLLM container provenance was not a JSON object")
    return provenance


def wait_for_server(base_url: str, timeout_secs: int = 900) -> None:
    """Block until the vLLM server answers, or fail with the last error. A TPU
    model load includes XLA compilation, hence the generous default timeout.
    """
    models_url = base_url.rstrip("/") + "/models"
    deadline = time.monotonic() + timeout_secs
    last_error = "no attempt made"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(models_url, timeout=10) as response:
                if response.status == 200:
                    LOGGER.info("Server is answering at %s", models_url)
                    return
                last_error = f"HTTP {response.status}"
        except (urllib.error.URLError, OSError) as error:
            last_error = str(error)
        time.sleep(5)
    raise TimeoutError(
        f"vLLM server at {models_url} did not become ready within "
        f"{timeout_secs}s; last error: {last_error}"
    )


def main() -> None:
    """Print the vLLM server command for a recipe and its overrides."""
    args = parse_recipe_args(recipe_parser(main.__doc__))
    settings = resolve_settings(load_eval_config(args.config, args.overrides))
    print(shlex.join(vllm_serve_command(settings)))


if __name__ == "__main__":
    main()
