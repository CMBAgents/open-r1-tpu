"""Recipe validation, server-command building, and result reduction for a
fine-tuned checkpoint's evaluation.

Held-out loss is teacher-forced: every scored token is conditioned on ground
truth, so it cannot say whether the model closes its reasoning trace, stops,
or reaches the right answer on its own. Those are the questions a reasoning
stage is actually judged on, and only free generation scored against a
reference answers them.

The stack is three decoupled layers:

- generation is vLLM on the TPU, serving the merged export behind an
  OpenAI-compatible endpoint;
- the harness is `open_r1_tpu.evaluation.experiment`, Langfuse's own
  `dataset.run_experiment()` driving iteration and concurrency per task/seed
  (`open_r1_tpu.evaluation.task_fn`/`.runner` reach vLLM directly over the
  `openai` SDK -- no litellm, no subprocess);
- scoring is whatever metric the LightEval task declares, which for maths uses
  latex2sympy2-extended symbolic equivalence rather than string equality,
  called as a library through `open_r1_tpu.evaluation.scoring`.

This module owns the layers either side of the harness: it validates the
recipe, builds the vLLM server command from it, and reduces what
`open_r1_tpu.evaluation.reduce` read back from the harness's own JSONL output
into a single summary (`build_summary`, `aggregate_across_seeds`,
`log_summary_to_wandb` -- harness-agnostic, unchanged since before the
Langfuse-native rewrite). Nothing here imports JAX, Tunix, or vLLM -- the
server is reached over a socket, never in-process -- which is what keeps the
reducing and command-building logic testable on a laptop with no TPU stack
installed. Its CLI (`python -m open_r1_tpu.evaluation.run --print-server-command`)
does only that one thing now; the harness itself runs through
`scripts/run_eval_tpu.sh`, never through this module.

Install the frozen host stack and pinned inference image with
`scripts/setup_tpu_vm.sh --with-eval`.

Evaluation runs after training rather than beside it. Only one process can hold
the TPU chip, so the training job must have exited before the server starts.

Run through `scripts/run_eval_tpu.sh`, which owns the server's lifecycle::

    RECIPE=recipes/Qwen3-1.7B-Math/eval/tier1_core.yaml \
      TRACE_CONFIG=configs/tracing.yaml ./scripts/run_eval_tpu.sh

WHY SEEDS ARE MANDATORY. Seed variance alone moves small reasoning benchmarks
by 5-15 points (arXiv 2504.07086), which is more than most recipe changes are
worth. A single number off a 30-problem benchmark is not a measurement, so
every task is run once per seed and reported as mean and standard deviation.
`aggregate_across_seeds` reports a null standard deviation at one seed rather
than a reassuring 0.0.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import platform
import re
import shlex
import statistics
import subprocess
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

from open_r1_tpu.core.config import load_config, read_prompt_file
from open_r1_tpu.core.logging import LOG_LEVELS, configure_logging
from open_r1_tpu.evaluation.stack import (
    EVALUATION_PACKAGE_VERSIONS,
    VLLM_TPU_BASE_IMAGE,
    vllm_tpu_image_tag,
)

LOGGER = logging.getLogger(__name__)

# Every setting that changes what the model generates or how it is scored --
# sampling parameters, the system prompt, the reporting markers -- must be
# explicit in the recipe. There are no defaults for any of them here: a
# missing key is a ValueError naming it, never a silently filled-in value.
# See each recipe for the rationale behind its own numbers (temperature 0.6 /
# top_p 0.95 follow the DeepSeek-R1 recommendation the distilled models are
# tuned for; max_new_tokens is deliberately large so a trace is never cut off
# and scored as wrong).

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
# vLLM is an external service, not a dependency: nothing here imports it, and
# it cannot share this environment anyway because its Python/JAX/PyTorch stack
# differs from the host's. The default wrapper runs the locally built TPU image
# and owns model/cache mounts plus container cleanup. Set server.image to null
# when overriding this with an external non-container command.
DEFAULT_SERVE_COMMAND = ("scripts/run_vllm_tpu_container.sh",)


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def stack_versions() -> dict[str, str]:
    """Record the versions that give the numbers their meaning."""
    return {
        "python": platform.python_version(),
        **{name: _version(name) for name in EVALUATION_PACKAGE_VERSIONS},
    }


# The complete key set each section accepts. A key outside this set is either
# a typo or a stale setting from a schema that moved on, and both deserve an
# error rather than being silently ignored -- including a typo'd dotted
# override, since `load_config` applies overrides before validation runs.
EVAL_KEYS = {
    "tier",
    "tasks",
    "seeds",
    "max_samples",
    "consensus",
}
CONSENSUS_KEYS = {"n", "metric"}
SERVER_KEYS = {
    "model_path",
    "served_model_name",
    "turn_end_token",
    "serve_command",
    "image",
    "host",
    "port",
    "max_model_len",
    "tensor_parallel_size",
    "extra_args",
    "startup_timeout_secs",
    "base_url",
    # Required by `evaluation.runner`, the Langfuse-native generation loop.
    # No default for either: a concurrency width and an error budget are
    # deliberate per-deployment choices (a wider width saturates a bigger
    # server; a laxer budget is wrong for a flaky one), not values worth
    # guessing on a recipe's behalf.
    "max_concurrency",
    "fail_fast_after",
}
SAMPLING_KEYS = {"temperature", "top_p", "max_new_tokens", "system_prompt_file"}
REPORTING_KEYS = {
    "output_dir",
    "summary_path",
    "reasoning_start",
    "reasoning_end",
    "answer_marker",
    "wandb",
}
WANDB_KEYS = {
    "enabled",
    "project_name",
    "run_id",
    "run_name",
    "entity",
    "group",
    "mode",
    "job_type",
    "tags",
    "resume",
}


def _reject_unknown_keys(
    prefix: str, section: Mapping[str, Any], allowed: set[str]
) -> None:
    """Reject a key outside a section's schema, suggesting the nearest match."""
    for key in section:
        if key not in allowed:
            close = difflib.get_close_matches(str(key), sorted(allowed), n=1)
            hint = f"; did you mean {close[0]!r}?" if close else ""
            raise ValueError(f"Unknown key {prefix}.{key}{hint}")


def _validate_consensus(
    consensus: Any, tasks: Sequence[str], seeds: Sequence[int]
) -> None:
    """Check `eval.consensus`, the per-task consensus (cons@n) request.

    Explicit per task rather than inferred, and explicit about which metric
    judges the consensus answer, because both choices change a headline
    number: a task declares several metrics (`aime24` declares `pass@k:k=1`
    and `avg@n:n=1`), and picking one of them by position would make the
    reported `cons@n` depend on LightEval's declaration order.
    """
    if consensus is None:
        return
    if not isinstance(consensus, dict):
        raise ValueError(
            "eval.consensus must be a mapping of task -> {n, metric}, or null"
        )
    for task, request in consensus.items():
        if task not in tasks:
            raise ValueError(
                f"eval.consensus names task {task!r}, which eval.tasks does "
                f"not run (tasks: {sorted(tasks)})"
            )
        if not isinstance(request, dict):
            raise ValueError(
                f"eval.consensus[{task!r}] must be a mapping with keys "
                f"{sorted(CONSENSUS_KEYS)}"
            )
        _reject_unknown_keys(f"eval.consensus.{task}", request, CONSENSUS_KEYS)
        for key in CONSENSUS_KEYS:
            if key not in request:
                raise ValueError(f"eval.consensus[{task!r}].{key} is required")
        n = request["n"]
        if not isinstance(n, int) or isinstance(n, bool) or n < 2:
            raise ValueError(
                f"eval.consensus[{task!r}].n must be an integer of at least 2 "
                "-- a majority vote over one sample is that sample"
            )
        if n > len(seeds):
            # The replicates are the samples voted over, so asking for more
            # than the tier generates cannot be satisfied. Caught here rather
            # than after the generations have been paid for.
            raise ValueError(
                f"eval.consensus[{task!r}].n is {n} but eval.seeds has only "
                f"{len(seeds)} replicate(s) to vote over"
            )
        metric = request["metric"]
        if not isinstance(metric, str) or not metric:
            raise ValueError(
                f"eval.consensus[{task!r}].metric must name one of the task's "
                "own LightEval metrics (e.g. 'pass@k:k=1')"
            )


def validate_eval_config(config: dict[str, Any]) -> None:
    """Fail early for recipe mistakes that would otherwise waste TPU time."""
    for section in ("eval", "server", "sampling", "reporting"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")

    _reject_unknown_keys("eval", config["eval"], EVAL_KEYS)
    _reject_unknown_keys("server", config["server"], SERVER_KEYS)
    _reject_unknown_keys("sampling", config["sampling"], SAMPLING_KEYS)
    _reject_unknown_keys("reporting", config["reporting"], REPORTING_KEYS)

    tasks = config["eval"].get("tasks")
    if (
        not isinstance(tasks, list)
        or not tasks
        or not all(isinstance(task, str) and task for task in tasks)
    ):
        raise ValueError("eval.tasks must be a non-empty list of task strings")

    seeds = config["eval"].get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(isinstance(seed, int) for seed in seeds)
    ):
        raise ValueError("eval.seeds must be a non-empty list of integers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("eval.seeds must not repeat a seed")

    max_samples = config["eval"].get("max_samples")
    if max_samples is not None and (
        not isinstance(max_samples, int) or max_samples <= 0
    ):
        raise ValueError("eval.max_samples must be a positive integer or null")

    _validate_consensus(config["eval"].get("consensus"), tasks, seeds)

    model_path = config["server"].get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("server.model_path must be a non-empty string")
    turn_end_token = config["server"].get("turn_end_token")
    if not isinstance(turn_end_token, str) or not turn_end_token:
        raise ValueError(
            "server.turn_end_token must name the token the model's chat "
            "template closes each turn with (<|im_end|> on Qwen3)"
        )
    serve_command = config["server"].get("serve_command")
    if serve_command is not None and (
        not isinstance(serve_command, list)
        or not serve_command
        or not all(isinstance(part, str) and part for part in serve_command)
    ):
        raise ValueError(
            "server.serve_command must be a non-empty list of strings or null"
        )
    server_image = config["server"].get("image")
    if server_image is not None and (
        not isinstance(server_image, str) or not server_image
    ):
        raise ValueError("server.image must be a non-empty image string or null")
    if server_image is not None:
        local_tag = vllm_tpu_image_tag()
        prefix, marker, digest = server_image.rpartition("@sha256:")
        has_digest = (
            bool(marker)
            and bool(prefix)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        )
        if server_image != local_tag and not has_digest:
            raise ValueError(
                "server.image must be the derived local image tag or include an "
                "immutable @sha256:<64 hex> digest"
            )

    port = config["server"].get("port", DEFAULT_PORT)
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("server.port must be a TCP port number")

    max_concurrency = config["server"].get("max_concurrency")
    if (
        not isinstance(max_concurrency, int)
        or isinstance(max_concurrency, bool)
        or max_concurrency <= 0
    ):
        raise ValueError("server.max_concurrency must be a positive integer")
    fail_fast_after = config["server"].get("fail_fast_after")
    if (
        not isinstance(fail_fast_after, int)
        or isinstance(fail_fast_after, bool)
        or fail_fast_after <= 0
    ):
        raise ValueError("server.fail_fast_after must be a positive integer")

    sampling = config["sampling"]
    for key in ("temperature", "top_p", "max_new_tokens"):
        if key not in sampling:
            raise ValueError(f"sampling.{key} is required")
    # `.get()` cannot tell "absent" from "explicitly null", and tier 3 relies
    # on an explicit null to mean "no system prompt" -- so presence is checked
    # with `in` rather than a default.
    if "system_prompt_file" not in sampling:
        raise ValueError(
            "sampling.system_prompt_file is required (use null for no system prompt)"
        )

    temperature = sampling["temperature"]
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError("sampling.temperature must be a non-negative number")
    top_p = sampling["top_p"]
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError("sampling.top_p must be in (0.0, 1.0]")
    max_new_tokens = sampling["max_new_tokens"]
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("sampling.max_new_tokens must be a positive integer")
    system_prompt_file = sampling["system_prompt_file"]
    if system_prompt_file is not None and (
        not isinstance(system_prompt_file, str) or not system_prompt_file
    ):
        raise ValueError(
            "sampling.system_prompt_file must be a non-empty string or null"
        )

    max_model_len = config["server"].get("max_model_len")
    if max_model_len is not None:
        if not isinstance(max_model_len, int) or max_model_len <= 0:
            raise ValueError("server.max_model_len must be a positive integer or null")
        if max_model_len <= max_new_tokens:
            # The server budgets prompt plus completion against one window, so
            # a cap at or below the completion budget leaves no room for the
            # problem and truncates every trace.
            raise ValueError(
                f"server.max_model_len ({max_model_len}) must exceed "
                f"sampling.max_new_tokens ({max_new_tokens}) to leave room for "
                "the prompt"
            )

    reporting = config["reporting"]
    for key in ("reasoning_start", "reasoning_end", "answer_marker"):
        if key not in reporting:
            raise ValueError(f"reporting.{key} is required")
    # Null means the serving chat template opens the reasoning block inside
    # the prompt itself -- DeepSeek's distills append `<think>` to the
    # generation prompt -- so a completion can only ever carry the closing
    # tag and closure is judged on that alone.
    reasoning_start = reporting["reasoning_start"]
    if reasoning_start is not None and (
        not isinstance(reasoning_start, str) or not reasoning_start
    ):
        raise ValueError("reporting.reasoning_start must be a non-empty string or null")
    for key in ("reasoning_end", "answer_marker"):
        if not isinstance(reporting[key], str) or not reporting[key]:
            raise ValueError(f"reporting.{key} must be a non-empty string")

    wandb = reporting.get("wandb", {})
    if not isinstance(wandb, dict):
        raise ValueError("reporting.wandb must be a configuration mapping")
    _reject_unknown_keys("reporting.wandb", wandb, WANDB_KEYS)
    enabled = wandb.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("reporting.wandb.enabled must be a boolean")
    if enabled:
        # `log_summary_to_wandb` no longer falls back for either key: a run
        # logged to the wrong project or the wrong mode is a mistake worth
        # catching at load time, not a reasonable default to guess.
        for key in ("project_name", "mode"):
            if key not in wandb:
                raise ValueError(
                    f"reporting.wandb.{key} is required when "
                    "reporting.wandb.enabled is true"
                )
    mode = wandb.get("mode", "online")
    if mode not in {"online", "offline", "disabled"}:
        raise ValueError("reporting.wandb.mode must be online, offline, or disabled")


def load_eval_config(
    path: str | Path, overrides: list[str] | None = None
) -> dict[str, Any]:
    """Load an evaluation recipe and apply dotted command-line overrides."""
    return load_config(path, overrides, validator=validate_eval_config)


def resolve_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Normalize an evaluation recipe and apply defaults."""
    evaluation = config["eval"]
    server = config["server"]
    sampling = config["sampling"]
    reporting = config["reporting"]

    host = str(server.get("host", DEFAULT_HOST))
    port = int(server.get("port", DEFAULT_PORT))
    model_path = str(server["model_path"])
    # vLLM reports the model under this name and the harness must request the
    # same one, so it is derived once here rather than being set twice in the
    # recipe.
    served_model_name = str(server.get("served_model_name") or Path(model_path).name)
    output_dir = str(reporting.get("output_dir") or Path(model_path).parent / "eval")
    serve_command = server.get("serve_command")
    # The supported wrapper selects the local build spec whether it is implicit
    # or written explicitly in a recipe. Any other command remains the external
    # Python 3.12 escape hatch unless it explicitly supplies an image.
    uses_supported_wrapper = serve_command is None or (
        isinstance(serve_command, list)
        and bool(serve_command)
        and str(serve_command[0]).endswith("run_vllm_tpu_container.sh")
    )
    server_image = server.get(
        "image", vllm_tpu_image_tag() if uses_supported_wrapper else None
    )

    return {
        "tier": str(evaluation.get("tier", "unnamed")),
        "tasks": [str(task) for task in evaluation["tasks"]],
        "seeds": [int(seed) for seed in evaluation["seeds"]],
        "max_samples": evaluation.get("max_samples"),
        # `{task: {"n": int, "metric": str}}`; empty when the recipe asks for
        # no consensus number. See `evaluation.consensus`.
        "consensus": {
            str(task): {"n": int(request["n"]), "metric": str(request["metric"])}
            for task, request in (evaluation.get("consensus") or {}).items()
        },
        "model_path": model_path,
        "served_model_name": served_model_name,
        "turn_end_token": str(server["turn_end_token"]),
        "host": host,
        "port": port,
        "base_url": str(server.get("base_url") or f"http://{host}:{port}/v1"),
        "max_model_len": server.get("max_model_len"),
        "max_concurrency": int(server["max_concurrency"]),
        "fail_fast_after": int(server["fail_fast_after"]),
        "serve_command": [
            str(part) for part in (serve_command or DEFAULT_SERVE_COMMAND)
        ],
        "server_image": server_image,
        "temperature": float(sampling["temperature"]),
        "top_p": float(sampling["top_p"]),
        "max_new_tokens": int(sampling["max_new_tokens"]),
        # The recipe's own system prompt, read from the file the recipe names.
        # Chatting the model off-distribution changes its behaviour, so
        # evaluation renders the prompt training used.
        "system_prompt": (
            read_prompt_file(sampling["system_prompt_file"])
            if sampling["system_prompt_file"] is not None
            else None
        ),
        "tensor_parallel_size": int(server.get("tensor_parallel_size", 1)),
        "server_extra_args": [str(arg) for arg in (server.get("extra_args") or [])],
        "startup_timeout_secs": int(server.get("startup_timeout_secs", 900)),
        "output_dir": output_dir,
        "summary_path": str(
            reporting.get("summary_path")
            or Path(output_dir) / f"summary_{evaluation.get('tier', 'unnamed')}.json"
        ),
        "reasoning_start": (
            None
            if reporting["reasoning_start"] is None
            else str(reporting["reasoning_start"])
        ),
        "reasoning_end": str(reporting["reasoning_end"]),
        "answer_marker": str(reporting["answer_marker"]),
        "wandb": dict(reporting.get("wandb", {})),
    }


def vllm_serve_command(settings: Mapping[str, Any]) -> list[str]:
    """Build the server invocation for this recipe.

    Emitted from here rather than written into the shell launcher so the recipe
    stays the single source of truth for the port, the served name, and the
    context window. The launcher owns the process; this owns its arguments.

    `server.serve_command` supplies everything up to the model path, so the
    server can live outside this environment -- which it must, since
    tpu-inference does not support the Python this project runs on.
    """
    command = [*settings.get("serve_command", DEFAULT_SERVE_COMMAND)]
    if settings.get("server_image"):
        # The supported container wrapper owns all Docker-specific arguments.
        # `--` leaves every following option to vLLM, while the selected image
        # remains visible in the durable summary rather than hidden in a script.
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
    """Read image ID and service versions through the supported wrapper.

    The wrapper supplies its own Docker/sudo detection, so this works on a
    freshly provisioned TPU VM where Docker is intentionally not in the login
    user's group. The command runs Python in the already-built image only; it
    does not initialize vLLM or reserve the TPU.
    """
    image = settings.get("server_image")
    raw_command = settings.get("serve_command", DEFAULT_SERVE_COMMAND)
    command = [str(part) for part in raw_command]
    if (
        image is None
        or not command
        or not command[0].endswith("run_vllm_tpu_container.sh")
    ):
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
    """Block until the vLLM server answers, or fail with what went wrong.

    The shell launcher starts the server and this confirms it is actually
    serving before a long evaluation is committed to it. Model load on a TPU
    includes weight transfer and an XLA compilation, so the default timeout is
    generous.
    """
    # Imported here to keep the module importable where `time` monkeypatching
    # in tests would otherwise leak across cases.
    import time

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


def aggregate_across_seeds(
    per_seed: Mapping[int, Mapping[str, Mapping[str, float]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Reduce per-seed metrics to mean and standard deviation.

    The standard deviation is None at a single seed rather than 0.0. Reporting
    zero spread from one sample is the exact overclaim this pipeline exists to
    prevent.
    """
    tasks: dict[str, dict[str, list[float]]] = {}
    for seed in sorted(per_seed):
        for task, metrics in per_seed[seed].items():
            for name, value in metrics.items():
                tasks.setdefault(task, {}).setdefault(name, []).append(float(value))

    aggregated: dict[str, dict[str, dict[str, Any]]] = {}
    for task, metrics in tasks.items():
        aggregated[task] = {
            name: {
                "mean": statistics.fmean(values),
                "std": statistics.stdev(values) if len(values) > 1 else None,
                "n": len(values),
                "values": values,
            }
            for name, values in metrics.items()
        }
    return aggregated


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON file from a local path or a GCS URI."""
    return json.loads(_read_text(str(path)))


def _read_text(path: str) -> str:
    if path.startswith("gs://"):
        import gcsfs

        with gcsfs.GCSFileSystem().open(path, "rt") as handle:
            return str(handle.read())
    return Path(path).expanduser().read_text(encoding="utf-8")


def write_summary(path: str, summary: Mapping[str, Any]) -> None:
    """Write the summary as JSON, locally or to GCS beside the checkpoint."""
    payload = json.dumps(summary, indent=2, sort_keys=True, default=str)
    if path.startswith("gs://"):
        import gcsfs

        with gcsfs.GCSFileSystem().open(path, "wt") as handle:
            handle.write(payload)
        return
    local = Path(path).expanduser()
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(payload, encoding="utf-8")


def task_slug(task: str) -> str:
    """Directory-safe name for one task spec (`suite|name|k` and friends)."""
    return re.sub(r"[^\w.=@-]+", "-", task)


def build_summary(
    settings: Mapping[str, Any],
    per_seed_metrics: Mapping[int, Mapping[str, Mapping[str, float]]],
    per_seed_stats: Mapping[int, Mapping[str, Any]],
    server_provenance: Mapping[str, Any] | None = None,
    consensus: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assemble the durable record of one evaluation.

    `consensus` is kept out of `tasks_metrics` on purpose. Every entry there
    is a mean and standard deviation *across* replicates; a cons@n number is
    a single value computed *from* all of them jointly and has no spread to
    report, so filing it alongside would invite reading a null standard
    deviation as one-replicate noise rather than as a category difference.
    `summary_rows` flattens both, so W&B still receives it.
    """
    generation: dict[str, Any] = {}
    for name in (
        "format_rate",
        "reasoning_closed_rate",
        "answer_marker_rate",
        "truncation_rate",
        "mean_completion_tokens",
        "mean_completion_chars",
    ):
        values = [
            float(stats[name])
            for stats in per_seed_stats.values()
            if stats.get(name) is not None
        ]
        generation[name] = {
            "mean": statistics.fmean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "n": len(values),
        }

    local_image = vllm_tpu_image_tag()
    server_image = settings.get("server_image")
    image_provenance = (
        {
            "spec_tag": local_image,
            "image_id": (
                server_provenance.get("image_id") if server_provenance else None
            ),
            "base_image": VLLM_TPU_BASE_IMAGE,
            "service_versions": (
                server_provenance.get("service_versions") if server_provenance else None
            ),
        }
        if server_image == local_image
        else None
    )

    return {
        "tier": settings["tier"],
        "model_path": settings["model_path"],
        "served_model_name": settings["served_model_name"],
        "tasks": list(settings["tasks"]),
        "seeds": list(settings["seeds"]),
        # `seeds` indexes replicates; the backend rejects per-request seeds, so
        # an archived summary must not be read as reproducible sample-by-sample.
        "seeded_replicates": False,
        "sampling": {
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "max_new_tokens": settings["max_new_tokens"],
        },
        "max_samples": settings.get("max_samples"),
        "stack": stack_versions(),
        "serve_command": list(settings.get("serve_command", DEFAULT_SERVE_COMMAND)),
        "server_image": server_image,
        "server_image_provenance": image_provenance,
        "server_command": vllm_serve_command(settings),
        "tasks_metrics": aggregate_across_seeds(per_seed_metrics),
        "consensus": {task: dict(result) for task, result in (consensus or {}).items()},
        "generation": generation,
        "per_seed_generation": {str(k): v for k, v in per_seed_stats.items()},
    }


def summary_rows(summary: Mapping[str, Any]) -> list[list[Any]]:
    """Flatten a summary into one row per task and metric, for tabular logs."""
    rows: list[list[Any]] = []
    for task, metrics in sorted(summary.get("tasks_metrics", {}).items()):
        for name, stats in sorted(metrics.items()):
            rows.append(
                [
                    summary.get("tier"),
                    task,
                    name,
                    stats.get("mean"),
                    stats.get("std"),
                    stats.get("n"),
                ]
            )
    # Consensus rows carry no standard deviation (there is one value, not one
    # per replicate); the count column holds the vote width instead of a
    # replicate count, which is the number that makes `cons@64` mean what it
    # says.
    for task, result in sorted(summary.get("consensus", {}).items()):
        rows.append(
            [
                summary.get("tier"),
                task,
                result.get("name"),
                result.get("value"),
                None,
                result.get("n"),
            ]
        )
    return rows


def log_summary_to_wandb(
    summary: Mapping[str, Any], settings: Mapping[str, Any]
) -> None:
    """Attach the summary to the training run, or a standalone run.

    Passing `reporting.wandb.run_id` puts eval numbers on the same run as the
    loss curves, which is the only way to read them together. W&B resumes by
    id, not by name, so without an id this starts a separate run rather than
    silently appending to whichever run happens to share the name.
    """
    wandb_config = dict(settings.get("wandb", {}))
    if not wandb_config.get("enabled", False):
        return
    try:
        import wandb
    except ImportError:
        LOGGER.warning("wandb is not installed; skipping W&B logging")
        return

    run_id = wandb_config.get("run_id")
    run_name = wandb_config.get("run_name") or f"{settings['tier']}-eval"
    # No fallback for project_name or mode: validate_eval_config requires both
    # whenever reporting.wandb.enabled is true, so logging to the wrong project
    # or mode is a recipe mistake worth catching at load time, not a default
    # worth guessing here.
    init_kwargs: dict[str, Any] = {
        "project": wandb_config["project_name"],
        "mode": wandb_config["mode"],
        "job_type": wandb_config.get("job_type", "eval"),
    }
    for key in ("entity", "group", "tags"):
        if wandb_config.get(key) is not None:
            init_kwargs[key] = wandb_config[key]
    if run_id:
        init_kwargs["id"] = str(run_id)
        init_kwargs["resume"] = wandb_config.get("resume", "allow")
    else:
        init_kwargs["name"] = run_name
        LOGGER.info(
            "No reporting.wandb.run_id set; logging to a standalone run named "
            "%s rather than the training run",
            run_name,
        )

    run = wandb.init(**init_kwargs)
    try:
        flat = {
            f"eval/{summary['tier']}/{task}/{name}": stats["mean"]
            for task, metrics in summary.get("tasks_metrics", {}).items()
            for name, stats in metrics.items()
            if stats.get("mean") is not None
        }
        flat.update(
            {
                f"eval/{summary['tier']}/generation/{name}": stats["mean"]
                for name, stats in summary.get("generation", {}).items()
                if stats.get("mean") is not None
            }
        )
        # Summary rather than a stepped log: evaluation happens after the last
        # optimizer step, so it has no step of its own, and a stepped write
        # after resume would land on an arbitrary one.
        run.summary.update(flat)
        table = wandb.Table(
            columns=["tier", "task", "metric", "mean", "std", "seeds"],  # pyright: ignore[reportArgumentType]
            data=summary_rows(summary),
        )
        run.log({f"eval/{summary['tier']}/table": table})
    finally:
        run.finish()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML evaluation recipe")
    parser.add_argument(
        "--print-server-command",
        action="store_true",
        required=True,
        help=(
            "Print the configured vLLM server command for this recipe and "
            "exit. The only mode this CLI has left: the harness itself runs "
            "through open_r1_tpu.evaluation.experiment, driven by "
            "scripts/run_eval_tpu.sh, not through this module."
        ),
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("OPEN_R1_TPU_LOG_LEVEL", "info"),
        choices=sorted(LOG_LEVELS),
    )
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    configure_logging(LOG_LEVELS[args.log_level])
    config = load_eval_config(args.config, args.overrides)
    settings = resolve_settings(config)
    print(shlex.join(vllm_serve_command(settings)))


if __name__ == "__main__":
    main()
