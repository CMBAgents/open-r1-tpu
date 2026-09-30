"""Evaluation recipes: their schema, validation, and resolved settings.

Every setting that changes what the model generates or how it is scored --
sampling parameters, the system prompt, the reporting markers -- must be
explicit in the recipe: a missing key is a ValueError naming it, never a
silently filled-in default. An unknown key is an error too, including a
typo'd dotted override, since overrides apply before validation.
"""

from __future__ import annotations

import difflib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from open_r1_tpu.core.config import load_config, read_prompt_file
from open_r1_tpu.evaluation.stack import vllm_tpu_image_tag

DEFAULT_HOST = "127.0.0.1"

DEFAULT_PORT = 8000

# vLLM is an external service, not a dependency: nothing here imports it, and
# it cannot share this environment anyway because its Python/JAX/PyTorch stack
# differs from the host's. The default wrapper runs the locally built TPU image
# and owns model/cache mounts plus container cleanup. Set server.image to null
# when overriding this with an external non-container command.
DEFAULT_SERVE_COMMAND = ("scripts/run_vllm_tpu_container.sh",)


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
    # Required by `evaluation.experiment`'s generation loop. No default for
    # either: a concurrency width and an error budget are deliberate
    # per-deployment choices (a wider width saturates a bigger server; a
    # laxer budget is wrong for a flaky one), not values worth guessing on a
    # recipe's behalf.
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


def task_slug(task: str) -> str:
    """Directory-safe name for one task spec (`suite|name|k` and friends)."""
    return re.sub(r"[^\w.=@-]+", "-", task)
