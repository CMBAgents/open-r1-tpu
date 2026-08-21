"""Benchmark evaluation for a fine-tuned checkpoint.

Held-out loss is teacher-forced: every scored token is conditioned on ground
truth, so it cannot say whether the model closes its reasoning trace, stops,
or reaches the right answer on its own. Those are the questions a reasoning
stage is actually judged on, and only free generation scored against a
reference answers them.

The stack is three decoupled layers:

- generation is vLLM on the TPU, serving the merged export behind an
  OpenAI-compatible endpoint;
- the harness is LightEval, reached over HTTP through its litellm backend;
- scoring is whatever metric the LightEval task declares, which for maths is
  Math-Verify's symbolic equivalence rather than string equality.

This module owns the layers either side of LightEval: it validates the recipe,
runs the harness once per seed, and then reduces what the harness wrote into a
single summary. Nothing here imports JAX, Tunix, or vLLM -- LightEval is reached
as a subprocess and the server over a socket -- which is what keeps the reducing
and command-building logic testable on a laptop with no TPU stack installed.

Install the harness with the `eval` extra: `pip install -e '.[eval]'`.

Evaluation runs after training rather than beside it. Only one process can hold
the TPU chip, so the training job must have exited before the server starts.

Run through `scripts/run_eval_tpu.sh`, which owns the server's lifecycle::

    RECIPE=recipes/Qwen3-1.7B-Math/eval/tier1_core.yaml ./scripts/run_eval_tpu.sh

WHY SEEDS ARE MANDATORY. Seed variance alone moves small reasoning benchmarks
by 5-15 points (arXiv 2504.07086), which is more than most recipe changes are
worth. A single number off a 30-problem benchmark is not a measurement, so
every task is run once per seed and reported as mean and standard deviation.
`aggregate_across_seeds` reports a null standard deviation at one seed rather
than a reassuring 0.0.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import statistics
import subprocess
import urllib.error
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from open_r1_tpu.config import load_config
from open_r1_tpu.logging import LOG_LEVELS, configure_logging

LOGGER = logging.getLogger(__name__)

# Sampling defaults follow the DeepSeek-R1 recommendation, which is what the
# distilled reasoning models are tuned for and what most published numbers use.
# Greedy decoding on a reasoning model collapses into repetition often enough
# that it is not a safe default, and it makes pass@k meaningless.
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
# Large on purpose. A trace cut off by the token cap is scored as wrong and
# reads as a reasoning failure, so an undersized cap silently understates the
# model. The chat test of the first math run hit exactly this at 2048.
DEFAULT_MAX_NEW_TOKENS = 16384

DEFAULT_REASONING_START = "<think>"
DEFAULT_REASONING_END = "</think>"
# Substring, not a regex: the LaTeX brace makes a regex needlessly fiddly and
# the presence of the marker is all that is being counted.
DEFAULT_ANSWER_MARKER = "\\boxed{"

# Qwen3-Base names <|endoftext|> as its EOS while the chat template closes a
# turn with <|im_end|>. A server left to the tokenizer's own EOS therefore runs
# past the end of the reply and writes the user's next turn too, which under a
# benchmark reads as a model that cannot stop reasoning.
DEFAULT_STOP_TOKENS = ("<|im_end|>",)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_LIGHTEVAL_BINARY = "lighteval"
# vLLM is an external service, not a dependency: nothing here imports it, and
# it cannot share this environment anyway because tpu-inference classifies as
# Python 3.10-3.12 while this project is on 3.13. Override this to reach it
# wherever it does live -- a 3.12 virtualenv's binary, or a container.
DEFAULT_SERVE_COMMAND = ("vllm", "serve")

# LightEval writes one Parquet row per evaluated document, with the model's
# generations nested under this column. Its inner field names have moved
# between releases, so they are probed rather than assumed; see
# `extract_completions`.
RESPONSE_COLUMN = "__model_response__"
_TEXT_KEYS = ("text", "final_text", "generated_text", "predictions", "result")
_TOKEN_KEYS = ("output_tokens", "generated_tokens", "num_generated_tokens")

# Distributions whose versions pin the meaning of a number. Recorded with every
# summary because a result that does not name its stack cannot be compared with
# one produced months later. vLLM is absent because it runs outside this
# environment and would always read back as "unknown"; the summary records the
# serve command instead, which is the honest record of what served the model.
_STACK_DISTRIBUTIONS = ("lighteval", "litellm", "math-verify")


def _version(distribution: str) -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return "unknown"


def stack_versions() -> dict[str, str]:
    """Record the versions that give the numbers their meaning."""
    return {name: _version(name) for name in _STACK_DISTRIBUTIONS}


def validate_eval_config(config: dict[str, Any]) -> None:
    """Fail early for recipe mistakes that would otherwise waste TPU time."""
    for section in ("eval", "server", "sampling", "reporting"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"Missing configuration section: {section}")

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

    model_path = config["server"].get("model_path")
    if not isinstance(model_path, str) or not model_path:
        raise ValueError("server.model_path must be a non-empty string")
    serve_command = config["server"].get("serve_command")
    if serve_command is not None and (
        not isinstance(serve_command, list)
        or not serve_command
        or not all(isinstance(part, str) and part for part in serve_command)
    ):
        raise ValueError(
            "server.serve_command must be a non-empty list of strings or null"
        )

    port = config["server"].get("port", DEFAULT_PORT)
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("server.port must be a TCP port number")

    temperature = config["sampling"].get("temperature", DEFAULT_TEMPERATURE)
    if not isinstance(temperature, (int, float)) or temperature < 0:
        raise ValueError("sampling.temperature must be a non-negative number")
    top_p = config["sampling"].get("top_p", DEFAULT_TOP_P)
    if not isinstance(top_p, (int, float)) or not 0 < top_p <= 1:
        raise ValueError("sampling.top_p must be in (0.0, 1.0]")
    max_new_tokens = config["sampling"].get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    if not isinstance(max_new_tokens, int) or max_new_tokens <= 0:
        raise ValueError("sampling.max_new_tokens must be a positive integer")

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

    wandb = config["reporting"].get("wandb", {})
    if not isinstance(wandb, dict):
        raise ValueError("reporting.wandb must be a configuration mapping")
    if not isinstance(wandb.get("enabled", False), bool):
        raise ValueError("reporting.wandb.enabled must be a boolean")
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
    # vLLM reports the model under this name and litellm must ask for the same
    # one, so it is derived once here rather than being set twice in the recipe.
    served_model_name = str(server.get("served_model_name") or Path(model_path).name)
    output_dir = str(reporting.get("output_dir") or Path(model_path).parent / "eval")

    return {
        "tier": str(evaluation.get("tier", "unnamed")),
        "tasks": [str(task) for task in evaluation["tasks"]],
        "seeds": [int(seed) for seed in evaluation["seeds"]],
        "max_samples": evaluation.get("max_samples"),
        "lighteval_binary": str(
            evaluation.get("lighteval_binary", DEFAULT_LIGHTEVAL_BINARY)
        ),
        # Passed through verbatim. LightEval's CLI moves between releases, so a
        # new flag is a recipe edit rather than a change here.
        "extra_args": [str(arg) for arg in (evaluation.get("extra_args") or [])],
        "model_path": model_path,
        "served_model_name": served_model_name,
        "host": host,
        "port": port,
        "base_url": str(server.get("base_url") or f"http://{host}:{port}/v1"),
        "max_model_len": server.get("max_model_len"),
        "serve_command": [
            str(part) for part in (server.get("serve_command") or DEFAULT_SERVE_COMMAND)
        ],
        "temperature": float(sampling.get("temperature", DEFAULT_TEMPERATURE)),
        "top_p": float(sampling.get("top_p", DEFAULT_TOP_P)),
        "max_new_tokens": int(sampling.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)),
        # The recipe's own system prompt. Chatting the model off-distribution
        # changes its behaviour, so evaluation renders the prompt training used.
        "system_prompt": sampling.get("system_prompt"),
        "stop": [
            str(token)
            for token in (
                sampling["stop"]
                if sampling.get("stop") is not None
                else DEFAULT_STOP_TOKENS
            )
        ],
        "tensor_parallel_size": int(server.get("tensor_parallel_size", 1)),
        "server_extra_args": [str(arg) for arg in (server.get("extra_args") or [])],
        "startup_timeout_secs": int(server.get("startup_timeout_secs", 900)),
        "output_dir": output_dir,
        "summary_path": str(
            reporting.get("summary_path")
            or Path(output_dir) / f"summary_{evaluation.get('tier', 'unnamed')}.json"
        ),
        "reasoning_start": str(
            reporting.get("reasoning_start", DEFAULT_REASONING_START)
        ),
        "reasoning_end": str(reporting.get("reasoning_end", DEFAULT_REASONING_END)),
        "answer_marker": str(reporting.get("answer_marker", DEFAULT_ANSWER_MARKER)),
        "wandb": dict(reporting.get("wandb", {})),
    }


def litellm_model_config(settings: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """Build the litellm model file LightEval reads for one seed.

    `hosted_vllm/` is the litellm provider prefix for a self-hosted server; the
    part after it must match the name vLLM serves the model under, or the
    server answers with a model-not-found rather than a completion.
    """
    generation: dict[str, Any] = {
        "temperature": settings["temperature"],
        "top_p": settings["top_p"],
        "max_new_tokens": settings["max_new_tokens"],
        "seed": seed,
    }
    if settings.get("stop"):
        generation["stop_tokens"] = list(settings["stop"])
    return {
        "model_parameters": {
            "provider": "hosted_vllm",
            "model_name": f"hosted_vllm/{settings['served_model_name']}",
            "base_url": settings["base_url"],
            # The server is local and unauthenticated, but litellm refuses to
            # send a request with no key at all.
            "api_key": "local",
            "generation_parameters": generation,
        }
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
    command = [
        *settings.get("serve_command", DEFAULT_SERVE_COMMAND),
        str(settings["model_path"]),
        "--served-model-name",
        str(settings["served_model_name"]),
        "--host",
        str(settings["host"]),
        "--port",
        str(settings["port"]),
        "--tensor-parallel-size",
        str(settings.get("tensor_parallel_size", 1)),
    ]
    if settings.get("max_model_len"):
        command += ["--max-model-len", str(settings["max_model_len"])]
    command += list(settings.get("server_extra_args", []))
    return command


def lighteval_command(
    settings: Mapping[str, Any], model_config_path: str | Path, output_dir: str | Path
) -> list[str]:
    """Build the LightEval invocation for one seed."""
    command = [
        settings["lighteval_binary"],
        "endpoint",
        "litellm",
        str(model_config_path),
        ",".join(settings["tasks"]),
        "--output-dir",
        str(output_dir),
        # Details carry the raw generations, which is the only source for the
        # truncation, format, and length metrics below.
        "--save-details",
    ]
    if settings.get("max_samples") is not None:
        command += ["--max-samples", str(settings["max_samples"])]
    if settings.get("system_prompt"):
        # The model was trained behind this prompt. Evaluating without it
        # measures the model off-distribution, which the chat test of the first
        # math run already showed changes its behaviour. Set it to null and use
        # eval.extra_args if the installed LightEval spells the flag
        # differently.
        command += ["--system-prompt", str(settings["system_prompt"])]
    command += list(settings.get("extra_args", []))
    return command


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


def _coerce_mapping(value: Any) -> dict[str, Any] | None:
    """Read a details cell that may be a struct or a JSON string."""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (str, bytes)):
        try:
            decoded = json.loads(value)
        except (ValueError, TypeError):
            return None
        return decoded if isinstance(decoded, dict) else None
    return None


def _first_present(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if mapping.get(key) is not None:
            return mapping[key]
    return None


def extract_completions(response: Any) -> list[str]:
    """Pull the generated strings out of one details row.

    LightEval has renamed these fields across releases, so the candidates are
    probed in order and an unrecognised shape raises with the keys that were
    actually there. Guessing would silently report a format rate of zero.
    """
    mapping = _coerce_mapping(response)
    if mapping is None:
        raise ValueError(f"Unreadable {RESPONSE_COLUMN} cell of type {type(response)}")
    value = _first_present(mapping, _TEXT_KEYS)
    if value is None:
        raise ValueError(
            f"No generation text in {RESPONSE_COLUMN}; tried "
            f"{list(_TEXT_KEYS)} but found keys {sorted(mapping)}"
        )
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        return [str(item) for item in value]
    return [str(value)]


def extract_token_counts(response: Any) -> list[int] | None:
    """Pull per-generation token counts, or None when the details omit them."""
    mapping = _coerce_mapping(response)
    if mapping is None:
        return None
    value = _first_present(mapping, _TOKEN_KEYS)
    if value is None:
        return None
    if isinstance(value, int):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        counts: list[int] = []
        for item in value:
            if isinstance(item, int):
                counts.append(item)
            elif isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
                # Some releases store the token ids rather than a count.
                counts.append(len(list(item)))
            else:
                return None
        return counts
    return None


def completion_stats(
    responses: Iterable[Any],
    *,
    max_new_tokens: int,
    reasoning_start: str = DEFAULT_REASONING_START,
    reasoning_end: str = DEFAULT_REASONING_END,
    answer_marker: str = DEFAULT_ANSWER_MARKER,
) -> dict[str, Any]:
    """Summarize generations along the axes accuracy alone cannot separate.

    Accuracy answers "was it right". These answer "why not":

    - `truncation_rate` separates a wrong answer from one that never finished.
      A truncated trace scores as wrong and looks like a reasoning failure.
    - `reasoning_closed_rate` and `answer_marker_rate` measure whether the
      model produces the shape SFT was teaching, independent of correctness.
    - `mean_completion_tokens` is the length-inflation signal.

    `truncation_rate` and `mean_completion_tokens` are None when the details
    carry no token counts, which is honest about the gap rather than filling it
    with a character-length guess.
    """
    completions = 0
    documents = 0
    closed = 0
    marked = 0
    formatted = 0
    chars = 0
    token_counts: list[int] = []
    missing_token_counts = False

    for response in responses:
        documents += 1
        texts = extract_completions(response)
        counts = extract_token_counts(response)
        if counts is None or len(counts) != len(texts):
            missing_token_counts = True
        else:
            token_counts.extend(counts)
        for text in texts:
            completions += 1
            chars += len(text)
            start = text.find(reasoning_start)
            end = text.find(reasoning_end)
            # An opening tag before a closing one. Order matters: a stray
            # closing tag alone is not a completed trace.
            is_closed = start != -1 and end != -1 and end > start
            has_marker = answer_marker in text
            closed += int(is_closed)
            marked += int(has_marker)
            formatted += int(is_closed and has_marker)

    def rate(count: int) -> float | None:
        return count / completions if completions else None

    truncated = (
        sum(1 for count in token_counts if count >= max_new_tokens)
        if token_counts
        else 0
    )
    have_tokens = bool(token_counts) and not missing_token_counts
    return {
        "documents": documents,
        "completions": completions,
        "format_rate": rate(formatted),
        "reasoning_closed_rate": rate(closed),
        "answer_marker_rate": rate(marked),
        "mean_completion_chars": (chars / completions) if completions else None,
        "mean_completion_tokens": (
            statistics.fmean(token_counts) if have_tokens else None
        ),
        "truncation_rate": (truncated / len(token_counts)) if have_tokens else None,
    }


def normalise_results(results: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Reduce a LightEval results file to numeric metrics per task.

    Whatever metric a task declares is carried through unchanged -- `avg@k`,
    `maj@k`, `pass@k`, `extractive_match` -- because hardcoding a metric name
    here would silently drop the one a newly added task reports. Standard-error
    columns are dropped: this module reports spread across seeds instead, which
    is the spread that actually moves.
    """
    scores = results.get("results")
    if not isinstance(scores, Mapping):
        raise ValueError("LightEval results file has no 'results' mapping")
    normalised: dict[str, dict[str, float]] = {}
    for task, metrics in scores.items():
        if not isinstance(metrics, Mapping):
            continue
        numeric = {
            name: float(value)
            for name, value in metrics.items()
            if isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not name.endswith("_stderr")
        }
        if numeric:
            normalised[str(task)] = numeric
    return normalised


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


def find_results_file(output_dir: str | Path) -> Path:
    """Locate the newest LightEval results file under an output directory."""
    candidates = sorted(Path(output_dir).glob("results/**/results_*.json"))
    if not candidates:
        raise FileNotFoundError(
            f"No LightEval results file under {output_dir}. The harness "
            "probably failed before writing results; check its output above."
        )
    return candidates[-1]


def find_details_files(output_dir: str | Path) -> list[Path]:
    """Locate the newest run's detail shards under an output directory."""
    candidates = sorted(Path(output_dir).glob("details/**/details_*.parquet"))
    if not candidates:
        return []
    # One directory per run timestamp; keep only the newest so a re-run in the
    # same output directory does not average old generations into new ones.
    newest = candidates[-1].parent
    return [path for path in candidates if path.parent == newest]


def read_detail_responses(paths: Iterable[Path]) -> list[Any]:
    """Read the generation column out of LightEval's Parquet detail shards."""
    import pyarrow.parquet as pq

    responses: list[Any] = []
    for path in paths:
        table = pq.read_table(path)
        if RESPONSE_COLUMN not in table.column_names:
            raise ValueError(
                f"{path} has no {RESPONSE_COLUMN} column; found {table.column_names}"
            )
        responses.extend(table.column(RESPONSE_COLUMN).to_pylist())
    return responses


def run_seed(
    settings: Mapping[str, Any], seed: int, work_dir: Path
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Run the harness once and reduce what it wrote."""
    seed_dir = work_dir / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_config_path = seed_dir / "litellm_model.yaml"
    model_config_path.write_text(
        yaml.safe_dump(litellm_model_config(settings, seed), sort_keys=False),
        encoding="utf-8",
    )

    command = lighteval_command(settings, model_config_path, seed_dir)
    LOGGER.info("seed %d: %s", seed, " ".join(command))
    subprocess.run(command, check=True)

    results = read_json(find_results_file(seed_dir))
    metrics = normalise_results(results)

    detail_paths = find_details_files(seed_dir)
    if detail_paths:
        stats = completion_stats(
            read_detail_responses(detail_paths),
            max_new_tokens=int(settings["max_new_tokens"]),
            reasoning_start=settings["reasoning_start"],
            reasoning_end=settings["reasoning_end"],
            answer_marker=settings["answer_marker"],
        )
    else:
        LOGGER.warning(
            "seed %d wrote no detail shards; generation-level metrics "
            "(truncation, format, length) are unavailable for it",
            seed,
        )
        stats = {}
    return metrics, stats


def build_summary(
    settings: Mapping[str, Any],
    per_seed_metrics: Mapping[int, Mapping[str, Mapping[str, float]]],
    per_seed_stats: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the durable record of one evaluation."""
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

    return {
        "tier": settings["tier"],
        "model_path": settings["model_path"],
        "served_model_name": settings["served_model_name"],
        "tasks": list(settings["tasks"]),
        "seeds": list(settings["seeds"]),
        "sampling": {
            "temperature": settings["temperature"],
            "top_p": settings["top_p"],
            "max_new_tokens": settings["max_new_tokens"],
        },
        "max_samples": settings.get("max_samples"),
        "stack": stack_versions(),
        "serve_command": list(settings.get("serve_command", DEFAULT_SERVE_COMMAND)),
        "tasks_metrics": aggregate_across_seeds(per_seed_metrics),
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
    init_kwargs: dict[str, Any] = {
        "project": wandb_config.get("project_name", "open-r1-tpu"),
        "mode": wandb_config.get("mode", "online"),
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


def run(settings: dict[str, Any]) -> dict[str, Any]:
    """Evaluate every seed, then reduce, persist, and report."""
    wait_for_server(settings["base_url"], int(settings["startup_timeout_secs"]))

    work_dir = Path(settings["output_dir"]).expanduser()
    work_dir.mkdir(parents=True, exist_ok=True)

    per_seed_metrics: dict[int, dict[str, dict[str, float]]] = {}
    per_seed_stats: dict[int, dict[str, Any]] = {}
    for seed in settings["seeds"]:
        metrics, stats = run_seed(settings, seed, work_dir)
        per_seed_metrics[seed] = metrics
        per_seed_stats[seed] = stats

    summary = build_summary(settings, per_seed_metrics, per_seed_stats)
    write_summary(settings["summary_path"], summary)
    LOGGER.info("Wrote evaluation summary to %s", settings["summary_path"])

    for task, metrics in sorted(summary["tasks_metrics"].items()):
        for name, stats in sorted(metrics.items()):
            std = stats["std"]
            spread = f" +/- {std:.4f}" if std is not None else " (1 seed, no spread)"
            LOGGER.info("%s %s: %.4f%s", task, name, stats["mean"], spread)

    log_summary_to_wandb(summary, settings)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML evaluation recipe")
    parser.add_argument(
        "--print-server-command",
        action="store_true",
        help="Print the vllm serve command for this recipe and exit",
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
    if args.print_server_command:
        print(shlex.join(vllm_serve_command(settings)))
        return
    run(settings)


if __name__ == "__main__":
    main()
