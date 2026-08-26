"""Compare vLLM service generation speed with Tunix's direct sampler.

Accuracy harness timings are not backend benchmarks. LightEval currently sends
one HTTP request at a time, while vLLM's main advantage is serving concurrent
requests; Tunix's sampler compiles once per static batch shape and then decodes
the whole batch in-process. This module measures both at matching batch sizes
with the same merged weights, rendered prompts, greedy decoding, and output
token budget.

Model loading/server startup and the first compilation are recorded separately
from steady-state generation. Fixed-length mode disables normal end-of-turn
stopping so both backends perform the same number of decode steps; these runs
measure speed, not answer quality or termination behavior.

The launcher runs the backends sequentially because only one process can own a
TPU chip::

    ./scripts/benchmark_generation_tpu.sh
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from open_r1_tpu.evaluation.run import (
    load_eval_config,
    wait_for_server,
)
from open_r1_tpu.evaluation.run import (
    resolve_settings as resolve_eval_settings,
)

DEFAULT_EVAL_CONFIG = "recipes/Qwen3-1.7B-Math/eval/tier0_smoke.yaml"
DEFAULT_SFT_CONFIG = "recipes/Qwen3-1.7B-Math/sft/config_distill.yaml"


@dataclass(frozen=True)
class BatchOutput:
    """The token counts and optional per-request latencies from one batch."""

    completion_tokens: tuple[int, ...]
    request_seconds: tuple[float, ...] = ()


def benchmark_questions(count: int) -> list[str]:
    """Build deterministic, distinct prompts so prefix caching cannot help."""
    if count <= 0:
        raise ValueError("prompt count must be positive")
    return [
        (
            f"A shop has {41 + index} boxes with {13 + (index % 11)} items in "
            "each box and then sells "
            f"{17 + 3 * index} items. How many items remain? Show your reasoning."
        )
        for index in range(count)
    ]


def render_prompts(
    tokenizer: Any, questions: Sequence[str], system_prompt: str | None
) -> list[str]:
    """Render the exact Qwen chat prefix consumed by both backends."""
    rendered: list[str] = []
    for question in questions:
        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": question})
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt, str):
            raise TypeError("tokenizer.apply_chat_template did not return text")
        rendered.append(prompt)
    return rendered


def prompt_digest(prompts: Sequence[str]) -> str:
    """Fingerprint rendered prompts so two result files prove comparability."""
    payload = json.dumps(list(prompts), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def split_batches(values: Sequence[str], batch_size: int) -> list[list[str]]:
    """Split a workload into equal static batches."""
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if len(values) % batch_size:
        raise ValueError(
            f"{len(values)} prompts is not divisible by batch size {batch_size}; "
            "a smaller final batch would trigger another Tunix compilation"
        )
    return [
        list(values[offset : offset + batch_size])
        for offset in range(0, len(values), batch_size)
    ]


def _percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(probability * len(ordered)) - 1)
    return float(ordered[index])


def _summarize_measurement(
    *,
    backend: str,
    batch_size: int,
    repeats: int,
    prompt_tokens: Sequence[int],
    elapsed_seconds: float,
    warmup_seconds: float,
    completion_tokens: Sequence[int],
    batch_seconds: Sequence[float],
    request_seconds: Sequence[float],
) -> dict[str, Any]:
    samples = len(completion_tokens)
    total_completion_tokens = sum(completion_tokens)
    total_prompt_tokens = sum(prompt_tokens) * repeats
    return {
        "backend": backend,
        "batch_size": batch_size,
        "repeats": repeats,
        "samples": samples,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "elapsed_seconds": elapsed_seconds,
        "warmup_seconds": warmup_seconds,
        "samples_per_second": samples / elapsed_seconds,
        "completion_tokens_per_second": total_completion_tokens / elapsed_seconds,
        "total_tokens_per_second": (total_prompt_tokens + total_completion_tokens)
        / elapsed_seconds,
        "batch_latency_seconds": {
            "mean": statistics.fmean(batch_seconds),
            "p50": _percentile(batch_seconds, 0.50),
            "p95": _percentile(batch_seconds, 0.95),
        },
        "request_latency_seconds": (
            {
                "mean": statistics.fmean(request_seconds),
                "p50": _percentile(request_seconds, 0.50),
                "p95": _percentile(request_seconds, 0.95),
            }
            if request_seconds
            else None
        ),
    }


def run_workload(
    *,
    backend: str,
    prompts: Sequence[str],
    prompt_tokens: Sequence[int],
    batch_sizes: Sequence[int],
    repeats: int,
    max_new_tokens: int,
    run_batch: Callable[[Sequence[str], int], BatchOutput],
) -> list[dict[str, Any]]:
    """Warm and time every static batch shape through one backend."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if len(prompts) != len(prompt_tokens):
        raise ValueError("prompts and prompt token counts must have equal length")

    measurements: list[dict[str, Any]] = []
    for batch_size in batch_sizes:
        batches = split_batches(prompts, batch_size)

        warmup_started = time.perf_counter()
        warmup = run_batch(batches[0], batch_size)
        warmup_seconds = time.perf_counter() - warmup_started
        if len(warmup.completion_tokens) != batch_size:
            raise RuntimeError(
                f"{backend} warmup returned {len(warmup.completion_tokens)} "
                f"completions for batch size {batch_size}"
            )

        completion_counts: list[int] = []
        batch_latencies: list[float] = []
        request_latencies: list[float] = []
        timed_started = time.perf_counter()
        for _ in range(repeats):
            for batch in batches:
                batch_started = time.perf_counter()
                output = run_batch(batch, batch_size)
                batch_latencies.append(time.perf_counter() - batch_started)
                if len(output.completion_tokens) != len(batch):
                    raise RuntimeError(
                        f"{backend} returned {len(output.completion_tokens)} "
                        f"completions for {len(batch)} prompts"
                    )
                completion_counts.extend(output.completion_tokens)
                request_latencies.extend(output.request_seconds)
        elapsed_seconds = time.perf_counter() - timed_started

        expected = len(prompts) * repeats
        if len(completion_counts) != expected:
            raise RuntimeError(
                f"{backend} produced {len(completion_counts)} completions, "
                f"expected {expected}"
            )
        short = [count for count in completion_counts if count != max_new_tokens]
        if short:
            raise RuntimeError(
                f"{backend} did not perform the fixed-length decode: "
                f"{len(short)} of {len(completion_counts)} completions were not "
                f"{max_new_tokens} tokens (range {min(short)}..{max(short)})"
            )

        measurements.append(
            _summarize_measurement(
                backend=backend,
                batch_size=batch_size,
                repeats=repeats,
                prompt_tokens=prompt_tokens,
                elapsed_seconds=elapsed_seconds,
                warmup_seconds=warmup_seconds,
                completion_tokens=completion_counts,
                batch_seconds=batch_latencies,
                request_seconds=request_latencies,
            )
        )
    return measurements


def _request_json(
    url: str, payload: Mapping[str, Any] | None = None, timeout: int = 900
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            decoded = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{url} returned HTTP {error.code}: {detail}") from error
    if not isinstance(decoded, dict):
        raise ValueError(f"{url} did not return a JSON object")
    return decoded


def vllm_completion_payload(
    *, model_name: str, prompt: str, max_new_tokens: int, seed: int
) -> dict[str, Any]:
    """Build a fixed-length greedy request accepted by vLLM's endpoint."""
    return {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        # vLLM extension: the speed run must execute the same number of decode
        # steps as Tunix, independent of whether the model emits an EOS token.
        "ignore_eos": True,
    }


def _run_vllm(
    *,
    base_url: str,
    model_name: str,
    prompts: Sequence[str],
    prompt_tokens: Sequence[int],
    batch_sizes: Sequence[int],
    repeats: int,
    max_new_tokens: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    wait_for_server(base_url)
    endpoint = base_url.rstrip("/") + "/completions"
    executors: dict[int, concurrent.futures.ThreadPoolExecutor] = {}

    def one_completion(prompt: str) -> tuple[int, float]:
        started = time.perf_counter()
        response = _request_json(
            endpoint,
            vllm_completion_payload(
                model_name=model_name,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                seed=seed,
            ),
        )
        request_seconds = time.perf_counter() - started
        usage = response.get("usage")
        if not isinstance(usage, Mapping) or not isinstance(
            usage.get("completion_tokens"), int
        ):
            raise ValueError("vLLM response omitted usage.completion_tokens")
        return int(usage["completion_tokens"]), request_seconds

    def run_batch(batch: Sequence[str], batch_size: int) -> BatchOutput:
        # Concurrent individual requests match how an async evaluation harness
        # would expose work to vLLM's continuous batching. Batch size one is the
        # current LightEval path.
        if batch_size == 1:
            results = [one_completion(batch[0])]
        else:
            executor = executors.setdefault(
                batch_size,
                concurrent.futures.ThreadPoolExecutor(max_workers=batch_size),
            )
            results = list(executor.map(one_completion, batch))
        return BatchOutput(
            completion_tokens=tuple(result[0] for result in results),
            request_seconds=tuple(result[1] for result in results),
        )

    try:
        measurements = run_workload(
            backend="vllm",
            prompts=prompts,
            prompt_tokens=prompt_tokens,
            batch_sizes=batch_sizes,
            repeats=repeats,
            max_new_tokens=max_new_tokens,
            run_batch=run_batch,
        )
    finally:
        for executor in executors.values():
            executor.shutdown()
    version: str | None = None
    try:
        root_url = base_url.rstrip("/").removesuffix("/v1")
        version_response = _request_json(root_url + "/version")
        if isinstance(version_response.get("version"), str):
            version = version_response["version"]
    except (OSError, RuntimeError, ValueError):
        pass
    return measurements, {"vllm": version}


def _run_tunix(
    *,
    model_path: str,
    sft_config_path: str,
    prompts: Sequence[str],
    prompt_tokens: Sequence[int],
    batch_sizes: Sequence[int],
    repeats: int,
    max_new_tokens: int,
    max_prompt_length: int,
    seed: int,
    use_flash_attention: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    startup_started = time.perf_counter()

    import jax
    from tunix.cli.utils import model as model_utils
    from tunix.generate import sampler as sampler_lib
    from tunix.utils import mesh as mesh_utils

    from open_r1_tpu.core.config import load_config
    from open_r1_tpu.training.run import _create_model

    config = copy.deepcopy(load_config(sft_config_path))
    model_config = config["model"]
    model_config["model_source"] = "local"
    model_config["model_path"] = model_path
    model_config["use_flash_attention"] = use_flash_attention
    # Rematerialization only benefits training and adds no value to inference.
    model_config["remat_config"] = "NONE"
    model_config.pop("lora_config", None)

    mesh_shape = tuple(int(value) for value in model_config["mesh"]["shape"])
    axis_names = tuple(str(value) for value in model_config["mesh"]["axis_names"])
    if math.prod(mesh_shape) != jax.device_count():
        raise ValueError(
            f"configured mesh {mesh_shape} needs {math.prod(mesh_shape)} devices, "
            f"but JAX sees {jax.device_count()}"
        )
    if use_flash_attention:
        block_size = int(model_config.get("flash_attention_block_size", 1024))
        if max_prompt_length < block_size or max_prompt_length % block_size:
            raise ValueError(
                "Tunix flash attention requires max_prompt_length to be at "
                f"least, and divisible by, its block size ({block_size})"
            )

    mesh = mesh_utils.create_mesh(mesh_shape, axis_names)
    model, tokenizer_path = _create_model(config, mesh)
    tokenizer_config = dict(config["tokenizer"])
    tokenizer_config["tokenizer_path"] = model_path
    tokenizer = model_utils.create_tokenizer(tokenizer_config, tokenizer_path)

    model_details = getattr(model, "config", None)
    if model_details is None:
        raise ValueError("Tunix model exposes no config for KV-cache sizing")
    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=max_prompt_length + max_new_tokens,
            num_layers=int(model_details.num_layers),
            num_kv_heads=int(model_details.num_kv_heads),
            head_dim=int(model_details.head_dim),
        ),
    )
    startup_seconds = time.perf_counter() - startup_started

    def run_batch(batch: Sequence[str], _batch_size: int) -> BatchOutput:
        output = sampler(
            input_strings=list(batch),
            max_generation_steps=max_new_tokens,
            max_prompt_length=max_prompt_length,
            temperature=0.0,
            seed=seed,
            # -1 cannot be emitted by the tokenizer, so this matches vLLM's
            # ignore_eos=True and forces exactly max_generation_steps.
            eos_tokens=[-1],
            pad_output=True,
        )
        return BatchOutput(
            completion_tokens=tuple(len(tokens) for tokens in output.tokens)
        )

    measurements = run_workload(
        backend="tunix",
        prompts=prompts,
        prompt_tokens=prompt_tokens,
        batch_sizes=batch_sizes,
        repeats=repeats,
        max_new_tokens=max_new_tokens,
        run_batch=run_batch,
    )
    runtime = {
        "google-tunix": importlib.metadata.version("google-tunix"),
        "jax": importlib.metadata.version("jax"),
        "devices": [
            {
                "platform": device.platform,
                "device_kind": device.device_kind,
            }
            for device in jax.devices()
        ],
        "use_flash_attention": use_flash_attention,
    }
    return measurements, runtime, startup_seconds


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return value


def build_result(
    *,
    backend: str,
    model_path: str,
    prompt_hash: str,
    max_new_tokens: int,
    max_prompt_length: int,
    prompt_count: int,
    repeats: int,
    batch_sizes: Sequence[int],
    measurements: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    startup_seconds: float | None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "backend": backend,
        "model_path": str(Path(model_path).expanduser().resolve()),
        "prompt_sha256": prompt_hash,
        "config": {
            "batch_sizes": list(batch_sizes),
            "prompt_count": prompt_count,
            "repeats": repeats,
            "max_new_tokens": max_new_tokens,
            "max_prompt_length": max_prompt_length,
            "temperature": 0.0,
            "fixed_length": True,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            **runtime,
        },
        "startup_seconds": startup_seconds,
        "measurements": list(measurements),
    }


def _comparison_rows(
    vllm: Mapping[str, Any], tunix: Mapping[str, Any]
) -> list[dict[str, Any]]:
    comparable = (
        "model_path",
        "prompt_sha256",
        "config",
    )
    for key in comparable:
        if vllm.get(key) != tunix.get(key):
            raise ValueError(f"result files differ on {key}; comparison is invalid")

    vllm_by_batch = {
        int(row["batch_size"]): row for row in vllm.get("measurements", [])
    }
    tunix_by_batch = {
        int(row["batch_size"]): row for row in tunix.get("measurements", [])
    }
    if set(vllm_by_batch) != set(tunix_by_batch):
        raise ValueError("result files do not contain the same batch sizes")

    rows: list[dict[str, Any]] = []
    for batch_size in sorted(vllm_by_batch):
        vllm_row = vllm_by_batch[batch_size]
        tunix_row = tunix_by_batch[batch_size]
        vllm_rate = float(vllm_row["completion_tokens_per_second"])
        tunix_rate = float(tunix_row["completion_tokens_per_second"])
        rows.append(
            {
                "batch_size": batch_size,
                "vllm_completion_tokens_per_second": vllm_rate,
                "tunix_completion_tokens_per_second": tunix_rate,
                "tunix_over_vllm": tunix_rate / vllm_rate,
                "vllm_samples_per_second": float(vllm_row["samples_per_second"]),
                "tunix_samples_per_second": float(tunix_row["samples_per_second"]),
                "vllm_elapsed_seconds": float(vllm_row["elapsed_seconds"]),
                "tunix_elapsed_seconds": float(tunix_row["elapsed_seconds"]),
            }
        )
    return rows


def build_comparison(
    vllm: Mapping[str, Any], tunix: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "model_path": vllm.get("model_path"),
        "prompt_sha256": vllm.get("prompt_sha256"),
        "config": vllm.get("config"),
        "startup_seconds": {
            "vllm": vllm.get("startup_seconds"),
            "tunix": tunix.get("startup_seconds"),
        },
        "runtime": {
            "vllm": vllm.get("runtime"),
            "tunix": tunix.get("runtime"),
        },
        "rows": _comparison_rows(vllm, tunix),
    }


def comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """Render the concise human-readable comparison beside the raw JSON."""
    rows = comparison.get("rows", [])
    lines = [
        "# vLLM versus Tunix generation speed",
        "",
        "Steady-state greedy generation using the same merged weights and "
        "rendered prompts. Startup and one warm-up batch per shape are excluded "
        "from timed throughput. Generation is forced to the configured token "
        "cap, so this measures decode speed rather than answer quality or "
        "termination.",
        "",
        "| Batch/concurrency | vLLM output tok/s | Tunix output tok/s | "
        "Tunix / vLLM | vLLM samples/s | Tunix samples/s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['batch_size']} | "
            f"{row['vllm_completion_tokens_per_second']:.2f} | "
            f"{row['tunix_completion_tokens_per_second']:.2f} | "
            f"{row['tunix_over_vllm']:.2f}x | "
            f"{row['vllm_samples_per_second']:.3f} | "
            f"{row['tunix_samples_per_second']:.3f} |"
        )

    startup = comparison.get("startup_seconds", {})
    lines.extend(
        [
            "",
            "Startup (weight loading and service/sampler construction, before "
            f"warm-up): vLLM {_format_seconds(startup.get('vllm'))}; Tunix "
            f"{_format_seconds(startup.get('tunix'))}.",
            "",
            "Batch/concurrency 1 represents the current serial LightEval request "
            "path. Higher values compare vLLM concurrent HTTP requests with one "
            "static Tunix batch. HTTP client and response decoding are included "
            "for vLLM; Tunix output transfer and decoding are included for Tunix.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_seconds(value: Any) -> str:
    return "not recorded" if value is None else f"{float(value):.2f}s"


def _load_tokenizer(model_path: str) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_path)


def _parse_batch_sizes(values: Sequence[int]) -> list[int]:
    batch_sizes = [int(value) for value in values]
    if not batch_sizes or any(value <= 0 for value in batch_sizes):
        raise ValueError("batch sizes must be positive")
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("batch sizes must not repeat")
    return batch_sizes


def _add_run_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("run", help="run one generation backend")
    parser.add_argument("--backend", choices=("vllm", "tunix"), required=True)
    parser.add_argument("--eval-config", default=DEFAULT_EVAL_CONFIG)
    parser.add_argument("--sft-config", default=DEFAULT_SFT_CONFIG)
    parser.add_argument("--model-path")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--prompt-count", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--startup-seconds", type=float)
    parser.add_argument(
        "--tunix-flash-attention",
        action="store_true",
        help="Use Tunix splash attention; requires a block-aligned prompt length",
    )


def _add_compare_parser(subparsers: Any) -> None:
    parser = subparsers.add_parser("compare", help="combine two backend results")
    parser.add_argument("--vllm", required=True, help="vLLM result JSON")
    parser.add_argument("--tunix", required=True, help="Tunix result JSON")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_parser(subparsers)
    _add_compare_parser(subparsers)
    return parser.parse_args()


def _run_command(args: argparse.Namespace) -> None:
    if args.prompt_count <= 0:
        raise ValueError("prompt count must be positive")
    if args.max_new_tokens <= 0 or args.max_prompt_length <= 0:
        raise ValueError("token lengths must be positive")
    batch_sizes = _parse_batch_sizes(args.batch_sizes)
    for batch_size in batch_sizes:
        if args.prompt_count % batch_size:
            raise ValueError(
                f"prompt count {args.prompt_count} must be divisible by batch "
                f"size {batch_size}"
            )

    eval_settings = resolve_eval_settings(load_eval_config(args.eval_config))
    model_path = str(args.model_path or eval_settings["model_path"])
    tokenizer = _load_tokenizer(model_path)
    questions = benchmark_questions(args.prompt_count)
    # The recipe's own prompt, verbatim -- None renders with no system message.
    # A recipe that deliberately sets no prompt must be benchmarked that way;
    # substituting one here would benchmark a prompt the recipe never asked for.
    prompts = render_prompts(tokenizer, questions, eval_settings["system_prompt"])
    prompt_tokens = [len(tokenizer.encode(prompt)) for prompt in prompts]
    longest = max(prompt_tokens)
    if longest > args.max_prompt_length:
        raise ValueError(
            f"longest rendered prompt is {longest} tokens but "
            f"max_prompt_length is {args.max_prompt_length}"
        )

    startup_seconds = args.startup_seconds
    if args.backend == "vllm":
        served_model_name = (
            Path(model_path).name
            if args.model_path is not None
            else eval_settings["served_model_name"]
        )
        measurements, runtime = _run_vllm(
            base_url=eval_settings["base_url"],
            model_name=served_model_name,
            prompts=prompts,
            prompt_tokens=prompt_tokens,
            batch_sizes=batch_sizes,
            repeats=args.repeats,
            max_new_tokens=args.max_new_tokens,
            seed=args.seed,
        )
    else:
        measurements, runtime, startup_seconds = _run_tunix(
            model_path=model_path,
            sft_config_path=args.sft_config,
            prompts=prompts,
            prompt_tokens=prompt_tokens,
            batch_sizes=batch_sizes,
            repeats=args.repeats,
            max_new_tokens=args.max_new_tokens,
            max_prompt_length=args.max_prompt_length,
            seed=args.seed,
            use_flash_attention=args.tunix_flash_attention,
        )

    result = build_result(
        backend=args.backend,
        model_path=model_path,
        prompt_hash=prompt_digest(prompts),
        max_new_tokens=args.max_new_tokens,
        max_prompt_length=args.max_prompt_length,
        prompt_count=args.prompt_count,
        repeats=args.repeats,
        batch_sizes=batch_sizes,
        measurements=measurements,
        runtime=runtime,
        startup_seconds=startup_seconds,
    )
    write_json(args.output, result)
    print(f"Wrote {args.backend} benchmark to {args.output}")
    for row in measurements:
        print(
            f"batch {row['batch_size']}: "
            f"{row['completion_tokens_per_second']:.2f} output tok/s, "
            f"{row['samples_per_second']:.3f} samples/s"
        )


def _compare_command(args: argparse.Namespace) -> None:
    comparison = build_comparison(read_json(args.vllm), read_json(args.tunix))
    write_json(args.output_json, comparison)
    markdown_path = Path(args.output_markdown).expanduser()
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(comparison_markdown(comparison), encoding="utf-8")
    print(f"Wrote comparison to {args.output_json} and {args.output_markdown}")


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        _run_command(args)
    else:
        _compare_command(args)


if __name__ == "__main__":
    main()
