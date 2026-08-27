"""Score pass: attach LightEval results to the traces the ingester created.

Walks a completed tier's output directory the same way
`open_r1_tpu.evaluation.run.run_seed` wrote it -- one `seed-{seed}/{task_slug}`
directory per seed/task pair, read from the tier's own summary JSON so this
module never re-derives `eval.seeds`/`eval.tasks` itself -- and, for every
completion in every detail-parquet row, computes the same
`open_r1_tpu.tracing.hashing.content_sha256` the ingester indexed traces by.
A hit posts scores to that trace: whatever per-document numeric metric
column(s) the detail shard carries (probed, never assumed -- see
`_metric_columns`), plus `truncated` and `completion_tokens`, and tags the
trace with tier/seed/task metadata the proxy could not have known at capture
time. A miss -- the proxy was off, or there is a capture gap -- creates the
trace from the parquet row instead, tagged `source: parquet`, so the system
of record is complete either way.

Reuses `open_r1_tpu.evaluation.run.find_details_files`,
`.read_detail_responses`, `.extract_completions`, `.extract_token_counts`,
`.read_json`, and `.task_slug` rather than duplicating any of their
LightEval-version-specific probing.

`find_details_files`/`read_detail_responses` only read a local filesystem
path (`pathlib.Path.glob`) -- true of every call site in this repo today,
since the harness always runs against its own local working directory. A
`gs://` `--output-dir` is mirrored to a local temporary directory first
(`_local_mirror`) so this module's own `--output-dir` contract (local or
`gs://`) holds without changing a file this repository's ground rules forbid
touching (`open_r1_tpu.evaluation.run`).

Guard the `langfuse` import behind the `tracing` extra; see
`open_r1_tpu.tracing.ingest`'s module docstring for the same note and for the
two Langfuse Python SDK v3 limitations this module also works around
(ingestion-time-only observation timestamps; trace tagging only through a
private, defensively-guarded method).
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import tempfile
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from open_r1_tpu.evaluation.run import (
    RESPONSE_COLUMNS,
    extract_completions,
    extract_token_counts,
    find_details_files,
    read_detail_responses,
    read_json,
    task_slug,
)
from open_r1_tpu.tracing.config import load_tracing_config
from open_r1_tpu.tracing.hashing import content_sha256
from open_r1_tpu.tracing.ingest import load_state

try:
    from langfuse import Langfuse
except ImportError as error:  # pragma: no cover - exercised via the extra
    raise ImportError(
        "open_r1_tpu.tracing.scores needs the 'langfuse' package; install it "
        "with `uv pip install -e '.[tracing]'` (see docker/langfuse/README.md)."
    ) from error

LOGGER = logging.getLogger(__name__)

_NON_METRIC_COLUMNS = frozenset({*RESPONSE_COLUMNS, "doc", "__doc__"})


def _metric_columns(paths: Iterable[Path]) -> list[dict[str, float]]:
    """Per-document numeric metric values, in the same file+row order
    `read_detail_responses` reads the response column in, so row *i* here
    lines up with completion *i* from that function.

    There is no fixed name for such a column across tasks or LightEval
    releases -- some may report only task-level aggregates in the results
    file and carry no per-document metric at all -- so every column outside
    the known non-metric ones is probed for a numeric dtype rather than one
    name being assumed. An empty per-row dict is a valid, expected outcome.
    """
    rows: list[dict[str, float]] = []
    for path in paths:
        table = pq.read_table(path)
        metric_names = [
            name
            for name in table.column_names
            if name not in _NON_METRIC_COLUMNS
            and (
                pa.types.is_floating(table.schema.field(name).type)
                or pa.types.is_integer(table.schema.field(name).type)
            )
        ]
        columns = {name: table.column(name).to_pylist() for name in metric_names}
        for index in range(table.num_rows):
            rows.append({name: values[index] for name, values in columns.items()})
    return rows


def _local_mirror(output_dir: str, local_root: Path) -> Path:
    """A local copy of `output_dir`'s detail parquet shards, preserving their
    relative layout, when `output_dir` is a `gs://` URI. `find_details_files`
    globs a local path only, so this is the one place that gap is bridged --
    by copying, not by changing that function.
    """
    if not output_dir.startswith("gs://"):
        return Path(output_dir)

    import gcsfs

    fs = gcsfs.GCSFileSystem()
    remote_root = output_dir[len("gs://") :].rstrip("/")
    for remote_path in fs.find(output_dir):
        if not remote_path.endswith(".parquet"):
            continue
        relative = Path(remote_path).relative_to(remote_root)
        local_path = local_root / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        fs.get(remote_path, str(local_path))
    return local_root


def _langfuse_client(config: Mapping[str, Any]) -> Langfuse:
    # `base_url`, not the deprecated `host` alias, per the pinned SDK's docs.
    base_url = f"http://{config['langfuse']['host']}:{config['langfuse']['port']}"
    return Langfuse(base_url=base_url)


# Namespace for deterministic score ids: the same trace and score name always
# yield the same id, so re-posting after a partial failure upserts the
# existing score instead of accumulating duplicates.
_SCORE_ID_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "open-r1-tpu-tracing-score")


def _score_id(trace_id: str, name: str) -> str:
    return str(uuid.uuid5(_SCORE_ID_NAMESPACE, f"{trace_id}:{name}"))


def _scores_state_path(state_dir: str, output_dir: str) -> Path:
    """Where this output directory's created-from-parquet trace ids live.

    Re-running the score pass over the same output directory (its natural
    recovery move after a partial failure, since it exits non-zero on any)
    must not append a second generation observation to every parquet-created
    trace: `start_observation` has no upsert semantics, so idempotency comes
    from remembering, per output directory, which completion hashes already
    have a created trace.
    """
    safe = re.sub(r"[^\w.-]+", "_", output_dir.rstrip("/"))
    return Path(state_dir) / f"scores_{safe}.json"


def load_combined_index(state_dir: str) -> dict[str, str]:
    """Merge every prefix's `content_sha256 -> trace_id` index under
    `ingester.state_dir` into one map.

    A hash is content-addressed, not run-scoped, so searching across every
    run the ingester has ever processed -- rather than requiring the caller
    to know which run's prefix produced a given tier's traces -- is both
    simpler and correct: a collision between two genuinely different
    completions across two different runs is exactly as unlikely as within
    one run (see `open_r1_tpu.tracing.hashing`'s module docstring).
    """
    index: dict[str, str] = {}
    directory = Path(state_dir)
    if not directory.is_dir():
        return index
    for state_file in sorted(directory.glob("*.json")):
        state = load_state(state_file)
        index.update(state.get("content_sha256_to_trace_id", {}))
    return index


def _tag_trace(
    client: Langfuse, trace_id: str, *, tier: str, seed: int, task: str, source: str
) -> None:
    """Best-effort trace-level tags via the only mechanism found for it
    (`Client._create_trace_tags_via_ingestion`, private -- see
    `open_r1_tpu.tracing.ingest`'s module docstring). Guarded so its removal
    in a future SDK release degrades tagging, not scoring.
    """
    tagger = getattr(client, "_create_trace_tags_via_ingestion", None)
    if tagger is None:
        return
    try:
        tagger(
            trace_id=trace_id,
            tags=[f"tier:{tier}", f"seed:{seed}", f"task:{task}", f"source:{source}"],
        )
    except Exception:
        LOGGER.warning("Could not tag trace %s", trace_id, exc_info=True)


def post_scores(
    client: Langfuse,
    trace_id: str,
    fields: Mapping[str, Any],
    *,
    tier: str,
    seed: int,
    task: str,
    source: str,
) -> None:
    """Post every present numeric field as a Langfuse score on `trace_id`,
    each carrying tier/seed/task/source in its `metadata` -- a public,
    stable field on `create_score` -- so that information reaches Langfuse
    even if `_tag_trace`'s private mechanism ever stops working.
    """
    metadata = {"tier": tier, "seed": seed, "task": task, "source": source}
    posted = False
    for name, value in fields.items():
        if value is None:
            continue
        client.create_score(
            trace_id=trace_id,
            name=name,
            value=float(value),
            metadata=metadata,
            # Deterministic on (trace, name): a re-run upserts, never
            # duplicates -- see _score_id.
            score_id=_score_id(trace_id, name),
        )
        posted = True
    if not posted:
        client.create_score(
            trace_id=trace_id,
            name="tagged",
            value=1.0,
            metadata=metadata,
            score_id=_score_id(trace_id, "tagged"),
        )
    _tag_trace(client, trace_id, tier=tier, seed=seed, task=task, source=source)


def create_trace_from_parquet(
    client: Langfuse,
    *,
    completion_text: str,
    model: str | None,
    tier: str,
    seed: int,
    task: str,
) -> tuple[str, str]:
    """A trace for a document the ingester never saw (proxy was off, or a
    capture gap), tagged `source: parquet`. The trace id is deterministic on
    the completion's own hash -- prefixed so it can never collide with a
    real litellm request id's seed space -- so a re-run addresses the same
    trace. The deterministic id alone does not make re-creation harmless,
    though: `start_observation` appends a fresh generation every call, so
    `score_pass` additionally remembers created hashes per output directory
    (`_scores_state_path`) and only ever calls this once per completion.
    """
    content_hash = content_sha256(completion_text)
    trace_id = client.create_trace_id(seed=f"parquet:{content_hash}")
    generation = client.start_observation(
        trace_context={"trace_id": trace_id},
        name="lighteval-detail-row",
        as_type="generation",
        output=completion_text,
        model=model,
        metadata={"content_sha256": content_hash, "source": "parquet"},
    )
    generation.end()
    return trace_id, content_hash


def score_pass(
    config: Mapping[str, Any],
    *,
    output_dir: str,
    summary_path: str,
    client: Langfuse | None = None,
) -> dict[str, int]:
    """Score every completion in a tier's output directory. Returns
    `{"matched": n, "created": n, "failed": n}`.
    """
    summary = read_json(summary_path)
    model = summary.get("served_model_name") or summary.get("model_path")
    max_new_tokens = int(summary["sampling"]["max_new_tokens"])
    index = load_combined_index(str(config["ingester"]["state_dir"]))

    state_path = _scores_state_path(str(config["ingester"]["state_dir"]), output_dir)
    parquet_created: dict[str, str] = {}
    if state_path.exists():
        parquet_created = dict(
            json.loads(state_path.read_text(encoding="utf-8")).get("parquet_traces", {})
        )
    seen_hashes: set[str] = set()

    owns_client = client is None
    resolved_client: Langfuse = (
        client if client is not None else _langfuse_client(config)
    )

    counts = {"matched": 0, "created": 0, "failed": 0}
    with tempfile.TemporaryDirectory(prefix="open-r1-tpu-tracing-mirror-") as tmp:
        local_output_dir = _local_mirror(output_dir, Path(tmp))
        try:
            for seed in summary["seeds"]:
                for task in summary["tasks"]:
                    task_dir = local_output_dir / f"seed-{seed}" / task_slug(task)
                    paths = find_details_files(task_dir)
                    if not paths:
                        continue
                    responses = read_detail_responses(paths)
                    metric_rows = _metric_columns(paths)
                    for response, metrics in zip(responses, metric_rows, strict=True):
                        try:
                            completions = extract_completions(response)
                        except ValueError:
                            LOGGER.warning(
                                "seed %s %s: unreadable detail row",
                                seed,
                                task,
                                exc_info=True,
                            )
                            counts["failed"] += 1
                            continue
                        token_counts = extract_token_counts(response)
                        for position, completion_text in enumerate(completions):
                            try:
                                completion_tokens = (
                                    token_counts[position]
                                    if token_counts and position < len(token_counts)
                                    else None
                                )
                                truncated = (
                                    int(completion_tokens >= max_new_tokens)
                                    if completion_tokens is not None
                                    else None
                                )
                                fields = {
                                    **metrics,
                                    "completion_tokens": completion_tokens,
                                    "truncated": truncated,
                                }
                                content_hash = content_sha256(completion_text)
                                if content_hash in seen_hashes:
                                    LOGGER.warning(
                                        "seed %s %s: completion hash %s repeats "
                                        "within this output directory -- "
                                        "identical completions collide on the "
                                        "content hash, so these documents' "
                                        "scores land on one shared trace",
                                        seed,
                                        task,
                                        content_hash,
                                    )
                                seen_hashes.add(content_hash)
                                trace_id = index.get(content_hash)
                                source = "proxy"
                                if trace_id is None:
                                    # Created by an earlier (partial) run of
                                    # this same score pass: reuse it rather
                                    # than appending a duplicate generation.
                                    trace_id = parquet_created.get(content_hash)
                                    source = "parquet"
                                if trace_id is not None:
                                    post_scores(
                                        resolved_client,
                                        trace_id,
                                        fields,
                                        tier=str(summary.get("tier")),
                                        seed=int(seed),
                                        task=str(task),
                                        source=source,
                                    )
                                    counts["matched"] += 1
                                else:
                                    trace_id, _ = create_trace_from_parquet(
                                        resolved_client,
                                        completion_text=completion_text,
                                        model=model,
                                        tier=str(summary.get("tier")),
                                        seed=int(seed),
                                        task=str(task),
                                    )
                                    parquet_created[content_hash] = trace_id
                                    post_scores(
                                        resolved_client,
                                        trace_id,
                                        fields,
                                        tier=str(summary.get("tier")),
                                        seed=int(seed),
                                        task=str(task),
                                        source="parquet",
                                    )
                                    counts["created"] += 1
                            except Exception:
                                LOGGER.warning(
                                    "seed %s %s: could not score a completion",
                                    seed,
                                    task,
                                    exc_info=True,
                                )
                                counts["failed"] += 1
        finally:
            if parquet_created:
                state_path.parent.mkdir(parents=True, exist_ok=True)
                state_path.write_text(
                    json.dumps(
                        {"parquet_traces": parquet_created},
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
            if owns_client:
                resolved_client.flush()
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML tracing config")
    parser.add_argument(
        "--output-dir", required=True, help="Tier output directory (local or gs://)"
    )
    parser.add_argument(
        "--summary", required=True, help="Tier summary JSON path (local or gs://)"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    config = load_tracing_config(args.config)
    counts = score_pass(config, output_dir=args.output_dir, summary_path=args.summary)
    LOGGER.info(
        "matched=%d created=%d failed=%d",
        counts["matched"],
        counts["created"],
        counts["failed"],
    )
    if counts["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
