"""GCS -> Langfuse ingester: turn captured trace-proxy payloads into Langfuse
traces.

Reads objects written by `docker/trace-proxy/gcs_logger.py` (see its module
docstring) from a GCS -- or, for tests, `file://` -- prefix, and creates one
Langfuse trace with one generation observation per object: input messages,
output text, token usage, and model name. Idempotent through two mechanisms
with distinct jobs: the per-prefix state file prevents replay against the
same Langfuse instance (an object recorded there is never re-sent, and an
object whose write failed is never recorded, so a crashed ingester restarts
safely), while the deterministic trace id -- Langfuse's own
`Client.create_trace_id(seed=...)`, seeded on the litellm request id --
makes a from-scratch backlog pass against a freshly rebuilt Langfuse (see
docker/langfuse/README.md's rebuild-on-provision story) reproduce every
trace under the same id it had before the VM was deleted.

Two disclosed limitations, both a consequence of the Langfuse Python SDK v3
being OpenTelemetry-native rather than built for backfilling historical
events (verified by reading `langfuse/_client/client.py` and
`langfuse/_client/span.py` at the pinned version -- there is no public
override for either):

- `Client.start_observation` has no public `start_time` parameter, so a
  generation it creates is always timestamped at ingestion time, not the
  original capture time. Replaying a backlog therefore does not reproduce
  the real wall-clock timing as the observation's own span. Nothing is lost,
  though: the real capture timing is carried in the generation's `metadata`
  (`capture_start_time`/`capture_end_time`) -- it just does not draw the
  timeline in the UI. (`.end(end_time=...)` does exist, but calling it with
  the true historical end time while the span's start time is "now" would
  usually make the span end before it starts, which is worse than not
  trying.)
- Trace-level tags have no public by-id API. `Client._create_trace_tags_via_ingestion`
  is the only mechanism found, and it is private (leading underscore, no
  public wrapper). Used defensively, guarded by `hasattr`: its absence in
  some future SDK release degrades tagging, not ingestion.

Guard the `langfuse` import behind the `tracing` extra:

    uv venv .venv-tracing && uv pip install --python .venv-tracing -e ".[tracing]"

Launch, on the VM, as a standalone loop in its own tmux session (matching how
tiers are launched elsewhere in this project) -- placeholders only:

    tmux new-session -d -s trace-ingest \
      '.venv-tracing/bin/python -m open_r1_tpu.tracing.ingest \
        --config configs/tracing.yaml'

`--once` processes the current backlog and exits; used by tests and by the
rebuild-on-provision pass in docker/langfuse/README.md. Without it, the
ingester polls forever at `ingester.poll_secs`.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import fsspec

from open_r1_tpu.tracing.config import load_tracing_config
from open_r1_tpu.tracing.hashing import content_sha256

try:
    from langfuse import Langfuse
except ImportError as error:  # pragma: no cover - exercised via the extra
    raise ImportError(
        "open_r1_tpu.tracing.ingest needs the 'langfuse' package; install it "
        "with `uv pip install -e '.[tracing]'` (see docker/langfuse/README.md)."
    ) from error

LOGGER = logging.getLogger(__name__)

# The common places litellm (or a future release of it) might put completion
# text in a StandardLoggingPayload's `response` field, probed in the same
# defensive style as `open_r1_tpu.evaluation.run.extract_completions`.
_TEXT_KEYS = ("text", "final_text", "generated_text", "content")


def _extract_response_text(response: Any) -> str | None:
    """Pull the completion text out of a payload's `response` field.

    Handles a plain string, an OpenAI-style
    `{"choices": [{"message": {"content": ...}}]}` object (what litellm's
    StandardLoggingPayload carries at the pinned version), and -- defensively,
    since litellm's shapes have moved before -- any mapping carrying one of
    the common text keys. Returns None, never raises, so one object with an
    unrecognised shape does not stop the rest of a backlog from ingesting.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, Mapping):
                message = first.get("message")
                if isinstance(message, Mapping) and isinstance(
                    message.get("content"), str
                ):
                    return message["content"]
        for key in _TEXT_KEYS:
            value = response.get(key)
            if isinstance(value, str):
                return value
    return None


def parse_payload(
    payload: Mapping[str, Any], *, object_name: str
) -> dict[str, Any] | None:
    """Reduce one captured StandardLoggingPayload object to what a trace
    needs. Returns None -- logged, never raised -- for a payload missing
    messages or a readable response, which is what a full/minimal/malformed
    fixture mix in the test suite exercises.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        LOGGER.warning("%s: no usable 'messages'; skipping", object_name)
        return None
    response_text = _extract_response_text(payload.get("response"))
    if response_text is None:
        LOGGER.warning("%s: no readable completion text; skipping", object_name)
        return None

    request_id = str(payload.get("id") or object_name)
    return {
        "request_id": request_id,
        "messages": messages,
        "response_text": response_text,
        "model": payload.get("model"),
        "prompt_tokens": payload.get("prompt_tokens"),
        "completion_tokens": payload.get("completion_tokens"),
        "total_tokens": payload.get("total_tokens"),
        "start_time": payload.get("startTime"),
        "end_time": payload.get("endTime"),
        "content_sha256": content_sha256(response_text),
    }


def prefix_url(config: Mapping[str, Any], prefix: str | None) -> str:
    """Resolve the fsspec URL to walk.

    An explicit `--prefix` that already names a scheme (`file://` in tests,
    `gs://` for a narrower real prefix) is used as-is. Otherwise, walk the
    bucket at `gcs.prefix_template`'s static leading segment -- everything
    before its first `{` -- which is where every run's timestamped
    sub-prefix lives.
    """
    if prefix:
        if "://" in prefix:
            return prefix
        return f"gs://{config['gcs']['bucket']}/{prefix.strip('/')}"
    template = str(config["gcs"]["prefix_template"])
    root = template.split("{", 1)[0].strip("/")
    bucket = str(config["gcs"]["bucket"])
    return f"gs://{bucket}/{root}" if root else f"gs://{bucket}"


def _state_path(state_dir: str, url: str) -> Path:
    safe = re.sub(r"[^\w.-]+", "_", url)
    return Path(state_dir) / f"{safe}.json"


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"processed": [], "content_sha256_to_trace_id": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def _langfuse_client(config: Mapping[str, Any]) -> Langfuse:
    """A client for this config's Langfuse instance.

    `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` are read from the environment
    by the SDK itself -- they are secrets, never config (see
    `open_r1_tpu.tracing.config`'s module docstring) -- so only the host,
    which *is* config, is passed explicitly.
    """
    # `base_url`, not the deprecated `host` alias, per the pinned SDK's docs.
    base_url = f"http://{config['langfuse']['host']}:{config['langfuse']['port']}"
    return Langfuse(base_url=base_url)


def write_trace(client: Langfuse, record: Mapping[str, Any], *, run_prefix: str) -> str:
    """Create one Langfuse trace + generation observation and return its
    (deterministic) trace id.
    """
    trace_id = client.create_trace_id(seed=record["request_id"])
    usage_details = {
        key: value
        for key, value in (
            ("input", record.get("prompt_tokens")),
            ("output", record.get("completion_tokens")),
            ("total", record.get("total_tokens")),
        )
        if isinstance(value, int)
    }
    generation = client.start_observation(
        trace_context={"trace_id": trace_id},
        name="litellm-completion",
        as_type="generation",
        input=record["messages"],
        output=record["response_text"],
        model=record.get("model"),
        usage_details=usage_details or None,
        metadata={
            "run_prefix": run_prefix,
            "content_sha256": record["content_sha256"],
            "capture_start_time": record.get("start_time"),
            "capture_end_time": record.get("end_time"),
        },
    )
    generation.end()
    return trace_id


def ingest_once(
    config: Mapping[str, Any],
    *,
    prefix: str | None = None,
    client: Langfuse | None = None,
) -> dict[str, int]:
    """Process every not-yet-seen object under the prefix once.

    Returns `{"ingested": n, "skipped": n, "failed": n}` for the caller to
    log or assert on. `skipped` counts objects already recorded in this
    prefix's state file (the idempotency the module docstring promises);
    `failed` counts objects read but not usable (`parse_payload` returned
    None) as well as objects that could not be read or parsed as JSON at all
    -- both are still marked processed, since retrying a malformed object
    forever would never succeed.
    """
    url = prefix_url(config, prefix)
    fs, root = fsspec.core.url_to_fs(url)
    state_path = _state_path(str(config["ingester"]["state_dir"]), url)
    state = load_state(state_path)
    processed = set(state["processed"])
    index = dict(state["content_sha256_to_trace_id"])

    owns_client = client is None
    resolved_client: Langfuse = (
        client if client is not None else _langfuse_client(config)
    )

    counts = {"ingested": 0, "skipped": 0, "failed": 0}
    try:
        for path in sorted(fs.find(root)):
            if not path.endswith(".json"):
                continue
            object_name = Path(path).name
            if object_name in processed:
                counts["skipped"] += 1
                continue

            try:
                with fs.open(path, "rt") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError) as error:
                LOGGER.warning("%s: could not read/parse: %s", path, error)
                processed.add(object_name)
                counts["failed"] += 1
                continue

            record = parse_payload(payload, object_name=object_name)
            if record is None:
                # Malformed forever: mark processed so it is not retried.
                processed.add(object_name)
                counts["failed"] += 1
                continue

            # Marked processed only after the write returns: if write_trace
            # raises mid-backlog, the `finally` below persists state without
            # this object, so a restarted ingester retries it instead of
            # skipping a trace that was never written.
            trace_id = write_trace(resolved_client, record, run_prefix=url)
            content_hash = record["content_sha256"]
            existing = index.get(content_hash)
            if existing is not None and existing != trace_id:
                LOGGER.warning(
                    "%s: completion hash %s already indexed to trace %s; "
                    "overwriting with %s -- identical completions from "
                    "different documents collide on this hash, and the score "
                    "pass will attach both documents' scores to one trace",
                    object_name,
                    content_hash,
                    existing,
                    trace_id,
                )
            index[content_hash] = trace_id
            processed.add(object_name)
            counts["ingested"] += 1
    finally:
        state["processed"] = sorted(processed)
        state["content_sha256_to_trace_id"] = index
        save_state(state_path, state)
        if owns_client:
            resolved_client.flush()
    return counts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML tracing config")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Narrow to one run's prefix (full URL or a sub-path under the bucket); "
        "default walks the whole configured prefix root",
    )
    parser.add_argument(
        "--once", action="store_true", help="Process the current backlog, then exit"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args()
    config = load_tracing_config(args.config)
    while True:
        counts = ingest_once(config, prefix=args.prefix)
        LOGGER.info(
            "ingested=%d skipped=%d failed=%d",
            counts["ingested"],
            counts["skipped"],
            counts["failed"],
        )
        if args.once:
            return
        time.sleep(int(config["ingester"]["poll_secs"]))


if __name__ == "__main__":
    main()
