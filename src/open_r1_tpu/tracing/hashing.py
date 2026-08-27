"""Content hash shared by the ingester and the score pass.

This is the join key between a trace captured by the proxy (from the
response it saw) and a LightEval detail-parquet row (from the harness's own
record of what it received) for the same document.

Hashes the completion text alone, not the prompt plus completion. The
completion is the one signal both sides can extract identically --
`open_r1_tpu.tracing.ingest.parse_payload` and
`open_r1_tpu.evaluation.run.extract_completions` both already probe a
response defensively for it -- whereas the exact prompt LightEval rendered
(few-shot exemplars, chat template, system prompt) is not reliably
reconstructable from a Parquet detail shard, which exposes only whatever a
given LightEval release happens to put in its `doc` column; `evaluation.run`
already documents how unstable those column names and shapes are across
releases. A free-form, multi-hundred-token reasoning completion is high
entropy enough that two genuinely different documents producing an identical
one is not a practical concern for this project's tasks, and the cost of
being wrong is bounded: `open_r1_tpu.tracing.scores` already falls back to
creating a `source: parquet` trace for a document whose hash finds no match,
so a false negative here only means a less-enriched trace, not lost data.
"""

from __future__ import annotations

import hashlib


def content_sha256(completion: str) -> str:
    """Deterministic hex digest of one completion's text."""
    return hashlib.sha256(completion.encode("utf-8")).hexdigest()
