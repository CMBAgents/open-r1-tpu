"""Optional trace capture: GCS-backed request/response logging and local
Langfuse ingestion for evaluation runs.

Importing this package costs nothing extra; importing `ingest` or `scores`
does, since both need the `langfuse` SDK -- declared in the `tracing` optional
dependency group, not the core install. See docker/langfuse/README.md for the
runtime picture.
"""
