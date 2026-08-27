"""Custom litellm proxy callback: write every captured request/response to
GCS, as a free substitute for litellm's own `gcs_bucket` success callback.

litellm's built-in `gcs_bucket` integration is gated behind an Enterprise
license: `litellm/integrations/gcs_bucket/gcs_bucket.py` (checked against the
litellm 1.97.0 lineage this project's evaluation stack already pins) raises
`ValueError("GCS Bucket logging is a premium feature...")` in both
`__init__` and `async_log_success_event` unless `premium_user is True`, which
requires a paid license key loaded into the process. Its own docs page
carries the same "Enterprise only feature" badge. There is no free path
through that integration.

This module is the substitute the tracing config's `config.yaml` actually
loads (`litellm_settings.callbacks: gcs_logger.proxy_handler_instance`).
litellm hands every custom callback the identical payload the gated
integration would have written -- `kwargs["standard_logging_object"]`, a
`StandardLoggingPayload` (documented at
https://docs.litellm.ai/docs/proxy/logging_spec) -- so what lands in GCS is
exactly what the plan this proxy implements originally called for; only the
delivery mechanism differs. `open_r1_tpu.tracing.ingest` parses that same
payload shape on the way into Langfuse.

The GCS write uses `httpx` against the JSON upload API plus a bearer token
from `google-auth` -- and nothing else -- because both are guaranteed present
in the pinned proxy image while nothing heavier is: the image's Dockerfile
(litellm v1.97.0) builds a venv with `uv sync --extra proxy --extra
proxy-runtime --extra extra_proxy ...` on a minimal wolfi base with no pip,
so a dependency this module needs must already be in that set. `httpx` is a
litellm core dependency; `google-auth` is required by the image's
`google-cloud-aiplatform` (proxy-runtime) and `google-cloud-kms`
(extra_proxy). `gcsfs`, by contrast, is NOT in the image -- which is why this
module must not import it. Both imports are at module top on purpose: a
missing dependency kills the proxy at startup, loudly, instead of silently
dropping every payload at write time.

Auth is ambient application-default credentials via `google.auth.default()`:
on the TPU VM this resolves through the instance's own service account with
no key file; elsewhere (e.g. a rehearsal on a workstation) set
`GOOGLE_APPLICATION_CREDENTIALS`, which `scripts/run_trace_proxy.sh` mounts
into the container when set. For tests, `STORAGE_EMULATOR_HOST` (the
standard Google emulator convention) redirects uploads to a local endpoint
and skips auth entirely, so the full write path runs against a stdlib HTTP
server with no network and no credentials.

Deployment-specific values -- the bucket and the run-scoped prefix -- arrive
through the environment (`TRACE_GCS_BUCKET`, `TRACE_GCS_PREFIX`), set by
`scripts/run_trace_proxy.sh` from the tracing config. This file carries no
deployment literal of its own, and reads no message-redaction setting: full
prompts and completions in the payload are the entire point.

Failure containment: `async_log_success_event` runs off litellm's response
path (dispatched after the response has been returned to the caller), and the
upload is genuinely async -- an `httpx.AsyncClient` request, with the rare
synchronous credential refresh pushed onto a worker thread -- so a slow GCS
never stalls the proxy's event loop the way a blocking write would. Transient
failures are retried a few times with backoff; a payload that still cannot be
written logs a warning and is dropped rather than raising anywhere a request
could be affected.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.parse
import uuid

import google.auth
import google.auth.transport.requests
import httpx
from litellm.integrations.custom_logger import CustomLogger

LOGGER = logging.getLogger("gcs_logger")

_STORAGE_SCOPE = "https://www.googleapis.com/auth/devstorage.read_write"
_UPLOAD_ATTEMPTS = 3
_REQUEST_TIMEOUT_SECS = 30.0

# Application-default credentials, resolved once and refreshed as needed.
# google-auth's loading and refresh are synchronous (metadata-server or token
# HTTP calls), so both run via asyncio.to_thread. Unset in emulator mode.
_credentials = None


def _emulator_host() -> str | None:
    return os.environ.get("STORAGE_EMULATOR_HOST") or None


def _sync_token() -> str:
    """Load-and-refresh ADC synchronously; called only from a worker thread."""
    global _credentials
    if _credentials is None:
        _credentials, _ = google.auth.default(scopes=[_STORAGE_SCOPE])
    if not _credentials.valid:
        _credentials.refresh(google.auth.transport.requests.Request())
    return _credentials.token


def _upload_url(bucket: str, object_name: str) -> str:
    base = _emulator_host() or "https://storage.googleapis.com"
    name = urllib.parse.quote(object_name, safe="")
    return (
        f"{base.rstrip('/')}/upload/storage/v1/b/{bucket}/o"
        f"?uploadType=media&name={name}"
    )


class GCSTraceLogger(CustomLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            await self._write(kwargs)
        except Exception:
            # Belt and braces: _write already contains its own handling, but
            # nothing may ever escape into litellm's logging worker.
            LOGGER.warning("Unexpected error writing a trace", exc_info=True)

    async def _write(self, kwargs: dict) -> None:
        payload = kwargs.get("standard_logging_object")
        if payload is None:
            LOGGER.warning("Request carried no standard_logging_object; skipping")
            return

        bucket = os.environ.get("TRACE_GCS_BUCKET")
        prefix = os.environ.get("TRACE_GCS_PREFIX")
        if not bucket or not prefix:
            LOGGER.warning(
                "TRACE_GCS_BUCKET/TRACE_GCS_PREFIX not set; dropping this trace"
            )
            return

        stem = payload.get("id") or uuid.uuid4().hex
        object_name = f"{prefix.rstrip('/')}/{stem}.json"
        url = _upload_url(bucket, object_name)
        body = json.dumps(payload, default=str).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if _emulator_host() is None:
            try:
                token = await asyncio.to_thread(_sync_token)
                headers["Authorization"] = f"Bearer {token}"
            except Exception:
                LOGGER.warning(
                    "Could not obtain GCS credentials; dropping this trace",
                    exc_info=True,
                )
                return

        for attempt in range(1, _UPLOAD_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT_SECS) as client:
                    response = await client.post(url, content=body, headers=headers)
                if response.status_code < 400:
                    return
                if response.status_code < 500:
                    # Auth or configuration problem: retrying cannot help.
                    LOGGER.warning(
                        "GCS rejected trace %s: HTTP %s %s",
                        object_name,
                        response.status_code,
                        response.text[:200],
                    )
                    return
                LOGGER.warning(
                    "GCS write of %s failed (attempt %d/%d): HTTP %s",
                    object_name,
                    attempt,
                    _UPLOAD_ATTEMPTS,
                    response.status_code,
                )
            except httpx.HTTPError as error:
                LOGGER.warning(
                    "GCS write of %s failed (attempt %d/%d): %s",
                    object_name,
                    attempt,
                    _UPLOAD_ATTEMPTS,
                    error,
                )
            if attempt < _UPLOAD_ATTEMPTS:
                await asyncio.sleep(attempt)
        LOGGER.warning(
            "Dropping trace %s after %d attempts", object_name, _UPLOAD_ATTEMPTS
        )


# The instance name litellm's proxy config resolves via a dotted
# `module.instance` reference in `litellm_settings.callbacks` -- see
# config.yaml in this directory.
proxy_handler_instance = GCSTraceLogger()
