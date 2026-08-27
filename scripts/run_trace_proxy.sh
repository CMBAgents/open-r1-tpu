#!/usr/bin/env bash
set -euo pipefail

# Start or stop the trace-capture litellm proxy container: a transparent
# relay between LightEval and vLLM that logs every request/response to GCS
# via a custom callback (docker/trace-proxy/gcs_logger.py), for later
# ingestion into Langfuse (docker/langfuse/). Opt-in per launch through
# scripts/run_eval_tpu.sh's TRACE_PROXY=1; see its header.
#
# Config-driven throughout: every port, image, bucket, and upstream URL comes
# from `python -m open_r1_tpu.tracing.config --export-env`, never a literal
# in this script. --recipe and --timestamp are required to start, because the
# run-scoped GCS prefix is rendered from them; without a rendered prefix
# there is nowhere correct to write.
#
# The container runs with --network host: the upstream vLLM (and the
# rehearsal test's stub) listens on the host's loopback, which a
# bridge-networked container cannot reach -- its 127.0.0.1 is its own
# namespace. litellm is told to bind 127.0.0.1 explicitly, so host networking
# does not widen exposure beyond the loopback-only default the published-port
# setup had.
#
# GCS auth inside the container: ambient ADC. On the TPU VM that is the
# instance's own service account via the metadata server, reachable from the
# container with nothing mounted. Elsewhere (a workstation rehearsal), set
# GOOGLE_APPLICATION_CREDENTIALS before calling this script and the key file
# is mounted read-only into the container.
#
# Failure containment: gcs_logger.py's write runs inside litellm's own async
# logging callback, already dispatched off the response path, as a genuinely
# async httpx upload -- a GCS outage logs a warning and drops one payload, it
# does not fail the evaluation request, stall the proxy's event loop, or stop
# this container. A missing Python dependency, by contrast, fails the proxy
# at startup on purpose (see that file's module docstring, which also cites
# why this is a custom callback at all: the built-in `gcs_bucket` integration
# is Enterprise-licensed).
#
#   scripts/run_trace_proxy.sh --config configs/tracing.yaml \
#     --recipe tier0-smoke --timestamp 20260101T000000Z
#   scripts/run_trace_proxy.sh --stop --config configs/tracing.yaml

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${TRACE_CONFIG:-}"
RECIPE=""
TIMESTAMP=""
ACTION=start
CONTAINER_NAME="${TRACE_PROXY_CONTAINER_NAME:-open-r1-tpu-trace-proxy}"

usage() {
  cat <<'USAGE'
Usage: scripts/run_trace_proxy.sh --config CONFIG --recipe NAME --timestamp TS
       scripts/run_trace_proxy.sh --stop --config CONFIG
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      [[ $# -ge 2 ]] || { echo "--config requires a value" >&2; exit 2; }
      CONFIG="$2"
      shift 2
      ;;
    --recipe)
      [[ $# -ge 2 ]] || { echo "--recipe requires a value" >&2; exit 2; }
      RECIPE="$2"
      shift 2
      ;;
    --timestamp)
      [[ $# -ge 2 ]] || { echo "--timestamp requires a value" >&2; exit 2; }
      TIMESTAMP="$2"
      shift 2
      ;;
    --stop)
      ACTION=stop
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "--config <tracing config path> is required (or set TRACE_CONFIG)" >&2
  exit 1
fi

DOCKER=()
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif command -v docker >/dev/null 2>&1 \
  && command -v sudo >/dev/null 2>&1 \
  && sudo -n docker info >/dev/null 2>&1; then
  DOCKER=(sudo -n docker)
else
  echo "Docker is unavailable or inaccessible." >&2
  echo "Install/start Docker, or grant this user Docker access (passwordless sudo is accepted)." >&2
  exit 1
fi

if [[ "${ACTION}" == stop ]]; then
  "${DOCKER[@]}" stop --time 10 "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  "${DOCKER[@]}" rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
  exit 0
fi

if [[ -z "${RECIPE}" || -z "${TIMESTAMP}" ]]; then
  echo "--recipe and --timestamp are required to start: they render the run-scoped GCS prefix" >&2
  exit 1
fi

# Populates TRACE_PROXY_PORT, TRACE_PROXY_UPSTREAM_BASE_URL, TRACE_PROXY_IMAGE,
# TRACE_GCS_BUCKET, and (rendered from --recipe/--timestamp above)
# TRACE_GCS_PREFIX. See open_r1_tpu.tracing.config's module docstring.
eval "$(PYTHONPATH="${REPO_ROOT}/src" python3 -m open_r1_tpu.tracing.config --export-env \
  --config "${CONFIG}" --recipe "${RECIPE}" --timestamp "${TIMESTAMP}")"

if "${DOCKER[@]}" container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container ${CONTAINER_NAME} already exists; stop it first (--stop)." >&2
  exit 1
fi

# A workstation's ADC key file, mounted only when the caller points at one;
# on the TPU VM this stays empty and the metadata server provides auth.
CRED_ARGS=()
if [[ -n "${GOOGLE_APPLICATION_CREDENTIALS:-}" ]]; then
  CRED_ARGS=(
    --volume "${GOOGLE_APPLICATION_CREDENTIALS}:/gcp-adc.json:ro"
    --env GOOGLE_APPLICATION_CREDENTIALS=/gcp-adc.json
  )
fi

"${DOCKER[@]}" run \
  --detach \
  --rm \
  --name "${CONTAINER_NAME}" \
  --network host \
  --env TRACE_PROXY_UPSTREAM_BASE_URL \
  --env TRACE_GCS_BUCKET \
  --env TRACE_GCS_PREFIX \
  --env PYTHONPATH=/etc/litellm \
  --volume "${REPO_ROOT}/docker/trace-proxy:/etc/litellm:ro" \
  ${CRED_ARGS[@]+"${CRED_ARGS[@]}"} \
  "${TRACE_PROXY_IMAGE}" \
  --config /etc/litellm/config.yaml \
  --host 127.0.0.1 \
  --port "${TRACE_PROXY_PORT}"

echo "Trace proxy listening on 127.0.0.1:${TRACE_PROXY_PORT}, writing to gs://${TRACE_GCS_BUCKET}/${TRACE_GCS_PREFIX}" >&2
