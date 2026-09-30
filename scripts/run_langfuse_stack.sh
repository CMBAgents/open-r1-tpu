#!/usr/bin/env bash
set -euo pipefail

# Start, stop, or inspect the self-hosted Langfuse stack
# (docker/langfuse/docker-compose.yaml; see docker/langfuse/README.md).
#
#   scripts/run_langfuse_stack.sh up
#   scripts/run_langfuse_stack.sh down
#   scripts/run_langfuse_stack.sh ps
#   scripts/run_langfuse_stack.sh logs [service]
#
# Every port, password and key comes from docker/langfuse/.env, which
# scripts/gen_langfuse_env.sh writes.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_DIR="${REPO_ROOT}/docker/langfuse"
COMPOSE_FILE="${COMPOSE_DIR}/docker-compose.yaml"
ENV_FILE="${COMPOSE_DIR}/.env"

usage() {
  cat <<'USAGE'
Usage: scripts/run_langfuse_stack.sh {up|down|ps|logs} [logs: service]
USAGE
}

ACTION="${1:-}"
if [[ -z "${ACTION}" ]]; then
  usage >&2
  exit 2
fi
shift || true

if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Missing ${ENV_FILE}; run scripts/gen_langfuse_env.sh first." >&2
  exit 1
fi

source "${REPO_ROOT}/scripts/lib/docker.sh"
resolve_docker

COMPOSE=("${DOCKER[@]}" compose --file "${COMPOSE_FILE}" --env-file "${ENV_FILE}")

case "${ACTION}" in
  up)
    "${COMPOSE[@]}" up --detach --wait
    ;;
  down)
    "${COMPOSE[@]}" down
    ;;
  ps)
    "${COMPOSE[@]}" ps
    ;;
  logs)
    "${COMPOSE[@]}" logs --follow --tail 200 "$@"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
