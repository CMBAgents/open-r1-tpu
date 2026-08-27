#!/usr/bin/env bash
set -euo pipefail

# Start, stop, or inspect the local Langfuse self-host stack
# (docker/langfuse/docker-compose.yaml). See docker/langfuse/README.md for
# the full picture: what the stack is for, how it is viewed from a
# workstation, and why it carries no durability expectations.
#
#   scripts/run_langfuse_stack.sh up
#   scripts/run_langfuse_stack.sh down
#   scripts/run_langfuse_stack.sh ps
#   scripts/run_langfuse_stack.sh logs [service]
#
# Needs docker/langfuse/.env (gitignored; copy docker/langfuse/.env.example
# and fill in real values). No port, password, or key is read from this
# script -- every one comes from that file.

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
  echo "Missing ${ENV_FILE}; copy docker/langfuse/.env.example and fill in real values." >&2
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
