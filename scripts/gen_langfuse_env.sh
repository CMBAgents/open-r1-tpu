#!/usr/bin/env bash
# Generate the two gitignored, deployment-local files the Langfuse-native eval
# path needs, with every cross-reference already consistent:
#
#   docker/langfuse/.env   -- the only input docker/langfuse/docker-compose.yaml
#                             reads (see docker/langfuse/.env.example).
#   configs/tracing.yaml   -- the langfuse host/port open_r1_tpu.evaluation
#                             .experiment / .dataset_sync build a client from.
#
# WHY A GENERATOR. docker/langfuse/.env has no variable interpolation, so three
# pairs of values that MUST be equal are easy to set inconsistently by hand:
#
#   DATABASE_URL ................... embeds POSTGRES_PASSWORD verbatim
#   LANGFUSE_S3_*_ACCESS_KEY_ID .... equals MINIO_ROOT_USER
#   LANGFUSE_S3_*_SECRET_ACCESS_KEY  equals MINIO_ROOT_PASSWORD
#
# and the tracing config's langfuse.port must equal LANGFUSE_WEB_PORT. A
# mismatch in any of them boots a stack that then fails partway -- the worker
# cannot reach Postgres, event uploads 403 against MinIO, or the harness
# connects to the wrong endpoint. This script derives all of them from one set
# of freshly generated secrets, so "pre-made scripts" is the whole setup on a
# fresh VM:
#
#   scripts/gen_langfuse_env.sh
#   scripts/run_langfuse_stack.sh up
#   scripts/gen_langfuse_env.sh --print-keys >> ~/.tpu-env   # or wherever the
#                                                            # eval launch sources
#
# Nothing here is committed: both output files are gitignored because a
# Langfuse host and these secrets are deployment identifiers. Re-running is
# refused unless --force, so an existing stack's secrets are never silently
# rotated out from under its volumes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${REPO_ROOT}/docker/langfuse/.env"
TRACING_FILE="${REPO_ROOT}/configs/tracing.yaml"

# Bound to 127.0.0.1 by the compose file; the stack is viewed through an SSH
# port-forward, never directly. Change here only if a port collides on the host.
LANGFUSE_WEB_PORT=3000
LANGFUSE_WORKER_PORT=3030
POSTGRES_PORT=5432
REDIS_PORT=6379
CLICKHOUSE_HTTP_PORT=8123
CLICKHOUSE_NATIVE_PORT=9000
MINIO_API_PORT=9090
MINIO_CONSOLE_PORT=9091

FORCE=0
PRINT_KEYS=0
WRITE_TRACING=1

usage() {
  cat <<'USAGE'
Usage: scripts/gen_langfuse_env.sh [--force] [--no-tracing-config]
       scripts/gen_langfuse_env.sh --print-keys

  (no args)             Write docker/langfuse/.env and configs/tracing.yaml with
                        a fresh, internally consistent set of secrets. Refuses
                        to overwrite either file.
  --force               Overwrite both files. Rotates every secret; only safe
                        with the stack down and its volumes discarded.
  --no-tracing-config   Write only docker/langfuse/.env.
  --print-keys          Do not generate anything. Read the existing
                        docker/langfuse/.env and print, to stdout, the two
                        `export LANGFUSE_PUBLIC_KEY=/SECRET_KEY=` lines the eval
                        harness authenticates with. Append them to the file the
                        eval launch sources (e.g. ~/.tpu-env).
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --no-tracing-config) WRITE_TRACING=0 ;;
    --print-keys) PRINT_KEYS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# --- --print-keys: read back an existing .env, emit nothing else -------------
if [[ "$PRINT_KEYS" == "1" ]]; then
  if [[ ! -f "$ENV_FILE" ]]; then
    echo "No ${ENV_FILE}; run scripts/gen_langfuse_env.sh first." >&2
    exit 1
  fi
  pk="$(sed -n 's/^LANGFUSE_INIT_PROJECT_PUBLIC_KEY=//p' "$ENV_FILE")"
  sk="$(sed -n 's/^LANGFUSE_INIT_PROJECT_SECRET_KEY=//p' "$ENV_FILE")"
  if [[ -z "$pk" || -z "$sk" ]]; then
    echo "${ENV_FILE} is missing LANGFUSE_INIT_PROJECT_PUBLIC_KEY/SECRET_KEY." >&2
    exit 1
  fi
  # Leading newline: the file this is appended to may lack a trailing one, and
  # a concatenated first line would corrupt whichever value it lands on.
  printf '\n%s\n%s\n' "export LANGFUSE_PUBLIC_KEY=${pk}" "export LANGFUSE_SECRET_KEY=${sk}"
  exit 0
fi

# --- generation -------------------------------------------------------------
if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required (used for every generated secret)." >&2
  exit 1
fi

collisions=()
[[ -f "$ENV_FILE" ]] && collisions+=("$ENV_FILE")
[[ "$WRITE_TRACING" == "1" && -f "$TRACING_FILE" ]] && collisions+=("$TRACING_FILE")
if [[ ${#collisions[@]} -gt 0 && "$FORCE" != "1" ]]; then
  printf 'Refusing to overwrite:\n' >&2
  printf '  %s\n' "${collisions[@]}" >&2
  echo "Pass --force (stack down, volumes discarded) to regenerate." >&2
  exit 1
fi

rand_hex() { openssl rand -hex "${1:-32}"; }

uuid() {
  if [[ -r /proc/sys/kernel/random/uuid ]]; then
    cat /proc/sys/kernel/random/uuid
  elif command -v uuidgen >/dev/null 2>&1; then
    uuidgen | tr 'A-Z' 'a-z'
  else
    python3 -c 'import uuid; print(uuid.uuid4())'
  fi
}

POSTGRES_PASSWORD="$(rand_hex 24)"
REDIS_AUTH="$(rand_hex 24)"
CLICKHOUSE_PASSWORD="$(rand_hex 24)"
MINIO_ROOT_PASSWORD="$(rand_hex 24)"
SALT="$(rand_hex 32)"
ENCRYPTION_KEY="$(rand_hex 32)"
NEXTAUTH_SECRET="$(rand_hex 32)"
INIT_USER_PASSWORD="$(rand_hex 16)"
INIT_PROJECT_PUBLIC_KEY="pk-lf-$(uuid)"
INIT_PROJECT_SECRET_KEY="sk-lf-$(uuid)"

mkdir -p "$(dirname "$ENV_FILE")"
umask 077

cat >"$ENV_FILE" <<EOF
# Generated by scripts/gen_langfuse_env.sh -- do not hand-edit the derived
# values below (see that script's header for which must stay equal). Gitignored.

# --- Ports (all bound to 127.0.0.1 by the compose file) ---
LANGFUSE_WEB_PORT=${LANGFUSE_WEB_PORT}
LANGFUSE_WORKER_PORT=${LANGFUSE_WORKER_PORT}
POSTGRES_PORT=${POSTGRES_PORT}
REDIS_PORT=${REDIS_PORT}
CLICKHOUSE_HTTP_PORT=${CLICKHOUSE_HTTP_PORT}
CLICKHOUSE_NATIVE_PORT=${CLICKHOUSE_NATIVE_PORT}
MINIO_API_PORT=${MINIO_API_PORT}
MINIO_CONSOLE_PORT=${MINIO_CONSOLE_PORT}

# --- Datastore credentials (internal to the compose network) ---
POSTGRES_USER=postgres
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=postgres
# No .env interpolation: this embeds POSTGRES_PASSWORD verbatim.
DATABASE_URL=postgresql://postgres:${POSTGRES_PASSWORD}@postgres:5432/postgres

REDIS_HOST=redis
REDIS_AUTH=${REDIS_AUTH}

CLICKHOUSE_USER=clickhouse
CLICKHOUSE_PASSWORD=${CLICKHOUSE_PASSWORD}
CLICKHOUSE_MIGRATION_URL=clickhouse://clickhouse:9000
CLICKHOUSE_URL=http://clickhouse:8123
CLICKHOUSE_CLUSTER_ENABLED=false

MINIO_ROOT_USER=minio
MINIO_ROOT_PASSWORD=${MINIO_ROOT_PASSWORD}

# Langfuse's S3 client for MinIO. ACCESS_KEY_ID must equal MINIO_ROOT_USER and
# SECRET_ACCESS_KEY must equal MINIO_ROOT_PASSWORD, for both the event and the
# media bucket.
LANGFUSE_S3_EVENT_UPLOAD_BUCKET=langfuse
LANGFUSE_S3_EVENT_UPLOAD_REGION=auto
LANGFUSE_S3_EVENT_UPLOAD_ENDPOINT=http://minio:9000
LANGFUSE_S3_EVENT_UPLOAD_FORCE_PATH_STYLE=true
LANGFUSE_S3_EVENT_UPLOAD_PREFIX=events/
LANGFUSE_S3_EVENT_UPLOAD_ACCESS_KEY_ID=minio
LANGFUSE_S3_EVENT_UPLOAD_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}
LANGFUSE_S3_MEDIA_UPLOAD_BUCKET=langfuse
LANGFUSE_S3_MEDIA_UPLOAD_REGION=auto
LANGFUSE_S3_MEDIA_UPLOAD_ENDPOINT=http://minio:9000
LANGFUSE_S3_MEDIA_UPLOAD_FORCE_PATH_STYLE=true
LANGFUSE_S3_MEDIA_UPLOAD_PREFIX=media/
LANGFUSE_S3_MEDIA_UPLOAD_ACCESS_KEY_ID=minio
LANGFUSE_S3_MEDIA_UPLOAD_SECRET_ACCESS_KEY=${MINIO_ROOT_PASSWORD}

# --- Langfuse application secrets (openssl rand -hex 32) ---
SALT=${SALT}
ENCRYPTION_KEY=${ENCRYPTION_KEY}
NEXTAUTH_SECRET=${NEXTAUTH_SECRET}
# Must name the same port as LANGFUSE_WEB_PORT.
NEXTAUTH_URL=http://localhost:${LANGFUSE_WEB_PORT}

# --- Non-secret compose toggles, pinned so this file fully defines the stack ---
TELEMETRY_ENABLED=false
LANGFUSE_ENABLE_EXPERIMENTAL_FEATURES=false
REDIS_TLS_ENABLED=false

# --- Headless init: a ready org/project/user/API-key pair on first boot
# against an empty Postgres. LANGFUSE_INIT_PROJECT_PUBLIC_KEY / _SECRET_KEY are
# what the eval harness authenticates with -- export them as
# LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY (scripts/gen_langfuse_env.sh
# --print-keys), never in a YAML config.
LANGFUSE_INIT_ORG_ID=default-org
LANGFUSE_INIT_ORG_NAME=Default Org
LANGFUSE_INIT_PROJECT_ID=default-project
LANGFUSE_INIT_PROJECT_NAME=Default Project
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=${INIT_PROJECT_PUBLIC_KEY}
LANGFUSE_INIT_PROJECT_SECRET_KEY=${INIT_PROJECT_SECRET_KEY}
LANGFUSE_INIT_USER_EMAIL=admin@example.invalid
LANGFUSE_INIT_USER_NAME=Admin
LANGFUSE_INIT_USER_PASSWORD=${INIT_USER_PASSWORD}
EOF
chmod 600 "$ENV_FILE"
echo "Wrote ${ENV_FILE}" >&2

if [[ "$WRITE_TRACING" == "1" ]]; then
  mkdir -p "$(dirname "$TRACING_FILE")"
  cat >"$TRACING_FILE" <<EOF
# Generated by scripts/gen_langfuse_env.sh. Gitignored. langfuse.port matches
# LANGFUSE_WEB_PORT in docker/langfuse/.env.
langfuse:
  host: 127.0.0.1
  port: ${LANGFUSE_WEB_PORT}
EOF
  chmod 600 "$TRACING_FILE"
  echo "Wrote ${TRACING_FILE}" >&2
fi

cat >&2 <<EOF

Next:
  1. scripts/run_langfuse_stack.sh up
  2. scripts/gen_langfuse_env.sh --print-keys >> <the file the eval launch sources>
  3. RECIPE=<recipe> TRACE_CONFIG=configs/tracing.yaml ./scripts/run_eval_tpu.sh
EOF
