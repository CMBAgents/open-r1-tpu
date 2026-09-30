#!/usr/bin/env bash
# Write the two gitignored files that tracing evaluations in Langfuse needs:
#
#   docker/langfuse/.env  -- the stack's settings: docker/langfuse/.env.example
#                            filled in with freshly generated secrets.
#   configs/tracing.yaml  -- where the evaluation's Langfuse client connects.
#
# .env has no variable interpolation, so values that must match (the password
# inside DATABASE_URL, the MinIO credentials Langfuse's S3 client uses, the web
# port in both files) are easy to get wrong by hand; this derives them all from
# one set of secrets. On one host:
#
#   scripts/gen_langfuse_env.sh
#   scripts/run_langfuse_stack.sh up
#   scripts/gen_langfuse_env.sh --print-keys >> ~/.open-r1-tpu.env
#
# With the stack and the evaluation on different hosts, each gets one file:
#
#   # stack host: publish the web port on an address the evaluation host reaches
#   scripts/gen_langfuse_env.sh --web-bind <addr> --no-tracing-config
#   scripts/run_langfuse_stack.sh up
#   scripts/gen_langfuse_env.sh --print-keys   # copy to the evaluation host
#
#   # evaluation host: the client config only, no secrets
#   scripts/gen_langfuse_env.sh --tracing-only --langfuse-host <addr>
#
# Neither address has a non-loopback default, since both are deployment
# values. Existing files are kept unless --force, so a running stack's secrets
# are never rotated out from under its volumes.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_TEMPLATE="${REPO_ROOT}/docker/langfuse/.env.example"
ENV_FILE="${REPO_ROOT}/docker/langfuse/.env"
TRACING_FILE="${REPO_ROOT}/configs/tracing.yaml"

# LANGFUSE_WEB_BIND is where langfuse-web listens (.env); LANGFUSE_HOST is
# where the evaluation's client dials (configs/tracing.yaml). On one host both
# are loopback, and the UI is reached through an SSH port-forward.
LANGFUSE_WEB_PORT=3000
LANGFUSE_WEB_BIND=127.0.0.1
LANGFUSE_HOST=127.0.0.1

FORCE=0
PRINT_KEYS=0
WRITE_TRACING=1
WRITE_ENV=1
HOST_GIVEN=0

usage() {
  cat <<'USAGE'
Usage: scripts/gen_langfuse_env.sh [--force] [--web-bind ADDR] [--langfuse-host ADDR]
                                  [--langfuse-port PORT] [--no-tracing-config]
       scripts/gen_langfuse_env.sh --tracing-only --langfuse-host ADDR [--langfuse-port PORT] [--force]
       scripts/gen_langfuse_env.sh --print-keys

  (no args)             Write docker/langfuse/.env and configs/tracing.yaml with
                        a fresh, internally consistent set of secrets, both
                        loopback. Refuses to overwrite either file.
  --force               Overwrite what this invocation would write. Rotates
                        every secret it regenerates; only safe with the stack
                        down and its volumes discarded.
  --web-bind ADDR       Interface langfuse-web publishes on (LANGFUSE_WEB_BIND
                        in .env). Default 127.0.0.1. A non-loopback address
                        makes the stack reachable off-host: firewall the port
                        to the evaluation host only.
  --langfuse-host ADDR  Address the eval client dials (langfuse.host in
                        configs/tracing.yaml). Default 127.0.0.1. Set it to the
                        stack host's address when the two are different
                        machines.
  --langfuse-port PORT  Web port, for both files. Default 3000.
  --no-tracing-config   Write only docker/langfuse/.env.
  --tracing-only        Write only configs/tracing.yaml, generating no secrets
                        and touching no .env -- the evaluation host's half of a
                        two-host deployment. Requires --langfuse-host.
  --print-keys          Do not generate anything. Read the existing
                        docker/langfuse/.env and print, to stdout, the two
                        `export LANGFUSE_PUBLIC_KEY=/SECRET_KEY=` lines the eval
                        harness authenticates with. Append them to the file the
                        eval launch sources (~/.open-r1-tpu.env). Run it on the
                        host holding .env; in a two-host deployment that is the
                        stack's host, and the output is copied to the other.
USAGE
}

# Both files want a bare host: the client adds the scheme and port itself
# (open_r1_tpu.evaluation.traced), so a URL here would boot a stack the client
# then fails to reach.
require_bare_host() {
  local flag="$1" value="$2"
  if [[ -z "$value" ]]; then
    echo "${flag} requires an address." >&2
    exit 2
  fi
  if [[ "$value" == *://* || "$value" == */* || "$value" == *:* || "$value" =~ [[:space:]] ]]; then
    echo "${flag} takes a bare host or IP, not a URL, port, or path: ${value}" >&2
    exit 2
  fi
}

require_port_value() {
  local flag="$1" value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 1 || value > 65535 )); then
    echo "${flag} takes a TCP port number: ${value}" >&2
    exit 2
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1 ;;
    --no-tracing-config) WRITE_TRACING=0 ;;
    --tracing-only) WRITE_ENV=0 ;;
    --web-bind)
      require_bare_host "--web-bind" "${2:-}"
      LANGFUSE_WEB_BIND="$2"; shift ;;
    --langfuse-host)
      require_bare_host "--langfuse-host" "${2:-}"
      LANGFUSE_HOST="$2"; HOST_GIVEN=1; shift ;;
    --langfuse-port)
      require_port_value "--langfuse-port" "${2:-}"
      LANGFUSE_WEB_PORT="$2"; shift ;;
    --print-keys) PRINT_KEYS=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [[ "$WRITE_ENV" == "0" && "$WRITE_TRACING" == "0" ]]; then
  echo "--tracing-only and --no-tracing-config together would write nothing." >&2
  exit 2
fi

# --tracing-only is for an evaluation host whose stack runs elsewhere, so a
# loopback default would point it at a machine with no Langfuse on it.
if [[ "$WRITE_ENV" == "0" && "$HOST_GIVEN" == "0" ]]; then
  echo "--tracing-only requires --langfuse-host ADDR (where the stack is reachable)." >&2
  exit 2
fi

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
# Only .env needs secrets, so --tracing-only generates no key material.
if [[ "$WRITE_ENV" == "1" ]]; then
  if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl is required (used for every generated secret)." >&2
    exit 1
  fi
  if [[ ! -f "$ENV_TEMPLATE" ]]; then
    echo "Missing template ${ENV_TEMPLATE}." >&2
    exit 1
  fi
fi

collisions=()
[[ "$WRITE_ENV" == "1" && -f "$ENV_FILE" ]] && collisions+=("$ENV_FILE")
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

if [[ "$WRITE_ENV" == "1" ]]; then
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

  # Names kept from the template that other values must repeat.
  template_value() { sed -n "s/^${1}=//p" "$ENV_TEMPLATE" | head -n 1; }
  POSTGRES_USER="$(template_value POSTGRES_USER)"
  POSTGRES_DB="$(template_value POSTGRES_DB)"
  MINIO_ROOT_USER="$(template_value MINIO_ROOT_USER)"

  # The value to write for a template key; fails for keys that keep the
  # template's value.
  generated_value() {
    case "$1" in
      LANGFUSE_WEB_BIND) printf '%s\n' "$LANGFUSE_WEB_BIND" ;;
      LANGFUSE_WEB_PORT) printf '%s\n' "$LANGFUSE_WEB_PORT" ;;
      POSTGRES_PASSWORD) printf '%s\n' "$POSTGRES_PASSWORD" ;;
      DATABASE_URL) printf '%s\n' "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}" ;;
      REDIS_AUTH) printf '%s\n' "$REDIS_AUTH" ;;
      CLICKHOUSE_PASSWORD) printf '%s\n' "$CLICKHOUSE_PASSWORD" ;;
      LANGFUSE_S3_*_ACCESS_KEY_ID) printf '%s\n' "$MINIO_ROOT_USER" ;;
      MINIO_ROOT_PASSWORD | LANGFUSE_S3_*_SECRET_ACCESS_KEY) printf '%s\n' "$MINIO_ROOT_PASSWORD" ;;
      SALT) printf '%s\n' "$SALT" ;;
      ENCRYPTION_KEY) printf '%s\n' "$ENCRYPTION_KEY" ;;
      NEXTAUTH_SECRET) printf '%s\n' "$NEXTAUTH_SECRET" ;;
      NEXTAUTH_URL) printf '%s\n' "http://localhost:${LANGFUSE_WEB_PORT}" ;;
      LANGFUSE_INIT_PROJECT_PUBLIC_KEY) printf '%s\n' "$INIT_PROJECT_PUBLIC_KEY" ;;
      LANGFUSE_INIT_PROJECT_SECRET_KEY) printf '%s\n' "$INIT_PROJECT_SECRET_KEY" ;;
      LANGFUSE_INIT_USER_PASSWORD) printf '%s\n' "$INIT_USER_PASSWORD" ;;
      *) return 1 ;;
    esac
  }

  mkdir -p "$(dirname "$ENV_FILE")"
  umask 077

  {
    printf '%s\n\n' "# Generated by scripts/gen_langfuse_env.sh from .env.example. Gitignored."
    # The template's opening comment, up to the first blank line, is about
    # filling it in by hand, so it is not copied.
    in_header=1
    while IFS= read -r line || [[ -n "$line" ]]; do
      if [[ "$in_header" == "1" ]]; then
        [[ -z "$line" ]] && in_header=0
        continue
      fi
      key="${line%%=*}"
      if [[ "$line" != \#* && "$line" == *=* ]] && value="$(generated_value "$key")"; then
        printf '%s=%s\n' "$key" "$value"
      else
        printf '%s\n' "$line"
      fi
    done <"$ENV_TEMPLATE"
  } >"$ENV_FILE"

  # A secret added to the template but not above must fail here, not ship as a
  # placeholder.
  if grep -n '^[^#]*changeme' "$ENV_FILE" >&2; then
    rm -f "$ENV_FILE"
    echo "${ENV_TEMPLATE} has placeholders this script does not generate (above)." >&2
    exit 1
  fi
  chmod 600 "$ENV_FILE"
  echo "Wrote ${ENV_FILE}" >&2

  if [[ "$LANGFUSE_WEB_BIND" != "127.0.0.1" && "$LANGFUSE_WEB_BIND" != "localhost" ]]; then
    cat >&2 <<WARN
Note: langfuse-web will publish on ${LANGFUSE_WEB_BIND}:${LANGFUSE_WEB_PORT}, not loopback.
  - Admit that port from the evaluation host only; nothing else should reach it.
  - The UI port-forward must now target that address:
      ssh ... -L ${LANGFUSE_WEB_PORT}:${LANGFUSE_WEB_BIND}:${LANGFUSE_WEB_PORT}
WARN
  fi
fi

if [[ "$WRITE_TRACING" == "1" ]]; then
  mkdir -p "$(dirname "$TRACING_FILE")"
  cat >"$TRACING_FILE" <<EOF
# Generated by scripts/gen_langfuse_env.sh. Gitignored. Where the evaluation
# reaches langfuse-web: the LANGFUSE_WEB_BIND:LANGFUSE_WEB_PORT in the stack
# host's docker/langfuse/.env.
langfuse:
  host: ${LANGFUSE_HOST}
  port: ${LANGFUSE_WEB_PORT}
EOF
  chmod 600 "$TRACING_FILE"
  echo "Wrote ${TRACING_FILE}" >&2
fi

if [[ "$WRITE_ENV" == "1" ]]; then
  cat >&2 <<EOF

Next:
  1. scripts/run_langfuse_stack.sh up
  2. scripts/gen_langfuse_env.sh --print-keys >> ~/.open-r1-tpu.env
     source ~/.open-r1-tpu.env
  3. RECIPE=<recipe> TRACE_CONFIG=configs/tracing.yaml ./scripts/run_eval_tpu.sh
EOF
else
  cat >&2 <<EOF

Next, on this host:
  1. Append the stack host's keys to ~/.open-r1-tpu.env and source it:
       scripts/gen_langfuse_env.sh --print-keys   # run there, paste here
  2. RECIPE=<recipe> TRACE_CONFIG=configs/tracing.yaml ./scripts/run_eval_tpu.sh
EOF
fi
