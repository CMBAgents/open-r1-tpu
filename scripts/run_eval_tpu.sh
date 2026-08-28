#!/usr/bin/env bash
set -euo pipefail

# Benchmark a fine-tuned checkpoint on the TPU.
#
# Owns the vLLM server's lifecycle and nothing else: the recipe decides what to
# serve and how, `open_r1_tpu.evaluation.run` decides what to run against it. Both halves
# read the same recipe and the same dotted overrides, so the port and the served
# model name cannot drift apart.
#
#   RECIPE=recipes/Qwen3-1.7B-Math/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
#
# RECIPE is required and has no default: an expensive run must name its tier
# on purpose rather than falling into whichever one happened to be the
# default.
#
# Overrides pass straight through, which is how the base model gets measured on
# the identical stack -- the only baseline worth comparing against:
#
#   RECIPE=recipes/Qwen3-1.7B-Math/eval/tier1_core.yaml ./scripts/run_eval_tpu.sh \
#     server.model_path=models/Qwen3-1.7B-Base
#
# Needs the locked host stack and pinned service image:
# `scripts/setup_tpu_vm.sh --with-eval`. Only one process can hold the TPU chip,
# so the training job must have exited before this starts. Set SKIP_SERVER=1 to
# reuse a server that is already up.
#
# Optional trace capture: set TRACE_PROXY=1 and TRACE_CONFIG=<tracing config
# path> to route LightEval through the trace-capture proxy
# (scripts/run_trace_proxy.sh) instead of talking to vLLM directly. It is
# started after the server and stopped on exit alongside it, and the only
# change to the harness invocation is one `server.base_url=` override pointing
# LightEval at the proxy's port -- read from the tracing config, never a
# literal here. With the proxy active, the probe request
# `check_sampling_accepted` sends before committing to a run also flows
# through it, which is desirable: it validates the proxy path before hours are
# spent on the harness, and needs no change here to do so. Leaving TRACE_PROXY
# unset is byte-identical to today's behaviour; setting it without
# TRACE_CONFIG is an error before anything is launched.
#
# EVAL_ENTRYPOINT selects what runs against the server once it is up.
# Defaults to `open_r1_tpu.evaluation.run` -- the LightEval subprocess path --
# so leaving it unset is byte-identical to today's behaviour. Set it to
# `open_r1_tpu.evaluation.experiment` to drive the Langfuse-native path
# instead (`dataset.run_experiment()` per task/seed); that path additionally
# needs `--tracing-config`, passed from TRACE_CONFIG (the same variable the
# trace-proxy branch above reads). Naming the experiment entry point without
# TRACE_CONFIG is an error before anything is launched, same as TRACE_PROXY
# above.
#
# One readiness gate moves into this script when the proxy is active: the
# harness's own `wait_for_server` polls `{base_url}/v1/models`, but the proxy
# answers that from its static model_list without contacting vLLM, so against
# the proxy the gate passes while vLLM may still be minutes into weight load
# and XLA compilation -- and the sampling probe would then abort the run. So
# before the proxy starts, this script polls vLLM's own endpoint (the tracing
# config's `proxy.upstream_base_url`) for up to TRACE_UPSTREAM_WAIT_SECS
# (default 900, matching the harness's gate; 0 skips the wait entirely, for
# callers that know the server is already up).

RECIPE="${RECIPE:-}"
SERVER_LOG="${SERVER_LOG:-artifacts/vllm-serve.log}"
SKIP_SERVER="${SKIP_SERVER:-0}"
TRACE_PROXY="${TRACE_PROXY:-0}"
TRACE_CONFIG="${TRACE_CONFIG:-}"
TRACE_UPSTREAM_WAIT_SECS="${TRACE_UPSTREAM_WAIT_SECS:-900}"
EVAL_ENTRYPOINT="${EVAL_ENTRYPOINT:-open_r1_tpu.evaluation.run}"

if [[ -z "$RECIPE" ]]; then
  echo "Usage: RECIPE=recipes/<model>/eval/<tier>.yaml ./scripts/run_eval_tpu.sh [overrides...]" >&2
  exit 1
fi

if [[ "$TRACE_PROXY" == "1" && -z "$TRACE_CONFIG" ]]; then
  echo "TRACE_PROXY=1 needs TRACE_CONFIG=<tracing config path>" >&2
  exit 1
fi

case "$EVAL_ENTRYPOINT" in
  open_r1_tpu.evaluation.run) ;;
  open_r1_tpu.evaluation.experiment)
    if [[ -z "$TRACE_CONFIG" ]]; then
      echo "EVAL_ENTRYPOINT=open_r1_tpu.evaluation.experiment needs TRACE_CONFIG=<tracing config path>" >&2
      exit 1
    fi
    ;;
  *)
    echo "EVAL_ENTRYPOINT must be open_r1_tpu.evaluation.run or open_r1_tpu.evaluation.experiment, got: $EVAL_ENTRYPOINT" >&2
    exit 1
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 -- "-$SERVER_PID" 2>/dev/null; then
    echo "Stopping vLLM server (process group $SERVER_PID)" >&2
    # TERM first so vLLM releases the TPU cleanly; a chip still held by a dead
    # process is the next run's problem, not this one's.
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [[ "$TRACE_PROXY" == "1" ]]; then
    "${REPO_ROOT}/scripts/run_trace_proxy.sh" --stop --config "$TRACE_CONFIG" || true
  fi
}
# Registered unconditionally -- including when SKIP_SERVER=1, where SERVER_PID
# is never set and this half of cleanup() stays a no-op -- so trace-proxy
# teardown always runs on exit whenever TRACE_PROXY=1, independent of whether
# this script is also managing the vLLM server's lifecycle.
trap cleanup EXIT INT TERM

if [[ "$SKIP_SERVER" != "1" ]]; then
  mkdir -p "$(dirname "$SERVER_LOG")"
  # Built from the recipe so there is one source of truth for the port, the
  # served name, and the context window.
  SERVER_CMD="$(python3 -m open_r1_tpu.evaluation.run --config "$RECIPE" \
    --print-server-command "$@")"
  echo "Starting: $SERVER_CMD" >&2
  echo "Server log: $SERVER_LOG" >&2
  # vLLM starts a separate EngineCore process. Give the service its own process
  # group so cleanup releases the TPU rather than stopping only the CLI parent.
  setsid bash -c "exec $SERVER_CMD" >"$SERVER_LOG" 2>&1 &
  SERVER_PID=$!

  # Fail fast on a server that dies during weight load or compilation, rather
  # than waiting out the full readiness timeout for a process that is gone.
  sleep 5
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "vLLM server exited during startup; last 40 lines of $SERVER_LOG:" >&2
    tail -n 40 "$SERVER_LOG" >&2
    exit 1
  fi
fi

TRACE_ARGS=()
if [[ "$TRACE_PROXY" == "1" ]]; then
  RECIPE_NAME="$(basename "$RECIPE" .yaml)"
  TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"

  # Populates TRACE_PROXY_PORT and TRACE_GCS_PREFIX (rendered from the
  # config's prefix_template by open_r1_tpu.tracing.config, never by shell
  # string-pasting) in this script's own shell, independently of
  # run_trace_proxy.sh doing the same in its own.
  eval "$(PYTHONPATH="${REPO_ROOT}/src" python3 -m open_r1_tpu.tracing.config \
    --export-env --config "$TRACE_CONFIG" --recipe "$RECIPE_NAME" --timestamp "$TIMESTAMP")"

  echo "Trace capture prefix: gs://${TRACE_GCS_BUCKET}/${TRACE_GCS_PREFIX}" >&2

  # Gate on vLLM itself before the proxy exists, because once the harness
  # talks to the proxy no endpoint reflects vLLM's readiness (see header).
  if [[ "${TRACE_UPSTREAM_WAIT_SECS}" -gt 0 ]]; then
    echo "Waiting up to ${TRACE_UPSTREAM_WAIT_SECS}s for vLLM at ${TRACE_PROXY_UPSTREAM_BASE_URL}" >&2
    WAIT_DEADLINE=$((SECONDS + TRACE_UPSTREAM_WAIT_SECS))
    until curl --silent --fail --output /dev/null "${TRACE_PROXY_UPSTREAM_BASE_URL%/}/models"; do
      if (( SECONDS >= WAIT_DEADLINE )); then
        echo "vLLM at ${TRACE_PROXY_UPSTREAM_BASE_URL} not ready after ${TRACE_UPSTREAM_WAIT_SECS}s" >&2
        exit 1
      fi
      sleep 2
    done
  fi

  "${REPO_ROOT}/scripts/run_trace_proxy.sh" --config "$TRACE_CONFIG" \
    --recipe "$RECIPE_NAME" --timestamp "$TIMESTAMP"

  # The only change to the harness invocation: LightEval talks to the proxy,
  # which relays to vLLM and logs on the way through. vLLM's own bind address
  # (used above to build SERVER_CMD) is untouched.
  TRACE_ARGS=("server.base_url=http://127.0.0.1:${TRACE_PROXY_PORT}/v1")
fi

# Only the experiment entry point takes --tracing-config; the default
# `evaluation.run` path has no such flag.
ENTRYPOINT_ARGS=()
if [[ "$EVAL_ENTRYPOINT" == "open_r1_tpu.evaluation.experiment" ]]; then
  ENTRYPOINT_ARGS=(--tracing-config "$TRACE_CONFIG")
fi

# The `${ARR[@]+"${ARR[@]}"}` form, not a plain `"${ARR[@]}"`, because bash 3.2
# (macOS's default /bin/bash) treats expanding an empty array as an unbound
# variable under `set -u`.
python3 -m "$EVAL_ENTRYPOINT" --config "$RECIPE" \
  ${ENTRYPOINT_ARGS[@]+"${ENTRYPOINT_ARGS[@]}"} "$@" ${TRACE_ARGS[@]+"${TRACE_ARGS[@]}"}
