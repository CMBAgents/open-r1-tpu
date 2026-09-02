#!/usr/bin/env bash
set -euo pipefail

# Benchmark a fine-tuned checkpoint on the TPU.
#
# Owns the vLLM server's lifecycle and nothing else: the recipe decides what to
# serve and how (the server command is always built via
# `open_r1_tpu.evaluation.run --print-server-command`), and
# `open_r1_tpu.evaluation.experiment` -- the Langfuse-native path,
# `dataset.run_experiment()` per task/seed -- runs against it. Both halves read
# the same recipe and the same dotted overrides, so the port and the served
# model name cannot drift apart.
#
#   RECIPE=recipes/Qwen3-1.7B-Math/eval/tier0_smoke.yaml \
#     TRACE_CONFIG=configs/tracing.yaml ./scripts/run_eval_tpu.sh
#
# RECIPE is required and has no default: an expensive run must name its tier
# on purpose rather than falling into whichever one happened to be the
# default. TRACE_CONFIG is required too -- the experiment path needs a
# Langfuse section to trace and score against -- and there is no default path
# for it: it is deployment-specific and gitignored (see
# `open_r1_tpu.tracing.config`'s module docstring), so a literal here would be
# either wrong or a leak.
#
# Overrides pass straight through, which is how the base model gets measured on
# the identical stack -- the only baseline worth comparing against:
#
#   RECIPE=recipes/Qwen3-1.7B-Math/eval/tier1_core.yaml \
#     TRACE_CONFIG=configs/tracing.yaml ./scripts/run_eval_tpu.sh \
#     server.model_path=models/Qwen3-1.7B-Base
#
# Needs the locked host stack and pinned service image:
# `scripts/setup_tpu_vm.sh --with-eval`. Only one process can hold the TPU chip,
# so the training job must have exited before this starts. Set SKIP_SERVER=1 to
# reuse a server that is already up.

RECIPE="${RECIPE:-}"
SERVER_LOG="${SERVER_LOG:-artifacts/vllm-serve.log}"
SKIP_SERVER="${SKIP_SERVER:-0}"
TRACE_CONFIG="${TRACE_CONFIG:-}"

if [[ -z "$RECIPE" ]]; then
  echo "Usage: RECIPE=recipes/<model>/eval/<tier>.yaml TRACE_CONFIG=<tracing config path> ./scripts/run_eval_tpu.sh [overrides...]" >&2
  exit 1
fi

if [[ -z "$TRACE_CONFIG" ]]; then
  echo "TRACE_CONFIG=<tracing config path> is required (the Langfuse section open_r1_tpu.evaluation.experiment traces and scores against)" >&2
  exit 1
fi

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
}
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

python3 -m open_r1_tpu.evaluation.experiment --config "$RECIPE" \
  --tracing-config "$TRACE_CONFIG" "$@"
