#!/usr/bin/env bash
set -euo pipefail

# Benchmark a model on the TPU: start the vLLM server the recipe describes
# (built by `open_r1_tpu.evaluation.server`), run `open_r1_tpu.evaluation.run`
# against it, and stop the server. Both read the same recipe and dotted
# overrides, so the port and served model name cannot drift apart.
#
#   RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml ./scripts/run_eval_tpu.sh
#
# RECIPE is required, so an expensive run names its tier on purpose. Overrides
# pass straight through, which is how a base model is measured on the same
# stack:
#
#   RECIPE=recipes/Qwen2.5-Math-1.5B/eval/tier1_core.yaml ./scripts/run_eval_tpu.sh \
#     server.model_path=models/Qwen2.5-Math-1.5B
#
# Set TRACE_CONFIG to a tracing config (configs/tracing.example.yaml,
# docker/langfuse/README.md) to trace the run in Langfuse. Set SKIP_SERVER=1 to
# reuse a server that is already up. Needs `scripts/setup_tpu_vm.sh
# --with-eval`, and no other process holding the TPU.

RECIPE="${RECIPE:-}"
SERVER_LOG="${SERVER_LOG:-artifacts/vllm-serve.log}"
SKIP_SERVER="${SKIP_SERVER:-0}"
TRACE_CONFIG="${TRACE_CONFIG:-}"

if [[ -z "$RECIPE" ]]; then
  echo "Usage: RECIPE=recipes/<model>/eval/<tier>.yaml [TRACE_CONFIG=<tracing config>] ./scripts/run_eval_tpu.sh [overrides...]" >&2
  exit 1
fi

TRACE_ARGS=()
if [[ -n "$TRACE_CONFIG" ]]; then
  TRACE_ARGS=(--tracing-config "$TRACE_CONFIG")
fi

source "$(dirname "${BASH_SOURCE[0]}")/lib/vllm_server.sh"
trap stop_vllm_server EXIT INT TERM

if [[ "$SKIP_SERVER" != "1" ]]; then
  mkdir -p "$(dirname "$SERVER_LOG")"
  SERVER_CMD="$(python3 -m open_r1_tpu.evaluation.server --config "$RECIPE" "$@")"
  start_vllm_server "$SERVER_CMD" "$SERVER_LOG"
fi

python3 -m open_r1_tpu.evaluation.run --config "$RECIPE" \
  ${TRACE_ARGS[@]+"${TRACE_ARGS[@]}"} "$@"
