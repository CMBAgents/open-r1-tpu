#!/usr/bin/env bash
set -euo pipefail

# Measure the same merged model through the vLLM service and Tunix's direct
# sampler, one after the other since only one process can hold the TPU.
# Startup and one warm-up batch per shape are recorded apart from steady-state
# throughput.
#
# Defaults time batch sizes 1 and 8 over 16 distinct prompts, twice, on the
# export the eval recipe names (server.model_path):
#
#   ./scripts/benchmark_generation_tpu.sh
#
# Point at another export or vLLM installation with environment values and
# eval-recipe overrides, for example:
#
#   MODEL_PATH=models/Qwen2.5-Math-1.5B \
#     ./scripts/benchmark_generation_tpu.sh \
#       server.serve_command='["/opt/vllm-venv/bin/vllm", "serve"]' \
#       server.image=null

EVAL_RECIPE="${EVAL_RECIPE:-recipes/Qwen2.5-Math-1.5B/eval/tier0_smoke.yaml}"
SFT_RECIPE="${SFT_RECIPE:-recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml}"
PROMPT_COUNT="${PROMPT_COUNT:-16}"
REPEATS="${REPEATS:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-256}"
BATCH_SIZES="${BATCH_SIZES:-1 8}"
MAX_MODEL_LEN=$((MAX_PROMPT_LENGTH + MAX_NEW_TOKENS))

# Every command below reads the eval recipe with the same overrides, so the
# served model, port and context window cannot drift apart.
OVERRIDES=(
  "server.max_model_len=$MAX_MODEL_LEN"
  "sampling.max_new_tokens=$MAX_NEW_TOKENS"
)
if [[ -n "${MODEL_PATH:-}" ]]; then
  OVERRIDES+=("server.model_path=$MODEL_PATH")
fi
OVERRIDES+=("$@")

read_setting() {
  python3 - "$EVAL_RECIPE" "$1" "${OVERRIDES[@]}" <<'PY'
import sys

from open_r1_tpu.evaluation.config import load_eval_config, resolve_settings

recipe, key, *overrides = sys.argv[1:]
print(resolve_settings(load_eval_config(recipe, overrides))[key])
PY
}

MODEL_PATH="$(read_setting model_path)"
BASE_URL="$(read_setting base_url)"
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "No merged export at $MODEL_PATH. Train the recipe that writes it, or" >&2
  echo "set MODEL_PATH to an existing export." >&2
  exit 1
fi
OUTPUT_DIR="${OUTPUT_DIR:-$(dirname "$MODEL_PATH")/generation-speed}"
SERVER_LOG="${SERVER_LOG:-${OUTPUT_DIR}/vllm-serve.log}"
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"

source "$SCRIPT_DIR/lib/vllm_server.sh"
trap stop_vllm_server EXIT INT TERM
mkdir -p "$OUTPUT_DIR"

SERVER_CMD="$(python3 -m open_r1_tpu.evaluation.server \
  --config "$EVAL_RECIPE" \
  "${OVERRIDES[@]}")"
SECONDS=0
start_vllm_server "$SERVER_CMD" "$SERVER_LOG"
python3 -c \
  "import sys; from open_r1_tpu.evaluation.server import wait_for_server; wait_for_server(sys.argv[1])" \
  "$BASE_URL"
VLLM_STARTUP_SECONDS="$SECONDS"

# shellcheck disable=SC2086 # BATCH_SIZES is intentionally a whitespace list.
python3 "$SCRIPT_DIR/benchmark_generation.py" run \
  --backend vllm \
  --eval-config "$EVAL_RECIPE" \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT_DIR/vllm.json" \
  --batch-sizes $BATCH_SIZES \
  --prompt-count "$PROMPT_COUNT" \
  --repeats "$REPEATS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  --startup-seconds "$VLLM_STARTUP_SECONDS" \
  "${OVERRIDES[@]}"

stop_vllm_server

# shellcheck disable=SC2086 # BATCH_SIZES is intentionally a whitespace list.
python3 "$SCRIPT_DIR/benchmark_generation.py" run \
  --backend tunix \
  --eval-config "$EVAL_RECIPE" \
  --sft-config "$SFT_RECIPE" \
  --model-path "$MODEL_PATH" \
  --output "$OUTPUT_DIR/tunix.json" \
  --batch-sizes $BATCH_SIZES \
  --prompt-count "$PROMPT_COUNT" \
  --repeats "$REPEATS" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-prompt-length "$MAX_PROMPT_LENGTH" \
  "${OVERRIDES[@]}"

python3 "$SCRIPT_DIR/benchmark_generation.py" compare \
  --vllm "$OUTPUT_DIR/vllm.json" \
  --tunix "$OUTPUT_DIR/tunix.json" \
  --output-json "$OUTPUT_DIR/comparison.json" \
  --output-markdown "$OUTPUT_DIR/comparison.md"

echo "Comparison: $OUTPUT_DIR/comparison.md"
