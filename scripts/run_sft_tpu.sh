#!/usr/bin/env bash
set -euo pipefail

# Train one SFT recipe. RECIPE is required: a long run names its recipe on
# purpose. Dotted overrides pass straight through:
#
#   RECIPE=recipes/Qwen2.5-Math-1.5B/sft/openr1-math-220k.yaml ./scripts/run_sft_tpu.sh \
#     training.max_steps=4
RECIPE="${RECIPE:-}"
if [[ -z "$RECIPE" ]]; then
  echo "Usage: RECIPE=recipes/<model>/sft/<recipe>.yaml ./scripts/run_sft_tpu.sh [overrides...]" >&2
  exit 1
fi

python3 -m open_r1_tpu.sft.run \
  --config "$RECIPE" \
  "$@"
