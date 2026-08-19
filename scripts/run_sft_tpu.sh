#!/usr/bin/env bash
set -euo pipefail

# Defaults to the reasoning-distillation recipe. Set RECIPE to train a different
# stage, for example the general instruction tuning that precedes it:
#   RECIPE=recipes/Qwen3-1.7B-Instruct/sft/config_instruct.yaml ./scripts/run_sft_tpu.sh
RECIPE="${RECIPE:-recipes/OpenR1-Distill-Qwen3-1.7B/sft/config_distill.yaml}"

python3 -m open_r1_tpu.sft \
  --config "$RECIPE" \
  "$@"
