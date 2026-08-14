#!/usr/bin/env bash
set -euo pipefail

python3 -m open_r1_tpu.sft \
  --config recipes/OpenR1-Distill-Qwen3-1.7B/sft/config_distill.yaml \
  "$@"

