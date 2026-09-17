#!/usr/bin/env bash
# Stage or run one arm. GCS_BUCKET is private deployment configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
arm="${1:?Usage: $0 rowanai|qwen stage|preflight|smoke|train}"
action="${2:?Specify stage, preflight, smoke, or train}"
case "$arm" in
  rowanai) model_dir=rowanai ;;
  qwen) model_dir=Qwen2.5-Math-1.5B-RoPE-300k ;;
  *) echo "Unknown arm: $arm" >&2; exit 2 ;;
esac
recipe="recipes/rowanai/sft/config_${arm}.yaml"
case "$action" in
  stage)
    : "${GCS_BUCKET:?Set GCS_BUCKET to the private bucket name or gs:// URI}"
    bucket="gs://${GCS_BUCKET#gs://}"
    gcloud storage rsync "$bucket/models/$model_dir" "models/$model_dir" --recursive
    mkdir -p data/rowanai
    gcloud storage cp "$bucket/datasets/rowanai/train.jsonl" data/rowanai/train.jsonl
    ;;
  preflight)
    python3 -m open_r1_tpu.sft.preflight --config "$recipe"
    ;;
  smoke)
    # Same sequence length, mesh and effective batch as full training.
    out="artifacts/rowanai-comparison/${arm}-smoke"
    if [[ -e "$out" ]]; then
      echo "Smoke output already exists: $out; use a fresh directory manually." >&2
      exit 1
    fi
    python3 -m open_r1_tpu.sft.preflight --config "$recipe"
    mkdir -p "$out"
    RECIPE="$recipe" ./scripts/run_sft_tpu.sh \
      dataset.max_examples=128 training.max_steps=4 \
      training.checkpointing_options.save_interval_steps=2 \
      "training.checkpoint_dir=$out/checkpoints" \
      "training.metrics_log_dir=$out/logs" \
      "training.run_name=rowanai-comparison-${arm}-smoke" \
      training.wandb.enabled=false "export.output_dir=$out/merged" \
      2>&1 | tee "$out/run.log"
    test -s "$out/merged/model.safetensors"
    touch "$out/smoke-passed"
    ;;
  train)
    : "${GCS_BUCKET:?Set GCS_BUCKET to the private bucket name or gs:// URI}"
    bucket="gs://${GCS_BUCKET#gs://}"
    out="artifacts/rowanai-comparison/$arm"
    if [[ ! -f "artifacts/rowanai-comparison/${arm}-smoke/smoke-passed" ]]; then
      echo "Complete this arm's TPU smoke on this VM before full training." >&2
      exit 1
    fi
    if [[ -e "$out" ]]; then
      echo "Run output already exists: $out; resume explicitly with the recipe." >&2
      exit 1
    fi
    # Direct GCS checkpoints survive VM expiry; merged export stays local.
    python3 -m open_r1_tpu.sft.preflight --config "$recipe"
    mkdir -p "$out"
    RECIPE="$recipe" ./scripts/run_sft_tpu.sh \
      "training.checkpoint_dir=$bucket/$out/checkpoints" \
      2>&1 | tee "$out/run.log"
    gcloud storage rsync "$out" "$bucket/$out" --recursive
    ;;
  *) echo "Unknown action: $action" >&2; exit 2 ;;
esac
