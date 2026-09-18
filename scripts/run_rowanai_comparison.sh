#!/usr/bin/env bash
# Stage or run one arm. GCS_BUCKET is private deployment configuration.
set -euo pipefail
cd "$(dirname "$0")/.."
arm="${1:?Usage: $0 rowanai|qwen stage|preflight|smoke|train [split]}"
action="${2:?Specify stage, preflight, smoke, or train}"
# The optional third argument selects the held-out-split variant, which trains
# on the 90% train side only and keeps its artifacts under a separate prefix.
variant="${3:-full}"
case "$arm" in
  rowanai) model_dir=rowanai ;;
  qwen) model_dir=Qwen2.5-Math-1.5B-RoPE-300k ;;
  *) echo "Unknown arm: $arm" >&2; exit 2 ;;
esac
case "$variant" in
  full)
    recipe="recipes/rowanai/sft/config_${arm}.yaml"
    prefix="artifacts/rowanai-comparison"
    ;;
  split)
    recipe="recipes/rowanai/sft/config_${arm}_split.yaml"
    prefix="artifacts/rowanai-split"
    ;;
  *) echo "Unknown variant: $variant" >&2; exit 2 ;;
esac
case "$action" in
  stage)
    : "${GCS_BUCKET:?Set GCS_BUCKET to the private bucket name or gs:// URI}"
    bucket="gs://${GCS_BUCKET#gs://}"
    gcloud storage rsync "$bucket/models/$model_dir" "models/$model_dir" --recursive
    mkdir -p data/rowanai
    gcloud storage cp "$bucket/datasets/rowanai/train.jsonl" data/rowanai/train.jsonl
    if [[ "$variant" == split ]]; then
      # Drawn here rather than downloaded, so the split is reproducible from
      # the corpus and the script alone.
      python3 scripts/make_rowanai_split.py \
        --input data/rowanai/train.jsonl \
        --train-out data/rowanai/train_split.jsonl \
        --test-out data/rowanai/test_split.jsonl \
        --test-fraction 0.1 --seed 42
    fi
    ;;
  preflight)
    python3 -m open_r1_tpu.sft.preflight --config "$recipe"
    ;;
  smoke)
    # Same sequence length, mesh and effective batch as full training.
    out="$prefix/${arm}-smoke"
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
      "training.run_name=$(basename "$prefix")-${arm}-smoke" \
      training.wandb.enabled=false "export.output_dir=$out/merged" \
      2>&1 | tee "$out/run.log"
    test -s "$out/merged/model.safetensors"
    touch "$out/smoke-passed"
    ;;
  train)
    : "${GCS_BUCKET:?Set GCS_BUCKET to the private bucket name or gs:// URI}"
    bucket="gs://${GCS_BUCKET#gs://}"
    out="$prefix/$arm"
    if [[ ! -f "$prefix/${arm}-smoke/smoke-passed" ]]; then
      echo "Complete this arm's TPU smoke on this VM before full training." >&2
      exit 1
    fi
    if [[ -e "$out" ]]; then
      echo "Run output already exists: $out; resume explicitly with the recipe." >&2
      exit 1
    fi
    # A full run's checkpoints do not fit next to the merged export on a VM
    # root disk, and a disk-full failure only surfaces at the export, after the
    # whole run has been paid for. Refuse early instead.
    free_gib=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    if (( free_gib < 15 )); then
      echo "Only ${free_gib}GiB free; the merged export needs headroom." >&2
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
