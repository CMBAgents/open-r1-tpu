#!/usr/bin/env bash
# Copy GCS bucket data (base model and dataset) onto the TPU VM's local disk.
#
# Tunix's Hugging Face loader contacts the Hub even when its download directory
# already holds the weights, so the local loaders need real files on disk. Run
# this on the VM, not on a workstation: the VM's service account authenticates
# to the bucket, and the copy goes straight from GCS to local disk.
#
# Safe to re-run: rsync only transfers what changed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${GCS_MODEL:-Qwen2.5-Math-1.5B-RoPE-300k}"
DATASET="${GCS_DATASET:-OpenR1-Math-220k}"
BUCKET="${GCS_BUCKET:-}"

usage() {
  cat <<'USAGE'
Usage: scripts/copy_gcs_bucket_data.sh [--bucket gs://BUCKET] [--model NAME]
                                       [--dataset NAME]

Copies a base model and one training dataset from a GCS bucket into the
repository's ignored models/ and data/ directories.

  --bucket gs://BUCKET   Source bucket. Defaults to $GCS_BUCKET.
  --model NAME           Model directory name. Defaults to $GCS_MODEL, or
                         Qwen2.5-Math-1.5B-RoPE-300k.
  --dataset NAME         Dataset directory name. Defaults to $GCS_DATASET, or
                         OpenR1-Math-220k, the distillation corpus. Also
                         accepts SimpleRL-Zoo-Data, the GRPO prompt set.

Objects are read from the bucket's models/NAME and datasets/NAME and written
to models/NAME and data/NAME. Override the bucket layout with
$GCS_MODEL_PREFIX, $GCS_DATA_PREFIX, and $GCS_DATA_GLOB.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bucket)
      if [[ $# -lt 2 ]]; then
        echo "--bucket requires a gs:// value" >&2
        exit 2
      fi
      BUCKET="$2"
      shift
      ;;
    --model)
      if [[ $# -lt 2 ]]; then
        echo "--model requires a name" >&2
        exit 2
      fi
      MODEL="$2"
      shift
      ;;
    --dataset)
      if [[ $# -lt 2 ]]; then
        echo "--dataset requires a name" >&2
        exit 2
      fi
      DATASET="$2"
      shift
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

MODEL_DIR="models/${MODEL}"
MODEL_PREFIX="${GCS_MODEL_PREFIX:-models/${MODEL}}"
DATA_DIR="data/${DATASET}"
DATA_PREFIX="${GCS_DATA_PREFIX:-datasets/${DATASET}}"
# The training glob is per-corpus because the shard layouts differ, and because
# dataset.data_files without a split mapping loads every match into the train
# split. Selecting a held-out shard here would train on the evaluation data.
case "${DATASET}" in
  # Only the `default` config's shards under data/. The repository's `extended`
  # and `all` views cover the same problems, so a wider glob would train on
  # duplicates.
  OpenR1-Math-220k) DATA_GLOB="${GCS_DATA_GLOB:-data/train-*.parquet}" ;;
  SimpleRL-Zoo-Data) DATA_GLOB="${GCS_DATA_GLOB:-simplelr_abel_level3to5/train.parquet}" ;;
  *) DATA_GLOB="${GCS_DATA_GLOB:-*.parquet}" ;;
esac

log() { printf '\n==> %s\n' "$*"; }

if [[ -z "${BUCKET}" ]]; then
  cat >&2 <<'ERR'
No bucket configured. Set GCS_BUCKET in your private environment file, or pass
--bucket:

  echo 'export GCS_BUCKET=gs://your-bucket' >> ~/.open-r1-tpu.env
  source ~/.open-r1-tpu.env

The bucket name is deployment-specific and is deliberately not committed.
ERR
  exit 1
fi
BUCKET="${BUCKET%/}"
if [[ "${BUCKET}" != gs://* ]]; then
  echo "Bucket must start with gs:// (got: ${BUCKET})" >&2
  exit 1
fi

if ! command -v gcloud >/dev/null 2>&1; then
  echo "gcloud not found. Run this on the TPU VM, where it is preinstalled." >&2
  exit 1
fi

cd "${REPO_ROOT}"

log "Copying the base model from ${BUCKET}/${MODEL_PREFIX}"
mkdir -p "${MODEL_DIR}"
gcloud storage rsync "${BUCKET}/${MODEL_PREFIX}" "${MODEL_DIR}" --recursive

log "Copying ${DATASET} from ${BUCKET}/${DATA_PREFIX}"
mkdir -p "${DATA_DIR}"
gcloud storage rsync "${BUCKET}/${DATA_PREFIX}" "${DATA_DIR}" --recursive

# Count what the training glob itself matches, not merely what was copied: an
# empty match fails much later, inside the loader, with a far less obvious error.
log "Verifying the copied inputs"
if [[ ! -s "${MODEL_DIR}/config.json" ]]; then
  echo "WARNING: ${MODEL_DIR}/config.json is missing or empty; check" >&2
  echo "         ${BUCKET}/${MODEL_PREFIX} and \$GCS_MODEL_PREFIX." >&2
fi
shopt -s nullglob
shards=(${DATA_DIR}/${DATA_GLOB})
shopt -u nullglob
if [[ ${#shards[@]} -eq 0 ]]; then
  echo "WARNING: ${DATA_DIR}/${DATA_GLOB} matched no Parquet shards; check" >&2
  echo "         ${BUCKET}/${DATA_PREFIX}, \$GCS_DATA_PREFIX, and" >&2
  echo "         \$GCS_DATA_GLOB." >&2
else
  echo "Found ${#shards[@]} Parquet shard(s) matching ${DATA_DIR}/${DATA_GLOB}"
fi
du -sh "${MODEL_DIR}" "${DATA_DIR}" 2>/dev/null || true

log "Done. Local-input overrides for preflight and training:"
cat <<NEXT

  model.model_source=local \\
  model.model_path=${MODEL_DIR} \\
  tokenizer.tokenizer_path=${MODEL_DIR} \\
  dataset.name=parquet \\
  dataset.config=null \\
  dataset.data_files='${DATA_DIR}/${DATA_GLOB}'

NEXT
