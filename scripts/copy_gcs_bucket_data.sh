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
MODEL_DIR="models/Qwen3-1.7B-Base"
DATA_DIR="data/Mixture-of-Thoughts"
MODEL_PREFIX="${GCS_MODEL_PREFIX:-models/Qwen3-1.7B-Base}"
DATA_PREFIX="${GCS_DATA_PREFIX:-datasets/Mixture-of-Thoughts}"
BUCKET="${GCS_BUCKET:-}"

usage() {
  cat <<'USAGE'
Usage: scripts/copy_gcs_bucket_data.sh [--bucket gs://BUCKET]

Copies the base model and the reasoning dataset from a GCS bucket into the
repository's ignored models/ and data/ directories.

  --bucket gs://BUCKET   Source bucket. Defaults to $GCS_BUCKET.

Object prefixes default to models/Qwen3-1.7B-Base and
datasets/Mixture-of-Thoughts; override with $GCS_MODEL_PREFIX and
$GCS_DATA_PREFIX if the bucket is laid out differently.
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
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

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

log "Copying the dataset from ${BUCKET}/${DATA_PREFIX}"
mkdir -p "${DATA_DIR}"
gcloud storage rsync "${BUCKET}/${DATA_PREFIX}" "${DATA_DIR}" --recursive

# The launcher selects Parquet shards by glob, so an empty match would fail
# later with a much less obvious error than a missing-file report here.
log "Verifying the copied inputs"
if [[ ! -s "${MODEL_DIR}/config.json" ]]; then
  echo "WARNING: ${MODEL_DIR}/config.json is missing or empty; check" >&2
  echo "         ${BUCKET}/${MODEL_PREFIX} and \$GCS_MODEL_PREFIX." >&2
fi
shard_count=$(find "${DATA_DIR}/all" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')
if [[ "${shard_count}" == "0" ]]; then
  echo "WARNING: no Parquet shards under ${DATA_DIR}/all; check" >&2
  echo "         ${BUCKET}/${DATA_PREFIX} and \$GCS_DATA_PREFIX." >&2
else
  echo "Found ${shard_count} Parquet shard(s) under ${DATA_DIR}/all"
fi
du -sh "${MODEL_DIR}" "${DATA_DIR}" 2>/dev/null || true

log "Done. Local-input overrides for preflight and training:"
cat <<'NEXT'

  model.model_source=local \
  model.model_path=models/Qwen3-1.7B-Base \
  tokenizer.tokenizer_path=models/Qwen3-1.7B-Base \
  dataset.name=parquet \
  dataset.config=null \
  dataset.data_files='data/Mixture-of-Thoughts/all/*.parquet'

NEXT
