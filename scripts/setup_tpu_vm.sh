#!/usr/bin/env bash
# Provision the open-r1-tpu Python environment on a TPU VM.
#
# Installs uv, the pinned CPython build from .python-version, a project
# virtualenv, and the project itself (which pulls in Tunix and jax[tpu]). With
# --with-eval it also installs the locked LightEval stack and pulls the pinned
# vLLM TPU service image.
# Safe to re-run: existing components are reused unless --recreate is passed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
ENV_FILE="${HOME}/.open-r1-tpu.env"
UV_BIN_DIR="${HOME}/.local/bin"
UV_VERSION="0.12.5"
RECREATE=0
VERIFY=1
WITH_EVAL=0

usage() {
  cat <<'USAGE'
Usage: scripts/setup_tpu_vm.sh [--recreate] [--skip-verify] [--with-eval]

  --recreate      Delete and rebuild .venv from scratch.
  --skip-verify   Skip the JAX device check and unit tests.
  --with-eval     Install the locked evaluation stack and pull vLLM TPU.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate) RECREATE=1 ;;
    --skip-verify) VERIFY=0 ;;
    --with-eval) WITH_EVAL=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

log() { printf '\n==> %s\n' "$*"; }

# --- System prerequisites -----------------------------------------------
# curl fetches the uv installer; git is required to build the pinned Tunix
# dependency from its GitHub revision.
missing_pkgs=()
command -v curl >/dev/null 2>&1 || missing_pkgs+=(curl)
command -v git >/dev/null 2>&1 || missing_pkgs+=(git)
if [[ ${#missing_pkgs[@]} -gt 0 ]]; then
  log "Installing system packages: ${missing_pkgs[*]}"
  sudo apt-get update -qq
  sudo apt-get install -y "${missing_pkgs[@]}"
fi

# --- uv ------------------------------------------------------------------
export PATH="${UV_BIN_DIR}:${PATH}"
if command -v uv >/dev/null 2>&1 \
  && [[ "$(uv --version)" == "uv ${UV_VERSION}"* ]]; then
  log "uv already installed: $(uv --version)"
else
  log "Installing pinned uv ${UV_VERSION}"
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" \
    | env UV_INSTALL_DIR="${UV_BIN_DIR}" sh
  hash -r
  log "Installed $(uv --version)"
fi

# --- Interpreter ---------------------------------------------------------
PYTHON_VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/.python-version")"
if [[ -z "${PYTHON_VERSION}" ]]; then
  echo "Could not read a version from ${REPO_ROOT}/.python-version" >&2
  exit 1
fi
if [[ "${PYTHON_VERSION}" == *t ]]; then
  echo "Refusing to install the free-threaded build ${PYTHON_VERSION}" >&2
  exit 1
fi

log "Installing CPython ${PYTHON_VERSION}"
uv python install "${PYTHON_VERSION}"

# --- Virtualenv ----------------------------------------------------------
if [[ ${RECREATE} -eq 1 && -d "${VENV_DIR}" ]]; then
  log "Removing existing ${VENV_DIR}"
  rm -rf "${VENV_DIR}"
fi
if [[ -x "${VENV_DIR}/bin/python" ]]; then
  log "Reusing ${VENV_DIR} ($("${VENV_DIR}/bin/python" --version))"
else
  log "Creating ${VENV_DIR} on CPython ${PYTHON_VERSION}"
  uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
fi

# --- Project dependencies ------------------------------------------------
# jax[tpu] and libtpu arrive transitively through the pinned google-tunix
# revision in pyproject.toml; no CUDA or PyTorch wheels are installed on the
# host. --frozen makes the checked-in uv.lock authoritative.
SYNC_ARGS=(--frozen --extra test)
if [[ ${WITH_EVAL} -eq 1 ]]; then
  SYNC_ARGS+=(--extra eval)
fi
log "Installing the locked open-r1-tpu environment"
cd "${REPO_ROOT}"
uv sync "${SYNC_ARGS[@]}"

# --- Containerised inference service ------------------------------------
if [[ ${WITH_EVAL} -eq 1 ]]; then
  log "Pulling the immutable vLLM TPU service image"
  scripts/run_vllm_tpu_container.sh --pull-only
fi

# --- Run-time environment file -------------------------------------------
# Kept outside the repository so entity/project names and tokens never land in
# git. W&B reads WANDB_ENTITY directly; the project name has to be passed to
# the launcher as an override because sft.py sets it explicitly.
if [[ -f "${ENV_FILE}" ]]; then
  log "Leaving existing ${ENV_FILE} untouched"
else
  log "Writing ${ENV_FILE} template"
  cat > "${ENV_FILE}" <<'ENVFILE'
# Sourced before training. Not tracked by git; fill in and keep private.
export PATH="${HOME}/.local/bin:${PATH}"
# export GCS_BUCKET=gs://your-bucket
# export WANDB_ENTITY=your-wandb-entity
# export WANDB_PROJECT=your-wandb-project
# export HF_TOKEN=hf_...
ENVFILE
  chmod 600 "${ENV_FILE}"
fi

# --- Verification --------------------------------------------------------
if [[ ${VERIFY} -eq 1 ]]; then
  log "Checking JAX sees exactly one TPU device"
  if ! "${VENV_DIR}/bin/python" - <<'PY'
import jax

devices = jax.devices()
print(f"jax {jax.__version__}: {devices}")
assert devices and all(d.platform == "tpu" for d in devices), "no TPU devices visible"
assert len(devices) == 1, f"expected 1 device, found {len(devices)}"
PY
  then
    echo "WARNING: TPU check failed. If a training job is running it holds the" >&2
    echo "         accelerator and only one process can claim it at a time." >&2
  fi

  log "Running host-independent unit tests"
  "${VENV_DIR}/bin/python" -m pytest -q
fi

log "Done. Next steps:"
cat <<NEXT

  # 1. Fill in the bucket, W&B names, and token; the template ships commented out.
  \$EDITOR ${ENV_FILE}
  source ${ENV_FILE}
  source ${VENV_DIR}/bin/activate

  # 2. Copy GCS bucket data to local disk (skip to train from the Hub instead):
  scripts/copy_gcs_bucket_data.sh

  # 3. Preflight, then launch. See "Quick start on a TPU VM" in README.md for
  #    the local-input overrides both commands need after step 2.
  python -m open_r1_tpu.check_env
  scripts/run_sft_tpu.sh training.project_name="\${WANDB_PROJECT}"
NEXT

if [[ ${WITH_EVAL} -eq 1 ]]; then
  cat <<'NEXT_EVAL'

  # Evaluation is installed and the immutable service image is cached. Once a
  # merged export exists, preflight and run the smoke tier:
  python -m open_r1_tpu.check_eval_env \
    --config recipes/Qwen3-1.7B-Math/eval/tier0_smoke.yaml
  RECIPE=recipes/Qwen3-1.7B-Math/eval/tier0_smoke.yaml scripts/run_eval_tpu.sh
NEXT_EVAL
fi
