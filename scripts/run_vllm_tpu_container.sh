#!/usr/bin/env bash
set -euo pipefail

# Run the locally built vLLM TPU OpenAI server without installing its Python 3.12
# dependency tree into the host's Python 3.13 evaluation environment.
#
# The model export is mounted read-only at the same absolute path. Hugging Face
# and vLLM/XLA caches use named volumes so an ephemeral container does not
# download or compile everything again, and so root-owned cache files do not
# leak into the user's home directory. HF_TOKEN is forwarded by variable name,
# never expanded into the command or written to a recipe/log.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_IMAGE="$(
  PYTHONPATH="${REPO_ROOT}/src" python3 -c \
    'from open_r1_tpu.evaluation.stack import vllm_tpu_image_tag; print(vllm_tpu_image_tag())'
)"

IMAGE="${VLLM_TPU_IMAGE:-${DEFAULT_IMAGE}}"
ACTION=run

usage() {
  cat <<'USAGE'
Usage: scripts/run_vllm_tpu_container.sh [OPTIONS] -- MODEL_PATH [VLLM_ARGS...]

Options:
  --image IMAGE  Use the derived local image tag or a digest-pinned remote image.
  --build        Build the derived local service image, then exit.
  --check        Verify Docker access and that the selected image is present.
  --provenance   Print the selected image ID and service versions as JSON, then exit.
  --pull-only    Pull a digest-pinned remote image, verify it, then exit.
  --print-image  Print the derived local image tag, then exit.
  -h, --help     Show this help.

Environment:
  VLLM_TPU_CONTAINER_NAME   Container name (default: open-r1-tpu-vllm).
  VLLM_TPU_HF_CACHE_VOLUME Persistent Hub cache volume.
  VLLM_TPU_CACHE_VOLUME    Persistent vLLM/XLA cache volume.
  HF_TOKEN                 Forwarded by name when set; its value is not logged.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      [[ $# -ge 2 ]] || { echo "--image requires a value" >&2; exit 2; }
      IMAGE="$2"
      shift 2
      ;;
    --check)
      ACTION=check
      shift
      ;;
    --build)
      ACTION=build
      shift
      ;;
    --provenance)
      ACTION=provenance
      shift
      ;;
    --pull-only)
      ACTION=pull
      shift
      ;;
    --print-image)
      printf '%s\n' "${DEFAULT_IMAGE}"
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      break
      ;;
  esac
done

is_remote_image() {
  local reference="$1"
  local repository="${reference%%@sha256:*}"
  [[ "${repository}" == */* ]] || return 1
  local first_component="${repository%%/*}"
  [[ "${first_component}" == "localhost" || "${first_component}" == *.* || "${first_component}" == *:* ]]
}

if is_remote_image "${IMAGE}"; then
  if [[ ! "${IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]; then
    echo "Refusing mutable remote vLLM image reference: ${IMAGE}" >&2
    echo "Use a remote release tag followed by its immutable @sha256 digest." >&2
    exit 2
  fi
elif [[ "${IMAGE}" != "${DEFAULT_IMAGE}" ]]; then
  echo "Refusing local vLLM image tag that does not match this build spec: ${IMAGE}" >&2
  echo "Expected the derived tag: ${DEFAULT_IMAGE}" >&2
  exit 2
fi

DOCKER=()
if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  DOCKER=(docker)
elif command -v docker >/dev/null 2>&1 \
  && command -v sudo >/dev/null 2>&1 \
  && sudo -n docker info >/dev/null 2>&1; then
  DOCKER=(sudo -n docker)
else
  echo "Docker is unavailable or inaccessible." >&2
  echo "Install/start Docker, or grant this user Docker access (passwordless sudo is accepted)." >&2
  exit 1
fi

if [[ "${ACTION}" == pull ]] && ! is_remote_image "${IMAGE}"; then
  echo "--pull-only is only for a digest-pinned remote image." >&2
  echo "Build the local service image with: scripts/run_vllm_tpu_container.sh --build" >&2
  exit 2
fi

if [[ "${ACTION}" == build ]]; then
  "${DOCKER[@]}" build --tag "${IMAGE}" "${REPO_ROOT}/docker/vllm-tpu"
  printf 'Built vLLM TPU image: %s\n' "${IMAGE}"
  exit 0
fi

if [[ "${ACTION}" == pull ]]; then
  "${DOCKER[@]}" pull "${IMAGE}"
fi

if [[ "${ACTION}" == check || "${ACTION}" == pull || "${ACTION}" == provenance ]]; then
  if ! "${DOCKER[@]}" image inspect "${IMAGE}" >/dev/null 2>&1; then
    echo "vLLM TPU image is not present: ${IMAGE}" >&2
    if is_remote_image "${IMAGE}"; then
      echo "Run scripts/run_vllm_tpu_container.sh --image '${IMAGE}' --pull-only first." >&2
    else
      echo "Run scripts/run_vllm_tpu_container.sh --build first." >&2
    fi
    exit 1
  fi
  if [[ "${ACTION}" == provenance ]]; then
    IMAGE_ID="$("${DOCKER[@]}" image inspect --format '{{.Id}}' "${IMAGE}")"
    SERVICE_VERSIONS="$("${DOCKER[@]}" run --rm --entrypoint python3 "${IMAGE}" -c \
      "import importlib.metadata as m; print(m.version('vllm-tpu'), m.version('tpu-inference'))")"
    read -r VLLM_TPU_VERSION TPU_INFERENCE_VERSION <<< "${SERVICE_VERSIONS}"
    if [[ -z "${VLLM_TPU_VERSION}" || -z "${TPU_INFERENCE_VERSION}" ]]; then
      echo "Could not read vLLM TPU service versions from ${IMAGE}." >&2
      exit 1
    fi
    printf '{"image_id":"%s","service_versions":{"vllm-tpu":"%s","tpu-inference":"%s"}}\n' \
      "${IMAGE_ID}" "${VLLM_TPU_VERSION}" "${TPU_INFERENCE_VERSION}"
  else
    printf 'Docker server: %s\n' "$("${DOCKER[@]}" version --format '{{.Server.Version}}')"
    printf 'vLLM TPU image: %s\n' "${IMAGE}"
  fi
  exit 0
fi

if ! is_remote_image "${IMAGE}" \
  && ! "${DOCKER[@]}" image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Derived local vLLM TPU image is not present: ${IMAGE}" >&2
  echo "Run scripts/run_vllm_tpu_container.sh --build first." >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  usage >&2
  exit 2
fi

MODEL_PATH="$1"
shift
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Model export is not a directory: ${MODEL_PATH}" >&2
  exit 1
fi
MODEL_PATH="$(cd "${MODEL_PATH}" && pwd -P)"

CONTAINER_NAME="${VLLM_TPU_CONTAINER_NAME:-open-r1-tpu-vllm}"
HF_CACHE_VOLUME="${VLLM_TPU_HF_CACHE_VOLUME:-open-r1-tpu-huggingface-cache}"
VLLM_CACHE_VOLUME="${VLLM_TPU_CACHE_VOLUME:-open-r1-tpu-vllm-cache}"

if "${DOCKER[@]}" container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container ${CONTAINER_NAME} already exists; stop it before evaluating." >&2
  exit 1
fi

CID_DIR="$(mktemp -d /tmp/open-r1-tpu-vllm-cid.XXXXXX)"
CID_FILE="${CID_DIR}/container.cid"
DOCKER_RUN_PID=""

cleanup() {
  local container_id=""
  if [[ -s "${CID_FILE}" ]]; then
    container_id="$(tr -d '[:space:]' < "${CID_FILE}")"
  fi
  if [[ -n "${container_id}" ]]; then
    "${DOCKER[@]}" stop --time 30 "${container_id}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${DOCKER_RUN_PID}" ]]; then
    wait "${DOCKER_RUN_PID}" 2>/dev/null || true
  fi
  rm -f "${CID_FILE}"
  rmdir "${CID_DIR}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

TOKEN_ARGS=()
if [[ -n "${HF_TOKEN:-}" ]]; then
  TOKEN_ARGS=(--env HF_TOKEN)
fi

"${DOCKER[@]}" run \
  --rm \
  --name "${CONTAINER_NAME}" \
  --cidfile "${CID_FILE}" \
  --stop-timeout 30 \
  --privileged \
  --network host \
  --shm-size 150g \
  --volume /dev/shm:/dev/shm \
  --volume "${MODEL_PATH}:${MODEL_PATH}:ro" \
  --volume "${HF_CACHE_VOLUME}:/root/.cache/huggingface" \
  --volume "${VLLM_CACHE_VOLUME}:/root/.cache/vllm" \
  --env HF_HOME=/root/.cache/huggingface \
  --env VLLM_XLA_CACHE_PATH=/root/.cache/vllm/xla_cache \
  "${TOKEN_ARGS[@]}" \
  --entrypoint vllm \
  "${IMAGE}" \
  serve "${MODEL_PATH}" "$@" &
DOCKER_RUN_PID=$!

RUN_STATUS=0
wait "${DOCKER_RUN_PID}" || RUN_STATUS=$?
DOCKER_RUN_PID=""
exit "${RUN_STATUS}"
