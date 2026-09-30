# Find how this user reaches the Docker daemon. Sourced, not run.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/docker.sh"
#   resolve_docker
#   "${DOCKER[@]}" ps

# Set the DOCKER array to `docker`, or to `sudo -n docker` when only
# passwordless sudo reaches the daemon. Exit the calling script if neither does.
resolve_docker() {
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
}
