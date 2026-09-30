# Start and stop a vLLM server in its own process group. Sourced, not run.
#
#   source "$(dirname "${BASH_SOURCE[0]}")/lib/vllm_server.sh"
#   trap stop_vllm_server EXIT INT TERM
#   start_vllm_server "$SERVER_CMD" "$SERVER_LOG"

VLLM_SERVER_PID=""

# Run the shell command $1 with its output in the log file $2, and exit the
# calling script if it dies within five seconds (a failed weight load or
# compilation) rather than leaving the caller to wait out a readiness timeout.
start_vllm_server() {
  local cmd="$1" log="$2"
  echo "Starting: $cmd" >&2
  echo "Server log: $log" >&2
  # vLLM starts a separate EngineCore process. A process group of its own lets
  # stop_vllm_server release the TPU rather than stop only the CLI parent.
  setsid bash -c "exec $cmd" >"$log" 2>&1 &
  VLLM_SERVER_PID=$!

  sleep 5
  if ! kill -0 "$VLLM_SERVER_PID" 2>/dev/null; then
    echo "vLLM server exited during startup; last 40 lines of $log:" >&2
    tail -n 40 "$log" >&2
    exit 1
  fi
}

# Stop the server's process group, if one is running: TERM first so vLLM
# releases the TPU cleanly, then KILL after 30 seconds.
stop_vllm_server() {
  if [[ -n "$VLLM_SERVER_PID" ]] && kill -0 -- "-$VLLM_SERVER_PID" 2>/dev/null; then
    echo "Stopping vLLM server (process group $VLLM_SERVER_PID)" >&2
    kill -TERM -- "-$VLLM_SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 -- "-$VLLM_SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-$VLLM_SERVER_PID" 2>/dev/null || true
    wait "$VLLM_SERVER_PID" 2>/dev/null || true
  fi
  VLLM_SERVER_PID=""
}
