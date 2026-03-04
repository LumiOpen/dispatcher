#!/bin/bash
###############################################################################
# launch_dispatcher_server.sh - Launch dispatcher server on the login node in tmux
#
# Usage:
#   ./launch_dispatcher_server.sh          # Start server in a tmux session
#   ./launch_dispatcher_server.sh --stop   # Stop the server
#   ./launch_dispatcher_server.sh --status # Check server status
#
# The server address is saved to .dispatcher_address for the worker scripts.
# Server stdout/stderr are logged to logs/dispatcher-server.{out,err}.
# The server is lightweight (FastAPI) and safe to run on the login node.
###############################################################################

set -euo pipefail

###############################################################################
# Configuration (mirrors launch_translation_task_dsv3.sh)
###############################################################################

INPUT_FILE=/shared_silo/scratch/adamhrin@amd.com/dispatcher/examples/reasoning_traces/data/Llama-Nemotron-SFT-math-1.5mplus.jsonl
OUTPUT_FILE=/shared_silo/scratch/adamhrin@amd.com/dispatcher/examples/reasoning_traces/data/Llama-Nemotron-SFT-math-1.5mplus_translated_DeepSeek-V3_fi.jsonl

WORK_TIMEOUT=${WORK_TIMEOUT:-1800}
DISPATCHER_PORT=${DISPATCHER_PORT:-9999}
MAX_RETRIES=${MAX_RETRIES:-10}
SESSION_NAME="dispatcher-server-test-retry-reissue"

# Dispatcher package source
DISPATCHER_PKG=/shared_silo/scratch/adamhrin@amd.com/dispatcher

# Resolve working directory (reasoning_traces/)
WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ADDRESS_FILE="$WORK_DIR/.dispatcher_address"
LOG_OUT="$WORK_DIR/logs/dispatcher-server.out"
LOG_ERR="$WORK_DIR/logs/dispatcher-server.err"

###############################################################################
# --stop: kill the tmux session and clean up
###############################################################################
if [ "${1:-}" = "--stop" ]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    tmux kill-session -t "$SESSION_NAME"
    echo "Server session '$SESSION_NAME' stopped."
  else
    echo "No active session '$SESSION_NAME' found."
  fi
  rm -f "$ADDRESS_FILE"
  exit 0
fi

###############################################################################
# --status: check the server
###############################################################################
if [ "${1:-}" = "--status" ]; then
  if [ -f "$ADDRESS_FILE" ]; then
    ADDR=$(cat "$ADDRESS_FILE")
    echo "Address file: $ADDR"
    echo "Querying server..."
    curl -s "http://$ADDR/status" 2>/dev/null && echo "" || echo "Server not reachable."
  else
    echo "No address file found. Server may not be running."
  fi
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Tmux session '$SESSION_NAME' is active."
  else
    echo "Tmux session '$SESSION_NAME' not found."
  fi
  echo ""
  echo "Log files:"
  echo "  stdout: $LOG_OUT"
  echo "  stderr: $LOG_ERR"
  exit 0
fi

###############################################################################
# --run-server: internal entry point (called inside the tmux session)
###############################################################################
if [ "${1:-}" = "--run-server" ]; then
  cd "$WORK_DIR"
  mkdir -p "$WORK_DIR/logs"

  # Log all output to files while also showing in the tmux terminal
  exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

  export DISPATCHER_SERVER=$(hostname)
  export DISPATCHER_PORT

  # Write address file early so workers can discover us.
  # The main script waits for the TCP port to actually open before declaring
  # success, so this is safe even though the server isn't listening yet.
  echo "${DISPATCHER_SERVER}:${DISPATCHER_PORT}" > "$ADDRESS_FILE"

  # Install dispatcher package in user site-packages (no singularity on login node).
  echo "Installing dispatcher package from $DISPATCHER_PKG..."
  pip install --user -e "$DISPATCHER_PKG"

  echo "=========================================="
  echo "Dispatcher server starting"
  echo "  Address:      $DISPATCHER_SERVER:$DISPATCHER_PORT"
  echo "  Input:        $INPUT_FILE"
  echo "  Output:       $OUTPUT_FILE"
  echo "  Work timeout: $WORK_TIMEOUT"
  echo "  Log stdout:   $LOG_OUT"
  echo "  Log stderr:   $LOG_ERR"
  echo "=========================================="

  python3 -m dispatcher.server \
    --infile "$INPUT_FILE" \
    --outfile "$OUTPUT_FILE" \
    --work-timeout "$WORK_TIMEOUT" \
    --max-retries "$MAX_RETRIES" \
    --host 0.0.0.0 \
    --port "$DISPATCHER_PORT"

  echo "Server exited."
  rm -f "$ADDRESS_FILE"
  exit 0
fi

###############################################################################
# Main: start the server in a new tmux session
###############################################################################
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Tmux session '$SESSION_NAME' already exists."
  echo "  Attach:  tmux attach -t $SESSION_NAME"
  echo "  Stop:    $0 --stop"
  echo "  Status:  $0 --status"
  exit 1
fi

echo "Starting dispatcher server in tmux session '$SESSION_NAME'..."
# Use --login so the tmux shell inherits the full login environment
# (module loads, PATH, etc.) just like an interactive session.
tmux new-session -d -s "$SESSION_NAME" "bash --login $0 --run-server"

# Wait for the TCP port to actually open (not just the address file).
echo "Waiting for server to come up (this may take a few minutes on first run)..."
for i in $(seq 1 150); do
  if [ -f "$ADDRESS_FILE" ]; then
    ADDR=$(cat "$ADDRESS_FILE")
    IFS=: read -r HOST PORT <<< "$ADDR"
    if (echo >/dev/tcp/"$HOST"/"$PORT") >/dev/null 2>&1; then
      echo ""
      echo "Server is up at: $ADDR"
      echo ""
      echo "  Monitor: tmux attach -t $SESSION_NAME"
      echo "  Logs:    tail -f $LOG_OUT"
      echo "  Status:  $0 --status"
      echo "  Stop:    $0 --stop"
      echo "  Check:   curl http://$ADDR/status"
      exit 0
    fi
  fi
  printf "."
  sleep 2
done

echo ""
echo "WARNING: Server did not become reachable within timeout."
echo "Check the tmux session: tmux attach -t $SESSION_NAME"
echo "Check logs: $LOG_ERR"
exit 1
