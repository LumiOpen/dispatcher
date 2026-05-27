#!/bin/bash
###############################################################################
# launch_annotation_workers.sh - Single-node preemptable dispatcher worker
#
# Runs one dispatcher worker against the central
# dispatcher server launched by start_dispatcher_server.sh.
#
# When preempted, any in-flight work items time out on the server side and are
# automatically reissued to another (or the same requeued) worker. No data is lost.
#
# Usage:
# sbatch jobs/launch_translation_workers.sh [configs/your-config.conf]
#
# If a config file is provided, it is sourced before applying defaults.
# Config files are shell-sourceable files with KEY=VALUE pairs.
###############################################################################

#SBATCH --job-name=inference
#SBATCH --partition=amd-tw-verification
#SBATCH --time=10-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=112G
#SBATCH --cpus-per-task=7
#SBATCH --hint=nomultithread
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --open-mode=append

set -euo pipefail

###############################################################################
# Config file: source if provided as first argument
###############################################################################

CONFIG_FILE="${1:-}"

# Resolve working directory (reasoning_traces/)
WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"

if [ -n "$CONFIG_FILE" ]; then
  if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$(cd "$(dirname "$CONFIG_FILE")" && pwd)/$(basename "$CONFIG_FILE")"
  fi
  if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE" >&2
    exit 1
  fi
  source "$CONFIG_FILE"
fi

###############################################################################
# Configuration - defaults for anything not set by config file or environment
###############################################################################

LAUNCHER="${LAUNCHER:-$WORK_DIR/tw_singularity_launcher.sh}"
SERVER_ADDRESS_FILE="${SERVER_ADDRESS_FILE:-.dispatcher-server}"
ADDRESS_FILE="$WORK_DIR/$SERVER_ADDRESS_FILE"

# Dispatcher package source (mirrors start_dispatcher_server.sh). Default:
# repo root two levels above WORK_DIR (examples/backlog_inference), so jobs
# submitted from that dir pick up the in-tree dispatcher.
DISPATCHER_PKG="${DISPATCHER_PKG:-$(cd "$WORK_DIR/../.." && pwd)}"

# Skip pip install when dispatcher is already installed (avoids race conditions when
# multiple workers share the same user site-packages). Set SKIP_DISPATCHER_INSTALL=1
# to use pre-installed dispatcher (e.g. from a one-time setup job).
SKIP_DISPATCHER_INSTALL="${SKIP_DISPATCHER_INSTALL:-1}"

TASK=${TASK:-tasks.task.Task}

MODEL=${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}
GPUS_PER_TASK=${GPUS_PER_TASK:-8}
LANGUAGE=${LANGUAGE:-}
PROMPT_TRANSLATION_MAX_TOKENS=${PROMPT_TRANSLATION_MAX_TOKENS:-}
TRACE_TRANSLATION_MAX_TOKENS=${TRACE_TRANSLATION_MAX_TOKENS:-}
ANSWER_TRANSLATION_MAX_TOKENS=${ANSWER_TRANSLATION_MAX_TOKENS:-}
GROUND_TRUTH_TRANSLATION_MAX_TOKENS=${GROUND_TRUTH_TRANSLATION_MAX_TOKENS:-}
HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-$(seq -s, 0 $((GPUS_PER_TASK - 1)))}
VLLM_EXTRA_ARGS=${VLLM_EXTRA_ARGS:-}

WORKERS=${WORKERS:-256}
BATCH_SIZE=${BATCH_SIZE:-1}

REQUEST_TIMEOUT=${REQUEST_TIMEOUT:-3600}
STARTUP_TIMEOUT=${STARTUP_TIMEOUT:-7200}

MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}

###############################################################################
# Resolve dispatcher server address
###############################################################################

if [ -z "${DISPATCHER_SERVER:-}" ] || [ -z "${DISPATCHER_PORT:-}" ]; then
  if [ -f "$ADDRESS_FILE" ]; then
    IFS=: read -r DISPATCHER_SERVER DISPATCHER_PORT < "$ADDRESS_FILE"
    echo "Read dispatcher address from $ADDRESS_FILE: $DISPATCHER_SERVER:$DISPATCHER_PORT"
  else
    echo "ERROR: No dispatcher address found."
    echo "Either set DISPATCHER_SERVER and DISPATCHER_PORT env vars,"
    echo "or run start_dispatcher_server.sh first."
    exit 1
  fi
fi

export DISPATCHER_SERVER
export DISPATCHER_PORT

###############################################################################
# Set up working directory and source launcher
###############################################################################

cd "$WORK_DIR"
source "$LAUNCHER"

###############################################################################
# Wait for dispatcher server to be reachable
###############################################################################

echo "Connecting to dispatcher server at $DISPATCHER_SERVER:$DISPATCHER_PORT..."
for i in $(seq 1 120); do
  curl -sf -o /dev/null --max-time 2 "http://$DISPATCHER_SERVER:$DISPATCHER_PORT/status" && break || true
  sleep 1
done
curl -sf -o /dev/null --max-time 2 "http://$DISPATCHER_SERVER:$DISPATCHER_PORT/status" || {
  echo "[ERROR] Dispatcher server not reachable at $DISPATCHER_SERVER:$DISPATCHER_PORT"
  exit 1
}
echo "Server is reachable."

###############################################################################
# Install dispatcher package in container (or skip if already installed)
# When multiple workers run in parallel, they share the same user site-packages.
# Concurrent pip install/uninstall causes race conditions (ModuleNotFoundError).
# Use flock to serialize installs, or set SKIP_DISPATCHER_INSTALL=1 to skip.
###############################################################################

if [ "$SKIP_DISPATCHER_INSTALL" = "1" ]; then
  echo "Skipping dispatcher install (SKIP_DISPATCHER_INSTALL=1). Adding $DISPATCHER_PKG to PYTHONPATH."
  DISPATCHER_PYTHONPATH_EXTRA="$DISPATCHER_PKG"
else
  INSTALL_LOCK="$WORK_DIR/.dispatcher_install.lock"
  echo "Installing dispatcher package from $DISPATCHER_PKG (lock: $INSTALL_LOCK)..."
  (
    flock -x -w 300 9 || { echo "[ERROR] Failed to acquire install lock"; exit 1; }
    run_sing_python -m pip install --user -e "$DISPATCHER_PKG"
  ) 9>"$INSTALL_LOCK"
  echo "Dispatcher install complete."
  DISPATCHER_PYTHONPATH_EXTRA=""
fi

###############################################################################
# Derive vLLM / MASTER ports from the physical GPU ID assigned by SLURM.
###############################################################################

GPU_ID="${SLURM_JOB_GPUS%%,*}"
if [ -z "$GPU_ID" ]; then
  echo "[ERROR] SLURM_JOB_GPUS is empty; cannot derive a deterministic port." >&2
  exit 1
fi

VLLM_PORT=$((20000 + GPU_ID * 100))
MASTER_PORT=$((19000 + GPU_ID * 100))

if [ -n "$(ss -Hltn "sport = :$VLLM_PORT" 2>/dev/null)" ]; then
  echo "[ERROR] Port $VLLM_PORT already in use on $(hostname); refusing to start." >&2
  exit 1
fi

###############################################################################
# Run worker in container
#
# Signal handling for preemption:
# Singularity tears down the container (including network) when it receives
# SIGTERM directly, which prevents the Python signal handler from POST-ing
# /release.  We use job control (set -m) to run the worker in its own process
# group so SLURM's SIGTERM only hits the outer bash.  The trap forwards SIGTERM
# to the worker process group, and the inner bash ignores it so only Python
# handles the signal while the container is still alive.
###############################################################################

_CHILD_PID=
trap '[ -n "$_CHILD_PID" ] && { kill -TERM -- -"$_CHILD_PID" 2>/dev/null; wait "$_CHILD_PID" 2>/dev/null; }' TERM INT

set -m
run_sing_bash "
  set -uo pipefail
  trap : TERM HUP
  DISPATCHER_PYTHONPATH_EXTRA=\"$DISPATCHER_PYTHONPATH_EXTRA\"
  if [ -n \"\$DISPATCHER_PYTHONPATH_EXTRA\" ]; then
    export PYTHONPATH=\"\$DISPATCHER_PYTHONPATH_EXTRA\":\${PYTHONPATH:-}
  fi

  export HIP_VISIBLE_DEVICES=\"$HIP_VISIBLE_DEVICES\"
  export LANGUAGE=\"$LANGUAGE\"
  export MODEL=\"$MODEL\"
  [ -n \"$PROMPT_TRANSLATION_MAX_TOKENS\" ] && export PROMPT_TRANSLATION_MAX_TOKENS=\"$PROMPT_TRANSLATION_MAX_TOKENS\"
  [ -n \"$TRACE_TRANSLATION_MAX_TOKENS\" ] && export TRACE_TRANSLATION_MAX_TOKENS=\"$TRACE_TRANSLATION_MAX_TOKENS\"
  [ -n \"$ANSWER_TRANSLATION_MAX_TOKENS\" ] && export ANSWER_TRANSLATION_MAX_TOKENS=\"$ANSWER_TRANSLATION_MAX_TOKENS\"
  [ -n \"$GROUND_TRUTH_TRANSLATION_MAX_TOKENS\" ] && export GROUND_TRUTH_TRANSLATION_MAX_TOKENS=\"$GROUND_TRUTH_TRANSLATION_MAX_TOKENS\"
  VLLM_EXTRA_ARGS=\"$VLLM_EXTRA_ARGS\"

  export MASTER_ADDR=\${MASTER_ADDR:-127.0.0.1}
  export MASTER_PORT=$MASTER_PORT
  export VLLM_PORT=$VLLM_PORT

  VLLM_ARGS=()
  if [ -n \"\$VLLM_EXTRA_ARGS\" ]; then
    VLLM_ARGS=(--vllm-extra-args \"\$VLLM_EXTRA_ARGS\")
  fi

  echo \"Launching worker on \$(hostname) with GPUs \$HIP_VISIBLE_DEVICES (job \${SLURM_JOB_ID:-unknown})\"

  run_python -m dispatcher.taskmanager.cli \
    --dispatcher $DISPATCHER_SERVER:$DISPATCHER_PORT \
    --task $TASK \
    --batch-size $BATCH_SIZE \
    --workers $WORKERS \
    --max-model-len $MAX_MODEL_LEN \
    --tensor-parallel $GPUS_PER_TASK \
    --model $MODEL \
    --port \$VLLM_PORT \
    --request-timeout $REQUEST_TIMEOUT \
    --startup-timeout $STARTUP_TIMEOUT \
    \"\${VLLM_ARGS[@]}\" " &

_CHILD_PID=$!
# Double-wait idiom: when a signal interrupts the first wait, it returns
# 128+signum (not the child's real status) and the trap fires.  The second
# wait then retrieves the child's actual exit status from bash's cache.
wait "$_CHILD_PID" 2>/dev/null
wait "$_CHILD_PID" 2>/dev/null
