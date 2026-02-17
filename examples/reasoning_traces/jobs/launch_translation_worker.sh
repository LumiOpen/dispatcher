#!/bin/bash
###############################################################################
# launch_translation_worker.sh - Single-node preemptable dispatcher worker
#
# Runs one dispatcher worker (vLLM backend on 8 GPUs) against the central
# dispatcher server launched by launch_dispatcher_server.sh.
#
# When preempted, any in-flight work items time out on the server side and are
# automatically reissued to another (or the same requeued) worker. No data is lost.
#
# Usage:
#   sbatch launch_translation_worker.sh              # submit one worker
#   ./submit_translation_workers.sh                  # submit 16 (see helper script)
###############################################################################

#SBATCH --job-name=trans-array
#SBATCH --nodes=1
#SBATCH --partition=amd-tw-verification
#SBATCH --time=10-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --cpus-per-task=7
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=8
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err

set -euo pipefail

# Resolve working directory (reasoning_traces/)
WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"
LAUNCHER="$WORK_DIR/singularity_launcher.sh"
ADDRESS_FILE="$WORK_DIR/.dispatcher_address"

# Dispatcher package source (mirrors launch_dispatcher_server.sh)
DISPATCHER_PKG="${DISPATCHER_PKG:-/shared_silo/scratch/adamhrin@amd.com/dispatcher}"

# Skip pip install when dispatcher is already installed (avoids race conditions when
# multiple workers share the same user site-packages). Set SKIP_DISPATCHER_INSTALL=1
# to use pre-installed dispatcher (e.g. from a one-time setup job).
SKIP_DISPATCHER_INSTALL="${SKIP_DISPATCHER_INSTALL:-1}"

###############################################################################
# Configuration (mirrors launch_translation_task_dsv3.sh)
###############################################################################

TASK=tasks.reasoning_translation_task.ReasoningTranslationTask
export LANGUAGE=fi

export MODEL=/shared_silo/scratch/models/DeepSeek-V3
GPUS_PER_TASK=8

# Generation parameters (override via: sbatch --export=ALL,WORKERS=64,... launch_translation_worker.sh)
WORKERS=${WORKERS:-128}
BATCH_SIZE=${BATCH_SIZE:-1}

# Timeouts
REQUEST_TIMEOUT=4800
STARTUP_TIMEOUT=7200

# vLLM parameters
MAX_MODEL_LEN=${MAX_MODEL_LEN:-65536}
MAX_SEQ_LEN_TO_CAPTURE=${MAX_SEQ_LEN_TO_CAPTURE:-65536}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-160}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-262144}

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
    echo "or run launch_dispatcher_server.sh first."
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
  (echo >/dev/tcp/$DISPATCHER_SERVER/${DISPATCHER_PORT}) >/dev/null 2>&1 && break || true
  sleep 1
done
(echo >/dev/tcp/$DISPATCHER_SERVER/${DISPATCHER_PORT}) >/dev/null 2>&1 || {
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
# Run worker in container (single node, all 8 GPUs)
###############################################################################

run_sing_bash "
  set -euo pipefail
  ${DISPATCHER_PYTHONPATH_EXTRA:+export PYTHONPATH=\"$DISPATCHER_PKG\":\${PYTHONPATH:-}}

  export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
  export MASTER_ADDR=\${MASTER_ADDR:-127.0.0.1}
  export MASTER_PORT=7000
  export VLLM_PORT=8000

  export LANGUAGE=$LANGUAGE
  export MODEL=$MODEL

  echo \"Launching worker on \$(hostname) with GPUs \$HIP_VISIBLE_DEVICES (job \${SLURM_JOB_ID:-unknown})\"

  # Based on https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-vllm-deepseek-r1-fp8.html
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
    --vllm-extra-args \"--swap-space 0 --max-num-seqs ${MAX_NUM_SEQS} --no-enable-prefix-caching --max-num-batched-tokens ${MAX_NUM_BATCHED_TOKENS} --block-size 1 --gpu-memory-utilization 0.95 --async-scheduling --quantization fp8\"
"
