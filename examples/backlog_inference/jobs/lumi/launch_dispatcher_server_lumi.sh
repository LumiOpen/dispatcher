#!/bin/bash
###############################################################################
# launch_dispatcher_server_lumi.sh - LUMI version of launch_dispatcher_server.sh
#
# Runs the dispatcher server (a lightweight, CPU-only FastAPI process) as a
# batch job instead of the login-node tmux flow used by
# start_dispatcher_server.sh directly. Runs inside the LUMI AI Framework
# container (via lumi_singularity_launcher.sh) so the server sees the same
# Python environment as the workers.
#
# Usage:
#   sbatch jobs/lumi/launch_dispatcher_server_lumi.sh [configs/your-server-config.conf]
#
# Note: LUMI has no CPU partition with a multi-day time limit like TW's
# amd-tw-verification (10 days) -- `small` caps at 3 days. Long campaigns
# will need to resubmit this job; start_dispatcher_server.sh's own
# restart-on-exit loop and on-disk address/checkpoint files make that safe.
###############################################################################

#SBATCH --job-name=dispatcher-server
#SBATCH --account=project_462001516
#SBATCH --partition=small
#SBATCH --time=3-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/dispatcher-server-%j.out
#SBATCH --error=logs/dispatcher-server-%j.err

set -euo pipefail
loginctl enable-linger "$(id -un)" 2>/dev/null || true

CONFIG_FILE="${1:-configs/dispatcher-server-sft-pipeline-stage5.conf}"
WORK_DIR="${WORK_DIR:-$SLURM_SUBMIT_DIR}"
cd "$WORK_DIR"

LAUNCHER="${LAUNCHER:-$WORK_DIR/lumi_singularity_launcher.sh}"
LAUNCHER_PASSTHROUGH_VARS="${LAUNCHER_PASSTHROUGH_VARS:-} OUTPUT_FILE DISPATCHER_PORT SESSION_NAME"
source "$LAUNCHER"

echo "Starting dispatcher server (config: $CONFIG_FILE) on $(hostname)"
run_sing_bash "cd '$WORK_DIR' && exec ./start_dispatcher_server.sh '$CONFIG_FILE' --run-server"
