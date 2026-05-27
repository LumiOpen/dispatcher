#!/bin/bash
#SBATCH --job-name=dispatcher-server
#SBATCH --partition=amd-tw-verification
#SBATCH --gpus=0
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=10-00:00:00
#SBATCH --output=logs/dispatcher-server-%j.out
#SBATCH --error=logs/dispatcher-server-%j.err

set -euo pipefail
loginctl enable-linger "$(id -un)" 2>/dev/null || true
CONFIG_FILE="${1:-configs/dispatcher-server-sft-pipeline-stage5.conf}"
cd "$SLURM_SUBMIT_DIR"
exec ./start_dispatcher_server.sh "$CONFIG_FILE" --run-server
