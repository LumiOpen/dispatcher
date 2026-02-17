#!/bin/bash
# Submit an array of translation worker jobs to SLURM.
# Each array task runs launch_translation_worker.sh on one exclusive node.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch --qos=backlog --requeue -a 1-16%16 -N1 --exclusive -t 10-00:00:00 \
  "$SCRIPT_DIR/launch_translation_worker.sh"
