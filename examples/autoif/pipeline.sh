#!/bin/bash
#
# AutoIF Pipeline - Simple wrapper for pipeline.py
#
# Usage: ./pipeline.sh --out-dir <OUT_DIR> [--force|--resubmit-failed|--continue]
#

set -euo pipefail

# Load Python on Lumi
module use /appl/local/csc/modulefiles
module load pytorch/2.5

# Run pipeline
python pipeline.py "$@"
