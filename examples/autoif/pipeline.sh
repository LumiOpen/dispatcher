#!/bin/bash
#
# AutoIF Pipeline - Simple wrapper for pipeline.py
#
# Usage: ./pipeline.sh --out-dir <OUT_DIR> [--force|--resubmit-failed|--continue]
#

# Run pipeline
python3 pipeline.py "$@"
