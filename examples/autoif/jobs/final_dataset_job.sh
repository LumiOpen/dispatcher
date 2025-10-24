#!/bin/bash
#SBATCH --job-name=sft
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --account=project_462000963

set -euo pipefail

echo "================================"
echo "AutoIF Final SFT Dataset Job"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

# Task configuration
INPUT_FILE="${scored_responses:-data/scored_responses.jsonl}"
OUTPUT_DIR="${sft_dataset_dir:-data/sft_dataset}"
SCORE_THRESHOLD="${score_threshold:-4}"

# Clean environment
unset VIRTUAL_ENV
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONSTARTUP
unset PYTHONNOUSERSITE
unset PYTHONEXECUTABLE

# Set up environment
mkdir -p logs pythonuserbase
export PYTHONUSERBASE="$(pwd)/pythonuserbase"

module use /appl/local/csc/modulefiles
module load pytorch/2.5

# Activate virtual environment
VENV_DIR="${venv_dir:-.venv}"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

export HF_HOME="${hf_home:-/scratch/project_462000353/hf_cache}"
export SSL_CERT_FILE=$(python -m certifi)

# Check input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Please ensure the response generation step completed successfully."
    exit 1
fi

if [ ! -s "$INPUT_FILE" ]; then
    echo "ERROR: Input file is empty: $INPUT_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Building SFT dataset..."
echo "  Input file: $INPUT_FILE"
echo "  Output directory: $OUTPUT_DIR"
echo "  Score threshold: $SCORE_THRESHOLD"
echo ""

python src/build_sft.py \
    --input-file "$INPUT_FILE" \
    --output-dir "$OUTPUT_DIR" \
    --score-threshold "$SCORE_THRESHOLD" \
    --test

if [ $? -ne 0 ]; then
    echo "ERROR: SFT dataset building failed"
    exit 1
fi

echo ""
echo "SFT dataset built successfully!"
echo "Output directory: $OUTPUT_DIR"
echo "Finished: $(date)"
