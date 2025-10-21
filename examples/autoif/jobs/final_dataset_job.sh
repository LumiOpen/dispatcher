#!/bin/bash
#SBATCH --job-name=autoif_sft
#SBATCH --output=logs/%j_sft.out
#SBATCH --error=logs/%j_sft.err

# SLURM parameters passed via environment variables:
# - partition, time, nodes, ntasks_per_node, account
# Script parameters:
# - input_file, output_dir, score_threshold

#SBATCH --partition=${partition:-small}
#SBATCH --time=${time:-00:30:00}
#SBATCH --nodes=${nodes:-1}
#SBATCH --ntasks-per-node=${ntasks_per_node:-1}
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --account=${account}

set -euo pipefail

echo "================================"
echo "AutoIF Final SFT Dataset Job"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

# Environment setup
module use /appl/local/csc/modulefiles
module load pytorch/2.5

# Activate virtual environment
VENV_DIR="${VENV_DIR:-.venv}"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

# Check input file exists (scored_responses.jsonl)
echo "Checking for input file: $input_file"
if [ ! -f "$input_file" ]; then
    echo "ERROR: Input file not found: $input_file"
    echo "Please ensure the response generation step completed successfully."
    exit 1
fi

if [ ! -s "$input_file" ]; then
    echo "ERROR: Input file is empty: $input_file"
    exit 1
fi
echo "Input file found and valid."
echo ""

# Create output directory
mkdir -p "$output_dir"

# Build SFT dataset
echo "Building SFT dataset..."
echo "  Input file: $input_file"
echo "  Output directory: $output_dir"
echo "  Score threshold: $score_threshold"
echo ""

python src/build_sft.py \
    --input-file "$input_file" \
    --output-dir "$output_dir" \
    --score-threshold "$score_threshold" \
    --test

if [ $? -ne 0 ]; then
    echo "ERROR: SFT dataset building failed"
    exit 1
fi

echo ""
echo "SFT dataset built successfully!"
echo "Output directory: $output_dir"
echo "Finished: $(date)"
