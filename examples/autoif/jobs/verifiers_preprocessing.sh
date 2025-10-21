#!/bin/bash
#SBATCH --job-name=autoif_ver_pre
#SBATCH --output=logs/%j_verifiers_preprocessing.out
#SBATCH --error=logs/%j_verifiers_preprocessing.err

# SLURM parameters passed via environment variables:
# - partition, time, account
# Script parameters:
# - augmented_instructions_file (JSONL from augmentation post-processing)
# - verifiers_input_file (output JSONL for verifiers task)

#SBATCH --partition=${partition:-small}
#SBATCH --time=${time:-00:10:00}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --account=${account}

set -euo pipefail

echo "======================================="
echo "AutoIF Verifiers Pre-processing"
echo "======================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

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
VENV_DIR="${VENV_DIR:-.venv}"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

export SSL_CERT_FILE=$(python -m certifi)

# Validate inputs
if [ -z "${augmented_instructions_file:-}" ]; then
    echo "ERROR: augmented_instructions_file not set"
    exit 1
fi

if [ ! -f "$augmented_instructions_file" ]; then
    echo "ERROR: Input file not found: $augmented_instructions_file"
    echo "Please ensure the augmentation post-processing step completed successfully."
    exit 1
fi

if [ -z "${verifiers_input_file:-}" ]; then
    echo "ERROR: verifiers_input_file not set"
    exit 1
fi

echo "Pre-processing configuration:"
echo "  Input file (augmented instructions JSONL): $augmented_instructions_file"
echo "  Output file (verifiers input JSONL): $verifiers_input_file"
echo ""

# Create verifier input with prompts
echo "Creating verifier generation input..."
python src/create_verifiers_input.py \
    --instructions_file "$augmented_instructions_file" \
    --output_file "$verifiers_input_file"

if [ $? -ne 0 ]; then
    echo "ERROR: Pre-processing failed"
    exit 1
fi

echo ""
echo "Pre-processing completed successfully!"
echo "  Output file: $verifiers_input_file"
echo "Finished: $(date)"
