#!/bin/bash
#SBATCH --job-name=cross_val
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --account=project_462000963

set -euo pipefail

echo "======================================="
echo "AutoIF Verifiers Post-processing"
echo "======================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

# Task configuration
INPUT_FILE="${verifiers_output:-data/verifiers_output.jsonl}"
OUTPUT_ALL_FILE="${verifiers_all:-data/verifiers_all.jsonl}"
OUTPUT_FILTERED_FILE="${verifiers_filtered:-data/verifiers_filtered.jsonl}"
LANGUAGE="${language:-en}"
FUNCTION_TIMEOUT="${function_timeout:-10}"
MIN_FUNCTIONS="${min_functions:-1}"
MIN_TEST_CASES="${min_test_cases:-1}"
FUNCTION_PASS_RATE="${function_pass_rate:-0.8}"

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
    echo "Please ensure the verifiers generation task completed successfully."
    exit 1
fi

echo "Post-processing configuration:"
echo "  Input file (raw dispatcher output): $INPUT_FILE"
echo "  Output file (all verifiers): $OUTPUT_ALL_FILE"
echo "  Output file (filtered verifiers): $OUTPUT_FILTERED_FILE"
echo "  Language: $LANGUAGE"
echo "  Function timeout: ${FUNCTION_TIMEOUT}s"
echo "  Min functions: $MIN_FUNCTIONS"
echo "  Min test cases: $MIN_TEST_CASES"
echo "  Function pass rate: $FUNCTION_PASS_RATE"
echo ""

# Export environment variables for cross-validation script
export FUNCTION_TIMEOUT="$FUNCTION_TIMEOUT"
export MIN_FUNCTIONS="$MIN_FUNCTIONS"
export MIN_TEST_CASES="$MIN_TEST_CASES"
export FUNCTION_PASS_RATE="$FUNCTION_PASS_RATE"
export LANGUAGE="$LANGUAGE"

# Cross-validate verification functions
echo "Cross-validating verification functions..."
python src/verifiers_cross_validation.py \
    --verifiers-file "$INPUT_FILE" \
    --output-all-file "$OUTPUT_ALL_FILE" \
    --output-filtered-file "$OUTPUT_FILTERED_FILE"

if [ $? -ne 0 ]; then
    echo "ERROR: Cross-validation failed"
    exit 1
fi

echo ""
echo "Post-processing completed successfully!"
echo "  All verifiers output: $OUTPUT_ALL_FILE"
echo "  Filtered verifiers output: $OUTPUT_FILTERED_FILE"
echo "Finished: $(date)"
