#!/bin/bash
#SBATCH --job-name=autoif_ver_post
#SBATCH --output=logs/%j_verifiers_postprocessing.out
#SBATCH --error=logs/%j_verifiers_postprocessing.err

# SLURM parameters passed via environment variables:
# - partition, time, account
# Script parameters:
# - output_file (raw dispatcher output from verifiers job)
# - verifiers_all_file (all verifiers output JSONL)
# - verifiers_filtered_file (filtered verifiers output JSONL)
# - language, function_timeout

#SBATCH --partition=${partition:-small}
#SBATCH --time=${time:-01:00:00}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --account=${account}

set -euo pipefail

echo "======================================="
echo "AutoIF Verifiers Post-processing"
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
if [ -z "${output_file:-}" ]; then
    echo "ERROR: output_file not set"
    exit 1
fi

if [ ! -f "$output_file" ]; then
    echo "ERROR: Input file not found: $output_file"
    echo "Please ensure the verifiers generation task completed successfully."
    exit 1
fi

if [ -z "${verifiers_all_file:-}" ]; then
    echo "ERROR: verifiers_all_file not set"
    exit 1
fi

if [ -z "${verifiers_filtered_file:-}" ]; then
    echo "ERROR: verifiers_filtered_file not set"
    exit 1
fi

if [ -z "${language:-}" ]; then
    echo "ERROR: language not set"
    exit 1
fi

if [ -z "${function_timeout:-}" ]; then
    echo "ERROR: function_timeout not set"
    exit 1
fi

echo "Post-processing configuration:"
echo "  Input file (raw dispatcher output): $output_file"
echo "  Output file (all verifiers): $verifiers_all_file"
echo "  Output file (filtered verifiers): $verifiers_filtered_file"
echo "  Language: $language"
echo "  Function timeout: ${function_timeout}s"
echo ""

# Export environment variables for cross-validation
export FUNCTION_TIMEOUT="$function_timeout"
export MIN_FUNCTIONS="${MIN_FUNCTIONS:-1}"
export MIN_TEST_CASES="${MIN_TEST_CASES:-1}"
export FUNCTION_PASS_RATE="${FUNCTION_PASS_RATE:-0.8}"
export LANGUAGE="$language"

echo "Cross-validation parameters:"
echo "  Min functions: $MIN_FUNCTIONS"
echo "  Min test cases: $MIN_TEST_CASES"
echo "  Function pass rate: $FUNCTION_PASS_RATE"
echo ""

# Cross-validate verification functions
echo "Cross-validating verification functions..."
python src/verifiers_cross_validation.py \
    --verifiers-file "$output_file" \
    --output-all-file "$verifiers_all_file" \
    --output-filtered-file "$verifiers_filtered_file"

if [ $? -ne 0 ]; then
    echo "ERROR: Cross-validation failed"
    exit 1
fi

echo ""
echo "Post-processing completed successfully!"
echo "  All verifiers output: $verifiers_all_file"
echo "  Filtered verifiers output: $verifiers_filtered_file"
echo "Finished: $(date)"
