#!/bin/bash
#SBATCH --job-name=autoif_aug_post
#SBATCH --output=logs/%j_augmentation_postprocessing.out
#SBATCH --error=logs/%j_augmentation_postprocessing.err

# SLURM parameters passed via environment variables:
# - partition, time, account
# Script parameters:
# - output_file (raw dispatcher output), augmented_instructions_file (final JSONL)
# - language, max_augmented_instructions, seed_file

#SBATCH --partition=${partition:-small}
#SBATCH --time=${time:-00:15:00}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --account=${account}

set -euo pipefail

echo "======================================="
echo "AutoIF Augmentation Post-processing"
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
    exit 1
fi

if [ -z "${augmented_instructions_file:-}" ]; then
    echo "ERROR: augmented_instructions_file not set"
    exit 1
fi

if [ -z "${language:-}" ]; then
    echo "ERROR: language not set"
    exit 1
fi

if [ -z "${max_augmented_instructions:-}" ]; then
    echo "ERROR: max_augmented_instructions not set"
    exit 1
fi

if [ -z "${seed_file:-}" ]; then
    echo "ERROR: seed_file not set"
    exit 1
fi

echo "Post-processing configuration:"
echo "  Input file (raw dispatcher output): $output_file"
echo "  Output file (final JSONL): $augmented_instructions_file"
echo "  Language: $language"
echo "  Max instructions: $max_augmented_instructions"
echo "  Seed file: $seed_file"
echo ""

# Post-processing: Filter and deduplicate instructions
echo "Filtering and deduplicating instructions..."
python src/process_instructions_output.py \
    --input-file "$output_file" \
    --output-file "$augmented_instructions_file" \
    --language "$language" \
    --max-instructions "$max_augmented_instructions" \
    --seed-file "$seed_file"

if [ $? -ne 0 ]; then
    echo "ERROR: Post-processing failed"
    exit 1
fi

echo ""
echo "Post-processing completed successfully!"
echo "  Final JSONL output: $augmented_instructions_file"
echo "Finished: $(date)"
