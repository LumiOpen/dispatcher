#!/bin/bash
#SBATCH --job-name=aug_post
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --account=project_462000963

set -euo pipefail

echo "======================================="
echo "AutoIF Augmentation Post-processing"
echo "======================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

# Task configuration
INPUT_FILE="${augmentation_output:-data/aug_output.jsonl}"
OUTPUT_FILE="${augmented_instructions:-data/augmented_instructions.jsonl}"
LANGUAGE="${language:-en}"
MAX_INSTRUCTIONS="${max_augmented_instructions:-200}"
SEED_FILE="${seed_file:-data/categorised_instructions.json}"

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
    exit 1
fi

echo "Post-processing configuration:"
echo "  Input file (raw dispatcher output): $INPUT_FILE"
echo "  Output file (final JSONL): $OUTPUT_FILE"
echo "  Language: $LANGUAGE"
echo "  Max instructions: $MAX_INSTRUCTIONS"
echo "  Seed file: $SEED_FILE"
echo ""

# Post-processing: Filter and deduplicate instructions
echo "Filtering and deduplicating instructions..."
python src/process_instructions_output.py \
    --input-file "$INPUT_FILE" \
    --output-file "$OUTPUT_FILE" \
    --language "$LANGUAGE" \
    --max-instructions "$MAX_INSTRUCTIONS" \
    --seed-file "$SEED_FILE"

if [ $? -ne 0 ]; then
    echo "ERROR: Post-processing failed"
    exit 1
fi

echo ""
echo "Post-processing completed successfully!"
echo "  Final JSONL output: $OUTPUT_FILE"
echo "Finished: $(date)"
