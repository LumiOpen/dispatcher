#!/bin/bash
#SBATCH --job-name=concat
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=small
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=project_462000963

set -euo pipefail

echo "================================"
echo "AutoIF Query Concatenation Job"
echo "================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

# Task configuration
VERIFIERS_FILE="${verifiers_filtered:-data/verifiers_filtered.jsonl}"
QUERIES_DATASET="${queries_dataset}"  # Required, no default
OUTPUT_FILE="${verifiers_queries:-data/verifiers_queries.jsonl}"
LANGUAGE="${language:-en}"
NUM_OUTPUT_LINES="${num_output_lines:-300000}"
INSTRUCTIONS_PER_QUERY="${instructions_per_query:-1}"
QUERY_MAX_LEN="${query_max_len:-200}"
QUERY_COLUMN_NAME="${query_column_name:-instruction}"
RESPONSE_COLUMN_NAME="${response_column_name:-response}"
MESSAGES_FORMAT="${messages_format:-true}"
MESSAGES_KEY="${messages_key:-messages}"
TURNS="${turns:-1}"
NO_FOLLOWUP="${no_followup:-true}"
BALANCE_CATEGORIES="${balance_categories:-true}"

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
if [ ! -f "$VERIFIERS_FILE" ]; then
    echo "ERROR: Input file not found: $VERIFIERS_FILE"
    echo "Please ensure the verification step completed successfully."
    exit 1
fi

if [ ! -s "$VERIFIERS_FILE" ]; then
    echo "ERROR: Input file is empty: $VERIFIERS_FILE"
    exit 1
fi

# Check queries dataset is set
if [ -z "$QUERIES_DATASET" ]; then
    echo "ERROR: queries_dataset not set in config"
    exit 1
fi

echo "Query concatenation configuration:"
echo "  Verifiers file: $VERIFIERS_FILE"
echo "  Queries dataset: $QUERIES_DATASET"
echo "  Output file: $OUTPUT_FILE"
echo "  Num output lines: $NUM_OUTPUT_LINES"
echo "  Instructions per query: $INSTRUCTIONS_PER_QUERY"
echo "  Language: $LANGUAGE"
echo ""

# Build command with all parameters
CMD="python src/concat_queries.py \
    --verifiers-file \"$VERIFIERS_FILE\" \
    --queries-dataset \"$QUERIES_DATASET\" \
    --output-file \"$OUTPUT_FILE\" \
    --language \"$LANGUAGE\" \
    --num-output-lines \"$NUM_OUTPUT_LINES\" \
    --instructions-per-query \"$INSTRUCTIONS_PER_QUERY\" \
    --query-max-len \"$QUERY_MAX_LEN\" \
    --query-column-name \"$QUERY_COLUMN_NAME\" \
    --response-column-name \"$RESPONSE_COLUMN_NAME\" \
    --messages-key \"$MESSAGES_KEY\" \
    --turns \"$TURNS\""

# Add boolean flags if set to true
if [ "$MESSAGES_FORMAT" = "true" ]; then
    CMD="$CMD --messages-format"
fi

if [ "$NO_FOLLOWUP" = "true" ]; then
    CMD="$CMD --no-followup"
fi

if [ "$BALANCE_CATEGORIES" = "true" ]; then
    CMD="$CMD --balance-categories"
fi

# Execute the command
eval $CMD

if [ $? -ne 0 ]; then
    echo "ERROR: Query concatenation failed"
    exit 1
fi

echo ""
echo "Query concatenation completed successfully!"
echo "Output: $OUTPUT_FILE"
echo "Finished: $(date)"
