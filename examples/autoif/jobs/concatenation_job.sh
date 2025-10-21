#!/bin/bash
#SBATCH --job-name=autoif_concat
#SBATCH --output=logs/%j_concatenation.out
#SBATCH --error=logs/%j_concatenation.err

# SLURM parameters passed via environment variables:
# - partition, time, nodes, ntasks_per_node, account
# Script parameters:
# - verifiers_file, queries_dataset, output_file, num_output_lines, instructions_per_query, language

#SBATCH --partition=${partition:-small}
#SBATCH --time=${time:-01:00:00}
#SBATCH --nodes=${nodes:-1}
#SBATCH --ntasks-per-node=${ntasks_per_node:-1}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --account=${account}

set -euo pipefail

echo "================================"
echo "AutoIF Query Concatenation Job"
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

# Check input file exists (verifiers_filtered.jsonl)
echo "Checking for input file: $verifiers_file"
if [ ! -f "$verifiers_file" ]; then
    echo "ERROR: Input file not found: $verifiers_file"
    echo "Please ensure the verification step completed successfully."
    exit 1
fi

if [ ! -s "$verifiers_file" ]; then
    echo "ERROR: Input file is empty: $verifiers_file"
    exit 1
fi
echo "Input file found and valid."
echo ""

# Run query concatenation
echo "Running query concatenation..."
echo "  Verifiers file: $verifiers_file"
echo "  Queries dataset: $queries_dataset"
echo "  Output file: $output_file"
echo "  Num output lines: $num_output_lines"
echo "  Instructions per query: $instructions_per_query"
echo "  Language: $language"
echo ""

# Build command with all parameters
CMD="python src/concat_queries.py \
    --verifiers-file \"$verifiers_file\" \
    --queries-dataset \"$queries_dataset\" \
    --output-file \"$output_file\" \
    --language \"$language\" \
    --num-output-lines \"$num_output_lines\" \
    --instructions-per-query \"$instructions_per_query\" \
    --query-max-len \"${query_max_len:-200}\" \
    --query-column-name \"${query_column_name:-instruction}\" \
    --response-column-name \"${response_column_name:-response}\" \
    --messages-key \"${messages_key:-messages}\" \
    --turns \"${turns:-1}\""

# Add boolean flags if set to true
if [ "${messages_format:-false}" = "true" ]; then
    CMD="$CMD --messages-format"
fi

if [ "${no_followup:-false}" = "true" ]; then
    CMD="$CMD --no-followup"
fi

if [ "${balance_categories:-false}" = "true" ]; then
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
echo "Output: $output_file"
echo "Finished: $(date)"
