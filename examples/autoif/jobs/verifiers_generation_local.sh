#!/bin/bash
set -euo pipefail

echo "======================================="
echo "AutoIF Verifiers Generation Job (Local vLLM Mode)"
echo "======================================="
echo "Started: $(date)"
echo ""
echo "This job connects to an existing vLLM server"
echo "instead of launching its own."
echo ""

# Task configuration
AUGMENTED_INSTRUCTIONS="${augmented_instructions:-data/augmented_instructions.jsonl}"
VERIFIERS_INPUT="${verifiers_input:-data/verifiers_input.jsonl}"
VERIFIERS_OUTPUT="${verifiers_output:-data/verifiers_output.jsonl}"
TASK="tasks.verifiers_task.GenerateVerifiersTask"
LANGUAGE="${language:-en}"
MIN_FUNCTIONS="${min_functions:-1}"
MIN_TEST_CASES="${min_test_cases:-1}"
NUM_VERIFIER_GENERATIONS="${num_verifier_generations:-10}"

# vLLM server connection
VLLM_HOST="${vllm_host:-127.0.0.1}"
VLLM_PORT="${vllm_port:-8000}"
MODEL="${model:-'meta-llama/Llama-3.3-70B-Instruct'}"

# Task manager
WORKERS="${workers:-8}"
BATCH_SIZE="${batch_size:-1}"
REQUEST_TIMEOUT="${request_timeout:-1800}"
MAX_MODEL_LEN="${max_model_len:-16384}"

# Check if input file exists
echo "Checking for augmented instructions file: $AUGMENTED_INSTRUCTIONS"
if [ ! -f "$AUGMENTED_INSTRUCTIONS" ]; then
    echo "ERROR: Augmented instructions file not found: $AUGMENTED_INSTRUCTIONS"
    echo "Please ensure the augmentation step completed successfully."
    exit 1
fi
echo "Augmented instructions file found."
echo ""

# Pre-processing: Create verifiers input
echo "Pre-processing: Creating verifiers input..."
python src/create_verifiers_input.py \
    --augmented-instructions "$AUGMENTED_INSTRUCTIONS" \
    --output-file "$VERIFIERS_INPUT" \
    --num-verifier-generations "$NUM_VERIFIER_GENERATIONS"

if [ $? -ne 0 ]; then
    echo "ERROR: Pre-processing failed"
    exit 1
fi
echo "Pre-processing complete. Input file: $VERIFIERS_INPUT"
echo ""

echo "Running verifiers generation task in local file mode..."
echo "  Model: $MODEL"
echo "  vLLM Server: $VLLM_HOST:$VLLM_PORT"
echo "  Language: $LANGUAGE"
echo "  Input JSONL: $VERIFIERS_INPUT"
echo "  Output JSONL: $VERIFIERS_OUTPUT"
echo "  Workers: $WORKERS"
echo "  Min functions: $MIN_FUNCTIONS"
echo "  Min test cases: $MIN_TEST_CASES"
echo ""

# Export environment variables for the task
export LANGUAGE="$LANGUAGE"
export MIN_FUNCTIONS="$MIN_FUNCTIONS"
export MIN_TEST_CASES="$MIN_TEST_CASES"

# Run task in local file mode (connects to existing vLLM server)
PYTHONPATH=. python -m dispatcher.taskmanager.cli \
    --task "$TASK" \
    --input "$VERIFIERS_INPUT" \
    --output "$VERIFIERS_OUTPUT" \
    --model "$MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --workers "$WORKERS" \
    --batch-size "$BATCH_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --no-launch
