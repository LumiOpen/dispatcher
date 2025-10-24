#!/bin/bash
set -euo pipefail

echo "======================================="
echo "AutoIF Instruction Augmentation Job (Local vLLM Mode)"
echo "======================================="
echo "Started: $(date)"
echo ""
echo "This job connects to an existing vLLM server"
echo "instead of launching its own."
echo ""

# Task configuration
AUGMENT_INPUT_FILE="${augmentation_input:-data/aug_input.jsonl}"
AUGMENT_OUTPUT_FILE="${augmentation_output:-data/aug_output.jsonl}"
TASK="tasks.augmentation_task.AugmentInstructionsTask"
SEED_FILE="${seed_file:-data/categorised_instructions.json}"
NUM_INSTRUCTIONS="${num_instructions_per_category:-4}"
LANGUAGE="${language:-en}"

# vLLM server connection
VLLM_HOST="${vllm_host:-127.0.0.1}"
VLLM_PORT="${vllm_port:-8000}"
MODEL="${model:-'meta-llama/Llama-3.3-70B-Instruct'}"

# Task manager
WORKERS="${workers:-8}"
BATCH_SIZE="${batch_size:-1}"
REQUEST_TIMEOUT="${request_timeout:-1800}"
MAX_MODEL_LEN="${max_model_len:-16384}"

# Check if seed file exists
echo "Checking for seed file: $SEED_FILE"
if [ ! -f "$SEED_FILE" ]; then
    echo "ERROR: Seed file not found: $SEED_FILE"
    exit 1
fi
echo "Seed file found."
echo ""

# Pre-processing: Create input file with prompts
echo "Pre-processing: Creating augmentation input..."
python src/create_instructions_input.py \
    --seed-file "$SEED_FILE" \
    --output-file "$AUGMENT_INPUT_FILE" \
    --num-instructions-per-category "$NUM_INSTRUCTIONS" \
    --language "$LANGUAGE"

if [ $? -ne 0 ]; then
    echo "ERROR: Pre-processing failed"
    exit 1
fi
echo "Pre-processing complete. Input file: $AUGMENT_INPUT_FILE"
echo ""

echo "Running augmentation task in local file mode..."
echo "  Model: $MODEL"
echo "  vLLM Server: $VLLM_HOST:$VLLM_PORT"
echo "  Language: $LANGUAGE"
echo "  Input JSONL: $AUGMENT_INPUT_FILE"
echo "  Output JSONL: $AUGMENT_OUTPUT_FILE"
echo "  Workers: $WORKERS"
echo ""

# Run task in local file mode (connects to existing vLLM server)
PYTHONPATH=. python -m dispatcher.taskmanager.cli \
    --task "$TASK" \
    --input "$AUGMENT_INPUT_FILE" \
    --output "$AUGMENT_OUTPUT_FILE" \
    --model "$MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --workers "$WORKERS" \
    --batch-size "$BATCH_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --no-launch
