#!/bin/bash
set -euo pipefail

echo "======================================="
echo "AutoIF Response Generation Job (Local vLLM Mode)"
echo "======================================="
echo "Started: $(date)"
echo ""
echo "This job connects to an existing vLLM server"
echo "and runs a dispatcher server for failure management."
echo ""

# Task configuration
INPUT_FILE="${verifiers_queries:-data/verifiers_queries.jsonl}"
OUTPUT_FILE="${scored_responses:-data/scored_responses.jsonl}"
TASK="tasks.responses_task.GenerateQueryResponsesTask"
LANGUAGE="${language:-en}"
FUNCTION_TIMEOUT="${function_timeout:-10}"
SCORE_THRESHOLD="${score_threshold:-4}"

# Dispatcher
WORKERS="${workers:-32}"
BATCH_SIZE="${batch_size:-1}"
WORK_TIMEOUT="${work_timeout:-7200}"

# vLLM server connection
VLLM_HOST="${vllm_host:-127.0.0.1}"
VLLM_PORT="${vllm_port:-8000}"
MODEL="${model:-'meta-llama/Llama-3.3-70B-Instruct'}"
REQUEST_TIMEOUT="${request_timeout:-3600}"
MAX_MODEL_LEN="${max_model_len:-16384}"

# Check if input file exists
echo "Checking for input file: $INPUT_FILE"
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $INPUT_FILE"
    echo "Please ensure the concatenation step completed successfully."
    exit 1
fi
echo "Input file found."
echo ""

echo "Running response generation task..."
echo "  Model: $MODEL"
echo "  vLLM Server: $VLLM_HOST:$VLLM_PORT"
echo "  Input file: $INPUT_FILE"
echo "  Output file: $OUTPUT_FILE"
echo "  Language: $LANGUAGE"
echo "  Function timeout: $FUNCTION_TIMEOUT"
echo "  Workers: $WORKERS"
echo ""

# Export environment variables for the task
export LANGUAGE="$LANGUAGE"
export FUNCTION_TIMEOUT="$FUNCTION_TIMEOUT"
export SCORE_THRESHOLD="$SCORE_THRESHOLD"

# Start dispatcher server
export DISPATCHER_SERVER=$(hostname)
export DISPATCHER_PORT=9999

python -m dispatcher.server \
    --infile "$INPUT_FILE" \
    --outfile "$OUTPUT_FILE" \
    --work-timeout "$WORK_TIMEOUT" \
    --host 0.0.0.0 \
    --port ${DISPATCHER_PORT} &

sleep 10

# Run task worker (connects to existing vLLM server)
PYTHONPATH=. python -m dispatcher.taskmanager.cli \
    --dispatcher ${DISPATCHER_SERVER}:${DISPATCHER_PORT} \
    --task "$TASK" \
    --model "$MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_PORT" \
    --workers "$WORKERS" \
    --batch-size "$BATCH_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --request-timeout "$REQUEST_TIMEOUT" \
    --no-launch
