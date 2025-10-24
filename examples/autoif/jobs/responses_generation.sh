#!/bin/bash
#SBATCH --job-name=resp_gen
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=standard-g
#SBATCH --time=18:00:00
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --mem=480G
#SBATCH --cpus-per-task=7
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000963

set -euo pipefail

echo "======================================="
echo "AutoIF Response Generation Job"
echo "======================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
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

# vLLM
MODEL="${model:-'meta-llama/Llama-3.3-70B-Instruct'}"
STARTUP_TIMEOUT="${startup_timeout:-7200}"
REQUEST_TIMEOUT="${request_timeout:-3600}"
GPUS_PER_TASK="${tensor_parallel:-4}"
MAX_MODEL_LEN="${max_model_len:-16384}"

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

# Activate virtual environment for task dependencies
VENV_DIR="${venv_dir:-.venv}"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

export HF_HOME="${hf_home:-/scratch/project_462000353/hf_cache}"
export SSL_CERT_FILE=$(python -m certifi)

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
echo "  Input file: $INPUT_FILE"
echo "  Output file: $OUTPUT_FILE"
echo "  Language: $LANGUAGE"
echo "  Function timeout: $FUNCTION_TIMEOUT"
echo "  Workers: $WORKERS"
echo ""

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

# Run task workers
srun -l \
    bash -c '
    # Compute the starting GPU index for this task
    start_gpu=$(( SLURM_LOCALID * '"$GPUS_PER_TASK"' ))
    GPU_IDS=""
    for (( i=0; i < '"$GPUS_PER_TASK"'; i++ )); do
        if [ -z "$GPU_IDS" ]; then
            GPU_IDS="$(( start_gpu + i ))"
        else
            GPU_IDS="${GPU_IDS},$(( start_gpu + i ))"
        fi
    done
    export CUDA_VISIBLE_DEVICES=$GPU_IDS

    # Set ports uniquely per task
    export MASTER_PORT=$(( 7000 + SLURM_LOCALID ))
    export VLLM_PORT=$(( 8000 + SLURM_LOCALID * 100 ))

    echo "Launching task $SLURM_LOCALID (global id: $SLURM_PROCID) with GPU $GPU_IDS on $(hostname)"

    module use /appl/local/csc/modulefiles
    module load pytorch/2.5
    export PYTHONUSERBASE=./pythonuserbase
    export HF_HOME="${HF_HOME:-/scratch/project_462000353/hf_cache}"
    export LANGUAGE="'"$LANGUAGE"'"
    export FUNCTION_TIMEOUT="'"$FUNCTION_TIMEOUT"'"
    export SCORE_THRESHOLD="'"$SCORE_THRESHOLD"'"

    PYTHONPATH=. python -m dispatcher.taskmanager.cli \
        --dispatcher ${DISPATCHER_SERVER}:${DISPATCHER_PORT} \
        --task '"$TASK"' \
        --batch-size '"$BATCH_SIZE"' \
        --workers '"$WORKERS"' \
        --max-model-len '"$MAX_MODEL_LEN"' \
        --tensor-parallel '"$GPUS_PER_TASK"' \
        --model '"$MODEL"' \
        --port $VLLM_PORT \
        --startup-timeout '"$STARTUP_TIMEOUT"' \
        --request-timeout '"$REQUEST_TIMEOUT"'
'

