#!/bin/bash
#SBATCH --job-name=ver_gen
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=dev-g
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --mem=480G
#SBATCH --cpus-per-task=7
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000963

set -euo pipefail

echo "======================================="
echo "AutoIF Verifier Generation Job"
echo "======================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

# Task configuration
AUGMENTED_INSTRUCTIONS_FILE="${augmented_instructions:-data/augmented_instructions.jsonl}"
VERIFIERS_INPUT_FILE="${verifiers_input:-data/verifiers_input.jsonl}"
VERIFIERS_OUTPUT_FILE="${verifiers_output:-data/verifiers_output.jsonl}"
TASK="tasks.verifiers_task.GenerateVerifiersTask"
NUM_GENERATIONS="${num_verifier_generations:-3}"

# Dispatcher
WORKERS="${workers:-8}"
BATCH_SIZE="${batch_size:-1}"
WORK_TIMEOUT="${work_timeout:-7200}"

# vLLM
MODEL="${model:-'meta-llama/Llama-3.3-70B-Instruct'}"
STARTUP_TIMEOUT="${startup_timeout:-1800}"
REQUEST_TIMEOUT="${request_timeout:-1800}"
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

# Check if augmented instructions file exists
echo "Checking for augmented instructions file: $AUGMENTED_INSTRUCTIONS_FILE"
if [ ! -f "$AUGMENTED_INSTRUCTIONS_FILE" ]; then
    echo "ERROR: Augmented instructions file not found: $AUGMENTED_INSTRUCTIONS_FILE"
    exit 1
fi
echo "Augmented instructions file found."
echo ""

# Pre-processing: Create verifiers input file
echo "Pre-processing: Creating verifiers input..."
python src/create_verifiers_input.py \
    --instructions_file "$AUGMENTED_INSTRUCTIONS_FILE" \
    --output_file "$VERIFIERS_INPUT_FILE"

if [ $? -ne 0 ]; then
    echo "ERROR: Pre-processing failed"
    exit 1
fi
echo "Pre-processing complete. Input file: $VERIFIERS_INPUT_FILE"
echo ""

echo "Running verifier generation task..."
echo "  Model: $MODEL"
echo "  Input JSONL: $VERIFIERS_INPUT_FILE"
echo "  Output JSONL: $VERIFIERS_OUTPUT_FILE"
echo ""

# Start dispatcher server
export DISPATCHER_SERVER=$(hostname)
export DISPATCHER_PORT=9999

python -m dispatcher.server \
    --infile "$VERIFIERS_INPUT_FILE" \
    --outfile "$VERIFIERS_OUTPUT_FILE" \
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

    echo "Launching task $SLURM_LOCALID (global id: $SLURM_PROCID) with GPUs $GPU_IDS on $(hostname)"

    module use /appl/local/csc/modulefiles
    module load pytorch/2.5
    export PYTHONUSERBASE=./pythonuserbase
    export HF_HOME="${HF_HOME:-/scratch/project_462000353/hf_cache}"
    export NUM_GENERATIONS="${NUM_GENERATIONS:-3}"

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

