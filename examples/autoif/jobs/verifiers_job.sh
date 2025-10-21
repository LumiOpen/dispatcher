#!/bin/bash
#SBATCH --job-name=autoif_ver
#SBATCH --output=logs/%j_verification.out
#SBATCH --error=logs/%j_verification.err

# SLURM parameters passed via environment variables:
# - partition, time, nodes, ntasks_per_node, account
# Script parameters:
# - input_file (augmented_instructions.jsonl), output_file (raw dispatcher output)
# - verifiers_all_file, verifiers_filtered_file (final outputs from cross-validation)
# - model, language, function_timeout

#SBATCH --partition=${partition:-dev-g}
#SBATCH --time=${time:-02:00:00}
#SBATCH --nodes=${nodes:-1}
#SBATCH --ntasks-per-node=${ntasks_per_node:-2}
#SBATCH --mem=480G
#SBATCH --cpus-per-task=7
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=${account}

set -euo pipefail

echo "======================================="
echo "AutoIF Verifier Generation Job"
echo "======================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

# Configuration
VERIFIERS_INPUT_FILE="${verifiers_input_file}"  # Input from verifiers preprocessing
VERIFIERS_OUTPUT_FILE="${output_file}"  # Raw dispatcher output
TASK="tasks.verification_task.GenerateVerifiersTask"
WORKERS="${VER_WORKERS:-8}"
BATCH_SIZE=1
STARTUP_TIMEOUT="${VER_STARTUP_TIMEOUT:-1800}"
REQUEST_TIMEOUT="${VER_REQUEST_TIMEOUT:-1800}"
WORK_TIMEOUT="${VER_WORK_TIMEOUT:-7200}"
GPUS_PER_TASK=4
MAX_MODEL_LEN=16384

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
VENV_DIR="${VENV_DIR:-.venv}"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

export HF_HOME="${HF_HOME:-/scratch/project_462000353/hf_cache}"
export SSL_CERT_FILE=$(python -m certifi)

# Check if input file exists
echo "Checking for input file: $VERIFIERS_INPUT_FILE"
if [ ! -f "$VERIFIERS_INPUT_FILE" ]; then
    echo "ERROR: Input file not found: $VERIFIERS_INPUT_FILE"
    echo "Please ensure the verifiers pre-processing step completed successfully."
    exit 1
fi
echo "Input file found."
echo ""

echo "Running verifier generation task..."
echo "  Model: $model"
echo "  Workers: $WORKERS"
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

    PYTHONPATH=. python -m dispatcher.taskmanager.cli \
        --dispatcher ${DISPATCHER_SERVER}:${DISPATCHER_PORT} \
        --task '"$TASK"' \
        --batch-size '"$BATCH_SIZE"' \
        --workers '"$WORKERS"' \
        --max-model-len '"$MAX_MODEL_LEN"' \
        --tensor-parallel '"$GPUS_PER_TASK"' \
        --model '"$model"' \
        --port $VLLM_PORT \
        --startup-timeout '"$STARTUP_TIMEOUT"' \
        --request-timeout '"$REQUEST_TIMEOUT"'
'

if [ $? -ne 0 ]; then
    echo "ERROR: Verifier generation task failed"
    exit 1
fi

echo ""
echo "Verifier generation task completed successfully!"
echo "  Raw dispatcher output: $VERIFIERS_OUTPUT_FILE"
echo ""
echo "Note: Post-processing should be run separately via verifiers_postprocessing.sh"
echo "Finished: $(date)"
