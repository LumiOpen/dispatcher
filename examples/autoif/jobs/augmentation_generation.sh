#!/bin/bash
#SBATCH --job-name=aug_gen
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --partition=dev-g
#SBATCH --time=00:30:00
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
echo "AutoIF Instruction Augmentation Job"
echo "======================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Started: $(date)"
echo ""

# Task configuration
AUGMENT_INPUT_FILE="${augmentation_input:-data/aug_input.jsonl}"
AUGMENT_OUTPUT_FILE="${augmentation_output:-data/aug_output.jsonl}"
TASK="tasks.augmentation_task.AugmentInstructionsTask"
SEED_FILE="${seed_file:-data/categorised_instructions.json}"
NUM_INSTRUCTIONS="${num_instructions_per_category:-4}"
LANGUAGE="${language:-en}"

# Dispatcher
WORKERS="${workers:-8}"
BATCH_SIZE="${batch_size:-1}"
WORK_TIMEOUT="${work_timeout:-3600}"

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
VENV_DIR="${VENV_DIR:-.venv}"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: Virtual environment not found at $VENV_DIR"
    exit 1
fi

export HF_HOME="${HF_HOME:-/scratch/project_462000353/hf_cache}"
export SSL_CERT_FILE=$(python -m certifi)

# Check if seed file exists
echo "Checking for seed file: $SEED_FILE"
if [ ! -f "$SEED_FILE" ]; then
    echo "ERROR: Seed file not found: $SEED_FILE"
    exit 1
fi
echo "Seed file found."
echo ""

# Pre-processing: Create input file with prompts (done outside the task)
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

echo "Running augmentation task..."
echo "  Model: $MODEL"
echo "  Language: $LANGUAGE"
echo "  Input JSONL: $AUGMENT_INPUT_FILE"
echo "  Output JSONL: $AUGMENT_OUTPUT_FILE"
echo ""

# Start dispatcher server
export DISPATCHER_SERVER=$(hostname)
export DISPATCHER_PORT=9999

python -m dispatcher.server \
    --infile "$AUGMENT_INPUT_FILE" \
    --outfile "$AUGMENT_OUTPUT_FILE" \
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

