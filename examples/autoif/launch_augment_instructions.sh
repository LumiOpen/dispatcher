#!/bin/bash
#SBATCH --job-name=autoif_aug_instr
#SBATCH --nodes=1
#SBATCH --partition=dev-g
#SBATCH --time=00-00:30:00
#SBATCH --ntasks-per-node=2
#SBATCH --mem=480G
#SBATCH --cpus-per-task=7
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j_augment.out
#SBATCH --error=logs/%j_augment.err

###
# Configuration
###

# Use environment variables if already set by parents script ../phase1_pipeline.sh, otherwise use defaults
: "${AUGMENT_INPUT_FILE:=data/aug_input.jsonl}"
: "${AUGMENT_OUTPUT_FILE:=data/aug_output.jsonl}"

# jq-like path string to find the prompt within the input jsonl row.
PROMPT_PATH=".prompt"

# Prompting mode is "chat" or "completion"
MODE=chat
STOP_WORD=$'\n\n'  # $'' format allows escape chars to be interpreted.

# Generation parameters
BATCH_SIZE=4
NUM_GENERATIONS=1

# Sampling parameters
MIN_P=0
TOP_P=0.9
TEMPERATURE=0.6

# Model configuration
# Use environment variable if already set by parent script phase1_pipeline.sh, otherwise use default
: "${MODEL:=meta-llama/Llama-3.3-70B-Instruct}"
GPUS_PER_TASK=4
MAX_MODEL_LEN=8192
MAX_TOKENS=4096

###
# Job execution
###

# Clean environment
unset VIRTUAL_ENV
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONSTARTUP
unset PYTHONNOUSERSITE
unset PYTHONEXECUTABLE

# Set up environment
mkdir -p logs pythonuserbase
export PYTHONUSERBASE=./pythonuserbase
module use /appl/local/csc/modulefiles
module load pytorch/2.5
pip install git+https://github.com/LumiOpen/dispatcher.git

export HF_HOME="/scratch/project_462000353/hf_cache"

echo "Starting instruction augmentation job at $(date)"

# dispatcher server will run on the first node, before we launch the worker
# tasks.
export DISPATCHER_SERVER=$(hostname)
export DISPATCHER_PORT=9999


python -m dispatcher.server \
    --infile $AUGMENT_INPUT_FILE \
    --outfile $AUGMENT_OUTPUT_FILE \
    --host 0.0.0.0 \
    --port ${DISPATCHER_PORT} &

sleep 10

srun -l \
    bash -c '
    # Compute the starting GPU index for this task.
    # SLURM_LOCALID is the index of the task on this node.
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

    # Set ports uniquely per task (to avoid collisions)
    export MASTER_PORT=$(( 7000 + SLURM_LOCALID ))
    export VLLM_PORT=$(( 8000 + SLURM_LOCALID * 100 ))

    echo "Launching task $SLURM_LOCALID (global id: $SLURM_PROCID) with GPU $GPU_IDS on $(hostname)"

    module use /appl/local/csc/modulefiles
    module load pytorch/2.5
    export PYTHONUSERBASE=./pythonuserbase
    python ../inference.py \
        --batch_size '"$BATCH_SIZE"' \
        --dispatcher_server ${DISPATCHER_SERVER}:${DISPATCHER_PORT} \
        --prompt_path "'"$PROMPT_PATH"'" \
        --mode '"$MODE"' \
        --stop_word "'"$STOP_WORD"'" \
        --num_generations '"$NUM_GENERATIONS"' \
        --max_model_len '"$MAX_MODEL_LEN"' \
        --max_tokens '"$MAX_TOKENS"' \
        --min_p '"$MIN_P"' \
        --top_p '"$TOP_P"' \
        --temperature '"$TEMPERATURE"' \
        --tensor_parallel_size '"$GPUS_PER_TASK"' \
        --model_path '"$MODEL"'
'
