#!/bin/bash
#SBATCH --job-name=disp_traces
#SBATCH --nodes=1
#SBATCH --partition=dev-g
#SBATCH --time=1:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --mem=480G
#SBATCH --cpus-per-task=7
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000963
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


###
# configure the following.
export LANGUAGE="${language:-fi}"
export MODEL="${model:-Qwen/Qwen2.5-72B-Instruct}"

INPUT_FILE="${input_file:-/scratch/project_462000353/adamhrin/dispatcher/examples/translation/data/default-train-sample-100_translations_DeepSeek-V3_fi.jsonl}"
DATADIR=$(dirname "$INPUT_FILE")
FILE_NAME=$(basename "$INPUT_FILE" .jsonl)
MODEL_NAME=$(basename "$MODEL")
OUTPUT_FILE=${DATADIR}/${FILE_NAME}_traces_${MODEL_NAME}_${LANGUAGE}.jsonl
TASK=${TASK:-tasks.traces_task.TracesTask}

echo "Using model: $MODEL"
echo "Model base name: $MODEL_NAME"
echo "Language: $LANGUAGE"
echo "Input file: $INPUT_FILE"
echo "Output file: $OUTPUT_FILE"

# generation parameters
# These should be tuned so that you do not overload your backend vllm server,
# or run into any timeouts.  timeouts greatly affect the efficiency of the
# workflow.
WORKERS=${workers:-16}          # number of simultaneous backend requests
BATCH_SIZE=${batch_size:-1}     # amount of work to request from dispatcher. 1 is usually fine.

# Timeouts are safety valves and you should not hit them in the normal course
# of your workflow.  if you do, it suggests you need to change something about
# your configuration--tasks are usually written to expect success.
REQUEST_TIMEOUT=${request_timeout:-3600} # adjust as needed for your task so that you do not hit
WORK_TIMEOUT=${work_timeout:-7200}   # time for dispatcher to give up on a work item and reissue it.  ideally this should never be hit.

#
# If you are changing the model, be sure to update GPUS_PER_TASK and the
# sbatch --ntasks-per-node configuration appropriately.
# Typically on Lumi 70B = 4 GPUs, 34B = 2 GPUs, 8B = 1 GPU
# --ntasks-per-node should be int(8 / GPUS_PER_TASK)
#

GPUS_PER_TASK=${gpus_per_task:-4}    # enough for the model and large batch size
MAX_MODEL_LEN=${max_model_len:-16384} # for efficiency, only as much as you think you need for efficiency

# Optional vllm config if running externally
VLLM_HOST="${vllm_host:-}"
VLLM_PORT="${vllm_port:-}"

if [[ -n "$VLLM_HOST" ]]; then
    export DISPATCHER_VLLM=1
    export DISPATCHER_VLLM_HOST="$VLLM_HOST"
    export DISPATCHER_VLLM_PORT="${VLLM_PORT:-8000}"
else
    unset DISPATCHER_VLLM
    unset DISPATCHER_VLLM_HOST
    unset DISPATCHER_VLLM_PORT
fi

# end configuration
###################

# clean up any venv that might be inherited from the launch environment.
unset VIRTUAL_ENV
unset PYTHONHOME
unset PYTHONPATH
unset PYTHONSTARTUP
unset PYTHONNOUSERSITE
unset PYTHONEXECUTABLE

# set up environment
mkdir -p logs pythonuserbase
export PYTHONUSERBASE="./pythonuserbase" #"$(pwd)/pythonuserbase"
module use /appl/local/csc/modulefiles
module load pytorch/2.5
export HF_HOME="/scratch/project_462000353/hf_cache"
export SSL_CERT_FILE=$(python -m certifi)

pip install --user git+https://github.com/LumiOpen/dispatcher.git

# dispatcher server will run on the first node, before we launch the worker
# tasks.
export DISPATCHER_SERVER=$(hostname)
export DISPATCHER_PORT=9999
python -m dispatcher.server \
    --infile $INPUT_FILE \
    --outfile $OUTPUT_FILE \
    --work-timeout $WORK_TIMEOUT \
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

    if [[ -n "$DISPATCHER_VLLM" ]]; then
        echo "Using external vLLM server $DISPATCHER_VLLM_HOST:$DISPATCHER_VLLM_PORT"
    else
        TASK_VLLM_PORT=$(( 8000 + SLURM_LOCALID * 100 ))
        echo "Starting per-task vLLM server on port $TASK_VLLM_PORT"
    fi

    echo "Launching task $SLURM_LOCALID (global id: $SLURM_PROCID) with GPU $GPU_IDS on $(hostname)"

    module use /appl/local/csc/modulefiles
    module load pytorch/2.5
    export PYTHONUSERBASE=./pythonuserbase
    export HF_HOME="/scratch/project_462000353/hf_cache"

    CLI_ARGS=(
        --dispatcher "${DISPATCHER_SERVER}:${DISPATCHER_PORT}"
        --task "$TASK"
        --batch-size "$BATCH_SIZE"
        --workers "$WORKERS"
        --max-model-len "$MAX_MODEL_LEN"
        --tensor-parallel "$GPUS_PER_TASK"
        --model "$MODEL"
        --request-timeout "$REQUEST_TIMEOUT"
        # --silence-vllm-logs
    )

    if [[ -n "$DISPATCHER_VLLM" ]]; then
        CLI_ARGS+=(--host "$DISPATCHER_VLLM_HOST" --port "$DISPATCHER_VLLM_PORT" --no-launch)
    else
        CLI_ARGS+=(--port "$TASK_VLLM_PORT")
    fi

    PYTHONPATH=. python -m dispatcher.taskmanager.cli "${CLI_ARGS[@]}"
'

