#!/bin/bash
#SBATCH --job-name=resp
#SBATCH --nodes=4
#SBATCH --partition=standard-g
#SBATCH --time=16:00:00
#SBATCH --ntasks-per-node=2
#SBATCH --mem=480G
#SBATCH --cpus-per-task=7
#SBATCH --exclusive
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000963
#SBATCH --output=logs/%j_responses.out
#SBATCH --error=logs/%j_responses.err


###
# configure the following.
export LANGUAGE=${LANGUAGE:-eng}
export FUNCTION_TIMEOUT=${FUNCTION_TIMEOUT:-5}
INPUT_FILE=${VERIFIERS_QUERIES_FILE:-data/verifiers_queries.jsonl}
OUTPUT_FILE=${SCORED_RESPONSES_FILE:-data/scored_responses.jsonl}
TASK=autoif_generator_task.GenerateQueryResponsesTask

# generation parameters
# These should be tuned so that you do not overload your backend vllm server,
# or run into any timeouts.  timeouts greatly affect the efficiency of the
# workflow.
WORKERS=${RESP_WORKERS:-32}          # number of simultaneous backend requests
BATCH_SIZE=1        # amount of work to request from dispatcher. 1 is usually fine.

# Timeouts are safety valves and you should not hit them in the normal course
# of your workflow.  if you do, it suggests you need to change something about
# your configuration--tasks are usually written to expect success.
STARTUP_TIMEOUT=${RESP_STARTUP_TIMEOUT:-7200}
REQUEST_TIMEOUT=${RESP_REQUEST_TIMEOUT:-3600} # adjust as needed for your task so that you do not hit
WORK_TIMEOUT=${RESP_WORK_TIMEOUT:-7200}   # time for dispatcher to give up on a work item and reissue it.  ideally this should never be hit.

#
# If you are changing the model, be sure to update GPUS_PER_TASK and the
# sbatch --ntasks-per-node configuration appropriately.
# Typically on Lumi 70B = 4 GPUs, 34B = 2 GPUs, 8B = 1 GPU
# --ntasks-per-node should be int(8 / GPUS_PER_TASK)
#
MODEL=${MODEL:-meta-llama/Llama-3.3-70B-Instruct}
GPUS_PER_TASK=4     # enough for the model and large batch size
MAX_MODEL_LEN=16384 # for efficiency, only as much as you think you need for efficiency

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
export PYTHONUSERBASE="$(pwd)/pythonuserbase"

module use /appl/local/csc/modulefiles
module load pytorch/2.5
source .venv/bin/activate

# pip install fasttext
# pip install git+https://github.com/LumiOpen/dispatcher.git
export HF_HOME="${HF_HOME:-/scratch/project_462000353/hf_cache}"
export SSL_CERT_FILE=$(python -m certifi)

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
    export VLLM_PORT=$(( 8000 + SLURM_LOCALID * 100 ))

    echo "Launching task $SLURM_LOCALID (global id: $SLURM_PROCID) with GPU $GPU_IDS on $(hostname)"

    module use /appl/local/csc/modulefiles
    module load pytorch/2.5
    export PYTHONUSERBASE=./pythonuserbase
    export HF_HOME="${HF_HOME:-/scratch/project_462000353/hf_cache}"

    PYTHONPATH=. python -m dispatcher.taskmanager.cli \
        --dispatcher ${DISPATCHER_SERVER}:${DISPATCHER_PORT} \
        --task '"$TASK"' \
        --batch-size 1 \
        --workers '"$WORKERS"' \
        --max-model-len '"$MAX_MODEL_LEN"' \
        --tensor-parallel '"$GPUS_PER_TASK"' \
        --model '"$MODEL"' \
        --startup-timeout '"$STARTUP_TIMEOUT"' \
        --request-timeout '"$REQUEST_TIMEOUT"'
        # --silence-vllm-logs
'

