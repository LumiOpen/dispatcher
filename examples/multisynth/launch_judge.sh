#!/bin/bash
#SBATCH --job-name=multisynt_judge
#SBATCH --nodes=1
#SBATCH --partition=dev-g
#SBATCH --time=00-02:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --mem=480G
#SBATCH --cpus-per-task=7
#SBATCH --exclusive=user
#SBATCH --hint=nomultithread
#SBATCH --gpus-per-node=mi250:8
#SBATCH --account=project_462000353
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err


###
# configure the following.

INPUT_FILE=finnish.jsonl
OUTPUT_FILE=finnish_judged.jsonl
TASK=multisynt_judge.MultiSyntJudge

# use this if you need to run from a development/feature branch
DISPATCHER_BRANCH=feature/multisynt_judge_impl2

# singularity container configuration
SINGULARITY_IMAGE=/scratch/project_462000353/containers/sif_images/rocm-6.2.4-python-3.10-pytorch-2.8.sif
#export VLLM_USE_TRITON_FLASH_ATTN=0
export VLLM_USE_V1=1


# generation parameters
# These should be tuned so that you do not overload your backend vllm server,
# or run into any timeouts.  timeouts greatly affect the efficiency of the
# workflow.
WORKERS=32          # number of simultaneous backend requests
BATCH_SIZE=1        # amount of work to request from dispatcher. 1 is usually fine.

# Timeouts are safety valves and you should not hit them in the normal course
# of your workflow.  if you do, it suggests you need to change something about
# your configuration--tasks are usually written to expect success.
REQUEST_TIMEOUT=600 # adjust as needed for your task so that you do not hit
WORK_TIMEOUT=1800   # time for dispatcher to give up on a work item and reissue it.  ideally this should never be hit.

#
# If you are changing the model, be sure to update GPUS_PER_TASK and the
# sbatch --ntasks-per-node configuration appropriately.
# Typically on Lumi 70B = 4 GPUs, 34B = 2 GPUs, 8B = 1 GPU
# --ntasks-per-node should be int(8 / GPUS_PER_TASK)
#
MODEL=google/gemma-3-27b-it
#MODEL=google/gemma-3-12b-it
GPUS_PER_TASK=2     # enough for the model and large batch size
MAX_MODEL_LEN=9216

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

pip install --force-reinstall --no-cache-dir git+https://github.com/LumiOpen/dispatcher.git@${DISPATCHER_BRANCH}

# dispatcher server will run on the first node, before we launch the worker
# tasks.
export DISPATCHER_SERVER=$(hostname)
export DISPATCHER_PORT=9999
singularity exec  \
    -B /scratch/project_462000353 \
    $SINGULARITY_IMAGE \
    bash -c "
    \$WITH_CONDA

    # variables spread across different scopes are fun.
    export PYTHONUSERBASE=${PYTHONUSERBASE}
    echo PYTHONUSERBASE is $PYTHONUSERBASE
    pip install --force-reinstall --no-cache-dir git+https://github.com/LumiOpen/dispatcher.git@$DISPATCHER_BRANCH

    # dispatcher server will run on the first node, before we launch the worker
    # tasks.
    python -m dispatcher.server \
        --infile $INPUT_FILE \
        --outfile $OUTPUT_FILE \
        --work-timeout $WORK_TIMEOUT \
        --host 0.0.0.0 \
        --port $DISPATCHER_PORT &
"



echo "Waiting for dispatcher server to start..."
sleep 10
echo "Dispatcher server should be running. Launching workers."

echo HIP=$HIP_VISIBLE_DEVICES
echo CUDA=$CUDA_VISIBLE_DEVICES
echo ROCR=$ROCR_VISIBLE_DEVICES

srun -l \
    bash -c '
    echo HIP=$HIP_VISIBLE_DEVICES
    echo CUDA=$CUDA_VISIBLE_DEVICES
    echo ROCR=$ROCR_VISIBLE_DEVICES
    # Compute the starting GPU index for this task. This logic is unchanged.
    start_gpu=$(( SLURM_LOCALID * '"$GPUS_PER_TASK"' ))
    GPU_IDS=""
    for (( i=0; i < '"$GPUS_PER_TASK"'; i++ )); do
        if [ -z "$GPU_IDS" ]; then
            GPU_IDS="$(( start_gpu + i ))"
        else
            GPU_IDS="${GPU_IDS},$(( start_gpu + i ))" 
        fi
    done
    export HIP_VISIBLE_DEVICES=$GPU_IDS
    unset CUDA_VISIBLE_DEVICES
    unset ROCR_VISIBLE_DEVICES

    # Set ports uniquely per task (to avoid collisions). Unchanged.
    export MASTER_PORT=$(( 7000 + SLURM_LOCALID ))
    export VLLM_PORT=$(( 8000 + SLURM_LOCALID * 100 ))

    echo "Launching task $SLURM_LOCALID (global id: $SLURM_PROCID) with GPU $GPU_IDS on $(hostname)"
    echo HIP=$HIP_VISIBLE_DEVICES
    echo CUDA=$CUDA_VISIBLE_DEVICES
    echo ROCR=$ROCR_VISIBLE_DEVICES

    #    --env=VLLM_USE_TRITON_FLASH_ATTN="'"$VLLM_USE_TRITON_FLASH_ATTN"'" \
    singularity exec --rocm \
        --env=MASTER_PORT=$MASTER_PORT \
        --env=VLLM_USE_v1="'"$VLLM_USE_V1"'" \
        --env=VLLM_PORT=$VLLM_PORT \
        -B /scratch/project_462000353 \
        -B "/scratch/project_462000353/jburdge/singularity/999-custom.sh:/.singularity.d/env/999-custom.sh" \
        "'"$SINGULARITY_IMAGE"'" \
        bash -c "
        $WITH_CONDA
        export PYTHONUSERBASE=./pythonuserbase

        # Make sure the dispatcher server variable is available inside this shell
        export DISPATCHER_SERVER='"$DISPATCHER_SERVER"'
        export DISPATCHER_PORT='"$DISPATCHER_PORT"'

        PYTHONPATH=. python -m dispatcher.taskmanager.cli \
            --dispatcher \${DISPATCHER_SERVER}:\${DISPATCHER_PORT} \
            --task '"$TASK"' \
            --batch-size 1 \
            --workers '"$WORKERS"' \
            --max-model-len '"$MAX_MODEL_LEN"' \
            --tensor-parallel '"$GPUS_PER_TASK"' \
            --enforce-eager \
            --silence-vllm-logs \
            --model '"$MODEL"'
        "
'

echo "All tasks finished."

